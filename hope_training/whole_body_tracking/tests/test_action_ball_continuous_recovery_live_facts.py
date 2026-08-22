from __future__ import annotations

from dataclasses import fields, replace
import math
from pathlib import Path
import sys
import types

import pytest
import torch


_WBT_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _WBT_ROOT / "source" / "whole_body_tracking"
_MDP_ROOT = _SOURCE_ROOT / "whole_body_tracking" / "tasks" / "tracking" / "mdp"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))
if str(_MDP_ROOT) not in sys.path:
    sys.path.insert(0, str(_MDP_ROOT))

import action_ball_continuous_recovery_device as recovery  # noqa: E402
import action_ball_motion_cadence_device as cadence  # noqa: E402
import test_action_ball_motion_genesis_cadence_activation as motion_test  # noqa: E402
import test_action_ball_motion_rowwise_accept_writer as motion_row  # noqa: E402
epoch_v1 = motion_test.E


UPPER = (
    "torso_Link",
    "left_shoulder_roll_Link",
    "left_elbow_Link",
    "left_wrist_yaw_Link",
    "right_shoulder_roll_Link",
    "right_elbow_Link",
    "right_wrist_yaw_Link",
)
FEET = ("left_ankle_roll_Link", "right_ankle_roll_Link")
BODY_NAMES = ("root_Link", *UPPER, *FEET)


class _Scene(types.SimpleNamespace):
    def __getitem__(self, name):
        if name == "robot":
            return self.robot
        if name == "contact_forces":
            return self.sensors["contact_forces"]
        raise KeyError(name)


def _install_live_r07_plant(motion, upper: tuple[str, ...]) -> None:
    """Install the narrow live Isaac tensor surface consumed by R07."""

    robot = motion.robot
    data = robot.data
    device = torch.device(motion.device)
    dtype = data.joint_pos.dtype
    n = motion.num_envs
    motion._env.num_envs = n
    motion._env.device = device
    starts = motion.motion.seg_start[motion.clip_id]
    env_origins = motion._env.scene.env_origins
    root_position = motion.motion.body_pos_w[starts, 0] + env_origins
    root_orientation = motion.motion.body_quat_w[starts, 0]
    joint_position = motion.motion.joint_pos[starts]
    upper_position = (
        motion.motion.body_pos_w[starts][:, 1 : 1 + len(upper)]
        + env_origins[:, None, :]
    )
    upper_orientation = motion.motion.body_quat_w[starts][
        :, 1 : 1 + len(upper)
    ]

    robot.body_names = list(BODY_NAMES)
    motion.cfg.body_names = list(BODY_NAMES)
    data.root_pos_w = root_position.detach().clone()
    data.root_quat_w = root_orientation.detach().clone()
    data.root_lin_vel_w = torch.zeros((n, 3), dtype=dtype, device=device)
    data.root_ang_vel_w = torch.zeros((n, 3), dtype=dtype, device=device)
    data.joint_pos = joint_position.detach().clone()
    data.joint_vel = torch.zeros_like(data.joint_pos)
    data.body_pos_w = torch.zeros(
        (n, len(BODY_NAMES), 3), dtype=dtype, device=device
    )
    data.body_pos_w[:, 0] = data.root_pos_w
    data.body_pos_w[:, 1 : 1 + len(upper)] = upper_position
    data.body_quat_w = torch.zeros(
        (n, len(BODY_NAMES), 4), dtype=dtype, device=device
    )
    data.body_quat_w[..., 0] = 1.0
    data.body_quat_w[:, 0] = data.root_quat_w
    data.body_quat_w[:, 1 : 1 + len(upper)] = upper_orientation
    data.body_lin_vel_w = torch.zeros_like(data.body_pos_w)
    data.body_ang_vel_w = torch.zeros_like(data.body_pos_w)

    forces = torch.zeros_like(data.body_pos_w)
    forces[:, BODY_NAMES.index(FEET[0]), 2] = 10.0
    forces[:, BODY_NAMES.index(FEET[1]), 0] = 1000.0
    sensor = types.SimpleNamespace(
        body_names=list(BODY_NAMES),
        data=types.SimpleNamespace(net_forces_w=forces),
    )
    motion._env.scene = _Scene(
        env_origins=env_origins,
        robot=robot,
        sensors={"contact_forces": sensor},
    )


def _subject(monkeypatch, *, device: torch.device):
    command, _cadence_owner, device_owner, epoch_owner = (
        motion_test._fresh_command_and_owners(device)
    )
    command.bind_action_ball_continuous_motion_device_r05_reveal(device_owner)
    command.bind_action_ball_full_mdp_motion_epoch_owner(epoch_owner)
    upper = motion_test._install_frame0_body_contract(command)
    assert upper == UPPER
    motion_test._install_fake_a3_upper_module(monkeypatch, upper)
    _install_live_r07_plant(command, upper)
    command._env.common_step_counter = 7
    command._diagnostic_test_epoch_owner = epoch_owner
    command._diagnostic_test_device_r05_owner = device_owner
    epoch_owner._diagnostic_test_parent_identity = (
        command._diagnostic_test_motion_parent_authority
        .project_bound_action_identity(
            command._diagnostic_test_motion_parent_receipt,
            motion_owner=command,
        )
    )
    robot = command.robot
    sensor = command._env.scene.sensors["contact_forces"]
    robot.data.body_lin_vel_w[:, BODY_NAMES.index(FEET[0]), :2] = torch.tensor(
        (0.25, -0.5), dtype=robot.data.root_pos_w.dtype, device=device
    )
    sensor.data.net_forces_w[:, BODY_NAMES.index(FEET[1]), 0] = 1000.0
    sensor.data.net_forces_w[:, BODY_NAMES.index(FEET[1]), 2] = 0.0
    return command._env, command, robot, sensor


def _construct_bundle(
    *,
    env,
    motion,
    action_epoch_owner,
    motion_parent_authority=None,
    motion_parent_receipt=None,
):
    if motion_parent_authority is None:
        motion_parent_authority = (
            motion._diagnostic_test_motion_parent_authority
        )
    if motion_parent_receipt is None:
        motion_parent_receipt = (
            motion._diagnostic_test_motion_parent_receipt
        )
    return recovery.construct_action_ball_full_mdp_diagnostic_n2_recovery_owner(
        env=env,
        motion_owner=motion,
        action_epoch_owner=action_epoch_owner,
        motion_parent_authority=motion_parent_authority,
        motion_parent_receipt=motion_parent_receipt,
    )


def _publish_epoch_reward_facts(
    bundle,
    motion,
    *,
    cadence_tick: int,
    current_source_step: torch.Tensor,
):
    """Publish after installing this test's explicit Motion row clock."""

    assert type(cadence_tick) is int
    motion._action_ball_continuous_episode_step.fill_(cadence_tick)
    return bundle.publish_epoch_reward_facts(
        current_source_step=current_source_step
    )


def _refresh_epoch_readiness_without_keyed_facts(
    bundle,
    motion,
    *,
    cadence_tick: int,
    current_source_step: torch.Tensor,
):
    """Refresh bootstrap readiness without an ActionEpoch fact publication."""

    assert type(cadence_tick) is int
    motion._action_ball_continuous_episode_step.fill_(cadence_tick)
    return bundle.refresh_epoch_readiness_without_keyed_facts(
        current_source_step=current_source_step
    )


def _rowwise_settled_subject(
    monkeypatch,
    *,
    device: torch.device,
    construction_admissible: bool = True,
    playback_admissible: bool = True,
    producer_fault: int = 0,
):
    """Settle row 0 through the real D05 -> Motion -> Epoch writer chain.

    Row 1 is deliberately not due.  It is the byte-preserved bootstrap peer,
    so every R07 assertion exercises completed and upcoming lifecycles in the
    same N=2 publication instead of relying on the removed scalar fixture.
    """

    parent, parent_receipt, profile = (
        cadence.build_action_ball_full_mdp_diagnostic_motion_profile()
    )
    configure_unbound = motion_row.bridge._configure_unbound_command
    monkeypatch.setattr(
        motion_row.bridge,
        "_configure_unbound_command",
        lambda *, num_envs=1, profile=None: configure_unbound(
            num_envs=num_envs, profile=profile_for_construction
        ),
    )
    profile_for_construction = profile
    (
        motion,
        _d05_owner,
        epoch_owner,
        token,
        row_record,
        _racket_peer,
        _physical_peer,
    ) = motion_row._install_real_d05_record(
        device=device,
        corrupt_accept_mask=False,
    )
    parent.bind_exact_parent_schedule(motion, parent_receipt)
    motion._diagnostic_test_motion_parent_authority = parent
    motion._diagnostic_test_motion_parent_receipt = parent_receipt
    parent_identity = parent.project_bound_action_identity(
        parent_receipt, motion_owner=motion
    )
    epoch_owner._diagnostic_test_parent_identity = parent_identity
    row_record.candidate.identity.shot_key.action_uid[0, 0] = (
        parent_identity.action_uid
    )
    if not construction_admissible:
        row_record.candidate.construction_admissible[0, 0] = False
        row_record.accept_mask[0] = False
    if not playback_admissible:
        row_record.candidate.playback_admissible[0, 0] = False
        row_record.accept_mask[0] = False
    if producer_fault:
        r05_slot = epoch_v1.OWNER_ORDER.index("r05_runtime")
        row_record.candidate.owner_fault_bits[0, 0, r05_slot] = producer_fault
        row_record.accept_mask[0] = False

    upper = motion_test._install_frame0_body_contract(motion)
    assert upper == UPPER
    motion_test._install_fake_a3_upper_module(monkeypatch, upper)
    _install_live_r07_plant(motion, upper)
    motion._env.common_step_counter = 40
    robot = motion.robot
    sensor = motion._env.scene.sensors["contact_forces"]
    robot.data.body_lin_vel_w[:, BODY_NAMES.index(FEET[0]), :2] = torch.tensor(
        (0.25, -0.5), dtype=robot.data.root_pos_w.dtype, device=device
    )
    sensor.data.net_forces_w[:, BODY_NAMES.index(FEET[1]), 0] = 1000.0
    sensor.data.net_forces_w[:, BODY_NAMES.index(FEET[1]), 2] = 0.0

    # The focused Motion fixture has already armed its private row token.  Put
    # that token aside only while the real R07 owner performs the mandatory
    # canonical-genesis cold bind, then restore and settle the same token.
    active_d05 = epoch_owner._active_d05
    epoch_owner._active_d05 = None
    try:
        bundle = _construct_bundle(
            env=motion._env,
            motion=motion,
            action_epoch_owner=epoch_owner,
        )
    finally:
        epoch_owner._active_d05 = active_d05
    epoch_owner.settle_d05_transaction(token)
    return motion._env, motion, robot, sensor, bundle, epoch_owner


def _set_current_phase_and_deadline(
    epoch_owner,
    *,
    phase: int,
    deadline_tick: int,
) -> None:
    """Install one controlled lifecycle cell without changing its real key."""

    publication = epoch_owner._publication
    record = publication.current
    row_phase = record.phase.clone()
    row_deadline = record.clocks.deadline_tick.clone()
    slot = int(record.current_task_slot[0])
    row_phase[0, slot] = phase
    row_deadline[0, slot] = deadline_tick
    epoch_owner._publication = epoch_v1._Publication(
        replace(
            record,
            phase=row_phase,
            clocks=replace(record.clocks, deadline_tick=row_deadline),
        ),
        publication.pending_log,
    )


def _ready_reference(
    *,
    motion,
    epoch_owner,
    robot,
    reference_kind: int = recovery.R07_REFERENCE_COMPLETED_ACTION_FRAME0,
    reference_action_slot=None,
    reference_action_uid=None,
):
    parent_identity = epoch_owner._diagnostic_test_parent_identity
    if reference_action_slot is None:
        reference_action_slot = parent_identity.action_slot
    if reference_action_uid is None:
        reference_action_uid = parent_identity.action_uid
    produced = motion.project_action_ball_full_mdp_recovery_ready_reference()
    payload = {
        name: (value.detach().clone() if torch.is_tensor(value) else value)
        for name, value in vars(produced).items()
        if name != "shot_key"
    }
    payload["shot_key"] = produced.shot_key.clone()
    payload["reference_kind"] = torch.full_like(
        produced.reference_kind, reference_kind
    )
    payload["reference_action_slot"] = torch.full_like(
        produced.reference_action_slot, reference_action_slot
    )
    payload["reference_action_uid"] = torch.full_like(
        produced.reference_action_uid, reference_action_uid
    )
    return types.SimpleNamespace(**payload)


@pytest.mark.parametrize("device_name", ("cpu", "cuda:0"))
def test_diagnostic_n2_live_adapter_reads_same_tick_real_channels(
    monkeypatch, device_name
):
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    device = torch.device(device_name)
    env, motion, robot, sensor = _subject(monkeypatch, device=device)

    bundle = _construct_bundle(
        env=env,
        motion=motion,
        action_epoch_owner=motion._diagnostic_test_epoch_owner,
    )
    assert type(bundle) is recovery.DiagnosticN2ContinuousRecoveryBundle
    assert type(bundle.owner) is recovery.ContinuousRecoveryDeviceCoordinator
    assert bundle.owner.num_envs == 2
    assert bundle.owner.profile.ordered_joint_names == tuple(
        robot.data.joint_names
    )
    assert bundle.owner.profile.ordered_body_names == UPPER
    assert bundle.owner.profile.ordered_foot_names == FEET
    assert recovery.DIAGNOSTIC_UNAUTHORIZED is True
    assert recovery.RUNTIME_WIRING_CONNECTED is False
    assert recovery.LAUNCH_AUTHORIZED is False

    cold = bundle.action_epoch_observation_state()
    assert type(cold) is recovery.ContinuousRecoveryObservationState
    assert cold.postphysics_valid.tolist() == [False, False]
    assert cold.source_step.tolist() == [-1, -1]
    assert cold.reset_generation.tolist() == [-1, -1]
    assert cold.control_tick.tolist() == [-1, -1]
    assert cold.ready_streak.tolist() == [0, 0]
    assert not cold.foot_supported_lr.any()

    facts = bundle.plant_fact_adapter.read()
    assert type(facts) is recovery.DeviceContinuousRecoveryPlantFacts
    assert bundle.plant_fact_adapter.last_source_step == 7
    assert facts.body_position_m.shape == (2, len(UPPER), 3)
    assert facts.foot_contact_signal.tolist() == [[10.0, 0.0], [10.0, 0.0]]
    assert facts.foot_slip_velocity_xy_mps[:, 0].tolist() == [
        [0.25, -0.5],
        [0.25, -0.5],
    ]
    assert facts.facts_valid.tolist() == [True, True]
    assert facts.hard_safety_ok.tolist() == [True, True]

    # A finite fall is valid learning data, not an infrastructure fault.
    robot.data.root_pos_w[0, 2] = 0.49
    env.common_step_counter += 1
    fallen = bundle.plant_fact_adapter.read()
    assert fallen.facts_valid.tolist() == [True, True]
    assert fallen.hard_safety_ok.tolist() == [False, True]

    # Non-finite live state is an invalid producer row and cannot become ready.
    robot.data.joint_vel[1, 0] = math.nan
    env.common_step_counter += 1
    invalid = bundle.plant_fact_adapter.read()
    assert invalid.facts_valid.tolist() == [True, False]
    assert invalid.hard_safety_ok.tolist() == [False, False]

    env.scene.sensors["contact_forces"] = types.SimpleNamespace(
        body_names=sensor.body_names, data=sensor.data
    )
    with pytest.raises(
        recovery.ContinuousRecoveryDeviceError, match="identity changed"
    ):
        bundle.plant_fact_adapter.read()


def test_diagnostic_n2_constructor_rejects_missing_live_fact_sources(monkeypatch):
    env, motion, _robot, sensor = _subject(monkeypatch, device=torch.device("cpu"))
    env.scene.sensors["contact_forces"] = None
    with pytest.raises(
        recovery.ContinuousRecoveryConstructionHold,
        match="contact_sensor_identity_is_ambiguous|live_contact_forces_sensor_absent",
    ):
        _construct_bundle(
            env=env,
            motion=motion,
            action_epoch_owner=motion._diagnostic_test_epoch_owner,
        )

    env, motion, _robot, sensor = _subject(monkeypatch, device=torch.device("cpu"))
    sensor.body_names = tuple(name for name in BODY_NAMES if name != FEET[1])
    with pytest.raises(
        recovery.ContinuousRecoveryConstructionHold,
        match="A3_foot_missing_from_live_contact_sensor",
    ):
        _construct_bundle(
            env=env,
            motion=motion,
            action_epoch_owner=motion._diagnostic_test_epoch_owner,
        )


def test_diagnostic_n2_constructor_rejects_foreign_parent_or_receipt(
    monkeypatch,
):
    env, motion, _robot, _sensor = _subject(
        monkeypatch, device=torch.device("cpu")
    )
    epoch_owner = motion._diagnostic_test_epoch_owner
    bound_parent = motion._diagnostic_test_motion_parent_authority
    foreign_parent, foreign_receipt, _profile = (
        cadence.build_action_ball_full_mdp_diagnostic_motion_profile()
    )
    with pytest.raises(
        recovery.ContinuousRecoveryConstructionHold,
        match="exact_Motion_parent_projection_failed",
    ):
        _construct_bundle(
            env=env,
            motion=motion,
            action_epoch_owner=epoch_owner,
            motion_parent_authority=bound_parent,
            motion_parent_receipt=foreign_receipt,
        )
    with pytest.raises(
        recovery.ContinuousRecoveryConstructionHold,
        match="exact_Motion_parent_projection_failed",
    ):
        _construct_bundle(
            env=env,
            motion=motion,
            action_epoch_owner=epoch_owner,
            motion_parent_authority=foreign_parent,
            motion_parent_receipt=foreign_receipt,
        )


def test_equal_private_motion_schedule_replacement_cannot_change_parent_truth(
    monkeypatch,
):
    device = torch.device("cpu")
    env, motion, robot, _sensor = _subject(monkeypatch, device=device)
    env.common_step_counter = 0
    epoch_owner = motion._diagnostic_test_epoch_owner
    bundle = _construct_bundle(
        env=env, motion=motion, action_epoch_owner=epoch_owner
    )
    motion._action_ball_continuous_schedule_projection = types.MappingProxyType(
        dict(motion._action_ball_continuous_schedule_projection)
    )
    result = _publish_epoch_reward_facts(
        bundle,
        motion,
        cadence_tick=0,
        current_source_step=torch.zeros(2, dtype=torch.int64)
    )
    assert result.facts_valid.tolist() == [True, True]
    assert result.producer_fault_bits.tolist() == [0, 0]


@pytest.mark.parametrize("device_name", ("cpu", "cuda:0"))
def test_r07_reveal_keeps_fall_and_low_support_numeric_but_unpaid(
    monkeypatch, device_name
):
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    device = torch.device(device_name)
    env, motion, robot, _sensor, bundle, epoch_owner = (
        _rowwise_settled_subject(monkeypatch, device=device)
    )
    completed = epoch_owner.current()
    completed_slots = completed.current_task_slot
    completed_envs = torch.arange(2, dtype=torch.int64, device=device)
    selected_action_slots = completed.identity.action_slot[
        completed_envs, completed_slots
    ]
    selected_action_uids = completed.identity.action_uid[
        completed_envs, completed_slots
    ]
    assert selected_action_slots.tolist() == [motion.clip_id[0].item(), -1]
    assert selected_action_uids.tolist() == [
        bundle._project_parent_action_identity().action_uid,
        -1,
    ]

    result = _publish_epoch_reward_facts(
        bundle,
        motion,
        cadence_tick=40,
        current_source_step=torch.full(
            (2,), 40, dtype=torch.int64, device=device
        )
    )
    assert result.facts_valid.tolist() == [True, True]
    assert result.infrastructure_fault.tolist() == [False, False]
    assert result.producer_fault_bits.tolist() == [0, 0]
    assert result.ready_instant.tolist() == [False, False]
    assert result.recovery_age_tick.tolist() == [0, -1]
    assert result.reward_eligible.tolist() == [False, False]
    assert torch.isfinite(result.weighted_reward).all()
    assert torch.all(result.weighted_reward == 0)

    record = epoch_owner.current()
    owner_slot = epoch_v1.OWNER_ORDER.index("r07_recovery")
    assert record.fact_valid_bits[:, 0, owner_slot].tolist() == [3, 0]
    assert record.owner_fault_bits[:, 0, owner_slot].tolist() == [0, 0]
    assert record.fact_f32[0, 0, owner_slot, 0] == result.weighted_reward[0]
    assert record.fact_f32[1, 0, owner_slot, 0] == 0.0

    # A finite fall remains a low-value sample, not an infrastructure fault.
    robot.data.root_pos_w[0, 2] = 0.49
    env.common_step_counter = 41
    fallen = _publish_epoch_reward_facts(
        bundle,
        motion,
        cadence_tick=41,
        current_source_step=torch.full(
            (2,), 41, dtype=torch.int64, device=device
        )
    )
    assert fallen.facts_valid.tolist() == [True, True]
    assert fallen.infrastructure_fault.tolist() == [False, False]
    assert fallen.ready_instant.tolist() == [False, False]
    assert fallen.recovery_age_tick.tolist() == [1, -1]
    assert fallen.reward_eligible.tolist() == [False, False]
    assert torch.isfinite(fallen.weighted_reward).all()


@pytest.mark.parametrize(
    ("phase", "age", "expected_eligible"),
    (
        (epoch_v1.PHASE_REVEAL_COMMITTED, 10, False),
        (epoch_v1.PHASE_LAUNCH_SETTLED, 10, False),
        (epoch_v1.PHASE_OUTCOME_SETTLED, 9, False),
        (epoch_v1.PHASE_OUTCOME_SETTLED, 10, True),
        (epoch_v1.PHASE_OUTCOME_SETTLED, 77, True),
        (epoch_v1.PHASE_OUTCOME_SETTLED, 78, False),
        (epoch_v1.PHASE_RETIRED, 77, False),
    ),
    ids=(
        "reveal_age10",
        "launch_age10",
        "outcome_age9",
        "outcome_age10",
        "outcome_age77",
        "outcome_age78",
        "retired_age77",
    ),
)
def test_r07_reward_eligibility_is_exact_post_outcome_window(
    monkeypatch, phase, age, expected_eligible
):
    device = torch.device("cpu")
    env, motion, _robot, _sensor, bundle, epoch_owner = (
        _rowwise_settled_subject(monkeypatch, device=device)
    )
    source_step = 100
    _set_current_phase_and_deadline(
        epoch_owner,
        phase=phase,
        deadline_tick=source_step - age,
    )
    env.common_step_counter = source_step
    result = _publish_epoch_reward_facts(
        bundle,
        motion,
        cadence_tick=source_step,
        current_source_step=torch.full(
            (2,), source_step, dtype=torch.int64, device=device
        ),
    )
    assert result.facts_valid.tolist() == [True, True]
    assert result.recovery_age_tick.tolist() == [age, -1]
    assert result.reward_eligible.tolist() == [expected_eligible, False]
    assert (result.weighted_reward[0] > 0).item() is expected_eligible
    assert result.weighted_reward[1].item() == 0.0

    record = epoch_owner.current()
    owner_slot = epoch_v1.OWNER_ORDER.index("r07_recovery")
    expected_present = phase != epoch_v1.PHASE_RETIRED
    assert bool(record.fact_valid_bits[0, 0, owner_slot].ne(0)) is expected_present
    assert record.fact_f32[0, 0, owner_slot, 2].item() == float(
        expected_eligible
    )
    # The non-due IDLE peer is never turned into an R07 payment row.
    assert record.fact_valid_bits[1, 0, owner_slot].item() == 0
    assert record.fact_f32[1, 0, owner_slot].eq(0).all()


def test_r07_epoch_direct_publish_marks_nonfinite_plant_as_typed_fault(monkeypatch):
    device = torch.device("cpu")
    env, motion, robot, _sensor, bundle, epoch_owner = (
        _rowwise_settled_subject(monkeypatch, device=device)
    )
    robot.data.joint_vel[1, 0] = math.nan
    result = _publish_epoch_reward_facts(
        bundle,
        motion,
        cadence_tick=40,
        current_source_step=torch.full(
            (2,), 40, dtype=torch.int64, device=device
        )
    )
    assert result.facts_valid.tolist() == [True, False]
    assert result.infrastructure_fault.tolist() == [False, True]
    assert result.producer_fault_bits[1].item() & recovery.R07_EPOCH_FAULT_INVALID_PLANT_FACT
    assert result.weighted_reward[1].item() == 0.0


@pytest.mark.parametrize("device_name", ("cpu", "cuda:0"))
def test_cold_idle_bootstrap_requires_two_real_facts_for_control_two_ready(
    monkeypatch, device_name
):
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    device = torch.device(device_name)
    env, motion, robot, sensor = _subject(monkeypatch, device=device)
    env.common_step_counter = 0
    sensor.data.net_forces_w.zero_()
    for name in FEET:
        sensor.data.net_forces_w[:, BODY_NAMES.index(name), 2] = 10.0
    robot.data.body_lin_vel_w.zero_()
    epoch_owner = motion._diagnostic_test_epoch_owner
    bundle = _construct_bundle(
        env=env, motion=motion, action_epoch_owner=epoch_owner
    )
    commit_head = epoch_owner.commit_head
    original_current = epoch_v1.ActionEpochOwner.current

    def reject_full_epoch_clone():
        raise AssertionError("bootstrap readiness cloned the full ActionEpoch")

    monkeypatch.setattr(epoch_owner, "current", reject_full_epoch_clone)

    def reject_keyed_write(*_args, **_kwargs):
        raise AssertionError("bootstrap readiness reached a keyed epoch write")

    monkeypatch.setattr(
        epoch_owner, "merge_runtime_owner_fault", reject_keyed_write
    )
    monkeypatch.setattr(epoch_owner, "publish_owner_facts", reject_keyed_write)
    monkeypatch.setattr(
        epoch_owner, "publish_r07_first_ready", reject_keyed_write
    )
    first = _refresh_epoch_readiness_without_keyed_facts(
        bundle,
        motion,
        cadence_tick=0,
        current_source_step=torch.zeros(2, dtype=torch.int64, device=device)
    )
    first_view = bundle.require_owned_motion_ready_projection(
        bundle.motion_ready_projection(), owner_kind="motion"
    )
    assert first.ready_instant.tolist() == [True, True]
    assert first.reference_kind.tolist() == [1, 1]
    assert first.reward_eligible.tolist() == [False, False]
    assert first.weighted_reward.eq(0).all()
    assert first_view.ready.tolist() == [False, False]
    assert first_view.ready_streak.tolist() == [1, 1]
    assert first_view.required_dwell == 2
    assert first_view.control_tick.tolist() == [1, 1]
    assert epoch_owner.commit_head == commit_head
    # Genesis has no action row that could own reward facts.  Bootstrap
    # readiness therefore advances only the R07 owner-private dwell state.
    record = original_current(epoch_owner)
    owner_slot = epoch_v1.OWNER_ORDER.index("r07_recovery")
    assert record.fact_valid_bits[:, :, owner_slot].eq(0).all()
    assert record.owner_fault_bits[:, :, owner_slot].eq(0).all()

    env.common_step_counter = 1
    second = _refresh_epoch_readiness_without_keyed_facts(
        bundle,
        motion,
        cadence_tick=1,
        current_source_step=torch.ones(2, dtype=torch.int64, device=device)
    )
    second_view = bundle.require_owned_motion_ready_projection(
        bundle.motion_ready_projection(), owner_kind="motion"
    )
    assert second.ready_instant.tolist() == [True, True]
    assert second.reward_eligible.tolist() == [False, False]
    assert second.weighted_reward.eq(0).all()
    assert second_view.ready.tolist() == [True, True]
    assert second_view.ready_streak.tolist() == [2, 2]
    assert second_view.required_dwell == 2
    assert second_view.control_tick.tolist() == [2, 2]
    assert bundle.owner._action_epoch_ready_streak.tolist() == [2, 2]
    assert epoch_owner.commit_head == commit_head
    # Bootstrap readiness authorizes the next Motion reveal but has no current
    # full shot key, so it must not poison ActionEpoch by publishing keyed R07
    # telemetry against the neutral genesis rows.
    assert not bool(epoch_owner._undecoded_overflow.any())
    record = original_current(epoch_owner)
    owner_slot = epoch_v1.OWNER_ORDER.index("r07_recovery")
    assert record.fact_valid_bits[:, :, owner_slot].eq(0).all()
    assert record.owner_fault_bits[:, :, owner_slot].eq(0).all()


def test_bootstrap_fastpath_matches_generic_no_key_fixed_tape(monkeypatch):
    device = torch.device("cpu")

    def subject():
        env, motion, robot, sensor = _subject(monkeypatch, device=device)
        env.common_step_counter = 0
        sensor.data.net_forces_w.zero_()
        for name in FEET:
            sensor.data.net_forces_w[:, BODY_NAMES.index(name), 2] = 10.0
        robot.data.body_lin_vel_w.zero_()
        epoch_owner = motion._diagnostic_test_epoch_owner
        bundle = _construct_bundle(
            env=env, motion=motion, action_epoch_owner=epoch_owner
        )
        motion._action_ball_continuous_episode_step.zero_()
        return motion, bundle, epoch_owner

    old_motion, old_bundle, old_epoch = subject()
    new_motion, new_bundle, new_epoch = subject()
    old_before = old_epoch.current()
    new_before = new_epoch.current()
    source_step = torch.zeros(2, dtype=torch.int64, device=device)
    generic = old_bundle._publish_epoch_reward_facts(
        current_source_step=source_step,
        publish_keyed_action_epoch=False,
    )
    fast = new_bundle.refresh_epoch_readiness_without_keyed_facts(
        current_source_step=source_step,
    )
    for field in fields(recovery.R07EpochDirectRewardFacts):
        old_value = getattr(generic, field.name)
        new_value = getattr(fast, field.name)
        assert torch.equal(old_value, new_value), field.name

    generic_ready = old_bundle.require_owned_motion_ready_projection(
        old_bundle.motion_ready_projection(), owner_kind="motion"
    )
    fast_ready = new_bundle.require_owned_motion_ready_projection(
        new_bundle.motion_ready_projection(), owner_kind="motion"
    )
    for name in ("ready", "ready_streak", "control_tick"):
        assert torch.equal(
            getattr(generic_ready, name), getattr(fast_ready, name)
        ), name
    assert generic_ready.required_dwell == fast_ready.required_dwell
    assert old_bundle.owner._mutation_version == new_bundle.owner._mutation_version
    assert torch.equal(
        old_bundle.owner._ready_instant_total,
        new_bundle.owner._ready_instant_total,
    )
    assert torch.equal(
        old_bundle.owner._first_ready_total,
        new_bundle.owner._first_ready_total,
    )

    assert old_epoch.commit_head == new_epoch.commit_head == 1
    old_after = old_epoch.current()
    new_after = new_epoch.current()
    for name in (
        "owner_fault_bits",
        "fact_valid_bits",
        "fact_source_step",
        "fact_f32",
        "writes_started",
        "writes_committed",
    ):
        assert torch.equal(getattr(old_before, name), getattr(old_after, name))
        assert torch.equal(getattr(new_before, name), getattr(new_after, name))
    for key_field in fields(epoch_v1.ActionEpochShotKey):
        assert torch.equal(
            getattr(old_before.identity.shot_key, key_field.name),
            getattr(old_after.identity.shot_key, key_field.name),
        )
        assert torch.equal(
            getattr(new_before.identity.shot_key, key_field.name),
            getattr(new_after.identity.shot_key, key_field.name),
        )
    assert old_motion is not new_motion


def test_bootstrap_motion_reference_is_real_frame0_and_has_no_live_alias(
    monkeypatch,
):
    device = torch.device("cpu")
    _env, motion, _robot, _sensor = _subject(monkeypatch, device=device)
    reference = motion.project_action_ball_full_mdp_bootstrap_ready_reference()
    starts = motion.motion.seg_start[motion.clip_id]
    expected_root = (
        motion.motion.body_pos_w[starts, 0] + motion._env.scene.env_origins
    )
    assert torch.equal(reference.root_position_m, expected_root)
    assert torch.equal(reference.joint_position_rad, motion.motion.joint_pos[starts])
    assert reference.cadence_tick.data_ptr() != (
        motion._action_ball_continuous_episode_step.data_ptr()
    )
    assert reference.root_position_m.data_ptr() != motion.motion.body_pos_w.data_ptr()
    assert reference.joint_position_rad.data_ptr() != motion.motion.joint_pos.data_ptr()
    assert reference.body_position_m.data_ptr() != motion.motion.body_pos_w.data_ptr()
    before = motion.motion.joint_pos.clone()
    reference.joint_position_rad.add_(1.0)
    assert torch.equal(motion.motion.joint_pos, before)


def test_bootstrap_fastpath_subset_reset_restarts_only_selected_dwell(
    monkeypatch,
):
    device = torch.device("cpu")
    env, motion, robot, sensor = _subject(monkeypatch, device=device)
    sensor.data.net_forces_w.zero_()
    for name in FEET:
        sensor.data.net_forces_w[:, BODY_NAMES.index(name), 2] = 10.0
    robot.data.body_lin_vel_w.zero_()
    epoch_owner = motion._diagnostic_test_epoch_owner
    bundle = _construct_bundle(
        env=env, motion=motion, action_epoch_owner=epoch_owner
    )

    env.common_step_counter = 0
    first = _refresh_epoch_readiness_without_keyed_facts(
        bundle,
        motion,
        cadence_tick=0,
        current_source_step=torch.zeros(2, dtype=torch.int64),
    )
    assert first.reset_generation.tolist() == [0, 0]
    assert bundle.owner._action_epoch_ready_streak.tolist() == [1, 1]

    record = epoch_owner._publication.current
    assert record is not None
    reset_generation = record.reset_generation.clone()
    reset_generation[0] += 1
    selected = torch.tensor([True, False], dtype=torch.bool, device=device)
    reset_record = replace(
        record,
        version=record.version + 1,
        reset_generation=reset_generation,
        reset_selected_mask=selected,
    )
    epoch_owner._publication = replace(
        epoch_owner._publication, current=reset_record
    )
    epoch_owner._reset_generation = reset_record.reset_generation
    epoch_owner._commit_head += 1

    env.common_step_counter = 1
    second = _refresh_epoch_readiness_without_keyed_facts(
        bundle,
        motion,
        cadence_tick=1,
        current_source_step=torch.ones(2, dtype=torch.int64),
    )
    assert second.reset_generation.tolist() == [1, 0]
    ready = bundle.require_owned_motion_ready_projection(
        bundle.motion_ready_projection(), owner_kind="motion"
    )
    assert ready.ready_streak.tolist() == [1, 2]
    assert ready.ready.tolist() == [False, True]


@pytest.mark.parametrize(
    ("malformation", "expected_r07_fault"),
    (
        (
            "phase",
            recovery.R07_EPOCH_FAULT_BOOTSTRAP_SLOT_OR_PHASE,
        ),
        (
            "key",
            recovery.R07_EPOCH_FAULT_BOOTSTRAP_NONNEUTRAL_KEY,
        ),
        (
            "writer",
            recovery.R07_EPOCH_FAULT_BOOTSTRAP_DIRTY_WRITER,
        ),
    ),
)
def test_bootstrap_fastpath_preserves_malformed_epoch_fault_telemetry(
    monkeypatch, malformation, expected_r07_fault
):
    device = torch.device("cpu")
    env, motion, robot, sensor = _subject(monkeypatch, device=device)
    env.common_step_counter = 0
    sensor.data.net_forces_w.zero_()
    for name in FEET:
        sensor.data.net_forces_w[:, BODY_NAMES.index(name), 2] = 10.0
    robot.data.body_lin_vel_w.zero_()
    epoch_owner = motion._diagnostic_test_epoch_owner
    bundle = _construct_bundle(
        env=env, motion=motion, action_epoch_owner=epoch_owner
    )
    record = epoch_owner._publication.current
    assert record is not None
    if malformation == "phase":
        phase = record.phase.clone()
        phase[0, 0] = epoch_v1.PHASE_REVEAL_COMMITTED
        record = replace(record, phase=phase)
    elif malformation == "key":
        key = record.identity.shot_key.clone()
        key.action_uid[0, 0] = 21
        record = replace(
            record,
            identity=replace(record.identity, shot_key=key),
        )
    else:
        writes_started = record.writes_started.clone()
        writes_started[
            0, 0, epoch_v1.OWNER_ORDER.index("motion")
        ] = True
        record = replace(record, writes_started=writes_started)
    epoch_owner._publication = replace(epoch_owner._publication, current=record)

    result = _refresh_epoch_readiness_without_keyed_facts(
        bundle,
        motion,
        cadence_tick=0,
        current_source_step=torch.zeros(2, dtype=torch.int64),
    )
    assert result.facts_valid.tolist() == [False, True]
    assert result.ready_instant.tolist() == [False, True]
    assert int(result.producer_fault_bits[0]) & expected_r07_fault
    assert int(result.producer_fault_bits[0]) & (
        recovery.R07_EPOCH_FAULT_INVALID_REFERENCE
    )
    assert int(result.producer_fault_bits[1]) == 0


def test_bootstrap_view_rejects_snapshot_when_epoch_advances(monkeypatch):
    device = torch.device("cpu")
    env, motion, _robot, _sensor = _subject(monkeypatch, device=device)
    env.common_step_counter = 0
    epoch_owner = motion._diagnostic_test_epoch_owner
    bundle = _construct_bundle(
        env=env, motion=motion, action_epoch_owner=epoch_owner
    )
    epoch_facts = epoch_owner.snapshot_bootstrap_readiness_facts(owner=bundle)
    reference = motion.project_action_ball_full_mdp_bootstrap_ready_reference()
    facts = bundle.plant_fact_adapter.read()
    epoch_owner._commit_head += 1
    with pytest.raises(
        recovery.ContinuousRecoveryDeviceError,
        match="bootstrap epoch facts differ",
    ):
        bundle.owner.action_epoch_bootstrap_readiness_view(
            facts,
            reference=reference,
            epoch_facts=epoch_facts,
            current_source_step=torch.zeros(2, dtype=torch.int64),
            adapter_source_step=0,
            motion_owner=motion,
            action_epoch_owner=epoch_owner,
        )


def test_epoch_publish_reads_one_action_epoch_snapshot(monkeypatch):
    device = torch.device("cpu")
    env, motion, robot, sensor, bundle, epoch_owner = (
        _rowwise_settled_subject(monkeypatch, device=device)
    )
    sensor.data.net_forces_w.zero_()
    for name in FEET:
        sensor.data.net_forces_w[:, BODY_NAMES.index(name), 2] = 10.0
    robot.data.body_lin_vel_w.zero_()

    before_head = epoch_owner.commit_head
    current_calls = []
    original_current = epoch_v1.ActionEpochOwner.current

    def counted_current():
        current_calls.append(epoch_owner.commit_head)
        return original_current(epoch_owner)

    monkeypatch.setattr(epoch_owner, "current", counted_current)
    result = _publish_epoch_reward_facts(
        bundle,
        motion,
        cadence_tick=40,
        current_source_step=torch.full((2,), 40, dtype=torch.int64),
    )
    assert current_calls == [before_head]
    assert result.facts_valid.tolist() == [True, True]


def test_reward_view_rejects_internally_consistent_stale_epoch_reference(
    monkeypatch,
):
    device = torch.device("cpu")
    env, motion, _robot, _sensor, bundle, epoch_owner = (
        _rowwise_settled_subject(monkeypatch, device=device)
    )
    env.common_step_counter = 40
    motion._action_ball_continuous_episode_step.fill_(40)
    stale_epoch = epoch_owner.current()
    stale_reference = (
        motion.project_action_ball_full_mdp_recovery_ready_reference(
            action_epoch_snapshot=stale_epoch
        )
    )
    facts = bundle.plant_fact_adapter.read()
    epoch_owner.merge_runtime_owner_fault(
        "r07_recovery",
        torch.zeros(
            (epoch_owner.num_envs, epoch_owner.shot_slot_capacity),
            dtype=torch.int64,
            device=device,
        ),
        owner=bundle,
    )

    with pytest.raises(
        recovery.ContinuousRecoveryDeviceError,
        match="snapshot/reference is stale or foreign",
    ):
        bundle.owner.action_epoch_reward_view(
            facts,
            reference=stale_reference,
            epoch=stale_epoch,
            current_source_step=torch.full((2,), 40, dtype=torch.int64),
            adapter_source_step=40,
            motion_owner=motion,
            action_epoch_owner=epoch_owner,
        )


def test_epoch_publish_rejects_owner_advance_before_first_mutation(monkeypatch):
    device = torch.device("cpu")
    env, motion, robot, sensor, bundle, epoch_owner = (
        _rowwise_settled_subject(monkeypatch, device=device)
    )
    sensor.data.net_forces_w.zero_()
    for name in FEET:
        sensor.data.net_forces_w[:, BODY_NAMES.index(name), 2] = 10.0
    robot.data.body_lin_vel_w.zero_()
    before_head = epoch_owner.commit_head
    original_view = bundle.owner.action_epoch_reward_view

    def advancing_view(*args, **kwargs):
        result = original_view(*args, **kwargs)
        epoch_owner.merge_runtime_owner_fault(
            "r07_recovery",
            torch.zeros(
                (epoch_owner.num_envs, epoch_owner.shot_slot_capacity),
                dtype=torch.int64,
                device=device,
            ),
            owner=bundle,
        )
        return result

    monkeypatch.setattr(bundle.owner, "action_epoch_reward_view", advancing_view)
    with pytest.raises(
        recovery.ContinuousRecoveryDeviceError,
        match="snapshot advanced before publication",
    ):
        _publish_epoch_reward_facts(
            bundle,
            motion,
            cadence_tick=40,
            current_source_step=torch.full((2,), 40, dtype=torch.int64),
        )
    assert epoch_owner.commit_head == before_head + 1


def test_cold_idle_bad_reference_does_not_advance_or_self_authorize(monkeypatch):
    device = torch.device("cpu")
    env, motion, robot, sensor = _subject(monkeypatch, device=device)
    env.common_step_counter = 0
    sensor.data.net_forces_w.zero_()
    for name in FEET:
        sensor.data.net_forces_w[:, BODY_NAMES.index(name), 2] = 10.0
    robot.data.body_lin_vel_w.zero_()
    epoch_owner = motion._diagnostic_test_epoch_owner
    bundle = _construct_bundle(
        env=env, motion=motion, action_epoch_owner=epoch_owner
    )
    original_joint_frames = motion.motion.joint_pos.detach().clone()
    starts = motion.motion.seg_start[motion.clip_id]
    motion.motion.joint_pos[starts, 0] = math.nan

    invalid = _publish_epoch_reward_facts(
        bundle,
        motion,
        cadence_tick=0,
        current_source_step=torch.zeros(2, dtype=torch.int64)
    )
    invalid_view = bundle.require_owned_motion_ready_projection(
        bundle.motion_ready_projection(), owner_kind="motion"
    )
    assert invalid.facts_valid.tolist() == [False, False]
    assert torch.all(
        torch.bitwise_and(
            invalid.producer_fault_bits,
            recovery.R07_EPOCH_FAULT_INVALID_REFERENCE,
        ).ne(0)
    )
    assert bundle.owner._action_epoch_ready_streak.tolist() == [0, 0]
    assert invalid_view.ready.tolist() == [False, False]

    motion.motion.joint_pos.copy_(original_joint_frames)
    env.common_step_counter = 1
    _publish_epoch_reward_facts(
        bundle,
        motion,
        cadence_tick=1,
        current_source_step=torch.ones(2, dtype=torch.int64)
    )
    assert bundle.owner._action_epoch_ready_streak.tolist() == [1, 1]


@pytest.mark.parametrize("device_name", ("cpu", "cuda:0"))
@pytest.mark.parametrize(
    ("construction_admissible", "playback_admissible"),
    (
        (False, True),
        (True, False),
    ),
    ids=("construction_reject", "playback_defer"),
)
def test_nonaccepted_event_keeps_public_idle_bootstrap_r07_reference(
    monkeypatch,
    device_name,
    construction_admissible,
    playback_admissible,
):
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    device = torch.device(device_name)
    env, motion, robot, sensor, bundle, epoch_owner = (
        _rowwise_settled_subject(
            monkeypatch,
            device=device,
            construction_admissible=construction_admissible,
            playback_admissible=playback_admissible,
        )
    )
    env.common_step_counter = 0
    sensor.data.net_forces_w.zero_()
    for name in FEET:
        sensor.data.net_forces_w[:, BODY_NAMES.index(name), 2] = 10.0
    robot.data.body_lin_vel_w.zero_()
    record = epoch_owner.current()
    motion_slot = epoch_v1.OWNER_ORDER.index("motion")
    assert torch.all(record.phase.eq(epoch_v1.PHASE_IDLE))
    assert not torch.any(record.writes_started[:, :, motion_slot])
    assert not torch.any(record.writes_committed[:, :, motion_slot])
    env_ids = torch.arange(2, dtype=torch.int64, device=device)
    selected_slots = record.current_task_slot
    selected_uids = record.identity.action_uid[env_ids, selected_slots]
    assert selected_uids.tolist() == [-1, -1]
    result = _publish_epoch_reward_facts(
        bundle,
        motion,
        cadence_tick=0,
        current_source_step=torch.zeros(
            2, dtype=torch.int64, device=device
        )
    )
    assert result.reference_kind.tolist() == [1, 1]
    assert result.facts_valid.tolist() == [True, True]
    assert result.infrastructure_fault.tolist() == [False, False]
    assert result.producer_fault_bits.tolist() == [0, 0]


def test_r07_expected_kind_independently_rejects_wrong_bootstrap_kind(
    monkeypatch,
):
    device = torch.device("cpu")
    env, motion, robot, sensor, bundle, epoch_owner = (
        _rowwise_settled_subject(monkeypatch, device=device)
    )
    env.common_step_counter = 40
    sensor.data.net_forces_w.zero_()
    for name in FEET:
        sensor.data.net_forces_w[:, BODY_NAMES.index(name), 2] = 10.0
    robot.data.body_lin_vel_w.zero_()
    # The ActionEpoch lifecycle and the reference kind come from independent
    # producers: a real accepted settlement + Motion commit requires
    # COMPLETED, while this fixture deliberately supplies BOOTSTRAP.
    epoch = epoch_owner.current()
    motion._action_ball_continuous_episode_step.fill_(40)
    reference = _ready_reference(
        motion=motion,
        epoch_owner=epoch_owner,
        robot=robot,
        reference_kind=(
            recovery.R07_REFERENCE_BOOTSTRAP_UPCOMING_ACTION_FRAME0
        ),
    )
    facts = bundle.plant_fact_adapter.read()
    result = bundle.owner.action_epoch_reward_view(
        facts,
        reference=reference,
        epoch=epoch,
        current_source_step=torch.full((2,), 40, dtype=torch.int64),
        adapter_source_step=40,
        motion_owner=motion,
        action_epoch_owner=epoch_owner,
    )
    assert result.facts_valid.tolist() == [False, True]
    assert result.producer_fault_bits.tolist() == [
        recovery.R07_EPOCH_FAULT_INVALID_REFERENCE,
        0,
    ]


def test_r07_censor_event_keeps_idle_upcoming_reference(monkeypatch):
    device = torch.device("cpu")
    env, motion, _robot, _sensor, bundle, epoch_owner = (
        _rowwise_settled_subject(
            monkeypatch, device=device, producer_fault=1
        )
    )
    env.common_step_counter = 0
    epoch = epoch_owner.current()
    assert torch.all(epoch.phase.eq(epoch_v1.PHASE_IDLE))
    result = _publish_epoch_reward_facts(
        bundle,
        motion,
        cadence_tick=0,
        current_source_step=torch.zeros(2, dtype=torch.int64)
    )
    assert result.reference_kind.tolist() == [1, 1]
    assert result.facts_valid.tolist() == [True, True]
    assert result.producer_fault_bits.tolist() == [0, 0]
    owner_slot = epoch_v1.OWNER_ORDER.index("r07_recovery")
    assert epoch_owner.current().fact_valid_bits[:, :, owner_slot].eq(0).all()


@pytest.mark.parametrize(
    ("lifecycle", "prepare_kwargs", "reference_kind"),
    (
        (
            "construction_reject",
            {"construction_admissible": False},
            recovery.R07_REFERENCE_BOOTSTRAP_UPCOMING_ACTION_FRAME0,
        ),
        (
            "playback_defer",
            {"playback_admissible": False},
            recovery.R07_REFERENCE_BOOTSTRAP_UPCOMING_ACTION_FRAME0,
        ),
        (
            "completed",
            {},
            recovery.R07_REFERENCE_COMPLETED_ACTION_FRAME0,
        ),
    ),
)
def test_same_reference_writer_cannot_forge_epoch_or_parent_identity(
    monkeypatch, lifecycle, prepare_kwargs, reference_kind
):
    device = torch.device("cpu")
    env, motion, robot, sensor, bundle, epoch_owner = (
        _rowwise_settled_subject(
            monkeypatch, device=device, **prepare_kwargs
        )
    )
    env.common_step_counter = 40
    sensor.data.net_forces_w.zero_()
    for name in FEET:
        sensor.data.net_forces_w[:, BODY_NAMES.index(name), 2] = 10.0
    robot.data.body_lin_vel_w.zero_()
    epoch = epoch_owner.current()

    # One reference producer simultaneously claims validity, the lifecycle's
    # expected kind, a coherent foreign slot/UID, and finite self-referential
    # frame-0 values.  Independent epoch and parent Motion identities stay at
    # the parent's first slot, so none of those same-writer claims authorize
    # the different but otherwise real catalog member below.
    parent_identity = bundle._project_parent_action_identity()
    assert len(parent_identity.action_uids) > 1
    motion._action_ball_continuous_episode_step.fill_(40)
    reference = _ready_reference(
        motion=motion,
        epoch_owner=epoch_owner,
        robot=robot,
        reference_kind=reference_kind,
        reference_action_slot=1,
        reference_action_uid=parent_identity.action_uids[1],
    )
    reference.root_position_m.add_(0.025)
    facts = bundle.plant_fact_adapter.read()
    result = bundle.owner.action_epoch_reward_view(
        facts,
        reference=reference,
        epoch=epoch,
        current_source_step=torch.full((2,), 40, dtype=torch.int64),
        adapter_source_step=40,
        motion_owner=motion,
        action_epoch_owner=epoch_owner,
    )
    assert lifecycle in {"construction_reject", "playback_defer", "completed"}
    assert result.facts_valid.tolist() == [False, False]
    assert torch.all(
        torch.bitwise_and(
            result.producer_fault_bits,
            recovery.R07_EPOCH_FAULT_INVALID_REFERENCE,
        ).ne(0)
    )


def test_foreign_exact_motion_and_epoch_group_cannot_enter_bound_r07(
    monkeypatch,
):
    device = torch.device("cpu")
    env, motion, _robot, _sensor = _subject(monkeypatch, device=device)
    env.common_step_counter = 0
    bundle = _construct_bundle(
        env=env,
        motion=motion,
        action_epoch_owner=motion._diagnostic_test_epoch_owner,
    )
    (
        _foreign_env,
        foreign_motion,
        _foreign_robot,
        _foreign_sensor,
    ) = _subject(monkeypatch, device=device)
    foreign_epoch_owner = foreign_motion._diagnostic_test_epoch_owner
    foreign_epoch = foreign_epoch_owner.current()
    foreign_reference = (
        foreign_motion.project_action_ball_full_mdp_recovery_ready_reference()
    )
    facts = bundle.plant_fact_adapter.read()
    with pytest.raises(
        recovery.ContinuousRecoveryDeviceError,
        match="construction-bound parent/Motion/epoch identity differs",
    ):
        bundle.owner.action_epoch_reward_view(
            facts,
            reference=foreign_reference,
            epoch=foreign_epoch,
            current_source_step=torch.zeros(2, dtype=torch.int64),
            adapter_source_step=0,
            motion_owner=foreign_motion,
            action_epoch_owner=foreign_epoch_owner,
        )


@pytest.mark.parametrize("contradiction", ("nonneutral_bootstrap", "stale_completed"))
def test_r07_reference_requires_exact_lifecycle_full_key(
    monkeypatch, contradiction
):
    device = torch.device("cpu")
    if contradiction == "stale_completed":
        env, motion, robot, _sensor, bundle, epoch_owner = (
            _rowwise_settled_subject(monkeypatch, device=device)
        )
        reference_kind = recovery.R07_REFERENCE_COMPLETED_ACTION_FRAME0
        source_step = 40
    else:
        env, motion, robot, _sensor = _subject(monkeypatch, device=device)
        epoch_owner = motion._diagnostic_test_epoch_owner
        bundle = _construct_bundle(
            env=env, motion=motion, action_epoch_owner=epoch_owner
        )
        reference_kind = (
            recovery.R07_REFERENCE_BOOTSTRAP_UPCOMING_ACTION_FRAME0
        )
        source_step = 0
    env.common_step_counter = source_step
    epoch = epoch_owner.current()
    motion._action_ball_continuous_episode_step.fill_(source_step)
    reference = _ready_reference(
        motion=motion,
        epoch_owner=epoch_owner,
        robot=robot,
        reference_kind=reference_kind,
    )
    if contradiction == "stale_completed":
        reference.reference_kind[1] = (
            recovery.R07_REFERENCE_BOOTSTRAP_UPCOMING_ACTION_FRAME0
        )
    reference.shot_key.shot_index[0].add_(1)
    facts = bundle.plant_fact_adapter.read()
    result = bundle.owner.action_epoch_reward_view(
        facts,
        reference=reference,
        epoch=epoch,
        current_source_step=torch.full(
            (2,), source_step, dtype=torch.int64
        ),
        adapter_source_step=source_step,
        motion_owner=motion,
        action_epoch_owner=epoch_owner,
    )
    assert result.facts_valid.tolist() == [False, True]
    assert result.producer_fault_bits[0].item() & (
        recovery.R07_EPOCH_FAULT_INVALID_REFERENCE
    )
    assert result.producer_fault_bits[1].item() == 0


def test_same_writer_schedule_binding_clip_and_reference_cannot_replace_parent(
    monkeypatch,
):
    device = torch.device("cpu")
    env, motion, robot, sensor = _subject(monkeypatch, device=device)
    env.common_step_counter = 0
    sensor.data.net_forces_w.zero_()
    for name in FEET:
        sensor.data.net_forces_w[:, BODY_NAMES.index(name), 2] = 10.0
    robot.data.body_lin_vel_w.zero_()
    epoch_owner = motion._diagnostic_test_epoch_owner
    bundle = _construct_bundle(
        env=env, motion=motion, action_epoch_owner=epoch_owner
    )
    _publish_epoch_reward_facts(
        bundle,
        motion,
        cadence_tick=0,
        current_source_step=torch.zeros(2, dtype=torch.int64)
    )
    assert bundle.owner._action_epoch_ready_streak.tolist() == [1, 1]

    # Replace every same-writer identity input coherently.  Motion now emits a
    # finite, self-valid reference for the second real catalog member, while
    # the independent parent still retains the cold-bound first member.
    parent_identity = bundle._project_parent_action_identity()
    assert len(parent_identity.action_uids) > 1
    foreign_slot = 1
    foreign_uid = parent_identity.action_uids[foreign_slot]
    replaced_schedule = dict(
        motion._action_ball_continuous_schedule_projection
    )
    replaced_schedule["upcoming_action_slot"] = foreign_slot
    replaced_schedule["upcoming_action_uid"] = foreign_uid
    replaced_schedule = types.MappingProxyType(replaced_schedule)
    old_binding = motion._action_ball_continuous_parent_authority_binding
    motion._action_ball_continuous_schedule_projection = replaced_schedule
    motion._action_ball_continuous_parent_authority_binding = (
        old_binding[0],
        replaced_schedule,
        old_binding[2],
        old_binding[3],
    )
    motion.clip_id.fill_(foreign_slot)
    same_writer_reference = (
        motion.project_action_ball_full_mdp_recovery_ready_reference()
    )
    assert same_writer_reference.validity.tolist() == [True, True]
    assert same_writer_reference.reference_kind.tolist() == [1, 1]
    assert same_writer_reference.reference_action_slot.tolist() == [1, 1]
    assert same_writer_reference.reference_action_uid.tolist() == [
        foreign_uid,
        foreign_uid,
    ]
    env.common_step_counter = 1
    invalid = _publish_epoch_reward_facts(
        bundle,
        motion,
        cadence_tick=1,
        current_source_step=torch.ones(2, dtype=torch.int64)
    )
    changed_view = bundle.require_owned_motion_ready_projection(
        bundle.motion_ready_projection(), owner_kind="motion"
    )
    assert torch.all(
        torch.bitwise_and(
            invalid.producer_fault_bits,
            recovery.R07_EPOCH_FAULT_INVALID_REFERENCE,
        ).ne(0)
    )
    assert bundle.owner._action_epoch_ready_streak.tolist() == [0, 0]
    assert changed_view.ready.tolist() == [False, False]

    env.common_step_counter = 2
    still_invalid = _publish_epoch_reward_facts(
        bundle,
        motion,
        cadence_tick=2,
        current_source_step=torch.full((2,), 2, dtype=torch.int64)
    )
    stable_view = bundle.require_owned_motion_ready_projection(
        bundle.motion_ready_projection(), owner_kind="motion"
    )
    assert torch.all(
        torch.bitwise_and(
            still_invalid.producer_fault_bits,
            recovery.R07_EPOCH_FAULT_INVALID_REFERENCE,
        ).ne(0)
    )
    assert bundle.owner._action_epoch_ready_streak.tolist() == [0, 0]
    assert stable_view.ready.tolist() == [False, False]
    assert stable_view.control_tick.tolist() == [3, 3]


def test_class_substituted_motion_reference_producer_is_not_exact_source(
    monkeypatch,
):
    device = torch.device("cpu")
    env, motion, _robot, _sensor = _subject(monkeypatch, device=device)
    env.common_step_counter = 0
    bundle = _construct_bundle(
        env=env,
        motion=motion,
        action_epoch_owner=motion._diagnostic_test_epoch_owner,
    )
    forged_reference = (
        motion.project_action_ball_full_mdp_recovery_ready_reference()
    )
    assert forged_reference.validity.tolist() == [True, True]
    assert forged_reference.producer_fault_bits.tolist() == [0, 0]

    def substituted_reference_producer(self):
        assert self is motion
        return forged_reference

    monkeypatch.setattr(
        type(motion),
        "project_action_ball_full_mdp_recovery_ready_reference",
        substituted_reference_producer,
    )
    with pytest.raises(
        recovery.ContinuousRecoveryDeviceError,
        match="exact Motion frame-0 producer definition differs",
    ):
        _publish_epoch_reward_facts(
            bundle,
            motion,
            cadence_tick=0,
            current_source_step=torch.zeros(2, dtype=torch.int64)
        )


def test_class_substituted_parent_projection_cannot_join_same_writer_forge(
    monkeypatch,
):
    device = torch.device("cpu")
    env, motion, _robot, _sensor = _subject(monkeypatch, device=device)
    env.common_step_counter = 0
    bundle = _construct_bundle(
        env=env,
        motion=motion,
        action_epoch_owner=motion._diagnostic_test_epoch_owner,
    )
    parent_identity = bundle._project_parent_action_identity()
    assert len(parent_identity.action_uids) > 1
    foreign_slot = 1
    foreign_uid = parent_identity.action_uids[foreign_slot]
    replaced_schedule = dict(
        motion._action_ball_continuous_schedule_projection
    )
    replaced_schedule["upcoming_action_slot"] = foreign_slot
    replaced_schedule["upcoming_action_uid"] = foreign_uid
    replaced_schedule = types.MappingProxyType(replaced_schedule)
    old_binding = motion._action_ball_continuous_parent_authority_binding
    motion._action_ball_continuous_schedule_projection = replaced_schedule
    motion._action_ball_continuous_parent_authority_binding = (
        old_binding[0],
        replaced_schedule,
        old_binding[2],
        old_binding[3],
    )
    motion.clip_id.fill_(foreign_slot)
    forged_reference = (
        motion.project_action_ball_full_mdp_recovery_ready_reference()
    )
    assert forged_reference.validity.tolist() == [True, True]
    assert forged_reference.reference_action_slot.tolist() == [1, 1]
    assert forged_reference.reference_action_uid.tolist() == [
        foreign_uid,
        foreign_uid,
    ]
    authority = bundle.motion_parent_authority

    def substituted_parent_projection(self, receipt, *, motion_owner):
        assert self is authority
        assert receipt is bundle.motion_parent_receipt
        assert motion_owner is motion
        return cadence.DiagnosticMotionParentActionIdentity(
            authority=self,
            motion_owner=motion_owner,
            action_slot=foreign_slot,
            action_uid=foreign_uid,
            action_uids=parent_identity.action_uids,
        )

    forged_parent = substituted_parent_projection(
        authority,
        bundle.motion_parent_receipt,
        motion_owner=motion,
    )
    assert forged_parent.action_slot == foreign_slot
    assert forged_parent.action_uid == foreign_uid
    monkeypatch.setattr(
        type(authority),
        "project_bound_action_identity",
        substituted_parent_projection,
    )
    with pytest.raises(
        recovery.ContinuousRecoveryDeviceError,
        match="exact Motion parent projection definition differs",
    ):
        _publish_epoch_reward_facts(
            bundle,
            motion,
            cadence_tick=0,
            current_source_step=torch.zeros(2, dtype=torch.int64)
        )


@pytest.mark.parametrize("device_name", ("cpu", "cuda:0"))
def test_epoch_publish_owns_two_tick_dwell_and_next_tick_motion_projection(
    monkeypatch, device_name
):
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    device = torch.device(device_name)
    env, motion, robot, sensor, bundle, epoch_owner = (
        _rowwise_settled_subject(monkeypatch, device=device)
    )
    # Both real feet are supported and stationary; this makes readiness depend
    # only on the owner's two consecutive post-physics observations.
    sensor.data.net_forces_w[:, :, :] = 0.0
    for name in FEET:
        sensor.data.net_forces_w[:, BODY_NAMES.index(name), 2] = 10.0
    robot.data.body_lin_vel_w.zero_()
    first = _publish_epoch_reward_facts(
        bundle,
        motion,
        cadence_tick=40,
        current_source_step=torch.full(
            (2,), 40, dtype=torch.int64, device=device
        )
    )
    assert first.ready_instant.tolist() == [True, True]
    first_projection = bundle.motion_ready_projection()
    first_view = bundle.require_owned_motion_ready_projection(
        first_projection, owner_kind="motion"
    )
    first_observation = bundle.action_epoch_observation_state()
    assert first_view.ready.tolist() == [False, False]
    assert first_view.control_tick.tolist() == [41, 41]
    assert first_observation.postphysics_valid.tolist() == [True, True]
    assert torch.equal(first_observation.source_step, first.source_step)
    assert torch.equal(
        first_observation.reset_generation, first.reset_generation
    )
    assert first_observation.control_tick.tolist() == [41, 41]
    assert first_observation.ready_streak.tolist() == [1, 1]
    assert first_observation.foot_supported_lr.tolist() == [
        [True, True],
        [True, True],
    ]
    assert bundle.owner._action_epoch_ready_streak.tolist() == [1, 1]

    env.common_step_counter = 41
    second = _publish_epoch_reward_facts(
        bundle,
        motion,
        cadence_tick=41,
        current_source_step=torch.full(
            (2,), 41, dtype=torch.int64, device=device
        )
    )
    assert second.ready_instant.tolist() == [True, True]
    second_projection = bundle.motion_ready_projection()
    second_view = bundle.require_owned_motion_ready_projection(
        second_projection, owner_kind="motion"
    )
    second_observation = bundle.action_epoch_observation_state()
    assert second_view.ready.tolist() == [True, True]
    assert second_view.control_tick.tolist() == [42, 42]
    assert second_observation.control_tick.tolist() == [42, 42]
    assert second_observation.ready_streak.tolist() == [2, 2]
    assert bundle.owner._action_epoch_ready_streak.tolist() == [2, 2]
    assert bundle.owner._action_epoch_first_ready_source_step.tolist() == [41, 41]

    with pytest.raises(
        recovery.ContinuousRecoveryDeviceError, match="stale or foreign"
    ):
        bundle.require_owned_motion_ready_projection(
            first_projection, owner_kind="motion"
        )

    bundle.owner._latest_motion_ready_projection = None
    with pytest.raises(
        recovery.ContinuousRecoveryDeviceError,
        match="lost its post-physics publication",
    ):
        bundle.action_epoch_observation_state()


def test_epoch_invalid_fact_resets_dwell_and_cannot_mint_ready(monkeypatch):
    device = torch.device("cpu")
    env, motion, robot, sensor, bundle, epoch_owner = (
        _rowwise_settled_subject(monkeypatch, device=device)
    )
    sensor.data.net_forces_w[:, :, :] = 0.0
    for name in FEET:
        sensor.data.net_forces_w[:, BODY_NAMES.index(name), 2] = 10.0
    robot.data.body_lin_vel_w.zero_()
    _publish_epoch_reward_facts(
        bundle,
        motion,
        cadence_tick=40,
        current_source_step=torch.full((2,), 40, dtype=torch.int64)
    )
    robot.data.joint_vel[1, 0] = math.nan
    env.common_step_counter = 41
    invalid = _publish_epoch_reward_facts(
        bundle,
        motion,
        cadence_tick=41,
        current_source_step=torch.full((2,), 41, dtype=torch.int64)
    )
    view = bundle.require_owned_motion_ready_projection(
        bundle.motion_ready_projection(), owner_kind="motion"
    )
    assert invalid.facts_valid.tolist() == [True, False]
    assert view.ready.tolist() == [True, False]
    assert bundle.owner._action_epoch_ready_streak.tolist() == [2, 0]


@pytest.mark.parametrize(
    ("bad_step", "message"),
    ((40, "skipped, stale, or replayed"), (39, "source step regressed")),
)
def test_epoch_stale_or_replayed_step_fails_before_readiness_mutation(
    monkeypatch, bad_step, message
):
    device = torch.device("cpu")
    env, motion, robot, sensor, bundle, epoch_owner = (
        _rowwise_settled_subject(monkeypatch, device=device)
    )
    sensor.data.net_forces_w[:, :, :] = 0.0
    for name in FEET:
        sensor.data.net_forces_w[:, BODY_NAMES.index(name), 2] = 10.0
    robot.data.body_lin_vel_w.zero_()
    _publish_epoch_reward_facts(
        bundle,
        motion,
        cadence_tick=40,
        current_source_step=torch.full((2,), 40, dtype=torch.int64)
    )
    before_streak = bundle.owner._action_epoch_ready_streak.clone()
    before_projection = bundle.motion_ready_projection()
    before_head = epoch_owner.commit_head
    before_epoch_facts = tuple(
        value.detach().clone()
        for value in (
            epoch_owner.current().owner_fault_bits,
            epoch_owner.current().fact_valid_bits,
            epoch_owner.current().fact_source_step,
            epoch_owner.current().fact_f32,
        )
    )
    env.common_step_counter = bad_step
    with pytest.raises(
        recovery.ContinuousRecoveryDeviceError,
        match=message,
    ):
        _publish_epoch_reward_facts(
            bundle,
            motion,
            cadence_tick=41,
            current_source_step=torch.full((2,), bad_step, dtype=torch.int64)
        )
    assert torch.equal(bundle.owner._action_epoch_ready_streak, before_streak)
    assert bundle.motion_ready_projection() is before_projection
    assert epoch_owner.commit_head == before_head
    after = epoch_owner.current()
    for expected, actual in zip(
        before_epoch_facts,
        (
            after.owner_fault_bits,
            after.fact_valid_bits,
            after.fact_source_step,
            after.fact_f32,
        ),
    ):
        assert torch.equal(actual, expected)
