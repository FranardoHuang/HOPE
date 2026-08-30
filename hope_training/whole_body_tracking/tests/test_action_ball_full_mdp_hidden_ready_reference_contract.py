"""Fresh FullMDP hidden-balance teacher and atomic reveal regressions.

This is deliberately a dependency-light ``MotionCommand`` harness.  It keeps
the production fresh configuration (non-canonical, non-split) and exercises
the real Device-R05 genesis, D05 token writer, and selected-reset writer.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch


TESTS = Path(__file__).resolve().parent
WBT_ROOT = TESTS.parent
SOURCE = WBT_ROOT / "source" / "whole_body_tracking"
MDP = SOURCE / "whole_body_tracking" / "tasks" / "tracking" / "mdp"
for path in (TESTS, SOURCE, MDP):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import test_action_ball_continuous_motion_bridge as bridge  # noqa: E402
import test_action_ball_continuous_motion_selected_reset as selected_reset  # noqa: E402
import test_action_ball_motion_rowwise_accept_writer as accept_writer  # noqa: E402


C = bridge.C
DEVICE = torch.device("cpu")
BODY_COUNT = 2


def _raw_bytes(value: torch.Tensor) -> torch.Tensor:
    return (
        value.detach()
        .contiguous()
        .reshape(-1)
        .view(torch.uint8)
        .cpu()
        .clone()
    )


def _install_distinct_reset_and_frame_zero_tapes(command) -> None:
    """Attach the real getter surfaces omitted by the small Motion harness."""

    n = command.num_envs
    frame_count, joint_count = command.motion.joint_pos.shape
    origins = command._env.scene.env_origins

    # These are the exact fresh FullMDP settings.  This test must never borrow
    # either legacy ready-teacher implementation to obtain the desired result.
    command.canonical_ready_mode = False
    command.cfg.canonical_ready_mode = False
    command.action_ball_diagnostic_split_ready_teacher = False
    command.cfg.action_ball_diagnostic_split_ready_teacher = False

    command.body_indexes = torch.arange(BODY_COUNT, dtype=torch.long)
    command.motion_anchor_body_index = 0
    command.robot_anchor_body_index = 0
    command.cfg.body_names = ("anchor", "paddle")

    command.robot.data.default_joint_pos = (
        torch.arange(n * joint_count, dtype=torch.float32).reshape(
            n, joint_count
        )
        + 1000.0
    )
    reset_body_pos = torch.tensor(
        (
            (1.0, 2.0, 0.90),
            (1.4, 2.5, 1.20),
        ),
        dtype=torch.float32,
    ).reshape(1, BODY_COUNT, 3)
    command.robot.data.body_pos_w = (
        reset_body_pos.repeat(n, 1, 1) + origins[:, None, :]
    )
    command.robot.data.body_quat_w = torch.zeros(
        n, BODY_COUNT, 4, dtype=torch.float32
    )
    command.robot.data.body_quat_w[..., 0] = 1.0
    half_sqrt_two = 2.0**-0.5
    command.robot.data.body_quat_w[:, 1] = torch.tensor(
        (half_sqrt_two, 0.0, 0.0, half_sqrt_two),
        dtype=torch.float32,
    )

    frame_offsets = (
        torch.arange(frame_count, dtype=torch.float32).reshape(-1, 1, 1)
        * 0.01
    )
    frame_zero_body = torch.tensor(
        (
            (10.0, 20.0, 1.00),
            (10.4, 20.5, 1.30),
        ),
        dtype=torch.float32,
    ).reshape(1, BODY_COUNT, 3)
    command.motion.body_pos_w = frame_zero_body + frame_offsets
    command.motion.body_quat_w = torch.zeros(
        frame_count, BODY_COUNT, 4, dtype=torch.float32
    )
    command.motion.body_quat_w[..., 0] = 1.0
    command.motion.joint_vel = torch.full_like(
        command.motion.joint_pos, 7.0
    )
    command.motion.body_lin_vel_w = torch.full(
        (frame_count, BODY_COUNT, 3), 8.0, dtype=torch.float32
    )
    command.motion.body_ang_vel_w = torch.full(
        (frame_count, BODY_COUNT, 3), 9.0, dtype=torch.float32
    )

    command.body_pos_relative_w = torch.full(
        (n, BODY_COUNT, 3), -91.0, dtype=torch.float32
    )
    command.body_quat_relative_w = torch.zeros(
        n, BODY_COUNT, 4, dtype=torch.float32
    )
    command.body_quat_relative_w[..., 0] = 1.0
    # ``_configure_unbound_command`` starts from the legacy test constructor,
    # where the generic pending tensor exists but the generic body caches do
    # not.  Supplying the latter lets the real fresh genesis validate and reuse
    # all three members instead of inventing a test-only reference source.
    command._action_ball_safe_ready_body_pos_w = torch.zeros(
        n, BODY_COUNT, 3, dtype=torch.float32
    )
    command._action_ball_safe_ready_body_quat_w = torch.zeros(
        n, BODY_COUNT, 4, dtype=torch.float32
    )
    action_count = int(command.motion.num_segments)
    command._action_ball_full_mdp_source_strike_root_xy = torch.zeros(
        action_count, 2, dtype=torch.float32
    )
    command._action_ball_full_mdp_source_strike_yaw_wxyz = torch.zeros(
        action_count, 4, dtype=torch.float32
    )
    command._action_ball_full_mdp_source_strike_yaw_wxyz[:, 0] = 1.0
    command._action_ball_full_mdp_task_yaw_wxyz = torch.zeros(
        n, 4, dtype=torch.float32
    )
    command._action_ball_full_mdp_task_yaw_wxyz[:, 0] = 1.0
    command._action_ball_full_mdp_task_translation_w = torch.zeros(
        n, 3, dtype=torch.float32
    )
    command._action_ball_full_mdp_frozen_root_pos_w = torch.zeros(
        n, 3, dtype=torch.float32
    )
    command._action_ball_full_mdp_frozen_root_quat_wxyz = torch.zeros(
        n, 4, dtype=torch.float32
    )
    command._action_ball_full_mdp_frozen_root_quat_wxyz[:, 0] = 1.0
    command._action_ball_full_mdp_frozen_root_valid = torch.ones(
        n, dtype=torch.bool
    )


def _fresh_command(num_envs: int):
    command, env_ids = bridge._configure_unbound_command(num_envs=num_envs)
    _install_distinct_reset_and_frame_zero_tapes(command)
    schedule = bridge._schedule_projection(cadence_steps=81)
    schedule.update(
        upcoming_action_slot=0,
        upcoming_action_uid=(
            command._action_ball_continuous_code_owned_action_uids()[0]
        ),
    )
    command.bind_action_ball_continuous_parent_authorities(
        **bridge._parent_binding_kwargs(schedule)
    )
    command._reset_action_ball_continuous_motion_cadence(env_ids)
    device_owner, authority = selected_reset._bind_real_device_r05_owner(
        command
    )

    assert command.canonical_ready_mode is False
    assert command.cfg.canonical_ready_mode is False
    assert command.action_ball_diagnostic_split_ready_teacher is False
    assert (
        command.cfg.action_ball_diagnostic_split_ready_teacher is False
    )
    assert command._action_ball_continuous_fresh_motion_lane_bound is True
    return command, device_owner, authority


def _frame_zero(command):
    steps = command.motion.seg_start[command.clip_id]
    body_pos = (
        command.motion.body_pos_w[steps]
        + command._env.scene.env_origins[:, None, :]
    )
    return {
        "steps": steps,
        "joint_pos": command.motion.joint_pos[steps],
        "body_pos_w": body_pos,
        "body_quat_w": command.motion.body_quat_w[steps],
    }


def test_playback_active_accessor_is_exact_zero_copy_motion_authority():
    command, _device_owner, _authority = _fresh_command(2)
    active = command._action_ball_continuous_canonical_playback_started
    assert command._action_ball_continuous_canonical_phase.tolist() == [
        C.ACTION_BALL_CONTINUOUS_CANONICAL_RECOVER_HIDDEN,
        C.ACTION_BALL_CONTINUOUS_CANONICAL_RECOVER_HIDDEN,
    ]
    assert not bool(active.any())

    # Focused phase projection of the existing owner tensor: prepare is false,
    # swing/follow are true, and suffix/recovery clears it again.  The accessor
    # must return that exact storage rather than reconstructing from phase.
    command._action_ball_continuous_canonical_phase.copy_(
        torch.tensor(
            [
                C.ACTION_BALL_CONTINUOUS_CANONICAL_PREPARE_VISIBLE,
                C.ACTION_BALL_CONTINUOUS_CANONICAL_SWING,
            ],
            dtype=torch.int64,
        )
    )
    active.copy_(torch.tensor([False, True], dtype=torch.bool))

    exposed = command.action_ball_full_mdp_playback_active_mask()

    assert exposed is active
    assert exposed.tolist() == [False, True]
    active.copy_(torch.tensor([False, True], dtype=torch.bool))
    command._action_ball_continuous_canonical_phase.copy_(
        torch.tensor(
            [
                C.ACTION_BALL_CONTINUOUS_CANONICAL_RECOVER_HIDDEN,
                C.ACTION_BALL_CONTINUOUS_CANONICAL_FOLLOW_THROUGH,
            ],
            dtype=torch.int64,
        )
    )
    assert exposed.tolist() == [False, True]
    active.zero_()
    command._action_ball_continuous_canonical_phase.fill_(
        C.ACTION_BALL_CONTINUOUS_CANONICAL_READY_HOLD
    )
    assert not bool(exposed.any())


def _bind_exact_accept_test_owners(command, d05_owner, epoch_owner) -> None:
    """Finish the exact fresh ActionEpoch construction omitted by its unit stub."""

    epoch_owner.activate_reset_genesis(
        selected_mask=torch.ones(
            command.num_envs, dtype=torch.bool, device=DEVICE
        ),
        reset_generation=torch.zeros(
            command.num_envs, dtype=torch.int64, device=DEVICE
        ),
    )
    command._action_ball_continuous_motion_device_r05_owner = d05_owner
    command.bind_action_ball_full_mdp_motion_epoch_owner(epoch_owner)


def _reference_bytes(command) -> dict[str, torch.Tensor]:
    return {
        "mask": _raw_bytes(
            command._action_ball_full_mdp_initial_balance_reference_mask()
        ),
        "created": _raw_bytes(
            command._action_ball_continuous_policy_opportunities_created
        ),
        "safe_pos": _raw_bytes(
            command._action_ball_safe_ready_body_pos_w
        ),
        "safe_quat": _raw_bytes(
            command._action_ball_safe_ready_body_quat_w
        ),
        "pending": _raw_bytes(
            command._action_ball_safe_ready_reference_pending
        ),
        "relative_pos": _raw_bytes(command.body_pos_relative_w),
        "relative_quat": _raw_bytes(command.body_quat_relative_w),
        "joint_pos": _raw_bytes(command.joint_pos),
        "joint_vel": _raw_bytes(command.joint_vel),
        "body_pos_w": _raw_bytes(command.body_pos_w),
        "body_quat_w": _raw_bytes(command.body_quat_w),
        "body_lin_vel_w": _raw_bytes(command.body_lin_vel_w),
        "body_ang_vel_w": _raw_bytes(command.body_ang_vel_w),
        "anchor_pos_w": _raw_bytes(command.anchor_pos_w),
        "anchor_quat_w": _raw_bytes(command.anchor_quat_w),
        "anchor_lin_vel_w": _raw_bytes(command.anchor_lin_vel_w),
        "anchor_ang_vel_w": _raw_bytes(command.anchor_ang_vel_w),
    }


def _peer_bytes(command, rows: torch.Tensor) -> dict[str, torch.Tensor]:
    state = {}
    for name, value in selected_reset._tensor_snapshot(command).items():
        if value.ndim >= 1 and value.shape[0] == command.num_envs:
            state[name] = _raw_bytes(value[rows])
    for name in (
        "_action_ball_safe_ready_body_pos_w",
        "_action_ball_safe_ready_body_quat_w",
        "_action_ball_safe_ready_reference_pending",
        "body_pos_relative_w",
        "body_quat_relative_w",
    ):
        state[name] = _raw_bytes(getattr(command, name)[rows])
    state["getter_joint_pos"] = _raw_bytes(command.joint_pos[rows])
    state["getter_body_pos_w"] = _raw_bytes(command.body_pos_w[rows])
    state["getter_body_quat_w"] = _raw_bytes(command.body_quat_w[rows])
    state["getter_anchor_pos_w"] = _raw_bytes(command.anchor_pos_w[rows])
    state["getter_anchor_quat_w"] = _raw_bytes(command.anchor_quat_w[rows])
    return state


def _assert_zero_velocity_reference(command, rows) -> None:
    assert torch.count_nonzero(command.joint_vel[rows]) == 0
    assert torch.count_nonzero(command.body_lin_vel_w[rows]) == 0
    assert torch.count_nonzero(command.body_ang_vel_w[rows]) == 0
    assert torch.count_nonzero(command.anchor_lin_vel_w[rows]) == 0
    assert torch.count_nonzero(command.anchor_ang_vel_w[rows]) == 0


def test_fresh_hidden_balance_then_accept_is_atomic_and_suffix_keeps_frame_zero(
    monkeypatch,
):
    command, _device_owner, _authority = _fresh_command(2)
    frame_zero = _frame_zero(command)
    reset_joint = command.robot.data.default_joint_pos.clone()
    reset_body = command.robot.data.body_pos_w.clone()
    reset_quat = command.robot.data.body_quat_w.clone()

    assert command._action_ball_safe_ready_reference_pending.tolist() == [
        True,
        True,
    ]
    assert command._action_ball_safe_ready_pending_count == 2
    assert command._action_ball_full_mdp_initial_balance_reference_mask().tolist() == [
        True,
        True,
    ]
    assert not torch.equal(reset_joint, frame_zero["joint_pos"])
    assert not torch.equal(reset_body, frame_zero["body_pos_w"])

    # This is the same one-shot capture called by the first command update and
    # lazily by a reset observation that runs before that update.
    command._capture_action_ball_safe_ready_reference()
    assert not bool(command._action_ball_safe_ready_reference_pending.any())
    assert command._action_ball_safe_ready_pending_count == 0
    frozen_safe_pos = command._action_ball_safe_ready_body_pos_w.clone()
    frozen_safe_quat = command._action_ball_safe_ready_body_quat_w.clone()
    command.robot.data.body_pos_w.add_(123.0)
    command._capture_action_ball_safe_ready_reference()
    assert torch.equal(
        command._action_ball_safe_ready_body_pos_w, frozen_safe_pos
    )
    assert torch.equal(
        command._action_ball_safe_ready_body_quat_w, frozen_safe_quat
    )
    command.robot.data.body_pos_w.copy_(reset_body)

    assert torch.equal(command.joint_pos, reset_joint)
    assert torch.equal(command.body_pos_w, reset_body)
    assert torch.equal(command.body_quat_w, reset_quat)
    assert torch.equal(command.anchor_pos_w, reset_body[:, 0])
    assert torch.equal(command.anchor_quat_w, reset_quat[:, 0])
    _assert_zero_velocity_reference(command, torch.tensor([0, 1]))

    # Reuse the exact token/view owner seam from the focused D05 test, but
    # install it on this real fresh-genesis command.  Row 0 is ACCEPT; row 1
    # had an admissible candidate but is CENSOR and must be a byte no-op.
    (
        _unused_command,
        d05_owner,
        epoch_owner,
        token,
        _record,
        _active_calls,
    ) = accept_writer._install_exact_sources(
        monkeypatch,
        n=2,
        device=DEVICE,
        accept_mask=torch.tensor([True, False], dtype=torch.bool),
        candidate_valid=torch.tensor([True, True], dtype=torch.bool),
    )
    _bind_exact_accept_test_owners(command, d05_owner, epoch_owner)
    peer_writer_before = accept_writer._snapshot(command, row=1)
    peer_reference_before = {
        name: _raw_bytes(value[1])
        for name, value in (
            (
                "safe_pos",
                command._action_ball_safe_ready_body_pos_w,
            ),
            (
                "safe_quat",
                command._action_ball_safe_ready_body_quat_w,
            ),
            ("relative_pos", command.body_pos_relative_w),
            ("relative_quat", command.body_quat_relative_w),
            ("joint_pos", command.joint_pos),
            ("body_pos_w", command.body_pos_w),
            ("body_quat_w", command.body_quat_w),
            ("anchor_pos_w", command.anchor_pos_w),
            ("anchor_quat_w", command.anchor_quat_w),
        )
    }

    command.commit_action_ball_full_mdp_motion_epoch_rows(token)

    assert command._action_ball_continuous_policy_opportunities_created.tolist() == [
        1,
        0,
    ]
    assert command._action_ball_full_mdp_initial_balance_reference_mask().tolist() == [
        False,
        True,
    ]
    row_yaw = command._action_ball_full_mdp_task_yaw_wxyz[0].expand(
        BODY_COUNT, 4
    )
    expected_task_body_pos = (
        C.quat_apply(row_yaw, command.motion.body_pos_w[frame_zero["steps"][0]])
        + command._action_ball_full_mdp_task_translation_w[0]
        + command._env.scene.env_origins[0]
    )
    expected_task_body_quat = C.quat_mul(
        row_yaw, command.motion.body_quat_w[frame_zero["steps"][0]]
    )
    assert torch.equal(command.joint_pos[0], frame_zero["joint_pos"][0])
    assert torch.equal(command.body_pos_w[0], expected_task_body_pos)
    assert torch.equal(command.body_quat_w[0], expected_task_body_quat)
    assert torch.equal(command.anchor_pos_w[0], expected_task_body_pos[0])

    # Accepted-shot targets are a frozen scene transform.  Live robot motion is
    # an actor measurement and may not silently re-anchor the teacher.
    accepted_teacher = {
        "body_pos_w": command.body_pos_w[0].clone(),
        "body_quat_w": command.body_quat_w[0].clone(),
        "body_lin_vel_w": command.body_lin_vel_w[0].clone(),
        "body_ang_vel_w": command.body_ang_vel_w[0].clone(),
    }
    live_pos = command.robot.data.body_pos_w.clone()
    live_quat = command.robot.data.body_quat_w.clone()
    command.robot.data.body_pos_w[0].add_(torch.tensor((17.0, -9.0, 3.0)))
    command.robot.data.body_quat_w[0].copy_(
        torch.tensor(((0.5, 0.5, 0.5, 0.5),) * BODY_COUNT)
    )
    for name, expected in accepted_teacher.items():
        assert torch.equal(getattr(command, name)[0], expected), name
    command.robot.data.body_pos_w.copy_(live_pos)
    command.robot.data.body_quat_w.copy_(live_quat)

    assert torch.equal(command.body_pos_relative_w[0], expected_task_body_pos)
    assert torch.equal(command.body_quat_relative_w[0], expected_task_body_quat)
    assert not torch.equal(command.body_pos_relative_w[0], reset_body[0])

    peer_writer_after = accept_writer._snapshot(command, row=1)
    assert peer_writer_after.keys() == peer_writer_before.keys()
    for name, before in peer_writer_before.items():
        assert torch.equal(peer_writer_after[name], before), name
    for name, before in peer_reference_before.items():
        current = {
            "safe_pos": command._action_ball_safe_ready_body_pos_w,
            "safe_quat": command._action_ball_safe_ready_body_quat_w,
            "relative_pos": command.body_pos_relative_w,
            "relative_quat": command.body_quat_relative_w,
            "joint_pos": command.joint_pos,
            "body_pos_w": command.body_pos_w,
            "body_quat_w": command.body_quat_w,
            "anchor_pos_w": command.anchor_pos_w,
            "anchor_quat_w": command.anchor_quat_w,
        }[name]
        assert torch.equal(_raw_bytes(current[1]), before), name

    # The completed R07 recovery target is the completed action's frame 0,
    # never the episode-birth tuple and never the upcoming hidden action.
    command._action_ball_continuous_motion_active[0] = False
    command._action_ball_continuous_current_policy_opportunity[0] = False
    command._action_ball_continuous_suffix_complete[0] = True
    command._action_ball_continuous_ready_reference_active[0] = True
    command._hold_action_ball_continuous_ready_reference(
        torch.tensor([True, False], dtype=torch.bool)
    )
    assert command._action_ball_continuous_policy_opportunities_created[0] == 1
    assert not command._action_ball_full_mdp_initial_balance_reference_mask()[0]
    assert torch.equal(command.joint_pos[0], frame_zero["joint_pos"][0])
    assert torch.equal(command.body_pos_w[0], expected_task_body_pos)
    assert torch.equal(command.anchor_pos_w[0], expected_task_body_pos[0])
    _assert_zero_velocity_reference(command, torch.tensor([0]))


def test_fresh_censor_before_first_accept_preserves_complete_ready_reference(
    monkeypatch,
):
    command, _device_owner, _authority = _fresh_command(2)
    command._capture_action_ball_safe_ready_reference()
    before = _reference_bytes(command)
    (
        _unused_command,
        d05_owner,
        epoch_owner,
        token,
        _record,
        _active_calls,
    ) = accept_writer._install_exact_sources(
        monkeypatch,
        n=2,
        device=DEVICE,
        accept_mask=torch.tensor([False, False], dtype=torch.bool),
        candidate_valid=torch.tensor([True, True], dtype=torch.bool),
    )
    _bind_exact_accept_test_owners(command, d05_owner, epoch_owner)

    command.commit_action_ball_full_mdp_motion_epoch_rows(token)

    after = _reference_bytes(command)
    assert after.keys() == before.keys()
    for name, value in before.items():
        assert torch.equal(after[name], value), name
    assert command._action_ball_continuous_policy_opportunities_created.tolist() == [
        0,
        0,
    ]


def test_fresh_selected_reset_recaptures_only_selected_reset_ready_fk():
    command, _device_owner, _authority = _fresh_command(3)
    # Complete genesis capture before any later physical motion.  The selected
    # reset must explicitly create a new debt; it may not rely on this one.
    command._capture_action_ball_safe_ready_reference()
    assert not bool(command._action_ball_safe_ready_reference_pending.any())

    r05_owner = selected_reset._DeviceR05Authority(
        command=command,
        mask=torch.tensor([False, True, False], dtype=torch.bool),
    )
    command.bind_action_ball_continuous_motion_selected_reset(
        r05_owner,
        prepared_reset_validator=(
            r05_owner.require_owned_prepared_true_reset
        ),
        r05_receipt_validator=r05_owner.require_owned_true_reset_receipt,
        diagnostic=True,
    )
    command._action_ball_continuous_policy_opportunities_created.fill_(1)
    command.hold_counter.zero_()
    command.metrics["in_hold"].zero_()
    command.time_steps.copy_(command.motion.seg_start[command.clip_id])
    command.time_steps_f.copy_(command.time_steps.float())
    selected_reset._refresh_selection_generations(command, r05_owner)

    peers = torch.tensor([0, 2], dtype=torch.long)
    peer_before = _peer_bytes(command, peers)
    old_selected_safe = command._action_ball_safe_ready_body_pos_w[1].clone()
    new_selected_pos = command.robot.data.body_pos_w[1].clone()
    new_selected_pos += torch.tensor(
        ((30.0, 40.0, 0.20), (31.0, 41.0, 0.30)),
        dtype=torch.float32,
    )
    new_selected_quat = command.robot.data.body_quat_w[1].clone()
    new_selected_quat[0] = torch.tensor(
        (2.0**-0.5, 0.0, 0.0, -(2.0**-0.5)),
        dtype=torch.float32,
    )
    command.robot.data.body_pos_w[1].copy_(new_selected_pos)
    command.robot.data.body_quat_w[1].copy_(new_selected_quat)
    # Healthy rows may keep moving while row 1 resets.  If the pending mask is
    # accidentally widened, the lazy capture below will overwrite their bytes.
    command.robot.data.body_pos_w[peers].add_(77.0)

    stage = command.prepare_selected_reset(r05_owner.prepared)
    armed = command.arm_prevalidated_selected_reset(stage)
    terminal = command.commit_prevalidated_selected_reset(armed)
    completion = command.complete_selected_reset_after_r05(
        terminal, r05_owner.receipt
    )
    command.consume_owned_selected_reset_completion(
        completion,
        expected_prepared_true_reset=r05_owner.prepared,
    )

    assert command._action_ball_continuous_policy_opportunities_created.tolist() == [
        1,
        0,
        1,
    ]
    assert command._action_ball_full_mdp_initial_balance_reference_mask().tolist() == [
        False,
        True,
        False,
    ]
    assert command._action_ball_safe_ready_reference_pending.tolist() == [
        False,
        True,
        False,
    ]
    # This host scalar is intentionally only a conservative work flag; the
    # device mask above is the sole selected-row truth and avoids a D2H count.
    assert command._action_ball_safe_ready_pending_count > 0

    # The first same-return observation captures the newly reset selected FK.
    body_pos = command.body_pos_w
    body_quat = command.body_quat_w
    assert not bool(command._action_ball_safe_ready_reference_pending.any())
    assert command._action_ball_safe_ready_pending_count == 0
    assert not torch.equal(old_selected_safe, new_selected_pos)
    assert torch.equal(body_pos[1], new_selected_pos)
    assert torch.equal(body_quat[1], new_selected_quat)
    assert torch.equal(command.anchor_pos_w[1], new_selected_pos[0])
    assert torch.equal(command.anchor_quat_w[1], new_selected_quat[0])
    assert torch.equal(
        command.joint_pos[1], command.robot.data.default_joint_pos[1]
    )
    assert torch.equal(
        command._action_ball_safe_ready_body_pos_w[1], new_selected_pos
    )
    assert torch.equal(
        command._action_ball_safe_ready_body_quat_w[1], new_selected_quat
    )
    assert torch.equal(command.body_pos_relative_w[1], new_selected_pos)
    assert torch.equal(command.body_quat_relative_w[1], new_selected_quat)
    _assert_zero_velocity_reference(command, torch.tensor([1]))

    peer_after = _peer_bytes(command, peers)
    assert peer_after.keys() == peer_before.keys()
    for name, value in peer_before.items():
        assert torch.equal(peer_after[name], value), name
