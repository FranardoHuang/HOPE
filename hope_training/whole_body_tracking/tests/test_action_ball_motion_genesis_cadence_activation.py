"""Fresh Motion genesis activation and pre-D05 cadence regression.

The CPU row is host-runnable through the pinned Isaac test loader.  The CUDA
row is intended for the exact Pod GPU2 checkout; smoke/probe evidence remains
diagnostic and does not authorize launch.
"""

from __future__ import annotations

from dataclasses import fields
import os
from pathlib import Path
import sys
import types

import pytest
import torch


_ROOT = Path(__file__).resolve().parents[1]
_SOURCE = _ROOT / "source" / "whole_body_tracking"
_MDP = _SOURCE / "whole_body_tracking" / "tasks" / "tracking" / "mdp"
for path in (_SOURCE, _MDP):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import action_ball_continuous_runtime_transaction_device as r05  # noqa: E402
import test_action_ball_continuous_motion_bridge as bridge  # noqa: E402


_CANONICAL_EPOCH_BEFORE_TIMING_IMPORT = r05._require_action_epoch_module()
import action_ball_manifest as _manifest  # noqa: E402

_MDP_PACKAGE_NAME = "whole_body_tracking.tasks.tracking.mdp"
_MDP_PACKAGE = sys.modules[_MDP_PACKAGE_NAME]
for _name, _module in (
    ("action_ball_manifest", _manifest),
    ("commands", sys.modules[_MDP_PACKAGE_NAME + ".commands"]),
    ("hope_commands", sys.modules[_MDP_PACKAGE_NAME + ".hope_commands"]),
):
    sys.modules.setdefault(_MDP_PACKAGE_NAME + "." + _name, _module)
    setattr(_MDP_PACKAGE, _name, _module)
import action_ball_full_mdp_diagnostic_action_timing as timing  # noqa: E402
_CANONICAL_EPOCH_AFTER_TIMING_IMPORT = r05._require_action_epoch_module()
assert (
    _CANONICAL_EPOCH_AFTER_TIMING_IMPORT
    is _CANONICAL_EPOCH_BEFORE_TIMING_IMPORT
)
import action_ball_device_profile_authority as profile_authority  # noqa: E402
import action_ball_full_mdp_reset_genesis as reset_genesis  # noqa: E402
import action_ball_motion_cadence_device as cadence  # noqa: E402


C = bridge.C
# Device-R05 imports the canonical package module itself.  Use that same
# module identity when pytest collects this focused file in isolation.
E = r05._require_action_epoch_module()


def test_production_timing_import_preserves_canonical_epoch_module() -> None:
    assert E is _CANONICAL_EPOCH_BEFORE_TIMING_IMPORT
    assert E is _CANONICAL_EPOCH_AFTER_TIMING_IMPORT


def _synthetic_catalog(action_uids: tuple[int, ...]):
    clip_families = tuple(
        "forehand" if slot % 2 == 0 else "backhand"
        for slot in range(len(action_uids))
    )
    return C.ActionBallFullMdpDiagnosticCatalogTable(
        manifest_file_sha256="1" * 64,
        manifest_canonical_sha256="2" * 64,
        action_order=tuple(f"action_{slot}" for slot in range(len(action_uids))),
        action_uids=action_uids,
        motion_files=tuple(f"motion_{slot}" for slot in range(len(action_uids))),
        motion_sha256=tuple("3" * 64 for _ in action_uids),
        clip_family_per_clip=clip_families,
        strike_phase_per_clip=tuple(0.5 for _ in action_uids),
        mount_normal_sign_per_clip=tuple(1.0 for _ in action_uids),
    )


def test_real_ordered_catalog_is_the_exact_prebroker_schedule_identity() -> None:
    parent, receipt, profile = (
        cadence.build_action_ball_full_mdp_diagnostic_motion_profile()
    )
    command, _env_ids = bridge._configure_unbound_command(
        num_envs=2, profile=profile
    )
    broker_uids = command._action_ball_action_uids
    command._action_ball_full_mdp_diagnostic_catalog_table = (
        _synthetic_catalog(broker_uids)
    )
    command._action_ball_action_uids = None

    parent.bind_exact_parent_schedule(command, receipt)
    schedule = command._action_ball_continuous_schedule_projection
    assert schedule["upcoming_action_slot"] == 0
    assert schedule["upcoming_action_uid"] == broker_uids[0]


def test_catalog_broker_identity_drift_rejects_before_schedule_bind() -> None:
    parent, receipt, profile = (
        cadence.build_action_ball_full_mdp_diagnostic_motion_profile()
    )
    command, _env_ids = bridge._configure_unbound_command(
        num_envs=2, profile=profile
    )
    broker_uids = command._action_ball_action_uids
    command._action_ball_full_mdp_diagnostic_catalog_table = (
        _synthetic_catalog(tuple(uid + 1_000_000 for uid in broker_uids))
    )

    with pytest.raises(
        cadence.MotionCadenceAuthorityConflictError,
        match="upcoming Motion action identity differs",
    ):
        parent.bind_exact_parent_schedule(command, receipt)
    assert command._action_ball_continuous_schedule_projection is None
    assert parent._schedule.cadence_steps == 293
    assert (
        parent._schedule.cadence_steps
        == timing.diagnostic_catalog_max_task_close_ticks()
        + cadence._c02.RECOVERY_END_OFFSET_TICKS
        + 2
    )


def _cuda_gpu2_device() -> torch.device | None:
    if not torch.cuda.is_available():
        return None
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        physical = tuple(cell.strip() for cell in visible.split(","))
        if physical and physical[0] == "2":
            return torch.device("cuda", 0)
    if torch.cuda.device_count() > 2:
        return torch.device("cuda", 2)
    return None


_DEVICES = [torch.device("cpu")]
_GPU2 = _cuda_gpu2_device()
if _GPU2 is not None:
    _DEVICES.append(_GPU2)


def _move_command(command, device: torch.device) -> None:
    if device.type != "cuda":
        return
    for name, value in tuple(vars(command).items()):
        if torch.is_tensor(value):
            setattr(command, name, value.to(device))
    for name, value in tuple(vars(command.motion).items()):
        if torch.is_tensor(value):
            setattr(command.motion, name, value.to(device))
    for name, value in tuple(command.metrics.items()):
        if torch.is_tensor(value):
            command.metrics[name] = value.to(device)
    command._env.scene.env_origins = command._env.scene.env_origins.to(device)
    command.device = device


class _GenesisOnlyQuestionBoundary:
    """Current full-N D05 question protocol, unreachable in genesis tests."""

    def compose_r05_candidate_bank_inside_prepare(self, *_args, **_kwargs):
        raise AssertionError("genesis tests reached D05 question composition")


def _fresh_command_and_owners(
    device: torch.device,
):
    parent, parent_receipt, profile = (
        cadence.build_action_ball_full_mdp_diagnostic_motion_profile()
    )
    command, _env_ids = bridge._configure_unbound_command(
        num_envs=2, profile=profile
    )
    motion_action_uids = (
        command._action_ball_continuous_code_owned_action_uids()
    )
    motion_catalog = _synthetic_catalog(motion_action_uids)
    command.cfg.clip_family_per_clip = motion_catalog.clip_family_per_clip
    # Faithful CommandTerm construction begins with an all-zero resample
    # timer.  Fresh genesis must replace this inherited due state itself.
    command.time_left = torch.zeros(
        2, dtype=torch.float32, device=command.device
    )
    _move_command(command, device)
    parent.bind_exact_parent_schedule(command, parent_receipt)
    command._diagnostic_test_motion_parent_authority = parent
    command._diagnostic_test_motion_parent_receipt = parent_receipt
    cadence_owner = cadence.construct_production_motion_cadence_authority(
        motion_owner=command
    )
    action_family_catalog = (
        cadence_owner.project_action_stroke_family_catalog()
    )
    assert action_family_catalog.action_uids == motion_action_uids
    assert action_family_catalog.family_codes == tuple(
        1 if family == "forehand" else 2
        for family in motion_catalog.clip_family_per_clip
    )

    profile_spec = profile_authority.freeze_device_target_profile_spec(
        frame_id="hope_world_table_xy_m",
        frame_binding_sha256="a" * 64,
        cell_ids=("near_left", "near_right", "deep_center"),
        targets_xy_m=((2.10, -0.20), (2.10, 0.20), (2.80, 0.0)),
    )
    profile_owner, profile_receipt = (
        profile_authority.construct_device_profile_authority(
            profile_spec,
            device=device,
            expected_support_size=3,
        )
    )
    genesis_issue = reset_genesis.issue_action_ball_full_mdp_reset_genesis(
        num_envs=2,
        device=device,
    )
    epoch_genesis = genesis_issue.authority.require_owned_action_epoch_genesis(
        genesis_issue.receipt,
        device=device,
        num_envs=2,
    )
    reset_generation = epoch_genesis.reset_generations
    epoch_owner = E.ActionEpochOwner(
        num_envs=2,
        device=device,
        shot_slot_capacity=1,
        initial_reset_generation=reset_generation,
    )
    epoch_owner.activate_reset_genesis(
        selected_mask=torch.ones(2, dtype=torch.bool, device=device),
        reset_generation=reset_generation,
    )
    device_owner = r05.DeviceR05Owner(
        profile_owner,
        profile_receipt,
        seed=20260804,
        num_envs=2,
        journal_capacity=64,
        max_reveal_epochs_per_drain=64,
        genesis_authority=genesis_issue.authority,
        genesis_receipt=genesis_issue.receipt,
        cadence_authority=cadence_owner,
        question_authority=_GenesisOnlyQuestionBoundary(),
        reveal_boundary_authority=None,
        child_completion_authorities=(),
        diagnostic_epoch_owner=epoch_owner,
    )
    return command, cadence_owner, device_owner, epoch_owner


def _snapshot(command) -> dict[str, torch.Tensor]:
    values = {
        field: getattr(command, attr).detach().clone()
        for field, attr, _nonnegative in (
            C._ACTION_BALL_CONTINUOUS_MOTION_CHECKPOINT_TENSORS
        )
    }
    values.update(
        {
            "time_steps": command.time_steps.detach().clone(),
            "time_steps_f": command.time_steps_f.detach().clone(),
            "speed_scale": command.speed_scale.detach().clone(),
            "hold_counter": command.hold_counter.detach().clone(),
        }
    )
    return values


def _assert_same(command, before: dict[str, torch.Tensor]) -> None:
    after = _snapshot(command)
    assert after.keys() == before.keys()
    for name in before:
        assert torch.equal(after[name], before[name]), name


def _install_frame0_body_contract(command) -> tuple[str, ...]:
    upper = (
        "torso_Link",
        "left_shoulder_roll_Link",
        "left_elbow_Link",
        "left_wrist_yaw_Link",
        "right_shoulder_roll_Link",
        "right_elbow_Link",
        "right_wrist_yaw_Link",
    )
    command.robot.body_names = ["root_Link", *upper]
    command.cfg.body_names = list(command.robot.body_names)
    command.motion.body_pos_w = command.motion.body_pos_w.repeat(1, 8, 1)
    command.motion.body_quat_w = command.motion.body_quat_w.repeat(1, 8, 1)
    return upper


def _install_fake_a3_upper_module(monkeypatch, upper: tuple[str, ...]) -> None:
    feet = ("left_ankle_roll_Link", "right_ankle_roll_Link")
    robots = types.ModuleType("whole_body_tracking.robots")
    agibot_a3 = types.ModuleType("whole_body_tracking.robots.agibot_a3")
    agibot_a3.A3_UPPER_TRACKED = list(upper)
    agibot_a3.A3_FEET_BODIES = list(feet)
    robots.agibot_a3 = agibot_a3
    monkeypatch.setitem(sys.modules, "whole_body_tracking.robots", robots)
    monkeypatch.setitem(
        sys.modules, "whole_body_tracking.robots.agibot_a3", agibot_a3
    )
    monkeypatch.setattr(
        sys.modules["whole_body_tracking"], "robots", robots, raising=False
    )


def test_c01_post_balance_deadline_does_not_close_before_question_task_close() -> None:
    command, _cadence_owner, device_owner, epoch_owner = (
        _fresh_command_and_owners(torch.device("cpu"))
    )
    command.bind_action_ball_continuous_motion_device_r05_reveal(device_owner)
    command.bind_action_ball_full_mdp_motion_epoch_owner(epoch_owner)
    for common_step in range(296):
        command._env.common_step_counter = common_step
        command._advance_action_ball_continuous_motion_cadence()

    assert command._action_ball_continuous_current_deadline_step.tolist() == [297, 297]
    assert command._action_ball_continuous_next_reveal_step.tolist() == [588, 588]
    command._action_ball_continuous_motion_active.fill_(True)
    command._action_ball_continuous_current_policy_opportunity.fill_(True)
    command._action_ball_continuous_canonical_task_valid.fill_(True)
    command._action_ball_continuous_canonical_task_identity.fill_(1)
    command._action_ball_continuous_canonical_cadence_identity.fill_(1)
    command._action_ball_continuous_canonical_action_uid.copy_(
        torch.as_tensor(command._action_ball_action_uids)[command.clip_id]
    )
    command._action_ball_continuous_canonical_task_close_tick.fill_(299)
    command._action_ball_task_timing_active.fill_(True)
    command._action_ball_pre_swing_wait_s.fill_(100.0)
    command._action_ball_scaled_t_cycle_s.fill_(1.0)
    command._action_ball_teacher_rate.fill_(1.0)
    command._action_ball_continuous_canonical_phase.fill_(
        C.ACTION_BALL_CONTINUOUS_CANONICAL_PREPARE_VISIBLE
    )

    for common_step in (296, 297):
        command._env.common_step_counter = common_step
        command._advance_action_ball_continuous_motion_cadence()
    assert torch.all(command._action_ball_continuous_deadline_due)
    assert torch.all(command._action_ball_continuous_motion_active)
    assert torch.all(command._action_ball_continuous_canonical_task_valid)
    assert torch.all(command._action_ball_continuous_current_policy_opportunity)
    assert not torch.any(
        command._action_ball_continuous_canonical_playback_started
    )

    command._env.common_step_counter = 298
    command._advance_action_ball_continuous_motion_cadence()
    assert torch.all(command._action_ball_continuous_motion_active)
    command._env.common_step_counter = 299
    command._advance_action_ball_continuous_motion_cadence()
    assert not torch.any(command._action_ball_continuous_motion_active)
    assert not torch.any(command._action_ball_continuous_canonical_task_valid)
    assert not torch.any(command._action_ball_continuous_current_policy_opportunity)
    assert torch.all(command._action_ball_continuous_ready_reference_active)
    assert torch.all(
        command._action_ball_continuous_canonical_task_close_tick.eq(-1)
    )

@pytest.mark.parametrize("device", _DEVICES, ids=lambda value: str(value))
def test_fresh_compute_owns_initial_due_timer_without_legacy_resample(
    monkeypatch,
    device: torch.device,
) -> None:
    fresh, _cadence_owner, device_owner, _epoch_owner = (
        _fresh_command_and_owners(device)
    )
    assert torch.all(fresh.time_left.eq(0))
    fresh.bind_action_ball_continuous_motion_device_r05_reveal(device_owner)
    assert torch.all(torch.isinf(fresh.time_left))

    base_calls = []

    def _faithful_base_compute(owner, dt):
        base_calls.append(owner)
        owner._update_metrics()
        owner.time_left -= dt
        due = (owner.time_left <= 0.0).nonzero().flatten()
        if len(due) > 0:
            owner._resample(due)
        owner._update_command()

    monkeypatch.setattr(
        C.CommandTerm, "compute", _faithful_base_compute, raising=False
    )
    calls = {"metrics": 0, "update": 0, "resample": []}

    def _metrics(_owner):
        calls["metrics"] += 1

    def _update(_owner):
        calls["update"] += 1

    def _resample(_owner, env_ids):
        calls["resample"].append(env_ids.clone())

    fresh._update_metrics = types.MethodType(_metrics, fresh)
    fresh._update_command = types.MethodType(_update, fresh)
    fresh._resample = types.MethodType(_resample, fresh)
    fresh.compute(0.02)
    fresh.compute(0.02)

    assert base_calls == []
    assert calls == {"metrics": 2, "update": 2, "resample": []}
    assert torch.all(torch.isinf(fresh.time_left))

    legacy, _legacy_cadence, _legacy_owner, _legacy_epoch = (
        _fresh_command_and_owners(device)
    )
    legacy_calls = {"metrics": 0, "update": 0, "resample": []}
    legacy._update_metrics = types.MethodType(
        lambda _owner: legacy_calls.__setitem__(
            "metrics", legacy_calls["metrics"] + 1
        ),
        legacy,
    )
    legacy._update_command = types.MethodType(
        lambda _owner: legacy_calls.__setitem__(
            "update", legacy_calls["update"] + 1
        ),
        legacy,
    )
    legacy._resample = types.MethodType(
        lambda _owner, env_ids: legacy_calls["resample"].append(
            env_ids.clone()
        ),
        legacy,
    )

    legacy.compute(0.02)

    assert base_calls == [legacy]
    assert legacy_calls["metrics"] == 1
    assert legacy_calls["update"] == 1
    assert len(legacy_calls["resample"]) == 1
    assert legacy_calls["resample"][0].tolist() == [0, 1]
    with pytest.raises(RuntimeError, match="resample is tombstoned"):
        fresh._resample_command(torch.arange(2, device=device))


def test_fresh_compute_timer_drift_fails_before_update_or_resample() -> None:
    command, _cadence_owner, device_owner, _epoch_owner = (
        _fresh_command_and_owners(torch.device("cpu"))
    )
    command.bind_action_ball_continuous_motion_device_r05_reveal(device_owner)
    calls = {"metrics": 0, "update": 0}
    command._update_metrics = types.MethodType(
        lambda _owner: calls.__setitem__("metrics", calls["metrics"] + 1),
        command,
    )
    command._update_command = types.MethodType(
        lambda _owner: calls.__setitem__("update", calls["update"] + 1),
        command,
    )
    command.time_left.zero_()

    with pytest.raises(RuntimeError, match="resample timer drifted"):
        command.compute(0.02)

    assert calls == {"metrics": 0, "update": 0}
    assert command._action_ball_continuous_motion_poisoned is True


def test_fresh_genesis_allocates_motion_owned_buffers_without_legacy_broker() -> None:
    command, _cadence_owner, device_owner, _epoch_owner = (
        _fresh_command_and_owners(torch.device("cpu"))
    )
    fresh_fields = (
        "_action_ball_reset_generation",
        "_action_ball_swing_generation",
        "_action_ball_task_timing_active",
        "_action_ball_task_pending_elapsed_s",
        "_action_ball_task_age_s",
        "_action_ball_time_to_contact_s",
        "_action_ball_teacher_rate",
        "_action_ball_scaled_t_hit_s",
        "_action_ball_scaled_t_cycle_s",
        "_action_ball_pre_swing_wait_s",
    )
    command._action_ball_birth_broker = None
    for name in fresh_fields:
        setattr(command, name, None)
    command._action_ball_active_task_refs = None
    command._action_ball_diagnostic_pending_row_count = None

    command.bind_action_ball_continuous_motion_device_r05_reveal(device_owner)

    assert torch.equal(
        command._action_ball_reset_generation,
        torch.ones(2, dtype=torch.int64),
    )
    assert torch.all(command._action_ball_swing_generation.eq(0))
    assert not torch.any(command._action_ball_task_timing_active)
    for name in fresh_fields[3:]:
        value = getattr(command, name)
        assert value.dtype == torch.float64
        assert value.shape == (2,)
        assert torch.all(value.eq(0))
    assert command._action_ball_active_task_refs == [None, None]
    assert command._action_ball_diagnostic_pending_row_count == 0


def test_legacy_continuous_command0_still_requires_birth_broker() -> None:
    parent, receipt, profile = (
        cadence.build_action_ball_full_mdp_diagnostic_motion_profile()
    )
    command, _env_ids = bridge._configure_unbound_command(
        num_envs=2, profile=profile
    )
    parent.bind_exact_parent_schedule(command, receipt)
    command._action_ball_birth_broker = None
    command._stagger_ep_pending = False

    with pytest.raises(RuntimeError, match="birth authority"):
        command._update_command()


def test_stale_genesis_rejects_without_motion_mutation() -> None:
    command, _cadence_owner, device_owner, _epoch_owner = (
        _fresh_command_and_owners(torch.device("cpu"))
    )
    # Any runtime view closes the one construction-only genesis window.
    assert device_owner.runtime_wiring_connected is True
    before = _snapshot(command)
    with pytest.raises(RuntimeError, match="owner-issued Device-R05 genesis"):
        command.bind_action_ball_continuous_motion_device_r05_reveal(
            device_owner
        )
    _assert_same(command, before)
    assert command._action_ball_continuous_motion_device_r05_owner is None
    assert command._action_ball_continuous_fresh_motion_lane_bound is False


def test_genesis_idle_recovery_reference_is_upcoming_motion_frame0(
    monkeypatch,
) -> None:
    command, _cadence_owner, device_owner, epoch_owner = (
        _fresh_command_and_owners(torch.device("cpu"))
    )
    command.bind_action_ball_continuous_motion_device_r05_reveal(device_owner)
    command.bind_action_ball_full_mdp_motion_epoch_owner(epoch_owner)
    upper = _install_frame0_body_contract(command)
    import types

    robots = types.ModuleType("whole_body_tracking.robots")
    agibot_a3 = types.ModuleType("whole_body_tracking.robots.agibot_a3")
    agibot_a3.A3_UPPER_TRACKED = list(upper)
    robots.agibot_a3 = agibot_a3
    monkeypatch.setitem(sys.modules, "whole_body_tracking.robots", robots)
    monkeypatch.setitem(
        sys.modules, "whole_body_tracking.robots.agibot_a3", agibot_a3
    )
    monkeypatch.setattr(
        sys.modules["whole_body_tracking"], "robots", robots, raising=False
    )

    reference = command.project_action_ball_full_mdp_recovery_ready_reference()
    starts = command.motion.seg_start[command.clip_id]
    expected_root = (
        command.motion.body_pos_w[starts, 0]
        + command._env.scene.env_origins
    )
    assert reference.motion_owner is command
    assert reference.epoch_owner is epoch_owner
    assert not hasattr(reference, "epoch")
    assert reference.epoch_version == 0
    assert reference.reference_kind.tolist() == [1, 1]
    assert torch.equal(reference.reference_action_slot, command.clip_id)
    assert torch.equal(
        reference.reference_action_uid,
        torch.as_tensor(command._action_ball_action_uids)[command.clip_id],
    )
    assert torch.equal(reference.root_position_m, expected_root)
    assert torch.all(reference.validity)
    assert not torch.any(reference.producer_fault_bits)
    for field in fields(C._ACTION_BALL_ROW_IDENTITY.ActionEpochShotKey):
        assert getattr(reference.shot_key, field.name).tolist() == [-1, -1]


def test_genesis_idle_reference_allows_named_fact_version_not_identity_drift(
    monkeypatch,
) -> None:
    command, _cadence_owner, device_owner, epoch_owner = (
        _fresh_command_and_owners(torch.device("cpu"))
    )
    command.bind_action_ball_continuous_motion_device_r05_reveal(device_owner)
    command.bind_action_ball_full_mdp_motion_epoch_owner(epoch_owner)
    upper = _install_frame0_body_contract(command)
    import types

    robots = types.ModuleType("whole_body_tracking.robots")
    agibot_a3 = types.ModuleType("whole_body_tracking.robots.agibot_a3")
    agibot_a3.A3_UPPER_TRACKED = list(upper)
    robots.agibot_a3 = agibot_a3
    monkeypatch.setitem(sys.modules, "whole_body_tracking.robots", robots)
    monkeypatch.setitem(
        sys.modules, "whole_body_tracking.robots.agibot_a3", agibot_a3
    )
    monkeypatch.setattr(
        sys.modules["whole_body_tracking"], "robots", robots, raising=False
    )

    # A real IDLE owner-fact publication advances only the logical version.
    epoch_owner.bind_fact_owner("r07_recovery", object())
    fact_owner = epoch_owner._fact_owner_identities["r07_recovery"]
    epoch_owner.publish_owner_facts(
        "r07_recovery",
        owner=fact_owner,
        valid_bits=torch.zeros((2, 1), dtype=torch.int64),
        source_step=torch.full((2, 1), -1, dtype=torch.int64),
        values=torch.zeros((2, 1, E.OWNER_FACT_F32_WIDTH), dtype=torch.float32),
    )
    reference = command.project_action_ball_full_mdp_recovery_ready_reference()
    assert not hasattr(reference, "epoch")
    assert reference.epoch_version > 0
    assert torch.all(reference.validity)

    command.clip_id.fill_(1)
    slot_drifted = command.project_action_ball_full_mdp_recovery_ready_reference()
    assert not torch.any(slot_drifted.validity)
    assert torch.all(slot_drifted.producer_fault_bits.ne(0))

    command.clip_id.fill_(0)
    action_uids = command._action_ball_action_uids
    command._action_ball_action_uids = (
        action_uids[0] + 1,
        *action_uids[1:],
    )
    uid_drifted = command.project_action_ball_full_mdp_recovery_ready_reference()
    assert not torch.any(uid_drifted.validity)
    assert torch.all(uid_drifted.producer_fault_bits.ne(0))


def test_duplicate_bind_is_idempotent_and_foreign_rebind_is_zero_mutation() -> None:
    command, _cadence_owner, device_owner, epoch_owner = (
        _fresh_command_and_owners(torch.device("cpu"))
    )
    command.bind_action_ball_continuous_motion_device_r05_reveal(device_owner)
    command.bind_action_ball_full_mdp_motion_epoch_owner(epoch_owner)
    before = _snapshot(command)
    command.bind_action_ball_continuous_motion_device_r05_reveal(device_owner)
    _assert_same(command, before)

    _foreign_command, _foreign_cadence, foreign, _foreign_epoch = (
        _fresh_command_and_owners(torch.device("cpu"))
    )
    with pytest.raises(RuntimeError, match="may not be rebound"):
        command.bind_action_ball_continuous_motion_device_r05_reveal(foreign)
    _assert_same(command, before)
