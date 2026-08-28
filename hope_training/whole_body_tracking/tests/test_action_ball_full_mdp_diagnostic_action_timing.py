"""Focused contract for the cold generic-N diagnostic action-profile owner."""

from __future__ import annotations

import copy
from dataclasses import replace
import inspect
import math
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

import pytest
import numpy as np
import torch


_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "source" / "whole_body_tracking"
_MDP_ROOT = _SOURCE_ROOT / "whole_body_tracking" / "tasks" / "tracking" / "mdp"
for _path in (_SOURCE_ROOT, _MDP_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import test_reward_flags_mdp as loaded_mdp  # noqa: E402
import test_action_ball_continuous_racket_device_reveal_hold as ref_test  # noqa: E402
import test_action_ball_exact_face_timing_device as exact_face_test  # noqa: E402


_PKG = "whole_body_tracking.tasks.tracking.mdp"
if f"{_PKG}.action_ball_manifest" not in sys.modules:
    manifest_module = loaded_mdp._load(  # type: ignore[attr-defined]
        f"{_PKG}.action_ball_manifest", "action_ball_manifest.py"
    )
    setattr(sys.modules[_PKG], "action_ball_manifest", manifest_module)

import action_ball_full_mdp_diagnostic_action_timing as profile_mod  # noqa: E402


HC = loaded_mdp.hope_commands_mod
commands = loaded_mdp.commands_mod
_CATALOG_SLOTS = (0,)


def _catalog_rows():
    actions = profile_mod._load_pinned_catalog().manifest.actions
    return tuple(actions[index] for index in _CATALOG_SLOTS)


def test_pinned_a3p0807_catalog_is_grounded_on_the_articulation_root():
    rows = profile_mod._load_pinned_catalog().manifest.actions
    repo_root = Path(profile_mod.__file__).resolve().parents[4]
    for row in rows:
        with np.load(repo_root / row.motion_path, allow_pickle=False) as motion:
            assert motion["body_names"].tolist()[0] == "pelvis_link"
            w, x, y, z = (
                float(value) for value in motion["body_quat_w"][0, 0]
            )
        yaw = math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )
        assert abs(yaw) <= 1.0e-6


def _install_reference_table(
    racket,
    motion,
    rows,
    *,
    device: torch.device,
    angular_velocity_z_radps=None,
):
    count = len(rows)
    loader = motion.motion
    quat = torch.zeros(count, 4, dtype=torch.float32, device=device)
    quat[:, 0] = 1.0
    omega = torch.zeros(count, 3, dtype=torch.float32, device=device)
    if angular_velocity_z_radps is not None:
        omega[:, 2] = torch.as_tensor(
            angular_velocity_z_radps,
            dtype=torch.float32,
            device=device,
        )
    velocity = torch.zeros(count, 3, dtype=torch.float32, device=device)
    velocity[:, 0] = torch.tensor(
        [row.reference_racket_site_speed_mps for row in rows],
        dtype=torch.float32,
        device=device,
    )
    raw_normal = torch.zeros(count, 3, dtype=torch.float32, device=device)
    raw_normal[:, 1] = 1.0
    reach = torch.tensor(
        [[0.62, -0.11] for _row in rows],
        dtype=torch.float32,
        device=device,
    )
    base = quat.clone()
    segment_lengths = torch.tensor(
        [round(row.reference_t_cycle_s / racket._env.step_dt) + 1 for row in rows],
        dtype=torch.int64,
        device=device,
    )
    segment_starts = torch.cat(
        (
            torch.zeros(1, dtype=torch.int64, device=device),
            torch.cumsum(segment_lengths[:-1], dim=0),
        )
    )
    total_frames = int(segment_lengths.sum().item())
    loader.seg_len = segment_lengths
    loader.seg_start = segment_starts
    loader.time_step_total = total_frames
    loader._body_pos_w = torch.zeros(
        total_frames, 2, 3, dtype=torch.float32, device=device
    )
    loader._body_quat_w = torch.zeros(
        total_frames, 2, 4, dtype=torch.float32, device=device
    )
    loader._body_quat_w[..., 0] = 1.0
    loader._body_lin_vel_w = torch.zeros(
        total_frames, 2, 3, dtype=torch.float32, device=device
    )
    loader._body_ang_vel_w = torch.zeros_like(loader._body_lin_vel_w)
    for action_slot, row in enumerate(rows):
        start = int(segment_starts[action_slot].item())
        length = int(segment_lengths[action_slot].item())
        strike = start + round(float(row.strike_phase) * (length - 1))
        loader._body_pos_w[start : start + length, 1, :2] = reach[action_slot]
        loader._body_pos_w[strike - 1, 1] -= (
            velocity[action_slot] * racket._env.step_dt
        )
        loader._body_pos_w[strike + 1, 1] += (
            velocity[action_slot] * racket._env.step_dt
        )
        loader._body_quat_w[start : start + length, 1] = quat[action_slot]
        loader._body_quat_w[start : start + length, 0] = base[action_slot]
        loader._body_ang_vel_w[start : start + length, 1] = omega[action_slot]
    racket.cfg.clean_strike_vel_window = 1
    racket.cfg.clean_reference_strike_velocity = True
    racket.cfg.mount_normal_axis = 1
    racket._racket_mode = "body"
    racket._racket_body_index = 1
    racket.initialize_action_ball_full_mdp_racket_action_reference_cold()


def _reference_cadence(
    device: torch.device,
    action_slot: torch.Tensor,
) -> SimpleNamespace:
    num_envs = int(action_slot.shape[0])
    lane = torch.arange(num_envs, dtype=torch.int64, device=device)
    return SimpleNamespace(
        selected_env_index=lane,
        selected_count=num_envs,
        episode_tick=torch.full_like(lane, 2),
        reveal_tick=torch.full_like(lane, 2),
        deadline_tick=torch.full_like(lane, 4),
        next_reveal_tick=torch.full_like(lane, 48),
        swing_generation=lane + 1,
        ready_at_reveal=torch.ones(num_envs, dtype=torch.bool, device=device),
        action_slot=action_slot,
        pending_elapsed_s=torch.zeros(
            num_envs, dtype=torch.float32, device=device
        ),
        reset_generation=lane + 1,
        scheduled_ordinal=lane + 1,
        outcome_shot_index=lane,
        sampler_generation=lane + 1,
        task_identity=torch.full_like(lane, -1),
        cadence_identity=torch.full_like(lane, -1),
        cadence_producer_fault=torch.zeros_like(lane),
        action_uid=torch.full_like(lane, -1),
        contact_tick=torch.full_like(lane, 50),
        launch_tick=torch.full_like(lane, 42),
        chosen_horizon_ticks=torch.full_like(lane, 8),
        task_close_tick=torch.full_like(lane, 200),
        cadence_owner_receipt_identity=object(),
    )


def _harness(
    runtime_device: str = "cpu",
    *,
    slots=(0, 0),
    angular_velocity_z_radps=None,
):
    if runtime_device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    device = torch.device(runtime_device)
    rows = _catalog_rows()
    num_envs = len(slots)
    env = SimpleNamespace(
        num_envs=num_envs,
        device=device,
        step_dt=0.02,
        max_episode_length=500,
        common_step_counter=0,
    )
    loader = commands.MotionLoader.__new__(commands.MotionLoader)
    loader.kinematics_contract_exact = True
    loader.num_segments = len(rows)
    loader.seg_len = torch.tensor(
        [round(row.reference_t_cycle_s / env.step_dt) + 1 for row in rows],
        dtype=torch.int64,
        device=device,
    )
    motion = commands.MotionCommand.__new__(commands.MotionCommand)
    motion.motion = loader
    motion.num_envs = num_envs
    motion.device = device
    motion._env = env
    motion._canonical_diagnostic_unauthorized = True
    repo_root = Path(profile_mod.__file__).resolve().parents[4]
    motion._motion_files = tuple(
        str((repo_root / row.motion_path).resolve()) for row in rows
    )
    motion._motion_file_sha256 = tuple(row.motion_sha256 for row in rows)
    motion._motion_payloads = tuple(
        Path(path).read_bytes() for path in motion._motion_files
    )

    racket = HC.RacketTargetCommand.__new__(HC.RacketTargetCommand)
    racket.num_envs = num_envs
    racket.device = device
    racket._env = env
    racket._motion_term = motion
    racket._action_ball_enabled = False
    racket._action_ball_full_mdp_enabled = True
    racket.cfg = SimpleNamespace(
        strike_phase_per_clip=tuple(row.strike_phase for row in rows),
        strike_phase=0.46,
        mount_normal_sign_per_clip=tuple(
            row.mount_normal_sign for row in rows
        ),
        mount_normal_sign=1.0,
        motion_teacher_racket_source="robot_fk",
        cq_speed_budget=3.4,
    )
    _install_reference_table(
        racket,
        motion,
        rows,
        device=device,
        angular_velocity_z_radps=angular_velocity_z_radps,
    )

    slot_tensor = torch.tensor(slots, dtype=torch.int64, device=device)
    cadence = _reference_cadence(device, slot_tensor)
    safe_slots = slot_tensor.clamp(min=0, max=len(rows) - 1)
    action_uid = torch.tensor(
        [rows[int(index)].action_uid for index in safe_slots.cpu().tolist()],
        dtype=torch.int64,
        device=device,
    )
    cadence.action_uid = action_uid
    return rows, env, motion, racket, cadence


def _construct(runtime_device: str = "cpu", *, slots=(0, 0)):
    values = _harness(runtime_device, slots=slots)
    owner = profile_mod.construct_action_ball_full_mdp_diagnostic_action_timing_owner(
        racket_owner=values[3], cadence_projection=values[4]
    )
    return values, owner, owner.project()




@pytest.mark.parametrize("num_envs", (1, 2, 64))
def test_static_timing_accepts_positive_exact_generic_n(num_envs):
    rows, env, motion, racket, _cadence = _harness(
        slots=(0,) * num_envs
    )
    table = (
        profile_mod.
        construct_action_ball_full_mdp_diagnostic_action_timing_static_table(
            racket_owner=racket
        )
    )
    assert env.num_envs == motion.num_envs == racket.num_envs == num_envs
    assert type(table) is profile_mod.DiagnosticActionTimingStaticTableProjection
    assert table.action_uid.shape == (len(rows),)
    assert table.time_to_contact_ticks.shape == (len(rows),)
    assert table.teacher_rate_min.shape == (len(rows),)
    assert table.teacher_rate_max.shape == (len(rows),)
    assert torch.all(table.teacher_rate_min > 0.0)
    assert torch.all(table.teacher_rate_max >= table.teacher_rate_min)
    assert table.diagnostic_unauthorized is True
    assert table.runtime_integrated is False
    assert table.launch_authorized is False
    assert table.formal_admission is False


@pytest.mark.parametrize("invalid", (True, False, 0, -1, 2.0, "2"))
def test_static_timing_rejects_nonpositive_or_nonexact_racket_n(invalid):
    _rows, _env, _motion, racket, _cadence = _harness()
    racket.num_envs = invalid
    with pytest.raises(
        profile_mod.DiagnosticActionTimingError,
        match="positive exact-N",
    ):
        profile_mod.construct_action_ball_full_mdp_diagnostic_action_timing_static_table(
            racket_owner=racket
        )


@pytest.mark.parametrize("mutation", ("motion_n", "env_n", "foreign_motion"))
def test_static_timing_rejects_foreign_owner_cardinality(mutation):
    _rows, env, motion, racket, _cadence = _harness()
    if mutation == "motion_n":
        motion.num_envs = 1
    elif mutation == "env_n":
        env.num_envs = 1
    else:
        foreign = commands.MotionCommand.__new__(commands.MotionCommand)
        foreign.motion = motion.motion
        foreign.num_envs = motion.num_envs
        foreign.device = motion.device
        foreign._env = motion._env
        racket._motion_term = foreign
    with pytest.raises(
        profile_mod.DiagnosticActionTimingError,
        match="identity differs|explicit unauthorized",
    ):
        profile_mod.construct_action_ball_full_mdp_diagnostic_action_timing_static_table(
            racket_owner=racket
        )


def test_static_timing_has_no_parallel_dynamic_owner_or_constructor():
    for name in (
        "DiagnosticActionTimingOwner",
        "DiagnosticActionTimingProjection",
        "construct_action_ball_full_mdp_diagnostic_action_timing_owner",
    ):
        assert not hasattr(profile_mod, name)


def test_static_timing_catalog_and_production_hold_remain_explicit():
    signature = inspect.signature(
        profile_mod.construct_action_ball_full_mdp_diagnostic_action_timing_static_table
    )
    assert tuple(signature.parameters) == ("racket_owner",)
    assert profile_mod.diagnostic_catalog_max_task_close_ticks() == 106
    with pytest.raises(
        profile_mod.DiagnosticActionTimingProductionHold,
        match="formally admitted",
    ):
        profile_mod.construct_production_action_timing_owner()


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("time_to_contact_center_s", 1.231),
        ("time_to_contact_min_s", 2.0),
        ("time_to_contact_max_s", 0.1),
    ),
)
def test_time_to_contact_center_must_be_in_profile_and_on_policy_grid(
    monkeypatch, field, replacement
):
    rows = list(profile_mod._load_pinned_catalog().manifest.actions)
    ball = replace(rows[0].ball_profile, **{field: replacement})
    rows[0] = replace(rows[0], ball_profile=ball)
    with pytest.raises(
        profile_mod.DiagnosticActionTimingError,
        match="time-to-contact center",
    ):
        profile_mod._strict_time_to_contact_ticks(
            tuple(rows), policy_step_s=profile_mod.DIAGNOSTIC_POLICY_STEP_S
        )
