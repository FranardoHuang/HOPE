"""Fresh Racket timer, R03 arm, and cold-reference contract tests.

The active D05 ACCEPT writer is exercised through its token-only full-N ABI in
``test_action_ball_racket_rowwise_accept.py``.  This file retains the independent
fresh-command and MotionLoader-backed reference behavior; the deleted compact
preview/exact-face HOLD transaction has no compatibility tests here.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch


_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "source" / "whole_body_tracking"
_MDP_ROOT = _SOURCE_ROOT / "whole_body_tracking" / "tasks" / "tracking" / "mdp"
for _path in (_SOURCE_ROOT, _MDP_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import action_ball_continuous_runtime_transaction_device as device_r05  # noqa: E402
import test_action_ball_continuous_runtime_transaction_device as d05_test  # noqa: E402
_D05_EPOCH_BEFORE_RACKET_IMPORT = device_r05._require_action_epoch_module()

import test_reward_flags_mdp as loaded_mdp  # noqa: E402
import test_spdmix_per_clip_binding as spdmix  # noqa: E402
from whole_body_tracking.tasks.tracking.mdp import commands as command_module  # noqa: E402
import action_ball_full_mdp_portable_catalog as portable_catalog  # noqa: E402
import racket_contact_geometry as racket_geometry  # noqa: E402
_EPOCH_NAME = "whole_body_tracking.tasks.tracking.mdp.action_ball_full_mdp_epoch"
epoch = sys.modules.get(_EPOCH_NAME)
if epoch is None:
    _EPOCH_SPEC = importlib.util.spec_from_file_location(
        _EPOCH_NAME, _MDP_ROOT / "action_ball_full_mdp_epoch.py"
    )
    assert _EPOCH_SPEC is not None and _EPOCH_SPEC.loader is not None
    epoch = importlib.util.module_from_spec(_EPOCH_SPEC)
    sys.modules[_EPOCH_NAME] = epoch
    _EPOCH_SPEC.loader.exec_module(epoch)
setattr(
    sys.modules["whole_body_tracking.tasks.tracking.mdp"],
    "action_ball_full_mdp_epoch",
    epoch,
)
_R03_NAME = "whole_body_tracking.tasks.tracking.mdp.action_ball_strike_fact_device"
r03 = sys.modules.get(_R03_NAME)
if r03 is None:
    _R03_SPEC = importlib.util.spec_from_file_location(
        _R03_NAME, _MDP_ROOT / "action_ball_strike_fact_device.py"
    )
    assert _R03_SPEC is not None and _R03_SPEC.loader is not None
    r03 = importlib.util.module_from_spec(_R03_SPEC)
    sys.modules[_R03_NAME] = r03
    _R03_SPEC.loader.exec_module(r03)
setattr(
    sys.modules["whole_body_tracking.tasks.tracking.mdp"],
    "action_ball_strike_fact_device",
    r03,
)


HC = loaded_mdp.hope_commands_mod


def test_racket_import_preserves_canonical_d05_epoch_module_identity():
    assert epoch is _D05_EPOCH_BEFORE_RACKET_IMPORT
    assert device_r05._require_action_epoch_module() is epoch


_FRESH_COMMAND_LIVE_FIELDS = (
    "racket_target_pos_w",
    "racket_target_vel_w",
    "target_normal_cmd",
    "_action_ball_ball_contact_target_w",
    "_action_ball_face_center_velocity_target_w",
    "_action_ball_racket_command_quat_w",
    "base_target_pos_w",
    "vb_vel_in_w",
    "vb_spin_in_w",
    "_vb_target_xy_per_env",
    "time_to_strike",
    "pre_strike",
    "strike_window",
    "strike_window_pos",
    "strike_window_wide",
    "_action_ball_task_valid",
    "_action_ball_attempt_active",
    "_action_ball_action_uid",
    "_action_ball_action_slot",
    "_action_ball_reset_generation",
    "_action_ball_swing_generation",
    "_action_ball_full_mdp_racket_task_identity",
)


def _epoch_racket(r05_owner, *, device, bind_r03=False):
    target_device = torch.device(device)
    racket = HC.RacketTargetCommand.__new__(HC.RacketTargetCommand)
    racket.num_envs = 2
    racket.device = target_device
    racket._action_ball_enabled = False
    racket._action_ball_full_mdp_enabled = True
    racket._action_ball_full_mdp_device_r05_owner = None
    racket._action_ball_full_mdp_racket_epoch_owner = None
    racket._action_ball_continuous_racket_poisoned = False
    racket._action_ball_continuous_racket_poison_reason = None
    racket._action_ball_continuous_racket_drain_fault_count_device = torch.zeros(
        1, dtype=torch.int64, device=target_device
    )
    racket._action_ball_continuous_racket_active_ppo_drain_pack = None
    for name in (
        "_action_ball_continuous_racket_selected_reset_stage",
        "_action_ball_continuous_racket_selected_reset_record",
        "_action_ball_continuous_racket_selected_reset_prevalidated",
        "_action_ball_continuous_racket_selected_reset_swaps",
        "_action_ball_continuous_racket_selected_reset_sealed_afterimage",
        "_action_ball_continuous_racket_selected_reset_swap_receipts",
        "_action_ball_continuous_racket_selected_reset_version_receipt_after",
        "_action_ball_continuous_racket_selected_reset_logical_root_after",
        "_action_ball_continuous_racket_selected_reset_commit_token",
        "_action_ball_continuous_racket_selected_reset_completion",
        "_action_ball_continuous_racket_selected_reset_completion_prepared",
    ):
        setattr(racket, name, None)
    float_names = {
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
    for name, width in float_names.items():
        setattr(racket, name, torch.zeros(2, width, device=target_device))
    racket.time_to_strike = torch.zeros(2, device=target_device)
    # CommandTerm.__init__ begins with a zero inherited timer.  The exact
    # fresh D05/epoch cold bind below must replace it with the non-expiring
    # sentinel before the first external CommandManager.compute.
    racket.time_left = torch.zeros(2, device=target_device)
    racket.pre_strike = torch.zeros(2, dtype=torch.bool, device=target_device)
    racket.strike_window = torch.zeros_like(racket.pre_strike)
    racket.strike_window_pos = torch.zeros_like(racket.pre_strike)
    racket.strike_window_wide = torch.zeros_like(racket.pre_strike)
    racket._counter_rally_reward_terms = torch.zeros(2, 5, device=target_device)
    racket._counter_rally_accepted = torch.zeros_like(racket.pre_strike)
    racket._counter_rally_legal_first_landing = torch.zeros_like(racket.pre_strike)
    racket._counter_rally_primary_reason_code = torch.full(
        (2,), -1, dtype=torch.int64, device=target_device
    )
    racket._action_ball_prev_contact_valid = torch.zeros_like(racket.pre_strike)
    racket._action_ball_task_valid = torch.zeros_like(racket.pre_strike)
    racket._action_ball_attempt_active = torch.zeros_like(racket.pre_strike)
    racket._action_ball_attempt_legal = torch.zeros_like(racket.pre_strike)
    racket._action_ball_attempt_hit = torch.zeros_like(racket.pre_strike)
    for name in (
        "_action_ball_action_uid",
        "_action_ball_action_slot",
        "_action_ball_reset_generation",
        "_action_ball_swing_generation",
        "_action_ball_attempt_action",
        "_action_ball_full_mdp_racket_task_identity",
        "_action_ball_full_mdp_racket_outcome_shot_index",
    ):
        setattr(
            racket,
            name,
            torch.full((2,), -1, dtype=torch.int64, device=target_device),
        )
    racket.cfg = SimpleNamespace(
        strike_window_s=0.1,
        strike_window_pos_s=0.05,
        strike_window_wide_s=0.2,
    )
    origins = torch.tensor(
        [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]],
        dtype=torch.float32,
        device=target_device,
    )
    racket._env = SimpleNamespace(
        common_step_counter=4,
        scene=SimpleNamespace(env_origins=origins),
    )
    epoch_owner = epoch.ActionEpochOwner(
        num_envs=2,
        device=target_device,
        shot_slot_capacity=1,
        initial_reset_generation=torch.ones(
            2, dtype=torch.int64, device=target_device
        ),
    )
    epoch_owner.activate_reset_genesis(
        selected_mask=torch.ones(
            2, dtype=torch.bool, device=target_device
        ),
        reset_generation=torch.ones(
            2, dtype=torch.int64, device=target_device
        ),
    )
    if bind_r03:
        _bind_epoch_r03(racket, epoch_owner)
    racket.bind_action_ball_full_mdp_racket_epoch_sources(
        r05_owner, epoch_owner
    )
    return racket, epoch_owner


def _bind_epoch_r03(racket, epoch_owner):
    racket._action_ball_strike_fact_device_enabled = True
    racket._action_ball_strike_fact_source_step = torch.full(
        (racket.num_envs,), -1, dtype=torch.int64, device=racket.device
    )
    racket._action_ball_strike_fact_exact_eligibility = torch.zeros(
        racket.num_envs, dtype=torch.bool, device=racket.device
    )
    racket._action_ball_strike_fact_target_validity = torch.ones_like(
        racket._action_ball_strike_fact_exact_eligibility
    )
    racket._action_ball_strike_fact_target_validity[0] = True
    racket._action_ball_strike_fact_expected_publish_step = None
    racket._action_ball_full_mdp_r03_writer_active = False
    owner = r03.ActionBallStrikeFactDeviceCoordinator(
        num_envs=racket.num_envs,
        device=racket.device,
        observation_projection_mode=r03.OBSERVATION_PROJECTION_MODE_FRESH_FULL_MDP,
    )
    owner.bind_action_epoch_owner(epoch_owner)
    owner.bind_action_epoch_racket_owner(racket)
    racket._action_ball_strike_fact_device_coordinator = owner
    return owner


@pytest.mark.parametrize("runtime_device", ("cpu", "cuda:0"))
def test_fresh_first_external_reset_and_subsequent_compute_skip_resample(
    monkeypatch, runtime_device
):
    if runtime_device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    harness = d05_test._harness(2, device=runtime_device)
    racket, _epoch_owner = _epoch_racket(
        harness.owner, device=runtime_device
    )
    racket._env.common_step_counter = 0
    assert torch.isposinf(racket.time_left).all()
    before = {
        name: getattr(racket, name).clone()
        for name in _FRESH_COMMAND_LIVE_FIELDS
    }
    calls = []
    racket._update_metrics = lambda: calls.append("metrics")
    original_update = racket._update_command

    def update():
        calls.append("update")
        return original_update()

    racket._update_command = update
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
    racket._env.common_step_counter = 1
    assert racket.compute(0.02) is None

    assert calls == ["metrics", "update", "metrics", "update"]
    assert torch.isposinf(racket.time_left).all()
    assert command_module._tensor_matches_identity_version_receipt(
        racket.time_left,
        racket._action_ball_continuous_fresh_racket_time_left_receipt,
    )
    for name, expected in before.items():
        assert torch.equal(getattr(racket, name), expected), name


@pytest.mark.parametrize("runtime_device", ("cpu", "cuda:0"))
@pytest.mark.parametrize("mutation", ("replace", "inplace"))
def test_fresh_inherited_timer_drift_sticky_poisons_before_metrics(
    runtime_device, mutation
):
    if runtime_device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    harness = d05_test._harness(2, device=runtime_device)
    racket, _epoch_owner = _epoch_racket(
        harness.owner, device=runtime_device
    )
    racket._update_metrics = lambda: pytest.fail(
        "timer drift reached metrics before fail-stop"
    )
    if mutation == "replace":
        racket.time_left = racket.time_left.clone()
    else:
        racket.time_left[0] = 0.0

    with pytest.raises(
        HC.ActionBallContinuousRacketHotRevealHold,
        match="inherited resample timer drifted",
    ):
        racket.compute(0.02)

    assert racket._action_ball_continuous_racket_poisoned is True
    assert (
        racket._action_ball_continuous_racket_poison_reason
        == "fresh Racket inherited resample timer drifted"
    )
    assert torch.equal(
        racket._action_ball_continuous_racket_drain_fault_count_device,
        torch.ones(1, dtype=torch.int64, device=racket.device),
    )


def test_fresh_command_noop_precedes_legacy_logic_and_arm_is_post_d05_only():
    update_source = inspect.getsource(HC.RacketTargetCommand._update_command)
    assert update_source.index("fresh_racket_lane_bound") < update_source.index(
        "_ensure_action_ball_runtime_initialized"
    )
    compute_source = inspect.getsource(HC.RacketTargetCommand.compute)
    assert "if fresh_lane:" in compute_source
    assert compute_source.index("self._update_metrics()") < compute_source.index(
        "self.time_left -= dt"
    ) < compute_source.index("self._update_command()")
    assert (
        "self._arm_action_ball_strike_fact_for_next_transition()"
        in compute_source
    )


def test_full_mdp_initial_reset_defers_r03_arm_until_d05_reveal():
    racket = SimpleNamespace(
        _action_ball_strike_fact_device_enabled=True,
        _action_ball_full_mdp_enabled=True,
        _env=SimpleNamespace(common_step_counter=0),
        _action_ball_strike_fact_expected_publish_step=None,
        _arm_action_ball_strike_fact_for_next_transition=lambda: pytest.fail(
            "full-MDP initial reset armed R03 before D05 reveal"
        ),
    )
    HC.RacketTargetCommand._arm_action_ball_strike_fact_after_initial_reset(
        racket
    )
    assert racket._action_ball_strike_fact_expected_publish_step is None


def _exact_loader(action_count: int):
    loader = command_module.MotionLoader.__new__(command_module.MotionLoader)
    loader.kinematics_contract_exact = True
    loader.num_segments = action_count
    return loader


def _install_exact_motionloader_reference_rows(
    racket, motion, values, *, initialize: bool = True
):
    """Make exact MotionLoader FK facts, then use the frozen production builder.

    Tests may choose MotionLoader poses, but never call a seal helper or replace
    the production builder.  The closure-private publication is populated only
    through the same exact class construction callpoint as production.
    """

    cloned = tuple(value.clone().contiguous() for value in values)
    if len(cloned) == 6:
        legacy_rows = True
        quat, omega, velocity, _expected_normal, reach_offset_xy, base = cloned
        site_position = torch.zeros(
            quat.shape[0], 3, dtype=torch.float32, device=quat.device
        )
        site_position[:, :2] = reach_offset_xy
    elif len(cloned) == 7:
        legacy_rows = False
        (
            quat,
            omega,
            velocity,
            _expected_normal,
            reach_offset_xy,
            site_position,
            base,
        ) = cloned
    else:
        raise AssertionError("reference-row fixture expects six or seven tensors")
    device = quat.device
    action_count = quat.shape[0]
    frames_per_action = 5
    total_frames = action_count * frames_per_action
    loader = _exact_loader(action_count)
    loader.seg_start = torch.arange(
        0,
        total_frames,
        frames_per_action,
        dtype=torch.int64,
        device=device,
    )
    loader.seg_len = torch.full(
        (action_count,),
        frames_per_action,
        dtype=torch.int64,
        device=device,
    )
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
    dt = 0.02
    for action_slot in range(action_count):
        start = action_slot * frames_per_action
        strike = start + 2
        loader._body_pos_w[start : start + frames_per_action, 1] = (
            site_position[action_slot]
        )
        if not legacy_rows:
            loader._body_pos_w[start : start + frames_per_action, 0, :2] = (
                site_position[action_slot, :2] - reach_offset_xy[action_slot]
            )
        loader._body_pos_w[strike - 1, 1] -= velocity[action_slot] * dt
        loader._body_pos_w[strike + 1, 1] += velocity[action_slot] * dt
        loader._body_quat_w[
            start : start + frames_per_action, 1
        ] = quat[action_slot]
        loader._body_quat_w[
            start : start + frames_per_action, 0
        ] = base[action_slot]
        loader._body_ang_vel_w[
            start : start + frames_per_action, 1
        ] = omega[action_slot]
    motion.motion = loader
    racket._motion_term = motion
    racket._env = SimpleNamespace(step_dt=dt)
    racket.cfg = SimpleNamespace(
        strike_phase=0.5,
        strike_phase_per_clip=(),
        clean_strike_vel_window=1,
        clean_reference_strike_velocity=True,
        mount_normal_axis=1,
    )
    racket._racket_mode = "body"
    racket._racket_body_index = 1
    if initialize:
        racket.initialize_action_ball_full_mdp_racket_action_reference_cold()


def test_fresh_resample_holds_before_uniform_or_any_state_mutation():
    racket = HC.RacketTargetCommand.__new__(HC.RacketTargetCommand)
    racket._action_ball_full_mdp_enabled = True
    racket._action_ball_enabled = False
    before = vars(racket).copy()
    with pytest.raises(
        HC.ActionBallContinuousRacketHotRevealHold,
        match="awaits the Device-R05 Racket writer",
    ):
        racket._resample_command((0, 1))
    assert vars(racket) == before


def test_cold_materializer_uses_pure_fk_builder_and_not_legacy_bundle(
    monkeypatch,
):
    racket = HC.RacketTargetCommand.__new__(HC.RacketTargetCommand)
    racket.device = torch.device("cpu")
    racket._action_ball_full_mdp_enabled = True
    racket._action_ball_enabled = False
    motion = command_module.MotionCommand.__new__(command_module.MotionCommand)
    values = (
        torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
        torch.zeros(2, 3),
        torch.tensor([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]),
        torch.tensor([[0.5, -0.2], [0.7, 0.3]]),
        torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
    )
    _install_exact_motionloader_reference_rows(
        racket, motion, values, initialize=False
    )
    racket._action_ball_bundle = object()
    called = []

    def exact_fk_builder(owner, exact_motion, loader):
        called.append((owner, exact_motion, loader))
        return (
            torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
            ),
            torch.zeros(2, 3),
            torch.tensor(
            [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
            ),
            torch.tensor(
            [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]
            ),
            torch.tensor([[0.5, -0.2], [0.7, 0.3]]),
        )

    monkeypatch.setattr(
        HC.RacketTargetCommand,
        "_build_action_ball_full_mdp_racket_action_reference_cold_rows",
        exact_fk_builder,
    )
    racket.initialize_action_ball_full_mdp_racket_action_reference_cold()
    # The exact builder identity was captured at class definition; replacing
    # the public class attribute cannot inject rows into the sealed registry.
    assert called == []
    assert vars(HC.RacketTargetCommand)[
        "_action_ball_full_mdp_racket_action_reference_cold_public_methods"
    ]()[0](
        racket, motion, motion.motion
    )
    assert not hasattr(
        racket, "_action_ball_full_mdp_racket_action_reference_cold_table"
    )

    motion.motion.kinematics_contract_exact = False
    called.clear()
    with pytest.raises(
        HC.ActionBallContinuousRacketHotRevealHold,
        match="exact schema-2 MotionLoader FK",
    ):
        racket.initialize_action_ball_full_mdp_racket_action_reference_cold()
    assert called == []


def test_cold_materializer_builder_failure_installs_no_state(monkeypatch):
    racket = HC.RacketTargetCommand.__new__(HC.RacketTargetCommand)
    racket.device = torch.device("cpu")
    racket._action_ball_full_mdp_enabled = True
    racket._action_ball_enabled = False
    motion = command_module.MotionCommand.__new__(command_module.MotionCommand)
    motion.motion = _exact_loader(2)
    racket._motion = lambda: motion

    before = vars(racket).copy()
    with pytest.raises(AttributeError):
        racket.initialize_action_ball_full_mdp_racket_action_reference_cold()
    assert vars(racket) == before
    assert not vars(HC.RacketTargetCommand)[
        "_action_ball_full_mdp_racket_action_reference_cold_public_methods"
    ]()[0](racket, motion, motion.motion)


def test_cold_static_table_rejects_unsealed_owner_without_building_on_demand():
    racket = HC.RacketTargetCommand.__new__(HC.RacketTargetCommand)
    racket.device = torch.device("cpu")
    racket._action_ball_full_mdp_enabled = True
    racket._action_ball_enabled = False
    motion = command_module.MotionCommand.__new__(command_module.MotionCommand)
    motion.motion = _exact_loader(2)
    racket._motion_term = motion
    racket._motion = lambda: (_ for _ in ()).throw(
        AssertionError("cold projection must not build on demand")
    )

    with pytest.raises(
        HC.ActionBallContinuousRacketHotRevealHold,
        match="construction-sealed cold source",
    ):
        racket.project_action_ball_full_mdp_racket_action_reference_static_table()


def test_real_motionloader_backed_cold_materializer_publishes_static_table(tmp_path):
    files = [
        spdmix._write_motion_npz(
            str(tmp_path / f"clip{index}.npz"), frames=frames
        )
        for index, frames in enumerate(spdmix.SIX_FRAMES)
    ]
    motion, _robot = spdmix._make_motion_command(
        files, num_envs=2, clip_family_per_clip=spdmix.FAM6
    )
    assert motion.motion.kinematics_contract_exact is True
    racket = spdmix._make_strike_rt(motion)
    racket._action_ball_full_mdp_enabled = True
    racket._action_ball_enabled = False
    racket._motion_term = motion
    starts = motion.motion.seg_start
    motion._action_ball_dynamic_ready_physical_root_quat_wxyz = (
        motion.motion.body_quat_w[starts, 0].contiguous()
    )

    racket.initialize_action_ball_full_mdp_racket_action_reference_cold()
    table = (
        racket.
        project_action_ball_full_mdp_racket_action_reference_static_table()
    )
    for action_slot in range(6):
        strike_step, _phase, _start, _length = racket._strike_frame_for_clip(
            motion.motion, action_slot
        )
        contact_position = racket._ref_racket_pos_at(
            motion.motion, strike_step
        )
        base_position = racket._reference_body_state(
            motion.motion, strike_step, 0
        )[0]
        assert torch.equal(
            table.reference_racket_site_position_w_m[action_slot],
            contact_position,
        )
        assert torch.equal(
            table.reference_reach_offset_xy_m[action_slot],
            contact_position[:2] - base_position[:2],
        )
        _pos, reference_quat, _lin, reference_omega = (
            racket._reference_body_state(
                motion.motion,
                strike_step,
                racket._racket_body_index,
                require_ang_vel=True,
            )
        )
        assert torch.equal(
            table.reference_racket_quat_wxyz[action_slot], reference_quat
        )
        assert torch.equal(
            table.reference_racket_angular_velocity_w_radps[action_slot],
            reference_omega,
        )
    assert tuple(table.reference_racket_site_position_w_m.shape) == (6, 3)
    assert tuple(table.reference_racket_quat_wxyz.shape) == (6, 4)
    assert tuple(table.reference_racket_angular_velocity_w_radps.shape) == (6, 3)
    assert tuple(table.reference_racket_site_velocity_w_mps.shape) == (6, 3)
    assert tuple(table.reference_raw_face_normal_w.shape) == (6, 3)
    assert tuple(table.reference_reach_offset_xy_m.shape) == (6, 2)
    assert tuple(table.reference_base_root_quat_wxyz.shape) == (6, 4)


def test_all_portable_reference_rows_match_real_isaac_cold_fk_builder(
    monkeypatch,
):
    """The portable bank must mirror the shipped Isaac cold FK, not itself.

    Both implementations consume the same sealed NPZ bytes, but the expected
    side below is the production ``MotionLoader`` + ``RacketTargetCommand``
    builder.  This catches clip-boundary, strike-frame, point-offset, and
    centered-FD drift across all 73 actions.
    """

    portable = portable_catalog.load_portable_action_center_table()

    # ``test_reward_flags_mdp`` deliberately installs identity quaternion
    # stubs so its unrelated reward tests stay small.  Replace those three
    # imported globals with the actual wxyz algebra before exercising this
    # production FK path.
    def quat_apply(quaternion, vector):
        xyz = quaternion[..., 1:]
        return vector + 2.0 * (
            quaternion[..., :1] * torch.cross(xyz, vector, dim=-1)
            + torch.cross(xyz, torch.cross(xyz, vector, dim=-1), dim=-1)
        )

    def quat_mul(left, right):
        left_w, left_xyz = left[..., :1], left[..., 1:]
        right_w, right_xyz = right[..., :1], right[..., 1:]
        return torch.cat(
            (
                left_w * right_w
                - torch.sum(left_xyz * right_xyz, dim=-1, keepdim=True),
                left_w * right_xyz
                + right_w * left_xyz
                + torch.cross(left_xyz, right_xyz, dim=-1),
            ),
            dim=-1,
        )

    def matrix_from_quat(quaternion):
        basis = torch.eye(
            3, dtype=quaternion.dtype, device=quaternion.device
        ).expand(quaternion.shape[0], 3, 3)
        return torch.stack(
            [quat_apply(quaternion, basis[:, axis]) for axis in range(3)],
            dim=-1,
        )

    monkeypatch.setattr(HC, "quat_apply", quat_apply)
    monkeypatch.setattr(HC, "quat_mul", quat_mul)
    monkeypatch.setattr(HC, "matrix_from_quat", matrix_from_quat)

    with np.load(portable.actions[0].motion_file, allow_pickle=False) as data:
        body_names = tuple(str(value) for value in data["body_names"].tolist())
    loader = command_module.MotionLoader(
        [row.motion_file for row in portable.actions],
        list(range(len(body_names))),
        articulation_body_names=body_names,
        selected_body_names=body_names,
        device="cpu",
    )
    motion = command_module.MotionCommand.__new__(command_module.MotionCommand)
    motion.motion = loader
    racket = HC.RacketTargetCommand.__new__(HC.RacketTargetCommand)
    racket.device = torch.device("cpu")
    racket.cfg = SimpleNamespace(
        strike_phase=0.5,
        strike_phase_per_clip=tuple(
            row.strike_phase for row in portable.actions
        ),
        clean_strike_vel_window=2,
        clean_reference_strike_velocity=True,
        mount_normal_axis=1,
    )
    racket._env = SimpleNamespace(step_dt=1.0 / float(loader.fps))
    racket._motion = lambda: motion
    racket._racket_mode = "wrist_offset"
    racket._racket_body_index = -1
    racket._wrist_body_index = body_names.index(
        racket_geometry.GEOMETRY_SOURCE_PAYLOAD["official_wrist_body_name"]
    )
    racket._mount_offset = torch.tensor(
        [racket_geometry.RACKET_SITE_OFFSET_WRIST_M], dtype=torch.float32
    )
    racket._mount_quat = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32
    )

    quat, omega, velocity, raw_normal, reach_xy, position, base_quat = (
        racket._build_action_ball_full_mdp_racket_action_reference_cold_rows(
            motion, loader
        )
    )

    def portable_tensor(name):
        return torch.tensor(
            [getattr(row, name) for row in portable.actions],
            dtype=torch.float32,
        )

    expected = {
        "reference_racket_quat_wxyz": quat,
        "reference_racket_angular_velocity_w_radps": omega,
        "reference_racket_site_velocity_w_mps": velocity,
        "reference_raw_face_normal_w": raw_normal,
        "reference_reach_offset_xy_m": reach_xy,
        "reference_racket_site_position_w_m": position,
        "reference_base_root_quat_wxyz": base_quat,
    }
    for name, isaac_rows in expected.items():
        assert torch.allclose(
            portable_tensor(name), isaac_rows, rtol=0.0, atol=2.0e-6
        ), name
