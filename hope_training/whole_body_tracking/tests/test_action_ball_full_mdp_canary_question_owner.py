from __future__ import annotations

import inspect
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


_ROOT = Path(__file__).resolve().parents[1]
_SOURCE = _ROOT / "source" / "whole_body_tracking"
_MDP = _SOURCE / "whole_body_tracking" / "tasks" / "tracking" / "mdp"
if str(_SOURCE) not in sys.path:
    sys.path.insert(0, str(_SOURCE))

from test_reward_flags_mdp import _PKG, _load  # noqa: E402

sys.modules[_PKG].__path__ = [str(_MDP)]
_load(f"{_PKG}.virtual_ball", "virtual_ball.py")
_load(f"{_PKG}.continuous_questions", "continuous_questions.py")
_load(
    f"{_PKG}.action_ball_exact_face_timing_device",
    "action_ball_exact_face_timing_device.py",
)
fixed_question = _load(
    f"{_PKG}.action_ball_fixed_action_question_device",
    "action_ball_fixed_action_question_device.py",
)

import action_ball_device_profile_authority as profile  # noqa: E402
import action_ball_full_mdp_canary_question_owner as owner_mod  # noqa: E402
import action_ball_full_mdp_canary_target_profile as canary_profile  # noqa: E402
import action_ball_physical_question_device as physical  # noqa: E402
import test_action_ball_full_mdp_diagnostic_action_timing as timing_test  # noqa: E402
import test_action_ball_continuous_runtime_transaction_device as d05_test  # noqa: E402


def _physical_owner():
    return physical.make_test_physical_question_numeric_core(
        params=physical.PhysicalQuestionFlightParams(
            k_d=0.1261,
            k_m=0.00444,
            g=9.81,
            ball_radius_m=0.02,
        ),
        config=physical.PhysicalQuestionNumericConfig(
            motion_tick_s=0.02,
            integration_substeps_per_motion_tick=2,
            max_final_segment_motion_ticks=8,
            table_surface_z_m=0.76,
        ),
    )


def _full_horizon_physical_tape():
    params = physical.PhysicalQuestionFlightParams(
        k_d=0.0,
        k_m=0.0,
        g=9.81,
        ball_radius_m=0.02,
    )
    config = physical.PhysicalQuestionNumericConfig(
        motion_tick_s=0.02,
        integration_substeps_per_motion_tick=2,
        max_final_segment_motion_ticks=30,
        table_surface_z_m=0.76,
    )
    batch = physical.PhysicalQuestionCandidateBatch(
        candidate_identity=torch.tensor([501], dtype=torch.int64),
        contact_position_env_m=torch.tensor(
            [[0.0, 0.0, 5.0]], dtype=torch.float32
        ),
        incoming_linear_velocity_world_mps=torch.tensor(
            [[-1.0, 0.0, 0.0]], dtype=torch.float32
        ),
        incoming_angular_velocity_world_radps=torch.zeros(
            (1, 3), dtype=torch.float32
        ),
    )
    return batch, params, config


def _bundle_harness(*, num_envs: int = 2, angular_velocity_z_radps=None):
    # timing_test's structural fixture puts the racket at z=0 and seals that
    # cold table immediately.  Delay only that seal so this question fixture
    # can install a physically meaningful contact height before construction.
    racket_cls = timing_test.HC.RacketTargetCommand
    initialize = (
        racket_cls.initialize_action_ball_full_mdp_racket_action_reference_cold
    )
    racket_cls.initialize_action_ball_full_mdp_racket_action_reference_cold = (
        lambda _self: None
    )
    try:
        rows, env, motion, racket, cadence = timing_test._harness(
            slots=tuple(index % 2 for index in range(num_envs)),
            angular_velocity_z_radps=angular_velocity_z_radps,
        )
    finally:
        racket_cls.initialize_action_ball_full_mdp_racket_action_reference_cold = (
            initialize
        )
    initialize(racket)
    device = torch.device("cpu")
    motion.time_steps = torch.arange(num_envs, dtype=torch.int64, device=device)
    portable = timing_test.profile_mod._portable_catalog.load_portable_action_center_table()
    racket.racket_target_pos_w = torch.tensor(
        [row.reference_racket_site_position_w_m for row in portable.actions],
        dtype=torch.float32,
        device=device,
    )[torch.arange(num_envs).remainder(len(portable.actions))].contiguous()

    hope_commands = sys.modules[f"{_PKG}.hope_commands"]
    exact_cfg = hope_commands.RacketTargetCommandCfg()
    for name, value in vars(racket.cfg).items():
        setattr(exact_cfg, name, value)
    racket.cfg = exact_cfg
    racket.cfg.cq_n_iters = 12
    racket.cfg.cq_tol_m = 0.02
    racket.cfg.cq_speed_budget = 3.4
    racket.cfg.cq_max_redraw_rounds = 3
    racket.cfg.vb_rollout_h = 0.01
    racket.cfg.vb_rollout_steps = 100
    racket.cfg.vb_table_surface_z = 0.76
    racket.cfg.strike_phase_per_clip = tuple(row.strike_phase for row in rows)
    racket.cfg.mount_normal_sign_per_clip = tuple(
        row.mount_normal_sign for row in rows
    )
    racket.cfg.motion_teacher_racket_source = "robot_fk"
    racket._vb_net_x = 1.87
    racket._vb_net_top_z = 0.9125
    racket._vb_ball_r = 0.02

    class CommandManager:
        def get_term(self, name):
            assert name == racket.cfg.motion_command_name
            return motion

    env.command_manager = CommandManager()
    profile_owner, profile_receipt = profile.construct_device_profile_authority(
        canary_profile.build_action_ball_full_mdp_canary_target_profile(
            racket_cfg=racket.cfg
        ),
        device=device,
        expected_support_size=3,
    )
    physical_owner = _physical_owner()
    bundle = owner_mod.construct_recurring_d05_internal_question_bundle(
        profile_owner=profile_owner,
        profile_receipt=profile_receipt,
        racket_owner=racket,
        physical_owner=physical_owner,
    )
    env.scene = SimpleNamespace(
        env_origins=torch.zeros(num_envs, 3, dtype=torch.float32)
    )
    reference_root_xy = (
        bundle._contact_position_env_m[:, :2]
        - bundle._contact_reach_offset_xy
    )
    action_count = reference_root_xy.shape[0]
    slots = torch.arange(num_envs, dtype=torch.int64).remainder(action_count)
    motion._action_ball_full_mdp_frozen_root_pos_w = torch.cat(
        (
            reference_root_xy.index_select(0, slots),
            torch.ones(num_envs, 1, dtype=torch.float32),
        ),
        dim=1,
    ).contiguous()
    motion._action_ball_full_mdp_frozen_root_quat_wxyz = (
        bundle._base_yaw_quat.index_select(0, slots).contiguous()
    )
    motion._action_ball_full_mdp_frozen_root_valid = torch.ones(
        num_envs, dtype=torch.bool
    )
    return SimpleNamespace(
        bundle=bundle,
        cadence=cadence,
        profile_owner=profile_owner,
        profile_receipt=profile_receipt,
        physical_owner=physical_owner,
        racket=racket,
        motion=motion,
        env=env,
    )


def _new_device_cadence(values, selected_env_index):
    cls = d05_test.r05.DeviceCadenceProjection
    values = dict(values)
    values["selected_env_index"] = selected_env_index
    return cls(**values)


def _direct_projection(
    *,
    num_envs: int,
    source_rows: tuple[int, ...],
    selected_env_index: torch.Tensor,
    construction_mask: torch.Tensor | None = None,
    action_slot: torch.Tensor | None = None,
    cadence_producer_fault: torch.Tensor | None = None,
    frozen_root_xy: torch.Tensor | None = None,
    frozen_root_yaw_rad: torch.Tensor | None = None,
):
    harness = _bundle_harness(num_envs=num_envs)
    if frozen_root_xy is not None:
        harness.motion._action_ball_full_mdp_frozen_root_pos_w[
            :, :2
        ] = frozen_root_xy
    if frozen_root_yaw_rad is not None:
        half = 0.5 * frozen_root_yaw_rad
        harness.motion._action_ball_full_mdp_frozen_root_quat_wxyz.zero_()
        harness.motion._action_ball_full_mdp_frozen_root_quat_wxyz[:, 0] = (
            torch.cos(half)
        )
        harness.motion._action_ball_full_mdp_frozen_root_quat_wxyz[:, 3] = (
            torch.sin(half)
        )
    source_row = torch.tensor(source_rows, dtype=torch.int64)
    k = len(source_rows)
    values = {}
    cls = d05_test.r05.DeviceCadenceProjection
    for name in cls.__dataclass_fields__:
        value = getattr(harness.cadence, name, None)
        values[name] = (
            torch.index_select(value, 0, source_row).contiguous()
            if type(value) is torch.Tensor
            else value
        )
    values.update(
        selected_count=k,
        episode_tick=torch.full((k,), 2, dtype=torch.int64),
        reveal_tick=torch.full((k,), 2, dtype=torch.int64),
        deadline_tick=torch.full((k,), 4, dtype=torch.int64),
        next_reveal_tick=torch.full((k,), 435, dtype=torch.int64),
        action_slot=(
            torch.zeros(k, dtype=torch.int64)
            if action_slot is None
            else action_slot
        ),
        task_identity=torch.full((k,), -1, dtype=torch.int64),
        cadence_identity=torch.full((k,), -1, dtype=torch.int64),
        action_uid=torch.full((k,), -1, dtype=torch.int64),
        contact_tick=torch.full((k,), -1, dtype=torch.int64),
        launch_tick=torch.full((k,), -1, dtype=torch.int64),
        chosen_horizon_ticks=torch.full((k,), -1, dtype=torch.int64),
        cadence_producer_fault=(
            torch.zeros(k, dtype=torch.int64)
            if cadence_producer_fault is None
            else cadence_producer_fault
        ),
    )
    cadence = _new_device_cadence(values, selected_env_index)
    profile_projection = harness.profile_owner.require_owned_r05_profile(
        harness.profile_receipt
    )
    rounds = d05_test.r05.INTERNAL_QUESTION_REDRAW_ROUNDS
    width = d05_test.r05.INTERNAL_QUESTION_DRAW_WIDTH
    # D05 owns and consumes this fixed-width RNG tape.  The current Phase4
    # manifest declares zero ball-domain width, so the numerical question must
    # remain at the action-owned incoming centre regardless of these values.
    draw = torch.tensor(
        (0.4, 0.5, 7.0 / 15.0, 0.5, 0.5, 0.5),
        dtype=torch.float32,
    ).expand(k, rounds, width).contiguous()
    candidate_identity = torch.arange(
        1, 1 + k * rounds * 3, dtype=torch.int64
    ).reshape(k, rounds, 3)
    original_assert_async = torch._assert_async

    def compatible_assert_async(condition, *message):
        if torch.__version__.startswith("2.0."):
            return original_assert_async(condition)
        return original_assert_async(condition, *message)

    torch._assert_async = compatible_assert_async
    try:
        projection = owner_mod._compose_recurring_question_projection(
            bundle=harness.bundle,
            cadence_receipt=object(),
            cadence=cadence,
            profile=profile_projection,
            device=torch.device("cpu"),
            support=3,
            draw_u01=draw,
            candidate_identity=candidate_identity,
            construction_mask=(
                torch.ones(k, dtype=torch.bool)
                if construction_mask is None
                else construction_mask
            ),
            bank_sequence=1,
        )
    finally:
        torch._assert_async = original_assert_async
    return projection, harness, cadence


def test_recurring_question_no_move_goal_is_the_installed_physical_spawn():
    center = (
        timing_test.profile_mod._portable_catalog
        .load_portable_action_center_table().fresh_action
        .base_spawn_center_w_xy_m
    )
    frozen_xy = torch.tensor([center], dtype=torch.float32)
    projection, harness, _cadence = _direct_projection(
        num_envs=1,
        source_rows=(0,),
        selected_env_index=torch.tensor([0], dtype=torch.int64),
        frozen_root_xy=frozen_xy,
    )
    admitted = projection.round_bank.construction_reason.eq(-1)
    assert bool(admitted.any())
    base_goal = projection.round_bank.racket_task_f32[..., 19:21]
    torch.testing.assert_close(
        base_goal[admitted],
        frozen_xy.expand(int(admitted.sum()), -1),
        rtol=0.0,
        atol=2.0e-6,
    )
    assert harness.physical_owner._pending == {}


def test_recurring_question_adapter_consumes_one_shared_numeric_result(monkeypatch):
    captured = {}
    solve = fixed_question.construct_reference_center_question_device

    def capture(**kwargs):
        result = solve(**kwargs)
        captured["kwargs"] = kwargs
        captured["result"] = result
        return result

    monkeypatch.setattr(
        fixed_question, "construct_reference_center_question_device", capture
    )
    projection, harness, _cadence = _direct_projection(
        num_envs=1,
        source_rows=(0,),
        selected_env_index=torch.tensor([0], dtype=torch.int64),
    )
    result = captured["result"]
    torch.testing.assert_close(
        result.racket_task_f32[result.admitted, 21:24],
        captured["kwargs"]["incoming_linear_velocity_world_mps"][
            result.admitted
        ],
        rtol=0.0,
        atol=0.0,
    )
    bank = projection.round_bank
    torch.testing.assert_close(
        bank.motion_task_f32.reshape(-1, 5),
        result.motion_task_f32,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        bank.racket_task_f32.reshape(-1, 27),
        result.racket_task_f32,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        bank.physical_state_f32.reshape(-1, 13),
        result.physical_state_f32,
        rtol=0.0,
        atol=0.0,
    )
    assert torch.equal(
        bank.construction_reason.reshape(-1), result.construction_reason
    )
    assert torch.equal(
        projection.round_chronology.contact_tick.reshape(-1),
        result.contact_tick,
    )
    assert torch.equal(
        projection.round_chronology.launch_tick.reshape(-1),
        result.launch_tick,
    )
    assert torch.equal(
        projection.round_chronology.chosen_horizon_ticks.reshape(-1),
        result.chosen_horizon_ticks,
    )
    assert result.producer_fault.count_nonzero().item() == 0

    # The shared flat batch must retain the previous owner's numerical bytes,
    # even though lifecycle receipts no longer exist in the hot adapter.
    original_kwargs = captured["kwargs"]
    legacy_batch = physical.PhysicalQuestionCandidateBatch(
        candidate_identity=original_kwargs["candidate_identity"].reshape(1, 3, 3),
        contact_position_env_m=original_kwargs[
            "contact_position_env_m"
        ].reshape(1, 3, 3, 3),
        incoming_linear_velocity_world_mps=original_kwargs[
            "incoming_linear_velocity_world_mps"
        ].reshape(1, 3, 3, 3),
        incoming_angular_velocity_world_radps=original_kwargs[
            "incoming_angular_velocity_world_radps"
        ].reshape(1, 3, 3, 3),
    )
    receipt = harness.physical_owner.issue_horizon_for_test(legacy_batch)
    horizon = harness.physical_owner.project_horizon_for_test(receipt)
    legacy_reveal_tick = original_kwargs["reveal_tick"].reshape(1, 3, 3)
    legacy_contact_tick = original_kwargs["contact_tick"].reshape(1, 3, 3)
    legacy_remaining_tick = legacy_contact_tick - legacy_reveal_tick
    legacy_horizon = torch.minimum(
        horizon.max_feasible_motion_ticks,
        (legacy_remaining_tick - 1).clamp(min=0),
    ).contiguous()
    legacy_launch_tick = (legacy_contact_tick - legacy_horizon).contiguous()
    legacy = harness.physical_owner.finalize_exact_ticks_for_test(
        receipt,
        candidate_identity=legacy_batch.candidate_identity,
        contact_tick=legacy_contact_tick,
        launch_tick=legacy_launch_tick,
    )
    numeric_admitted = result.admitted
    assert bool(numeric_admitted.any())
    assert torch.equal(
        legacy.physical_state_f32.reshape(-1, 13)[numeric_admitted],
        result.physical_state_f32[numeric_admitted],
    )
    assert torch.equal(
        result.physical_state_f32[~numeric_admitted],
        torch.zeros_like(result.physical_state_f32[~numeric_admitted]),
    )
    assert torch.equal(
        legacy.construction_reason.reshape(-1), result.physical_reason
    )
    assert torch.equal(
        horizon.construction_reason.reshape(-1), result.physical_horizon_reason
    )
    assert harness.physical_owner._pending == {}

    attempted = result.construction_reason.numel()
    admitted = int(result.admitted.sum().item())
    rejected = int(result.construction_reason.ge(0).sum().item())
    assert attempted == admitted + rejected
    assert torch.equal(result.admitted, result.construction_reason.eq(-1))

    # This constructor owns only the sealed teacher-compatible centre.  A
    # changed aim or incoming ball must leave this exact lane and be rejected;
    # future non-zero curriculum support is solved by the general solver.
    changed_aim = dict(original_kwargs)
    changed_aim["landing_aim_xy_m"] = (
        original_kwargs["landing_aim_xy_m"]
        + torch.tensor((0.04, -0.03), dtype=torch.float32)
    ).contiguous()
    aim_result = solve(**changed_aim)
    assert bool((result.admitted & ~aim_result.admitted).any())
    assert not torch.equal(result.solver_residual_m, aim_result.solver_residual_m)

    changed_speed = dict(original_kwargs)
    changed_speed["incoming_linear_velocity_world_mps"] = (
        original_kwargs["incoming_linear_velocity_world_mps"] * 1.02
    ).contiguous()
    speed_result = solve(**changed_speed)
    assert bool((result.admitted & ~speed_result.admitted).any())
    assert not torch.equal(result.solver_residual_m, speed_result.solver_residual_m)


def test_physical_launch_horizon_is_reveal_relative_and_keeps_current_n1_bytes():
    batch, params, config = _full_horizon_physical_tape()
    current_reveal = torch.tensor([48], dtype=torch.int64)
    current_contact = current_reveal + 92
    current = physical.solve_max_final_segment_device(
        batch,
        candidate_identity=batch.candidate_identity,
        reveal_tick=current_reveal,
        contact_tick=current_contact,
        params=params,
        config=config,
    )
    assert current.max_feasible_motion_ticks.tolist() == [30]
    assert current.chosen_horizon_ticks.tolist() == [30]
    assert current.launch_tick.tolist() == [110]

    # The adopted N1 row has 92 ticks remaining, so the strict reveal-relative
    # cap cannot alter its legacy 30-tick launch state or any float32 byte.
    owner = physical.make_test_physical_question_numeric_core(
        params=params, config=config
    )
    receipt = owner.issue_horizon_for_test(batch)
    legacy = owner.finalize_exact_ticks_for_test(
        receipt,
        candidate_identity=batch.candidate_identity,
        contact_tick=current_contact,
        launch_tick=current_contact - 30,
    )
    assert torch.equal(
        current.physical_state_f32.view(torch.uint8),
        legacy.physical_state_f32.view(torch.uint8),
    )
    assert torch.equal(
        current.effective_contact_horizon_s.view(torch.uint8),
        legacy.effective_contact_horizon_s.view(torch.uint8),
    )

    # Future cadence counterexample: absolute contact=253 used to choose all
    # 30 ticks and launch at 223, ten ticks before reveal=233.  The strict Mu
    # semantics ttc_ticks > launch_horizon_ticks instead choose 19 and launch
    # one tick after reveal.
    short_reveal = torch.tensor([233], dtype=torch.int64)
    short_contact = short_reveal + 20
    short = physical.solve_max_final_segment_device(
        batch,
        candidate_identity=batch.candidate_identity,
        reveal_tick=short_reveal,
        contact_tick=short_contact,
        params=params,
        config=config,
    )
    assert short.max_feasible_motion_ticks.tolist() == [30]
    assert short.chosen_horizon_ticks.tolist() == [19]
    assert short.launch_tick.tolist() == [234]
    assert torch.all(short.launch_tick.gt(short_reveal))
    assert torch.all(
        (short_contact - short_reveal).gt(short.chosen_horizon_ticks)
    )
    assert torch.all(
        short.construction_reason.eq(physical.CONSTRUCTION_REASON_ADMITTED)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_shared_fixed_question_cuda_fixed_tape_is_deterministic_and_causal(
    monkeypatch,
):
    """Exercise the complete shared numerical owner on one real CUDA tape."""

    captured = {}
    solve = fixed_question.construct_reference_center_question_device

    def capture(**kwargs):
        captured["kwargs"] = kwargs
        return solve(**kwargs)

    monkeypatch.setattr(
        fixed_question, "construct_reference_center_question_device", capture
    )
    _direct_projection(
        num_envs=1,
        source_rows=(0,),
        selected_env_index=torch.tensor([0], dtype=torch.int64),
    )
    cuda_kwargs = {
        name: value.to(device="cuda").contiguous()
        if type(value) is torch.Tensor
        else value
        for name, value in captured["kwargs"].items()
    }
    first = solve(**cuda_kwargs)
    repeated = solve(**cuda_kwargs)
    for field in first.__dataclass_fields__:
        assert torch.equal(getattr(first, field), getattr(repeated, field)), field
    frozen_contact_tick = first.contact_tick.clone()
    cuda_kwargs["contact_tick"].add_(1)
    assert torch.equal(first.contact_tick, frozen_contact_tick)
    cuda_kwargs["contact_tick"].sub_(1)

    attempted = int(first.construction_reason.numel())
    admitted = int(first.admitted.sum().item())
    rejected = int(first.construction_reason.ge(0).sum().item())
    assert attempted == admitted + rejected
    assert first.producer_fault.count_nonzero().item() == 0
    assert bool(first.admitted.any())

    changed_aim = dict(cuda_kwargs)
    changed_aim["landing_aim_xy_m"] = (
        cuda_kwargs["landing_aim_xy_m"]
        + torch.tensor((0.04, -0.03), device="cuda", dtype=torch.float32)
    ).contiguous()
    aim_result = solve(**changed_aim)
    assert not torch.equal(
        first.solver_residual_m, aim_result.solver_residual_m
    )
    assert bool((first.admitted & ~aim_result.admitted).any())

    changed_speed = dict(cuda_kwargs)
    changed_speed["incoming_linear_velocity_world_mps"] = (
        cuda_kwargs["incoming_linear_velocity_world_mps"] * 1.02
    ).contiguous()
    speed_result = solve(**changed_speed)
    assert not torch.equal(
        first.solver_residual_m, speed_result.solver_residual_m
    )
    assert bool((first.admitted & ~speed_result.admitted).any())

    invalid = dict(cuda_kwargs)
    invalid["landing_aim_xy_m"] = cuda_kwargs["landing_aim_xy_m"].clone()
    invalid["landing_aim_xy_m"][0, 0] = float("nan")
    invalid_result = solve(**invalid)
    assert int(invalid_result.construction_reason[0].item()) == 12
    assert int(invalid_result.producer_fault[0].item()) != 0
    assert not bool(invalid_result.admitted[0])
    assert torch.equal(
        invalid_result.motion_task_f32[0],
        torch.zeros_like(invalid_result.motion_task_f32[0]),
    )
    assert torch.equal(
        invalid_result.racket_task_f32[0],
        torch.zeros_like(invalid_result.racket_task_f32[0]),
    )
    assert torch.equal(
        invalid_result.physical_state_f32[0],
        torch.zeros_like(invalid_result.physical_state_f32[0]),
    )
    attempted = int(invalid_result.construction_reason.numel())
    admitted = int(invalid_result.admitted.sum().item())
    rejected = int(invalid_result.construction_reason.ge(0).sum().item())
    assert attempted == admitted + rejected


@pytest.mark.parametrize("num_envs", (1, 2, 64))
def test_recurring_question_bundle_is_cardinality_generic(num_envs):
    harness = _bundle_harness(num_envs=num_envs)
    assert harness.bundle._num_envs == num_envs
    assert harness.motion.time_steps.shape == (num_envs,)
    assert harness.racket.racket_target_pos_w.shape == (num_envs, 3)


def test_recurring_question_ball_arrival_closes_measured_selected_face_geometry():
    harness = _bundle_harness(num_envs=2)
    hope_commands = sys.modules[f"{_PKG}.hope_commands"]
    geometry = sys.modules[f"{_PKG}.racket_contact_geometry"]
    static = (
        hope_commands.RacketTargetCommand.
        project_action_ball_full_mdp_racket_action_reference_static_table(
            harness.racket
        )
    )
    signs = harness.bundle._timing_table.mount_normal_sign
    local = torch.tensor(
        [
            geometry.ball_center_from_site_local(int(sign))
            for sign in signs.tolist()
        ],
        dtype=torch.float32,
    )
    rotated = owner_mod._quat_rotate_wxyz(
        static.reference_racket_quat_wxyz, local
    )
    torch.testing.assert_close(
        harness.bundle._contact_position_env_m,
        static.reference_racket_site_position_w_m + rotated,
    )
    torch.testing.assert_close(
        harness.bundle._contact_reach_offset_xy,
        static.reference_reach_offset_xy_m + rotated[:, :2],
    )


def test_n4096_constructs_the_same_recurring_graph_with_linear_row_storage():
    harness = _bundle_harness(num_envs=4096)
    assert type(harness.bundle) is owner_mod.RecurringD05InternalQuestionBundle
    assert harness.bundle._num_envs == 4096
    row_storage_bytes = (
        harness.motion.time_steps.numel()
        * harness.motion.time_steps.element_size()
        + harness.racket.racket_target_pos_w.numel()
        * harness.racket.racket_target_pos_w.element_size()
    )
    assert row_storage_bytes == 4096 * (8 + 3 * 4)
    assert not hasattr(harness.bundle, "_n4096_receipt")


@pytest.mark.parametrize("source_rows", ((0, 17, 63), (63, 17, 0)))
def test_n64_sparse_rows_and_permutation_share_one_recurring_graph(source_rows):
    selected = torch.tensor(source_rows, dtype=torch.int64)
    projection, harness, cadence = _direct_projection(
        num_envs=64,
        source_rows=source_rows,
        selected_env_index=selected,
    )
    assert harness.bundle._num_envs == 64
    assert cadence.selected_env_index.tolist() == list(source_rows)
    assert projection.selected_count == 3
    assert projection.producer_fault.tolist() == [0, 0, 0]
    assert projection.round_bank.candidate_identity.shape == (3, 3, 3)
    assert harness.physical_owner._pending == {}


@pytest.mark.parametrize(
    ("selected", "fault_mask"),
    (
        (torch.tensor([0, 0], dtype=torch.int64), (True, True)),
        (torch.tensor([0, 64], dtype=torch.int64), (False, True)),
        (torch.tensor([-1, 63], dtype=torch.int64), (True, False)),
    ),
)
def test_n64_foreign_or_duplicate_rows_censor_same_batch(selected, fault_mask):
    projection, harness, _cadence = _direct_projection(
        num_envs=64,
        source_rows=(0, 63),
        selected_env_index=selected,
    )
    expected_fault = torch.tensor(fault_mask, dtype=torch.bool)
    assert torch.equal(
        projection.producer_fault.eq(
            owner_mod.PRODUCER_FAULT_STATIC_ROW_BINDING
        ),
        expected_fault,
    )
    assert torch.all(
        projection.round_bank.construction_reason[expected_fault].eq(
            owner_mod.CONSTRUCTION_REASON_INVALID_PRODUCER
        )
    )
    assert harness.physical_owner._pending == {}


def _assert_active_projection_exact(actual, dense, construction_mask):
    active = construction_mask.nonzero(as_tuple=False).reshape(-1)
    assert torch.equal(
        actual.round_bank.candidate_identity,
        dense.round_bank.candidate_identity,
    )
    assert torch.equal(
        torch.index_select(actual.producer_fault, 0, active),
        torch.index_select(dense.producer_fault, 0, active),
    )
    for name in (
        "construction_reason",
        "producer_fault",
        "motion_task_f32",
        "racket_task_f32",
        "physical_state_f32",
    ):
        assert torch.equal(
            torch.index_select(getattr(actual.round_bank, name), 0, active),
            torch.index_select(getattr(dense.round_bank, name), 0, active),
        ), name
    for name in (
        "action_uid",
        "contact_tick",
        "launch_tick",
        "chosen_horizon_ticks",
        "task_close_tick",
    ):
        assert torch.equal(
            torch.index_select(getattr(actual.round_chronology, name), 0, active),
            torch.index_select(getattr(dense.round_chronology, name), 0, active),
        ), name

    inactive = (~construction_mask).nonzero(as_tuple=False).reshape(-1)
    inactive_reason = torch.index_select(
        actual.round_bank.construction_reason, 0, inactive
    )
    assert torch.all(
        inactive_reason.eq(owner_mod.CONSTRUCTION_REASON_INVALID_PRODUCER)
    )
    assert torch.count_nonzero(
        torch.index_select(actual.round_bank.producer_fault, 0, inactive)
    ) == 0
    assert torch.count_nonzero(
        torch.index_select(actual.producer_fault, 0, inactive)
    ) == 0
    for name in ("motion_task_f32", "racket_task_f32", "physical_state_f32"):
        assert torch.count_nonzero(
            torch.index_select(getattr(actual.round_bank, name), 0, inactive)
        ) == 0
    for name in ("launch_tick", "chosen_horizon_ticks", "task_close_tick"):
        assert torch.all(
            torch.index_select(
                getattr(actual.round_chronology, name), 0, inactive
            ).eq(-1)
        )


@pytest.mark.parametrize(
    "mask_values",
    (
        (True, False, True, False),
        (False, False, False, False),
        (True, True, True, True),
    ),
)
def test_mask_first_fixed_tape_matches_dense_reference(mask_values):
    mask = torch.tensor(mask_values, dtype=torch.bool)
    dense, _dense_harness, _ = _direct_projection(
        num_envs=4,
        source_rows=(0, 1, 2, 3),
        selected_env_index=torch.arange(4, dtype=torch.int64),
    )
    actual, harness, _ = _direct_projection(
        num_envs=4,
        source_rows=(0, 1, 2, 3),
        selected_env_index=torch.arange(4, dtype=torch.int64),
        construction_mask=mask,
    )
    _assert_active_projection_exact(actual, dense, mask)
    assert actual.round_bank.candidate_identity.shape == (4, 3, 3)
    assert actual.round_chronology.contact_tick.shape == (4, 3, 3)
    assert harness.physical_owner._pending == {}


def test_mask_first_dense_parity_includes_invalid_rows_slots_and_faults():
    mask = torch.tensor((True, False, True, True), dtype=torch.bool)
    selected = torch.tensor((0, 99, 0, 3), dtype=torch.int64)
    slots = torch.tensor((0, 99, -1, 0), dtype=torch.int64)
    faults = torch.tensor((0, 17, 1 << 40, 0), dtype=torch.int64)
    kwargs = dict(
        num_envs=4,
        source_rows=(0, 1, 2, 3),
        selected_env_index=selected,
        action_slot=slots,
        cadence_producer_fault=faults,
    )
    dense, _dense_harness, _ = _direct_projection(**kwargs)
    actual, harness, _ = _direct_projection(
        **kwargs,
        construction_mask=mask,
    )
    _assert_active_projection_exact(actual, dense, mask)
    assert torch.all(
        actual.round_bank.construction_reason[
            torch.tensor((0, 2), dtype=torch.int64)
        ].eq(
            owner_mod.CONSTRUCTION_REASON_INVALID_PRODUCER
        )
    )
    assert harness.physical_owner._pending == {}


@pytest.mark.parametrize(
    "mask_values",
    (
        (False, False, False, False),
        (True, False, True, False),
        (True, True, True, True),
    ),
)
def test_mask_first_numeric_owners_see_only_active_cells(monkeypatch, mask_values):
    questions = sys.modules[f"{_PKG}.continuous_questions"]
    exact_face = sys.modules[f"{_PKG}.action_ball_exact_face_timing_device"]
    calls = {"solver": [], "exact": [], "physical": []}

    original_solver = questions.solve_proposals_device
    original_exact = exact_face.solve_exact_face_timing_device
    original_physical = physical.solve_max_final_segment_device

    def spy_solver(*args, **kwargs):
        calls["solver"].append(args[0].shape[0])
        return original_solver(*args, **kwargs)

    def spy_exact(*args, **kwargs):
        calls["exact"].append(kwargs["ball_contact_w_m"].shape[0])
        return original_exact(*args, **kwargs)

    def spy_physical(batch, **kwargs):
        calls["physical"].append(batch.candidate_identity.numel())
        return original_physical(batch, **kwargs)

    monkeypatch.setattr(questions, "solve_proposals_device", spy_solver)
    monkeypatch.setattr(exact_face, "solve_exact_face_timing_device", spy_exact)
    monkeypatch.setattr(physical, "solve_max_final_segment_device", spy_physical)

    mask = torch.tensor(mask_values, dtype=torch.bool)
    projection, harness, _ = _direct_projection(
        num_envs=4,
        source_rows=(0, 1, 2, 3),
        selected_env_index=torch.arange(4, dtype=torch.int64),
        construction_mask=mask,
    )
    numeric_cells = int(mask.sum()) * 3 * 3
    expected_batch = [] if numeric_cells == 0 else [numeric_cells]
    assert calls == {
        "solver": [],
        "exact": expected_batch,
        "physical": expected_batch,
    }
    assert projection.round_bank.candidate_identity.shape == (4, 3, 3)
    assert harness.physical_owner._pending == {}


def test_mask_first_documents_its_single_dynamic_cuda_sync_boundary():
    hot = inspect.getsource(owner_mod._compose_recurring_question_projection)
    assert hot.count("construction_mask.nonzero(") == 1
    assert "synchronize while materializing its dynamic output size" in hot


@pytest.mark.parametrize(
    "selected",
    (
        torch.tensor([], dtype=torch.int64),
        torch.tensor([9], dtype=torch.int64),
        torch.tensor([4, 1, 4, 4, 7, 1], dtype=torch.int64),
        torch.tensor([3, 2, 1, 0], dtype=torch.int64),
    ),
)
def test_duplicate_row_mask_matches_pairwise_reference(selected):
    expected = torch.tensor(
        [
            sum(int(value == other) for other in selected.tolist()) > 1
            for value in selected.tolist()
        ],
        dtype=torch.bool,
    )
    actual = owner_mod._duplicate_valid_index_rows(selected, num_envs=64)
    assert torch.equal(actual, expected)


def test_duplicate_row_mask_rejects_non_i64_or_non_vector_inputs():
    with pytest.raises(owner_mod.CanaryQuestionError, match="index ABI"):
        owner_mod._duplicate_valid_index_rows(
            torch.tensor([0, 1], dtype=torch.int32), num_envs=2
        )
    with pytest.raises(owner_mod.CanaryQuestionError, match="index ABI"):
        owner_mod._duplicate_valid_index_rows(
            torch.tensor([[0, 1]], dtype=torch.int64), num_envs=2
        )
    with pytest.raises(owner_mod.CanaryQuestionError, match="capacity ABI"):
        owner_mod._duplicate_valid_index_rows(
            torch.tensor([0, 1], dtype=torch.int64), num_envs=0
        )


def test_duplicate_row_mask_leaves_foreign_rows_to_range_fault():
    selected = torch.tensor([-1, 63, -1, 64, 64], dtype=torch.int64)
    actual = owner_mod._duplicate_valid_index_rows(selected, num_envs=64)
    assert actual.tolist() == [False, False, False, False, False]


@pytest.mark.parametrize(
    "selected",
    (
        torch.tensor([True, False], dtype=torch.bool),
        torch.tensor([0, 9, 1, 9], dtype=torch.int64)[::2],
    ),
)
def test_malformed_due_row_metadata_rejects_before_physical_issue(selected):
    harness = _bundle_harness(num_envs=2)
    pending_before = dict(harness.physical_owner._pending)
    with pytest.raises(owner_mod.CanaryQuestionError, match="due environment index"):
        _direct_projection(
            num_envs=2,
            source_rows=(0, 1),
            selected_env_index=selected,
        )
    assert harness.physical_owner._pending == pending_before


def test_recurring_bundle_rejects_metadata_that_differs_from_live_rows():
    harness = _bundle_harness(num_envs=2)
    harness.env.num_envs = 64
    harness.racket.num_envs = 64
    harness.motion.num_envs = 64
    with pytest.raises(owner_mod.CanaryQuestionError, match="metadata differs"):
        owner_mod.construct_recurring_d05_internal_question_bundle(
            profile_owner=harness.profile_owner,
            profile_receipt=harness.profile_receipt,
            racket_owner=harness.racket,
            physical_owner=_physical_owner(),
        )


def test_recurring_path_has_no_parallel_receipt_rng_or_fixed_tape_owner():
    harness = _bundle_harness(num_envs=2)
    names = set(type(harness.bundle).__slots__)
    assert not any("receipt" in name or "rng" in name for name in names)
    for dead in (
        "CanaryQuestionReceipt",
        "DiagnosticTestOnlyCanaryQuestionOwner",
        "construct_diagnostic_test_only_canary_question_owner",
        "construct_n2_no_save_canary_question_owner",
    ):
        assert not hasattr(owner_mod, dead)
    hot = inspect.getsource(owner_mod._compose_recurring_question_projection)
    for forbidden in (".item(", ".cpu(", ".tolist(", ".numpy(", "torch.equal("):
        assert forbidden not in hot
