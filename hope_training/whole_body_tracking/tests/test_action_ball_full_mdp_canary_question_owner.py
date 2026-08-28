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
    contact_z = torch.tensor((1.02, 1.08), dtype=torch.float32)
    for action_slot in range(len(rows)):
        start = int(motion.motion.seg_start[action_slot].item())
        length = int(motion.motion.seg_len[action_slot].item())
        motion.motion._body_pos_w[
            start : start + length, 1, 2
        ] = contact_z[action_slot]
    initialize(racket)
    device = torch.device("cpu")
    motion.time_steps = torch.arange(num_envs, dtype=torch.int64, device=device)
    racket.racket_target_pos_w = torch.tensor(
        [[0.51, -0.11, 1.02], [0.57, 0.13, 1.08]],
        dtype=torch.float32,
        device=device,
    )[torch.arange(num_envs).remainder(2)].contiguous()

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
):
    harness = _bundle_harness(num_envs=num_envs)
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
        next_reveal_tick=torch.full((k,), 48, dtype=torch.int64),
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
    # A real finite, solvable incoming ball repeated across rows.  These are
    # the normalized ContinuousQuestionCfg draws for v=(-3.5, 0.0, -0.3)
    # m/s and zero spin; row-binding tests must not fail earlier in the
    # production numerical solver because their arbitrary tape was singular.
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
    calls = {"solver": [], "exact": [], "issue": [], "project": [], "final": []}

    original_solver = questions.solve_proposals_device
    original_exact = exact_face.solve_exact_face_timing_device
    original_issue = physical.PhysicalQuestionNumericCore.issue_horizon_for_test
    original_project = physical.PhysicalQuestionNumericCore.project_horizon_for_test
    original_final = physical.PhysicalQuestionNumericCore.finalize_exact_ticks_for_test

    def spy_solver(*args, **kwargs):
        calls["solver"].append(args[0].shape[0])
        return original_solver(*args, **kwargs)

    def spy_exact(*args, **kwargs):
        calls["exact"].append(kwargs["ball_contact_w_m"].shape[0])
        return original_exact(*args, **kwargs)

    def spy_issue(self, batch):
        calls["issue"].append(batch.candidate_identity.numel())
        return original_issue(self, batch)

    def spy_project(self, receipt):
        calls["project"].append(1)
        return original_project(self, receipt)

    def spy_final(self, receipt, **kwargs):
        calls["final"].append(kwargs["candidate_identity"].numel())
        return original_final(self, receipt, **kwargs)

    monkeypatch.setattr(questions, "solve_proposals_device", spy_solver)
    monkeypatch.setattr(exact_face, "solve_exact_face_timing_device", spy_exact)
    monkeypatch.setattr(
        physical.PhysicalQuestionNumericCore, "issue_horizon_for_test", spy_issue
    )
    monkeypatch.setattr(
        physical.PhysicalQuestionNumericCore, "project_horizon_for_test", spy_project
    )
    monkeypatch.setattr(
        physical.PhysicalQuestionNumericCore,
        "finalize_exact_ticks_for_test",
        spy_final,
    )

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
        "solver": expected_batch,
        "exact": expected_batch,
        "issue": expected_batch,
        "project": [] if numeric_cells == 0 else [1],
        "final": expected_batch,
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
