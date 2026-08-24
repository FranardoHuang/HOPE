"""Counterexamples for the small full-MDP milestone tensor leaf."""

from pathlib import Path
import inspect
import types
import sys

import pytest
import torch


MDP = Path(__file__).resolve().parents[1] / "source" / "whole_body_tracking" / "whole_body_tracking" / "tasks" / "tracking" / "mdp"
sys.path.insert(0, str(MDP))
import action_ball_full_mdp_milestone_tensors as M  # noqa: E402


def _term_names():
    return tuple(map(str, range(M.REWARD_TERM_COUNT)))


def _add_step(leaf, *, payments=None, scales=None, actual=None, eligible=None):
    num_envs = leaf.open_episode_return.numel()
    payments = payments or tuple(
        torch.zeros(num_envs) for _ in range(M.REWARD_TERM_COUNT)
    )
    scales = scales or (1.0,) * M.REWARD_TERM_COUNT
    eligible = torch.ones(num_envs, dtype=torch.bool) if eligible is None else eligible
    finite = torch.ones(num_envs, dtype=torch.bool)
    for ordinal, (payment, scale) in enumerate(zip(payments, scales)):
        leaf.add_reward(
            ordinal, payment, payment, eligible, finite,
            torch.tensor(scale, dtype=torch.float64),
        )
    configured = sum(
        (payment.to(torch.float64) * scale for payment, scale in zip(payments, scales)),
        torch.zeros(num_envs, dtype=torch.float64),
    )
    leaf.close_actual_step(configured.to(torch.float32) if actual is None else actual)


def test_reward_keeps_eligible_zero_distinct_and_names_only_configured_income():
    leaf = M.MilestoneTensorAccumulator(2, torch.device("cpu"))
    _add_step(leaf, eligible=torch.tensor([True, False]))
    payload = M.decode_host_window(*leaf.pack_views()).as_json(_term_names())
    row = payload["reward_terms"][0]
    assert payload["schema_version"] == 7
    assert payload["sample_unit"] == "reward_manager_payment_sample"
    assert "ladder_counts" not in payload and "ladder_stats" not in payload
    assert "per_action_event_strata" in payload["not_produced"]
    assert payload["actual_reward"]["conservation_unit"] == "per_env_control_step"
    assert (row["evaluated"], row["eligible"], row["finite"], row["nonzero"]) == (2, 1, 1, 0)
    assert row["configured_income_sum"] == 0.0
    assert all("manager_income" not in name for name in row)


def test_actual_reward_is_checked_per_env_step_and_catches_cancelling_errors():
    payments = tuple(
        torch.tensor([1.0, 1.0]) if index == 0 else torch.zeros(2)
        for index in range(M.REWARD_TERM_COUNT)
    )
    legal = M.MilestoneTensorAccumulator(2, torch.device("cpu"))
    _add_step(legal, payments=payments, actual=torch.tensor([1.0, 1.0]))
    payload = M.decode_host_window(*legal.pack_views()).as_json(_term_names())
    assert payload["actual_reward"]["sample_count"] == 2
    assert payload["actual_reward"]["sum"] == 2.0

    cancelling = M.MilestoneTensorAccumulator(2, torch.device("cpu"))
    _add_step(cancelling, payments=payments, actual=torch.tensor([1.01, 0.99]))
    assert cancelling.i64[M._AI + 3].item() == 2
    with pytest.raises(ValueError, match="actual reward conservation"):
        M.decode_host_window(*cancelling.pack_views())


def test_actual_reward_underflow_bound_is_derived_not_tiny_times_epsilon():
    payments = tuple(
        torch.ones(1) if index == 0 else torch.zeros(1)
        for index in range(M.REWARD_TERM_COUNT)
    )
    legal_ftz = M.MilestoneTensorAccumulator(1, torch.device("cpu"))
    _add_step(
        legal_ftz,
        payments=payments,
        scales=(1.0e-40,) + (1.0,) * (M.REWARD_TERM_COUNT - 1),
        actual=torch.zeros(1),
    )
    M.decode_host_window(*legal_ftz.pack_views())
    assert legal_ftz.f64[M._AF + 6].item() >= M.REWARD_TERM_COUNT * torch.finfo(torch.float32).tiny

    illegal = M.MilestoneTensorAccumulator(1, torch.device("cpu"))
    outside = 2.0 * M.REWARD_TERM_COUNT * torch.finfo(torch.float32).tiny
    _add_step(illegal, actual=torch.tensor([outside], dtype=torch.float32))
    assert illegal.i64[M._AI + 3].item() == 1
    with pytest.raises(ValueError, match="actual reward conservation"):
        M.decode_host_window(*illegal.pack_views())


def test_actual_close_rejects_nonfinite_and_missing_or_double_close():
    leaf = M.MilestoneTensorAccumulator(2, torch.device("cpu"))
    with pytest.raises(RuntimeError, match="not open"):
        leaf.close_actual_step(torch.zeros(2))

    for dtype in (torch.float16, torch.float64):
        drift = M.MilestoneTensorAccumulator(1, torch.device("cpu"))
        for ordinal in range(M.REWARD_TERM_COUNT):
            drift.add_reward(
                ordinal, torch.zeros(1), torch.zeros(1),
                torch.ones(1, dtype=torch.bool),
                torch.ones(1, dtype=torch.bool),
                torch.ones((), dtype=torch.float64),
            )
        with pytest.raises(RuntimeError, match="tensor ABI"):
            drift.close_actual_step(torch.zeros(1, dtype=dtype))
    _add_step(leaf, actual=torch.tensor([float("nan"), 0.0]))
    with pytest.raises(ValueError, match="actual reward conservation"):
        M.decode_host_window(*leaf.pack_views())
    with pytest.raises(RuntimeError, match="not open"):
        leaf.close_actual_step(torch.zeros(2))


def test_decoder_rejects_negative_domains_and_independent_actual_verdicts():
    leaf = M.MilestoneTensorAccumulator(2, torch.device("cpu"))
    payments = tuple(
        torch.tensor([1.0, -2.0]) if index == 0 else torch.zeros(2)
        for index in range(M.REWARD_TERM_COUNT)
    )
    _add_step(leaf, payments=payments)
    leaf.close_episodes(
        torch.tensor([True, False]), torch.tensor([1, 0]),
        torch.tensor([1, 0]),
    )
    good_i, good_f = (value.clone() for value in leaf.pack_views())
    M.decode_host_window(good_i, good_f)
    mutants = []
    bad_count = good_i.clone()
    bad_count[3] = -1
    mutants.append((bad_count, good_f))
    for offset in (1, 4, 5, 6):
        negative = good_f.clone()
        negative[offset] = -5.0e-10
        mutants.append((good_i, negative))
    for offset in (1, 3, 4, 5, 6):
        negative = good_f.clone()
        negative[M._AF + offset] = -5.0e-13
        mutants.append((good_i, negative))
    bad_return_square = good_f.clone()
    bad_return_square[M._EPF + 1] = -5.0e-10
    mutants.append((good_i, bad_return_square))
    nonfinite_verdict = good_i.clone()
    nonfinite_verdict[M._AI + 2] = 1
    mutants.append((nonfinite_verdict, good_f))
    mismatch_verdict = good_i.clone()
    mismatch_verdict[M._AI + 3] = 1
    mutants.append((mismatch_verdict, good_f))
    for i64, f64 in mutants:
        with pytest.raises(ValueError):
            M.decode_host_window(i64, f64)


def test_paddle_motion_prior_playback_is_exactly_masked_non_gating_telemetry():
    leaf = M.MilestoneTensorAccumulator(3, torch.device("cpu"))
    playback = torch.tensor([True, False, True], dtype=torch.bool)
    first = M.REWARD_TERM_COUNT - len(M.PADDLE_PLAYBACK_TERM_NAMES)
    expected = []
    for index, ordinal in enumerate(range(first, M.REWARD_TERM_COUNT)):
        kernel = torch.tensor(
            [0.5 + 0.1 * index, 0.99, 0.25 + 0.1 * index],
            dtype=torch.float32,
        )
        leaf.add_paddle_motion_prior_playback(
            ordinal, kernel, playback
        )
        expected.append(kernel)

    payload = M.decode_host_window(*leaf.pack_views()).as_json(_term_names())
    rows = payload["paddle_motion_prior_playback"]["terms"]
    assert payload["paddle_motion_prior_playback"]["predicate"] == (
        "Motion.action_ball_full_mdp_playback_active_mask"
    )
    assert tuple(row["term"] for row in rows) == M.PADDLE_PLAYBACK_TERM_NAMES
    for row, kernel in zip(rows, expected):
        assert row["telemetry_unavailable_count"] == 0
        assert row["playback_count"] == 2
        assert row["finite_count"] == 2
        assert row["domain_violation_count"] == 0
        assert row["kernel_sum"] == pytest.approx(
            float(kernel[0].double() + kernel[2].double())
        )
        assert row["kernel_sum_sq"] == pytest.approx(
            float(kernel[0].double().square() + kernel[2].double().square())
        )
    # Telemetry does not open or mutate the configured-income reward step.
    assert leaf._scratch_open is False
    assert not bool(leaf.open_step_configured_income.any())


def test_paddle_motion_prior_unavailable_is_explicit_non_gating_telemetry():
    leaf = M.MilestoneTensorAccumulator(3, torch.device("cpu"))
    first = M.REWARD_TERM_COUNT - len(M.PADDLE_PLAYBACK_TERM_NAMES)
    for ordinal in range(first, M.REWARD_TERM_COUNT):
        leaf.add_paddle_motion_prior_unavailable(ordinal)
    rows = M.decode_host_window(*leaf.pack_views()).as_json(_term_names())[
        "paddle_motion_prior_playback"
    ]["terms"]
    assert [row["telemetry_unavailable_count"] for row in rows] == [3] * 4
    assert [row["playback_count"] for row in rows] == [0] * 4
    assert [row["finite_count"] for row in rows] == [0] * 4
    assert leaf._scratch_open is False


def test_paddle_motion_prior_playback_counts_active_nonfinite_without_pollution():
    leaf = M.MilestoneTensorAccumulator(3, torch.device("cpu"))
    first = M.REWARD_TERM_COUNT - len(M.PADDLE_PLAYBACK_TERM_NAMES)
    leaf.add_paddle_motion_prior_playback(
        first,
        torch.tensor([0.5, float("nan"), 0.0], dtype=torch.float32),
        torch.tensor([True, True, True], dtype=torch.bool),
    )
    row = M.decode_host_window(*leaf.pack_views()).as_json(_term_names())[
        "paddle_motion_prior_playback"
    ]["terms"][0]
    assert row["playback_count"] == 3
    assert row["finite_count"] == 2
    assert row["domain_violation_count"] == 0
    assert row["kernel_sum"] == 0.5


def test_paddle_motion_prior_domain_drift_is_telemetry_not_a_decoder_gate():
    leaf = M.MilestoneTensorAccumulator(3, torch.device("cpu"))
    first = M.REWARD_TERM_COUNT - len(M.PADDLE_PLAYBACK_TERM_NAMES)
    leaf.add_paddle_motion_prior_playback(
        first,
        torch.tensor([-2.0, 1.25, 0.0], dtype=torch.float32),
        torch.ones(3, dtype=torch.bool),
    )
    row = M.decode_host_window(*leaf.pack_views()).as_json(_term_names())[
        "paddle_motion_prior_playback"
    ]["terms"][0]
    assert row["playback_count"] == 3
    assert row["finite_count"] == 3
    assert row["domain_violation_count"] == 2
    assert row["kernel_sum"] == pytest.approx(-0.75)
    assert row["kernel_sum_sq"] == pytest.approx(5.5625)


def test_decoder_does_not_recheck_same_writer_analytic_relationships():
    i64 = torch.zeros(M.I64_NUMEL, dtype=torch.int64)
    f64 = torch.zeros(M.F64_NUMEL, dtype=torch.float64)
    i64[3] = 3
    i64[M._EPI:M._EPI + 7] = torch.tensor(
        [1, 0, 2, 3, 4, 5, 6], dtype=torch.int64
    )
    f64[0:7] = torch.tensor(
        [9.0, 0.0, -7.0, -5.0, 0.0, 4.0, 8.0], dtype=torch.float64
    )
    f64[M._AF:M._EPF] = torch.tensor(
        [12.0, 0.0, 99.0, 1.0, 0.0, 3.0, 2.0], dtype=torch.float64
    )
    f64[M._EPF:M._EPF + 2] = torch.tensor([7.0, 0.0], dtype=torch.float64)
    decoded = M.decode_host_window(i64, f64)
    assert decoded.i64 == tuple(i64.tolist())
    assert decoded.f64 == tuple(f64.tolist())


def test_event_schema_is_honest_and_sticky_incidence_survives_ack():
    leaf = M.MilestoneTensorAccumulator(2, torch.device("cpu"))
    first = torch.tensor([[True], [False]])
    leaf.add_first_fact_event("r07_recovery", first)
    leaf.add_first_fact_event("r07_recovery", torch.zeros_like(first))
    leaf.add_first_fact_event("r07_recovery", first)
    leaf.add_r07_first_ready(first)
    leaf.add_r07_first_ready(first)
    payload = M.decode_host_window(*leaf.pack_views()).as_json(_term_names())
    events = {row["event"]: row for row in payload["event_ladder"]["events"]}
    assert events["r07_recovery_outcome_first_valid"]["count"] == 1
    assert events["r07_recovery_first_ready"]["count"] == 1
    assert events["d05_due"]["unit"] == "env_slot_row_per_D05_SETTLED"
    assert events["d05_construction_admitted"]["unit"] == "env_slot_row_per_D05_SETTLED"
    assert events["d05_key_admitted"]["unit"] == "full_key_row_per_D05_SETTLED"
    assert "NUMERICALLY_VALID" in events["r07_recovery_outcome_first_valid"]["predicate"]

    leaf.freeze_window_()
    leaf.clear_window_()
    leaf.add_first_fact_event("r07_recovery", first)
    leaf.add_r07_first_ready(first)
    assert not bool(leaf.i64[M._EI:M._EPI].any())
    leaf.reset_event_envs(torch.tensor([True, False]))
    leaf.add_first_fact_event("r07_recovery", first)
    leaf.add_r07_first_ready(first)
    assert leaf.i64[M._EI + 14].item() == 1
    assert leaf.i64[M._EI + 15].item() == 1


def test_event_positions_decode_one_distinct_known_vector_and_reject_field_swap():
    i64 = torch.zeros(M.I64_NUMEL, dtype=torch.int64)
    f64 = torch.zeros(M.F64_NUMEL, dtype=torch.float64)
    known = (40, 30, 20, 10, 5, 6, 7, 70, 60, 50, 40, 35, 30, 20, 8, 9)
    i64[M._EI:M._EPI] = torch.tensor(known, dtype=torch.int64)
    decoded = M.decode_host_window(i64, f64).as_json(_term_names())
    actual = tuple(
        row["count"] for row in decoded["event_ladder"]["events"]
    )
    assert actual == known
    assert tuple(
        row["event"] for row in decoded["event_ladder"]["events"]
    ) == M.EVENT_NAMES

    swapped = i64.clone()
    swapped[M._EI + 8], swapped[M._EI + 9] = (
        i64[M._EI + 9], i64[M._EI + 8]
    )
    swapped_payload = M.decode_host_window(swapped, f64).as_json(_term_names())
    swapped_values = tuple(
        row["count"] for row in swapped_payload["event_ladder"]["events"]
    )
    assert swapped_values != known


def test_actual_residual_same_shape_fields_remain_telemetry_not_a_second_gate():
    leaf = M.MilestoneTensorAccumulator(2, torch.device("cpu"))
    payments = tuple(
        torch.tensor([1.0, 2.0]) if index == 0 else torch.zeros(2)
        for index in range(M.REWARD_TERM_COUNT)
    )
    _add_step(leaf, payments=payments)
    i64, f64 = (value.clone() for value in leaf.pack_views())
    mutated = f64.clone()
    mutated[M._AF], mutated[M._AF + 1] = f64[M._AF + 1], f64[M._AF]
    decoded = M.decode_host_window(i64, mutated)
    assert decoded.f64[M._AF:M._AF + 2] == (
        float(mutated[M._AF]), float(mutated[M._AF + 1])
    )


def test_episode_return_carry_survives_ack_clear_until_successful_reset():
    leaf = M.MilestoneTensorAccumulator(2, torch.device("cpu"))
    leaf.add_step_return(torch.tensor([1.0, 2.0]))
    leaf.add_step_return(torch.tensor([3.0, 4.0]))
    leaf.close_episodes(
        torch.tensor([True, False]), torch.tensor([2, 2]), torch.tensor([2, 0])
    )
    first = M.decode_host_window(*leaf.pack_views()).as_json(_term_names())
    assert first["episodes"]["return_sum"] == 4.0
    leaf.freeze_window_()
    leaf.clear_window_()
    assert torch.equal(leaf.open_episode_return, torch.tensor([0.0, 6.0], dtype=torch.float64))
    leaf.add_step_return(torch.tensor([0.0, 1.0]))
    leaf.close_episodes(
        torch.tensor([False, True]), torch.tensor([1, 3]), torch.tensor([0, 4 | 16])
    )
    second = M.decode_host_window(*leaf.pack_views()).as_json(_term_names())
    assert second["episodes"]["return_sum"] == 7.0
    assert second["episodes"]["reason_base_too_low"] == 1
    assert second["episodes"]["reason_robot_hit_table"] == 1


def test_frozen_window_rejects_all_writes_before_any_tensor_mutation():
    leaf = M.MilestoneTensorAccumulator(2, torch.device("cpu"))
    leaf.add_step_return(torch.tensor([1.0, 2.0]))
    leaf.freeze_window_()

    def all_tensors():
        return (
            *leaf.pack_views(), leaf.open_episode_return,
            leaf.open_step_configured_income, leaf.open_step_configured_abs_income,
            leaf.r03_seen, leaf.r07_outcome_seen, leaf.r07_ready_seen,
        )

    frozen = tuple(value.clone() for value in all_tensors())
    calls = (
        lambda: leaf.add_step_return(torch.ones(2)),
        lambda: leaf.close_episodes(torch.ones(2, dtype=torch.bool), torch.ones(2, dtype=torch.int64), torch.ones(2, dtype=torch.int64)),
        lambda: leaf.add_reward(0, torch.ones(2), torch.ones(2), torch.ones(2, dtype=torch.bool), torch.ones(2, dtype=torch.bool), torch.ones((), dtype=torch.float64)),
        lambda: leaf.add_first_fact_event("r03_strike_fact", torch.ones((2, 1), dtype=torch.bool)),
        lambda: leaf.add_r07_first_ready(torch.ones((2, 1), dtype=torch.bool)),
    )
    for call in calls:
        with pytest.raises(RuntimeError, match="frozen"):
            call()
        assert all(torch.equal(before, after) for before, after in zip(frozen, all_tensors()))
    leaf.abort_window_()
    leaf.add_step_return(torch.ones(2))
    leaf.freeze_window_()
    leaf.clear_window_()
    leaf.add_step_return(torch.ones(2))


def test_leaf_has_no_host_transfer_or_scalar_read_and_decoder_rejects_bad_host():
    source = inspect.getsource(M.MilestoneTensorAccumulator)
    assert all(token not in source for token in (".cpu(", ".item(", ".tolist("))
    assert '.to(device="cpu"' not in source
    assert "_single_d2h_checkpoint_carry" not in vars(M)
    with pytest.raises(ValueError, match="ABI differs"):
        M.decode_host_window(
            torch.zeros(M.I64_NUMEL - 1, dtype=torch.int64),
            torch.zeros(M.F64_NUMEL, dtype=torch.float64),
        )
    bad = torch.zeros(M.F64_NUMEL, dtype=torch.float64)
    bad[0] = float("nan")
    with pytest.raises(ValueError, match="values differ"):
        M.decode_host_window(torch.zeros(M.I64_NUMEL, dtype=torch.int64), bad)


def test_private_carry_leaf_only_defines_schema_capture_stage_and_copy_views():
    capture_source = inspect.getsource(
        M.MilestoneTensorAccumulator._lean_carry_capture
    )
    assert all(
        token not in capture_source
        for token in ("bool(", ".any(", ".item(", ".tolist(", "isfinite")
    )
    source = M.MilestoneTensorAccumulator(2, torch.device("cpu"))
    source.add_step_return(torch.tensor([3.0, 4.0]))
    source.add_first_fact_event(
        "r03_strike_fact", torch.tensor([[True], [False]])
    )
    source.add_first_fact_event(
        "r07_recovery", torch.tensor([[False], [True]])
    )
    source.add_r07_first_ready(torch.tensor([[True], [True]]))
    source.freeze_window_()
    source.clear_window_()
    source_marker = object()
    source._lean_carry_coordinator = source_marker
    source_lease = types.SimpleNamespace(coordinator=source_marker, kind="capture")
    capture = source._lean_carry_capture(source_lease)
    assert capture.scalars == ()
    assert len(capture.tensors) == 8
    assert all(not bool(value.any()) for value in capture.tensors[:4])
    target = M.MilestoneTensorAccumulator(2, torch.device("cpu"))
    target_marker = types.SimpleNamespace(_active_lease=None)
    target._lean_carry_coordinator = target_marker
    target_lease = types.SimpleNamespace(coordinator=target_marker, kind="prepare")
    target_marker._active_lease = target_lease
    stage = target._lean_carry_stage(target_lease, (), capture.tensors)
    assert type(stage).__name__ == "_LeanCarryStage"
    assert stage.commit_started is False
    assert target._lean_carry_target_views(target_lease, stage) == stage.targets
    for field, staged, live in zip(
        target._lean_carry_schema().tensor_fields, stage.staging, stage.targets
    ):
        if field.disposition == "copy":
            live.copy_(staged)
    armed = type(stage)(stage.scalars, stage.staging, stage.targets, True)
    target._lean_carry_apply_scalars(target_lease, armed)
    assert torch.equal(target.open_episode_return, source.open_episode_return)


@pytest.mark.parametrize("field_index", range(8))
def test_private_carry_rejects_each_dirty_target_without_mutation(field_index):
    source = M.MilestoneTensorAccumulator(2, torch.device("cpu"))
    source.freeze_window_()
    source.clear_window_()
    source_marker = object()
    source._lean_carry_coordinator = source_marker
    capture = source._lean_carry_capture(
        types.SimpleNamespace(coordinator=source_marker, kind="capture")
    )
    target = M.MilestoneTensorAccumulator(2, torch.device("cpu"))
    target_marker = types.SimpleNamespace(_active_lease=None)
    target._lean_carry_coordinator = target_marker
    lease = types.SimpleNamespace(coordinator=target_marker, kind="prepare")
    target_marker._active_lease = lease
    dirty = target._checkpoint_device_views()[field_index]
    dirty.reshape(-1)[0] = True if dirty.dtype is torch.bool else 1
    before = tuple(value.clone() for value in target._checkpoint_device_views())

    with pytest.raises(RuntimeError, match="target is not dormant"):
        target._lean_carry_stage(lease, (), capture.tensors)

    assert all(
        torch.equal(value, expected)
        for value, expected in zip(target._checkpoint_device_views(), before)
    )
