from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from dataclasses import replace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
    / "lateral_perturbation.py"
)
PREREG_PATH = ROOT.parents[1] / "configs" / "phase1_lateral_balance_perturbation_prereg_20260715.json"


def _load_module():
    name = "lateral_perturbation_under_test"
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


L = _load_module()


def _cfg(**overrides):
    values = {
        "policy_dt_s": 0.02,
        "opportunity_interval_steps": 4,
        "pulse_duration_steps": 2,
        "selection_probability": 1.0,
        "normalized_impulse_min_mps": 0.04,
        "normalized_impulse_max_mps": 0.08,
        "seed": 20260715,
    }
    values.update(overrides)
    return L.LateralPerturbationConfig(**values)


def _inputs(
    n,
    episode_step,
    *,
    episode_index=0,
    eligible=True,
    strike=False,
    safe=100,
):
    def vector(value, dtype):
        if isinstance(value, torch.Tensor):
            return value.to(dtype=dtype)
        if isinstance(value, (list, tuple)):
            return torch.tensor(value, dtype=dtype)
        return torch.full((n,), value, dtype=dtype)

    return {
        "episode_indices": vector(episode_index, torch.long),
        "episode_steps": vector(episode_step, torch.long),
        "recovery_hold_eligible": vector(eligible, torch.bool),
        "strike_window": vector(strike, torch.bool),
        "safe_window_remaining_steps": vector(safe, torch.long),
    }


class _RecordingAdapter:
    body_name = "torso_link"
    input_force_frame = "world"
    application_point = "center_of_mass"
    full_batch_overwrite = True
    inactive_zero_overwrite = True
    world_to_backend_transform_identity_sha256 = "a" * 64

    def __init__(self, *, receipt_override=None):
        self.calls = []
        self.receipt_override = receipt_override

    def overwrite_world_wrench_at_body_com(
        self, *, step_token, total_mass_kg, force_w, torque_w
    ):
        self.calls.append((step_token, force_w.clone(), torque_w.clone()))
        receipt = L.LateralWrenchWriteReceipt(
            step_token=step_token,
            body_name=self.body_name,
            input_force_frame=self.input_force_frame,
            application_point=self.application_point,
            full_batch_overwrite=self.full_batch_overwrite,
            inactive_zero_overwrite=self.inactive_zero_overwrite,
            zero_torque=bool(torch.all(torque_w == 0.0)),
            world_to_backend_transform_identity_sha256=(
                self.world_to_backend_transform_identity_sha256
            ),
            actual_total_mass_kg=total_mass_kg.clone(),
            commanded_force_w=force_w.clone(),
            commanded_torque_w=torque_w.clone(),
            applied_force_mask=torch.any(
                force_w.reshape(force_w.shape[0], -1) != 0.0, dim=1
            ),
        )
        if self.receipt_override is not None:
            return self.receipt_override(receipt)
        return receipt


def _dispatch(scheduler, result, adapter, mass=None):
    if mass is None:
        mass = torch.full(
            (scheduler.num_envs,),
            40.0,
            dtype=result.normalized_accel_y_mps2.dtype,
        )
    return L.dispatch_lateral_wrench_fail_closed(
        scheduler=scheduler,
        result=result,
        total_mass_kg=mass,
        adapter=adapter,
    )


def _tensor_bits(value):
    return value.detach().contiguous().reshape(-1).view(torch.uint8).clone()


def _tensor_field_bits(dataclass_value):
    return {
        name: _tensor_bits(value)
        for name, value in vars(dataclass_value).items()
        if isinstance(value, torch.Tensor)
    }


def _assert_tensor_field_bits_unchanged(dataclass_value, expected):
    for name, bits in expected.items():
        assert torch.equal(_tensor_bits(getattr(dataclass_value, name)), bits), name


def test_config_freezes_recovery_hold_torso_world_com_and_rejects_anytime():
    cfg = _cfg()
    assert cfg.eligibility_mode == "recovery_hold"
    assert cfg.body_name == "torso_link"
    assert cfg.force_frame == "world"
    assert cfg.application_point == "center_of_mass"
    assert cfg.pulse_duration_s == pytest.approx(0.04)
    assert not cfg.is_zero_control
    assert _cfg(
        normalized_impulse_min_mps=0.0,
        normalized_impulse_max_mps=0.0,
    ).is_zero_control

    with pytest.raises(ValueError, match="separate future causal axis"):
        _cfg(eligibility_mode="anytime")
    with pytest.raises(ValueError, match="torso_link"):
        _cfg(body_name="pelvis")
    with pytest.raises(ValueError, match="positive plain int"):
        _cfg(pulse_duration_steps=True)
    with pytest.raises(ValueError, match="cannot exceed"):
        _cfg(
            normalized_impulse_min_mps=0.08,
            normalized_impulse_max_mps=0.04,
        )


def test_config_hard_safety_envelope_rejects_large_finite_and_derived_overflow():
    expected = {
        "schema_version": 1,
        "max_abs_normalized_impulse_mps": 0.15,
        "max_abs_normalized_accel_mps2": 2.0,
        "min_pulse_duration_s": 0.02,
        "max_pulse_duration_s": 0.2,
        "max_abs_world_force_y_N": 200.0,
        "world_force_xz_N": [0.0, 0.0],
        "explicit_torque_Nm": [0.0, 0.0, 0.0],
    }
    assert L.lateral_hard_safety_contract() == expected
    assert L.lateral_hard_safety_identity_sha256() == (
        "7de6f9a7ab418a63973e1680a56d7ca82d9b8c19cd1ac52d32d332cb6819dc45"
    )
    assert _cfg().hard_safety_identity_sha256 == L.lateral_hard_safety_identity_sha256()

    with pytest.raises(ValueError, match="normalized impulse exceeds"):
        _cfg(normalized_impulse_max_mps=1.0e300)
    with pytest.raises(ValueError, match="derived pulse_duration_s must be finite"):
        _cfg(policy_dt_s=1.0e308, pulse_duration_steps=2)
    with pytest.raises(ValueError, match="pulse_duration_s is outside"):
        _cfg(policy_dt_s=0.001, pulse_duration_steps=2)
    with pytest.raises(ValueError, match="derived normalized acceleration exceeds"):
        _cfg(policy_dt_s=0.02, pulse_duration_steps=1)

    lower_boundary = _cfg(
        policy_dt_s=0.02,
        pulse_duration_steps=1,
        normalized_impulse_min_mps=0.04,
        normalized_impulse_max_mps=0.04,
    )
    upper_duration_boundary = _cfg(
        policy_dt_s=0.02,
        opportunity_interval_steps=10,
        pulse_duration_steps=10,
    )
    assert lower_boundary.pulse_duration_s == 0.02
    assert upper_duration_boundary.pulse_duration_s == 0.2


def test_preregistered_train_and_eval_boundaries_are_machine_readable_and_blocked():
    payload = json.loads(PREREG_PATH.read_text())
    assert payload["schema_version"] == 1
    assert payload["launch_authorized"] is False
    assert payload["runtime_adapter"]["implemented"] is False
    assert payload["runtime_adapter"]["persistent_force_clearance_verified"] is False
    assert payload["eligibility"]["first_cell"] == "recovery_hold"
    assert payload["eligibility"]["anytime_axis_in_this_prereg"] is False
    assert payload["wrench_contract"] == {
        "body_name": "torso_link",
        "application_point": "center_of_mass",
        "force_frame": "world",
        "force_components": [0.0, "sampled_Fy", 0.0],
        "torque_components": [0.0, 0.0, 0.0],
        "normalization_mass": "total_articulation_mass_after_randomization",
        "torso_com_semantics": (
            "zero_explicit_torque_and_zero_torso_link_local_lever_arm_only; "
            "torso_com_is_not_whole_articulation_com; "
            "whole_articulation_r_cross_F_angular_impulse_and_contact_response_remain_physical"
        ),
        "direct_root_velocity_write": False,
    }
    assert payload["hard_safety_envelope"]["identity_sha256"] == (
        "7de6f9a7ab418a63973e1680a56d7ca82d9b8c19cd1ac52d32d332cb6819dc45"
    )
    safety_payload = dict(payload["hard_safety_envelope"])
    safety_payload.pop("identity_sha256")
    safety_payload.pop("basis")
    safety_payload.pop("fail_closed_checks")
    assert safety_payload == L.lateral_hard_safety_contract()
    control = payload["train_cells"][0]
    treatment = payload["train_cells"][1]
    assert control["normalized_impulse_mps"] == [0.0, 0.0]
    assert treatment["normalized_impulse_mps"] == [0.04, 0.08]
    assert control["schedule_seed"] == treatment["schedule_seed"]
    assert payload["pulse_schedule"]["sampling"] == (
        "philox4x32-10-domain-separated-v1"
    )
    assert payload["pulse_schedule"]["magnitude_distribution"].startswith(
        "affine_of_uniform_open_0_1"
    )
    expected_schedule_sha = (
        "d157bd6e7c063df80d41ca03b9eb4acae2a4b45c9ee0967b5dcbce5b76d14593"
    )
    assert payload["pulse_schedule"]["random_schedule_identity_sha256"] == (
        expected_schedule_sha
    )
    assert payload["common_random_numbers"][
        "random_schedule_identity_shared_by_L0_L1"
    ] == expected_schedule_sha
    prereg_cfg = L.LateralPerturbationConfig(
        policy_dt_s=payload["pulse_schedule"]["policy_dt_s"],
        opportunity_interval_steps=payload["pulse_schedule"][
            "opportunity_interval_steps"
        ],
        pulse_duration_steps=payload["pulse_schedule"]["pulse_duration_steps"],
        selection_probability=payload["pulse_schedule"][
            "selection_probability_per_eligible_opportunity"
        ],
        normalized_impulse_min_mps=0.0,
        normalized_impulse_max_mps=0.0,
        seed=control["schedule_seed"],
    )
    assert prereg_cfg.random_schedule_identity_sha256 == expected_schedule_sha
    assert payload["held_out_eval"]["clean"]["normalized_impulse_mps"] == [0.0, 0.0]
    assert payload["held_out_eval"]["strong"]["normalized_impulse_mps"] == [0.1, 0.14]
    assert payload["held_out_eval"]["strong"]["schedule_seed"] != treatment["schedule_seed"]
    paper = payload["held_out_eval"]["ball_by_action_family_paper"]
    assert paper["status"] == "pending"
    assert paper["required_before_launch"] is True
    assert paper["required_before_promotion"] is True
    assert paper["report_all_bins_and_worst_bin"] is True
    throughput = payload["runtime_unlock_gates"]["gpu_throughput"]
    assert throughput["status"] == "pending"
    assert throughput["required_before_launch"] is True
    assert throughput["minimum_environment_steps_per_second_ratio_vs_no_hook"] == 0.95
    assert throughput["host_device_sync_in_hot_path_allowed"] is False
    runtime_adapter = payload["runtime_adapter"]
    assert runtime_adapter[
        "adapter_receives_only_isolated_mass_force_and_torque_clones"
    ] is True
    assert runtime_adapter[
        "adapter_exception_or_rejection_must_leave_caller_tensors_bit_exact"
    ] is True
    assert runtime_adapter[
        "scheduler_application_cache_is_private_and_every_public_ledger_return_is_a_deep_clone"
    ] is True
    assert {
        "hard_safety_identity_sha256",
        "actual_total_articulation_mass_after_randomization_kg",
        "commanded_normalized_accel_y_mps2",
        "commanded_force_y_N",
        "commanded_world_impulse_y_Ns",
        "world_to_backend_transform_identity_sha256",
    } <= set(payload["per_step_ledger_required"])
    assert {
        "lateral_perturbation_interrupted_for_reset_count",
        "lateral_perturbation_reset_interrupted_sampled_impulse_abs_sum_mps",
        "lateral_perturbation_reset_abandoned_uncommanded_impulse_abs_sum_mps",
        "lateral_perturbation_reset_abandoned_unapplied_impulse_abs_sum_mps",
    } <= set(payload["activation_and_application_counters"])
    metrics = set(payload["held_out_eval"]["metrics"])
    assert {
        "recovery_time_to_ready_s_all_attempts",
        "capture_point_to_support_margin_min_m_all_attempts",
        "com_projection_to_support_margin_min_m_all_attempts",
        "stance_width_narrowing_from_episode_start_max_m_all_attempts",
        "left_right_foot_yaw_error_abs_max_rad_all_attempts",
        "physical_fall_rate_all_attempts",
        "ready_set_by_deadline_rate_all_attempts",
        "next_strike_composite_rate_all_attempts",
    } <= metrics


def test_philox_matches_random123_zero_counter_zero_key_vector():
    zero = torch.zeros(1, dtype=torch.long)
    lanes = L._philox4x32_10((zero, zero, zero, zero), (0, 0))
    assert [int(lane[0]) for lane in lanes] == [
        0x6627E8D5,
        0xE169C58D,
        0xBC57AC4C,
        0x9B00DBD8,
    ]


def _pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).mean() / (a.std(unbiased=False) * b.std(unbiased=False)))


def _uniform_diagnostics(values, bins=32):
    counts = torch.histc(values, bins=bins, min=0.0, max=1.0)
    expected = values.numel() / bins
    chi_square = float(torch.sum((counts - expected) ** 2 / expected))
    return float(values.mean()), float(values.var(unbiased=False)), chi_square


def test_philox_domains_and_neighbor_seeds_are_not_linearly_correlated():
    n = 131_072
    env_ids = torch.arange(n, dtype=torch.long)
    episode_indices = torch.remainder(env_ids * 17 + 3, 65_521)
    opportunity_indices = torch.remainder(env_ids * 29 + 11, 1_000_003)
    domains = {
        name: L._counter_uniform(
            seed=20260715,
            env_ids=env_ids,
            episode_indices=episode_indices,
            opportunity_indices=opportunity_indices,
            domain=name,
        )
        for name in ("phase_offset", "selection", "direction", "unit_magnitude")
    }
    for values in domains.values():
        mean, variance, chi_square = _uniform_diagnostics(values)
        assert mean == pytest.approx(0.5, abs=0.003)
        assert variance == pytest.approx(1.0 / 12.0, abs=0.002)
        assert chi_square < 70.0

    names = sorted(domains)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            assert abs(_pearson(domains[left], domains[right])) < 0.015

    neighboring_seed = L._counter_uniform(
        seed=20260716,
        env_ids=env_ids,
        episode_indices=episode_indices,
        opportunity_indices=opportunity_indices,
        domain="selection",
    )
    assert not torch.equal(domains["selection"], neighboring_seed)
    assert abs(_pearson(domains["selection"], neighboring_seed)) < 0.015


def test_crn_schedule_identity_and_potential_draws_are_exposed_for_l0_audit():
    control_cfg = _cfg(
        normalized_impulse_min_mps=0.0,
        normalized_impulse_max_mps=0.0,
    )
    treatment_cfg = _cfg()
    assert (
        control_cfg.random_schedule_identity_sha256
        == treatment_cfg.random_schedule_identity_sha256
    )
    assert (
        control_cfg.random_schedule_identity_sha256
        != _cfg(seed=20260716).random_schedule_identity_sha256
    )
    control = L.LateralPulseScheduler(512, control_cfg)
    treatment = L.LateralPulseScheduler(512, treatment_cfg)
    for step in range(8):
        inputs = _inputs(512, step)
        left = control.step(step_token=step, **inputs)
        right = treatment.step(step_token=step, **inputs)
        assert left.random_schedule_identity_sha256 == (
            right.random_schedule_identity_sha256
        )
        for name in (
            "potential_phase_offset_steps",
            "opportunity_indices",
            "potential_selection_u01",
            "potential_direction_u01",
            "potential_unit_magnitude_u01",
        ):
            assert torch.equal(getattr(left, name), getattr(right, name))
        assert torch.all((left.potential_selection_u01 > 0.0) & (left.potential_selection_u01 < 1.0))
        assert torch.all((left.potential_direction_u01 > 0.0) & (left.potential_direction_u01 < 1.0))
        assert torch.all(
            (left.potential_unit_magnitude_u01 > 0.0)
            & (left.potential_unit_magnitude_u01 < 1.0)
        )


def test_episode_reset_mid_pulse_has_reconciled_abandonment_ledger_and_no_restart():
    scheduler = L.LateralPulseScheduler(
        1,
        _cfg(opportunity_interval_steps=5, pulse_duration_steps=5),
        require_application_ack=True,
    )
    adapter = _RecordingAdapter()
    started = None
    start_token = None
    for token in range(5):
        result = scheduler.step(
            step_token=token,
            **_inputs(1, token, episode_index=0, safe=20 - token),
        )
        _dispatch(scheduler, result, adapter)
        if result.nonzero_start_mask.item():
            started = result
            start_token = token
            break
    assert started is not None and start_token is not None

    reset = scheduler.step(
        step_token=start_token + 1,
        **_inputs(1, 0, episode_index=1, eligible=True, strike=False, safe=20),
    )
    assert reset.interrupted_for_reset_mask.item()
    assert not reset.opportunity_mask.item()
    assert not reset.selected_start_mask.item()
    assert reset.normalized_accel_y_mps2.item() == 0.0
    sampled = reset.reset_interrupted_sampled_impulse_y_mps
    commanded = reset.reset_interrupted_commanded_impulse_y_mps
    applied = reset.reset_interrupted_applied_impulse_y_mps
    assert torch.allclose(
        sampled,
        commanded + reset.reset_abandoned_uncommanded_impulse_y_mps,
        atol=1e-15,
        rtol=0.0,
    )
    assert torch.allclose(
        commanded,
        applied + reset.reset_abandoned_unapplied_impulse_y_mps,
        atol=1e-15,
        rtol=0.0,
    )
    assert torch.allclose(commanded, applied, atol=1e-15, rtol=0.0)
    assert 0.0 < commanded.abs().item() < sampled.abs().item()
    _dispatch(scheduler, reset, adapter)
    counters = scheduler.consume_counters()
    assert counters["lateral_perturbation_interrupted_for_reset_count"].item() == 1
    assert counters[
        "lateral_perturbation_reset_interrupted_sampled_impulse_abs_sum_mps"
    ].item() == pytest.approx(sampled.abs().item())
    assert counters[
        "lateral_perturbation_reset_abandoned_uncommanded_impulse_abs_sum_mps"
    ].item() == pytest.approx(
        reset.reset_abandoned_uncommanded_impulse_y_mps.abs().item()
    )
    assert counters[
        "lateral_perturbation_reset_abandoned_unapplied_impulse_abs_sum_mps"
    ].item() == pytest.approx(0.0, abs=1e-15)


def test_plan_only_reset_ledger_exposes_commanded_but_unacknowledged_impulse():
    scheduler = L.LateralPulseScheduler(
        1,
        _cfg(opportunity_interval_steps=5, pulse_duration_steps=5),
    )
    started = None
    start_token = None
    for token in range(5):
        result = scheduler.step(
            step_token=token,
            **_inputs(1, token, episode_index=0, safe=20 - token),
        )
        if result.nonzero_start_mask.item():
            started = result
            start_token = token
            break
    assert started is not None and start_token is not None
    reset = scheduler.step(
        step_token=start_token + 1,
        **_inputs(1, 0, episode_index=1, eligible=True, safe=20),
    )
    assert reset.interrupted_for_reset_mask.item()
    assert reset.reset_interrupted_applied_impulse_y_mps.item() == 0.0
    assert torch.equal(
        reset.reset_abandoned_unapplied_impulse_y_mps,
        reset.reset_interrupted_commanded_impulse_y_mps,
    )


def test_source_hot_path_has_no_explicit_tensor_item_or_bool_any_sync():
    source = MODULE_PATH.read_text()
    assert ".item(" not in source
    assert "bool(torch.any" not in source


def test_stateless_schedule_reproducible_and_seed_sensitive():
    a = L.LateralPulseScheduler(256, _cfg())
    b = L.LateralPulseScheduler(256, _cfg())
    c = L.LateralPulseScheduler(256, _cfg(seed=20260716))
    different = False
    for step in range(12):
        inputs = _inputs(256, step)
        ra = a.step(step_token=step, **inputs)
        rb = b.step(step_token=step, **inputs)
        rc = c.step(step_token=step, **inputs)
        for name in (
            "opportunity_mask",
            "eligible_opportunity_mask",
            "selected_start_mask",
            "sampled_normalized_impulse_y_mps",
            "normalized_accel_y_mps2",
        ):
            assert torch.equal(getattr(ra, name), getattr(rb, name))
        different |= not torch.equal(
            ra.sampled_normalized_impulse_y_mps,
            rc.sampled_normalized_impulse_y_mps,
        )
    assert different


def test_control_and_treatment_share_schedule_but_only_treatment_applies_force():
    control_cfg = _cfg(
        normalized_impulse_min_mps=0.0,
        normalized_impulse_max_mps=0.0,
    )
    treatment_cfg = _cfg()
    control = L.LateralPulseScheduler(
        128, control_cfg, require_application_ack=True
    )
    treatment = L.LateralPulseScheduler(
        128, treatment_cfg, require_application_ack=True
    )
    control_adapter = _RecordingAdapter()
    treatment_adapter = _RecordingAdapter()
    for step in range(16):
        inputs = _inputs(128, step)
        rc = control.step(step_token=step, **inputs)
        rt = treatment.step(step_token=step, **inputs)
        assert torch.equal(rc.opportunity_mask, rt.opportunity_mask)
        assert torch.equal(rc.eligible_opportunity_mask, rt.eligible_opportunity_mask)
        assert torch.equal(rc.selected_start_mask, rt.selected_start_mask)
        assert torch.equal(
            torch.sign(rc.sampled_normalized_impulse_y_mps),
            torch.zeros_like(rc.sampled_normalized_impulse_y_mps),
        )
        _dispatch(control, rc, control_adapter)
        _dispatch(treatment, rt, treatment_adapter)

    cc = control.consume_counters()
    tc = treatment.consume_counters()
    for suffix in (
        "eligible_opportunity_count",
        "selected_start_count",
        "selected_left_count",
        "selected_right_count",
        "wrench_write_step_count",
    ):
        key = "lateral_perturbation_" + suffix
        assert cc[key].item() == tc[key].item()
    assert cc["lateral_perturbation_applied_pulse_count"].item() == 0
    assert tc["lateral_perturbation_applied_pulse_count"].item() > 0
    assert all(torch.all(force == 0.0) for _, force, _ in control_adapter.calls)
    assert any(torch.any(force != 0.0) for _, force, _ in treatment_adapter.calls)


def test_direction_and_magnitude_distribution_are_symmetric_and_bounded():
    n = 8192
    scheduler = L.LateralPulseScheduler(
        n,
        _cfg(policy_dt_s=0.04, opportunity_interval_steps=1, pulse_duration_steps=1),
    )
    result = scheduler.step(step_token=0, **_inputs(n, 0))
    impulses = result.sampled_normalized_impulse_y_mps
    assert result.selected_start_mask.all()
    assert torch.all(impulses.abs() >= 0.04)
    assert torch.all(impulses.abs() <= 0.08)
    assert torch.unique(impulses.abs()).numel() > 100
    left = int((impulses < 0).sum())
    right = int((impulses > 0).sum())
    assert abs(left - right) / n < 0.02


def test_first_cell_never_starts_in_strike_or_outside_recovery_hold():
    scheduler = L.LateralPulseScheduler(
        4,
        _cfg(policy_dt_s=0.04, opportunity_interval_steps=1, pulse_duration_steps=1),
    )
    result = scheduler.step(
        step_token=0,
        **_inputs(
            4,
            0,
            eligible=[True, True, False, False],
            strike=[False, True, False, True],
            safe=[1, 1, 1, 1],
        ),
    )
    assert result.eligible_opportunity_mask.tolist() == [True, False, False, False]
    assert result.nonzero_start_mask.tolist() == [True, False, False, False]
    assert torch.equal(
        result.normalized_accel_y_mps2.ne(0.0),
        torch.tensor([True, False, False, False]),
    )
    counters = scheduler.consume_counters()
    assert counters["lateral_perturbation_eligible_opportunity_count"].item() == 1
    assert counters["lateral_perturbation_skipped_strike_window_count"].item() == 2
    assert counters["lateral_perturbation_skipped_ineligible_phase_count"].item() == 1


def test_pulse_requires_full_safe_window_and_aborts_to_zero_if_strike_appears():
    short = L.LateralPulseScheduler(
        1,
        _cfg(opportunity_interval_steps=2, pulse_duration_steps=2),
    )
    # A stateless phase offset may place the first opportunity at step 0 or 1.
    no_start = None
    for token in range(2):
        candidate = short.step(step_token=token, **_inputs(1, token, safe=1))
        if candidate.opportunity_mask.item():
            no_start = candidate
            break
    assert no_start is not None
    assert not no_start.selected_start_mask.item()
    counters = short.consume_counters()
    assert counters["lateral_perturbation_skipped_short_window_count"].item() == 1

    scheduler = L.LateralPulseScheduler(1, _cfg())
    started = None
    start_token = None
    for token in range(4):
        candidate = scheduler.step(step_token=token, **_inputs(1, token, safe=10))
        if candidate.nonzero_start_mask.item():
            started = candidate
            start_token = token
            break
    assert started is not None and start_token is not None
    interrupted = scheduler.step(
        step_token=start_token + 1,
        **_inputs(1, start_token + 1, eligible=True, strike=True, safe=9),
    )
    assert interrupted.interrupted_for_strike_mask.item()
    assert interrupted.normalized_accel_y_mps2.item() == 0.0
    counters = scheduler.consume_counters()
    assert counters["lateral_perturbation_interrupted_for_strike_count"].item() == 1


def test_completed_pulse_obeys_sampled_impulse_budget_and_clears_next_step():
    scheduler = L.LateralPulseScheduler(1, _cfg())
    samples = []
    start_token = None
    sampled = None
    for token in range(4):
        result = scheduler.step(step_token=token, **_inputs(1, token, safe=20 - token))
        if result.nonzero_start_mask.item():
            start_token = token
            sampled = float(result.sampled_normalized_impulse_y_mps.abs().item())
            samples.append(result)
            break
    assert start_token is not None and sampled is not None
    samples.append(
        scheduler.step(
            step_token=start_token + 1,
            **_inputs(1, start_token + 1, safe=19 - start_token),
        )
    )
    cleared = scheduler.step(
        step_token=start_token + 2,
        **_inputs(1, start_token + 2, safe=18 - start_token),
    )
    integrated = sum(
        float(result.normalized_accel_y_mps2.abs().item())
        * scheduler.cfg.policy_dt_s
        for result in samples
    )
    assert integrated == pytest.approx(sampled, abs=1e-12)
    assert 0.04 <= sampled <= 0.08
    assert cleared.normalized_accel_y_mps2.item() == 0.0


def test_same_step_is_idempotent_but_changed_inputs_or_missing_step_fail_closed():
    scheduler = L.LateralPulseScheduler(
        8,
        _cfg(policy_dt_s=0.04, opportunity_interval_steps=1, pulse_duration_steps=1),
    )
    inputs = _inputs(8, 0)
    first = scheduler.step(step_token=0, **inputs)
    duplicate = scheduler.step(step_token=0, **inputs)
    assert torch.equal(
        first.sampled_normalized_impulse_y_mps,
        duplicate.sampled_normalized_impulse_y_mps,
    )
    counters = scheduler.consume_counters()
    assert counters["lateral_perturbation_selected_start_count"].item() == 8
    scheduler.step(step_token=0, **inputs)
    counters = scheduler.consume_counters()
    assert all(value.item() == 0 for value in counters.values())

    changed = _inputs(8, 0, strike=[True] + [False] * 7)
    with pytest.raises(RuntimeError, match="different perturbation inputs"):
        scheduler.step(step_token=0, **changed)
    with pytest.raises(RuntimeError, match="must be consecutive"):
        scheduler.step(step_token=2, **_inputs(8, 2))


def test_episode_clock_must_advance_or_reset_monotonically():
    repeated = L.LateralPulseScheduler(1, _cfg())
    repeated.step(step_token=0, **_inputs(1, 0, episode_index=4))
    with pytest.raises(RuntimeError, match="advance by exactly one"):
        repeated.step(step_token=1, **_inputs(1, 0, episode_index=4))

    bad_reset = L.LateralPulseScheduler(1, _cfg())
    bad_reset.step(step_token=0, **_inputs(1, 7, episode_index=4))
    with pytest.raises(RuntimeError, match="restart episode_steps at zero"):
        bad_reset.step(step_token=1, **_inputs(1, 8, episode_index=5))

    valid_reset = L.LateralPulseScheduler(1, _cfg())
    valid_reset.step(step_token=0, **_inputs(1, 7, episode_index=4))
    result = valid_reset.step(step_token=1, **_inputs(1, 0, episode_index=5))
    assert result.step_token == 1


def test_runtime_mode_requires_application_ack_before_next_step():
    scheduler = L.LateralPulseScheduler(
        2,
        _cfg(policy_dt_s=0.04, opportunity_interval_steps=1, pulse_duration_steps=1),
        require_application_ack=True,
    )
    result = scheduler.step(step_token=0, **_inputs(2, 0))
    with pytest.raises(RuntimeError, match="previous full-wrench application receipt"):
        scheduler.step(step_token=1, **_inputs(2, 1))
    adapter = _RecordingAdapter()
    _dispatch(scheduler, result, adapter)
    scheduler.step(step_token=1, **_inputs(2, 1))


def test_world_wrench_uses_total_mass_only_in_y_and_has_zero_torque():
    accel = torch.tensor([-1.5, 0.0, 2.0], dtype=torch.float64)
    total_mass = torch.tensor([20.0, 40.0, 60.0], dtype=torch.float64)
    force, torque = L.lateral_world_wrench_from_total_mass(accel, total_mass)
    assert force.shape == (3, 1, 3)
    assert torch.equal(force[:, 0, 1], accel * total_mass)
    assert torch.all(force[:, :, 0] == 0.0)
    assert torch.all(force[:, :, 2] == 0.0)
    assert torch.all(torque == 0.0)
    assert torch.equal(force[:, 0, 1] / total_mass, accel)

    force32, torque32 = L.lateral_world_wrench_from_total_mass(
        accel, total_mass.to(dtype=torch.float32)
    )
    assert force32.dtype == torch.float32
    assert torque32.dtype == torch.float32

    with pytest.raises(RuntimeError, match="strictly positive"):
        L.lateral_world_wrench_from_total_mass(accel, torch.tensor([20.0, 0.0, 60.0]))


def test_world_wrench_fails_closed_after_cast_multiply_and_final_force_bounds():
    with pytest.raises(RuntimeError, match="remain finite after cast"):
        L.lateral_world_wrench_from_total_mass(
            torch.tensor([1.0e40], dtype=torch.float64),
            torch.tensor([1.0], dtype=torch.float32),
        )
    with pytest.raises(RuntimeError, match="normalized acceleration exceeds"):
        L.lateral_world_wrench_from_total_mass(
            torch.tensor([2.0000001], dtype=torch.float64),
            torch.tensor([1.0], dtype=torch.float64),
        )
    with pytest.raises(RuntimeError, match="mass multiplication"):
        L.lateral_world_wrench_from_total_mass(
            torch.tensor([2.0], dtype=torch.float64),
            torch.tensor([3.0e38], dtype=torch.float32),
        )
    with pytest.raises(RuntimeError, match="WORLD-Y force exceeds"):
        L.lateral_world_wrench_from_total_mass(
            torch.tensor([2.0], dtype=torch.float64),
            torch.tensor([100.01], dtype=torch.float32),
        )

    force, _ = L.lateral_world_wrench_from_total_mass(
        torch.tensor([2.0], dtype=torch.float64),
        torch.tensor([100.0], dtype=torch.float32),
    )
    assert force[0, 0, 1] == 200.0


def test_typed_application_ledger_binds_mass_world_force_and_transform_identity():
    scheduler = L.LateralPulseScheduler(
        2,
        _cfg(policy_dt_s=0.04, opportunity_interval_steps=1, pulse_duration_steps=1),
        require_application_ack=True,
    )
    result = scheduler.step(step_token=0, **_inputs(2, 0))
    mass = torch.tensor([40.0, 55.0], dtype=torch.float32)
    adapter = _RecordingAdapter()
    ledger = _dispatch(scheduler, result, adapter, mass=mass)
    expected_accel = result.normalized_accel_y_mps2.to(dtype=torch.float32)
    expected_force_y = mass * expected_accel
    assert ledger.world_to_backend_transform_identity_sha256 == "a" * 64
    assert ledger.hard_safety_identity_sha256 == L.lateral_hard_safety_identity_sha256()
    assert torch.equal(ledger.actual_total_mass_kg, mass)
    assert torch.equal(ledger.commanded_normalized_accel_y_mps2, expected_accel)
    assert torch.equal(ledger.commanded_world_force_y_N, expected_force_y)
    assert torch.equal(
        ledger.commanded_world_impulse_y_Ns,
        expected_force_y * scheduler.cfg.policy_dt_s,
    )
    assert torch.equal(ledger.applied_force_mask, result.active_force_mask)
    assert torch.equal(
        ledger.commanded_world_force_y_N / ledger.actual_total_mass_kg,
        ledger.commanded_normalized_accel_y_mps2,
    )

    with pytest.raises(RuntimeError, match="same-step dispatch changed actual_total_mass_kg"):
        _dispatch(scheduler, result, adapter, mass=mass + 1.0)
    assert len(adapter.calls) == 1


def test_receipt_cannot_relabel_actual_mass_world_force_or_transform_identity():
    scheduler = L.LateralPulseScheduler(
        1,
        _cfg(policy_dt_s=0.04, opportunity_interval_steps=1, pulse_duration_steps=1),
    )
    result = scheduler.step(step_token=0, **_inputs(1, 0))

    def wrong_mass(receipt):
        return replace(receipt, actual_total_mass_kg=receipt.actual_total_mass_kg + 1.0)

    with pytest.raises(RuntimeError, match="actual_total_mass_kg does not match"):
        _dispatch(scheduler, result, _RecordingAdapter(receipt_override=wrong_mass))

    scheduler = L.LateralPulseScheduler(
        1,
        _cfg(policy_dt_s=0.04, opportunity_interval_steps=1, pulse_duration_steps=1),
    )
    result = scheduler.step(step_token=0, **_inputs(1, 0))

    def wrong_force(receipt):
        changed = receipt.commanded_force_w.clone()
        changed[:, 0, 1].add_(1.0)
        return replace(receipt, commanded_force_w=changed)

    with pytest.raises(RuntimeError, match="commanded_force_w does not match"):
        _dispatch(scheduler, result, _RecordingAdapter(receipt_override=wrong_force))

    scheduler = L.LateralPulseScheduler(
        1,
        _cfg(policy_dt_s=0.04, opportunity_interval_steps=1, pulse_duration_steps=1),
    )
    result = scheduler.step(step_token=0, **_inputs(1, 0))

    def wrong_transform(receipt):
        return replace(receipt, world_to_backend_transform_identity_sha256="b" * 64)

    with pytest.raises(RuntimeError, match="transform identity"):
        _dispatch(scheduler, result, _RecordingAdapter(receipt_override=wrong_transform))

    scheduler = L.LateralPulseScheduler(
        1,
        _cfg(policy_dt_s=0.04, opportunity_interval_steps=1, pulse_duration_steps=1),
    )
    result = scheduler.step(step_token=0, **_inputs(1, 0))

    class _MutatingAdapter(_RecordingAdapter):
        def overwrite_world_wrench_at_body_com(
            self, *, step_token, total_mass_kg, force_w, torque_w
        ):
            total_mass_kg.add_(1.0)
            force_w[:, 0, 1].add_(1.0)
            receipt = super().overwrite_world_wrench_at_body_com(
                step_token=step_token,
                total_mass_kg=total_mass_kg,
                force_w=force_w,
                torque_w=torque_w,
            )
            return replace(receipt, zero_torque=True)

    with pytest.raises(RuntimeError, match="actual_total_mass_kg does not match"):
        _dispatch(scheduler, result, _MutatingAdapter())


def test_mutating_or_raising_adapter_cannot_change_any_caller_owned_tensor_bits():
    class _MutateThenRejectAdapter(_RecordingAdapter):
        def overwrite_world_wrench_at_body_com(
            self, *, step_token, total_mass_kg, force_w, torque_w
        ):
            total_mass_kg.fill_(99.0)
            force_w.fill_(17.0)
            torque_w.fill_(-23.0)
            receipt = super().overwrite_world_wrench_at_body_com(
                step_token=step_token,
                total_mass_kg=total_mass_kg,
                force_w=force_w,
                torque_w=torque_w,
            )
            return replace(receipt, zero_torque=True)

    scheduler = L.LateralPulseScheduler(
        2,
        _cfg(policy_dt_s=0.04, opportunity_interval_steps=1, pulse_duration_steps=1),
    )
    result = scheduler.step(step_token=0, **_inputs(2, 0))
    mass = torch.tensor([40.0, 55.0], dtype=torch.float32)
    mass_bits = _tensor_bits(mass)
    result_bits = _tensor_field_bits(result)
    with pytest.raises(RuntimeError, match="actual_total_mass_kg does not match"):
        _dispatch(scheduler, result, _MutateThenRejectAdapter(), mass=mass)
    assert torch.equal(_tensor_bits(mass), mass_bits)
    _assert_tensor_field_bits_unchanged(result, result_bits)

    class _MutateThenRaiseAdapter(_RecordingAdapter):
        def overwrite_world_wrench_at_body_com(
            self, *, step_token, total_mass_kg, force_w, torque_w
        ):
            total_mass_kg.zero_()
            force_w.fill_(float("inf"))
            torque_w.fill_(float("nan"))
            raise RuntimeError("adversarial adapter exception")

    scheduler = L.LateralPulseScheduler(
        2,
        _cfg(policy_dt_s=0.04, opportunity_interval_steps=1, pulse_duration_steps=1),
    )
    result = scheduler.step(step_token=0, **_inputs(2, 0))
    mass = torch.tensor([40.0, 55.0], dtype=torch.float32)
    mass_bits = _tensor_bits(mass)
    result_bits = _tensor_field_bits(result)
    with pytest.raises(RuntimeError, match="adversarial adapter exception"):
        _dispatch(scheduler, result, _MutateThenRaiseAdapter(), mass=mass)
    assert torch.equal(_tensor_bits(mass), mass_bits)
    _assert_tensor_field_bits_unchanged(result, result_bits)
    # A rejected call cannot poison the same-step retry.
    ledger = _dispatch(scheduler, result, _RecordingAdapter(), mass=mass)
    assert torch.equal(ledger.actual_total_mass_kg, mass)


def test_public_ledger_mutation_never_reaches_private_cache_or_duplicate_return():
    scheduler = L.LateralPulseScheduler(
        2,
        _cfg(policy_dt_s=0.04, opportunity_interval_steps=1, pulse_duration_steps=1),
        require_application_ack=True,
    )
    result = scheduler.step(step_token=0, **_inputs(2, 0))
    mass = torch.tensor([40.0, 55.0], dtype=torch.float32)
    adapter = _RecordingAdapter()
    first = _dispatch(scheduler, result, adapter, mass=mass)
    pristine = _tensor_field_bits(first)

    for value in vars(first).values():
        if isinstance(value, torch.Tensor):
            if value.dtype == torch.bool:
                value.logical_not_()
            else:
                value.fill_(123.0)

    cached_public = scheduler.cached_application_ledger(0)
    assert cached_public is not None
    _assert_tensor_field_bits_unchanged(cached_public, pristine)
    for value in vars(cached_public).values():
        if isinstance(value, torch.Tensor):
            if value.dtype == torch.bool:
                value.logical_not_()
            else:
                value.zero_()

    duplicate = _dispatch(scheduler, result, adapter, mass=mass)
    _assert_tensor_field_bits_unchanged(duplicate, pristine)
    assert len(adapter.calls) == 1
    for name, bits in pristine.items():
        first_tensor = getattr(first, name)
        duplicate_tensor = getattr(duplicate, name)
        assert first_tensor.data_ptr() != duplicate_tensor.data_ptr(), name
        assert torch.equal(_tensor_bits(duplicate_tensor), bits), name


def test_adapter_seam_writes_zero_after_pulse_and_accounts_once():
    scheduler = L.LateralPulseScheduler(
        4,
        _cfg(policy_dt_s=0.04, opportunity_interval_steps=1, pulse_duration_steps=1),
        require_application_ack=True,
    )
    adapter = _RecordingAdapter()
    first = scheduler.step(step_token=0, **_inputs(4, 0))
    ledger = _dispatch(scheduler, first, adapter)
    duplicate = _dispatch(scheduler, first, adapter)
    assert duplicate is not ledger
    assert torch.equal(duplicate.actual_total_mass_kg, ledger.actual_total_mass_kg)
    assert len(adapter.calls) == 1
    second = scheduler.step(
        step_token=1,
        **_inputs(4, 1, eligible=False, strike=False, safe=0),
    )
    _dispatch(scheduler, second, adapter)
    assert len(adapter.calls) == 2
    assert torch.any(adapter.calls[0][1] != 0.0)
    assert torch.all(adapter.calls[1][1] == 0.0)
    assert torch.all(adapter.calls[1][2] == 0.0)
    counters = scheduler.consume_counters()
    assert counters["lateral_perturbation_wrench_write_step_count"].item() == 2
    assert counters["lateral_perturbation_applied_pulse_count"].item() == 4
    assert counters["lateral_perturbation_applied_force_env_step_count"].item() == 4


def test_adapter_seam_rejects_stale_force_contract_or_false_receipt():
    scheduler = L.LateralPulseScheduler(
        1,
        _cfg(policy_dt_s=0.04, opportunity_interval_steps=1, pulse_duration_steps=1),
    )
    result = scheduler.step(step_token=0, **_inputs(1, 0))
    bad_adapter = _RecordingAdapter()
    bad_adapter.inactive_zero_overwrite = False
    with pytest.raises(RuntimeError, match="not runtime-safe"):
        _dispatch(scheduler, result, bad_adapter)

    def wrong_mask(receipt):
        return L.LateralWrenchWriteReceipt(
            step_token=receipt.step_token,
            body_name=receipt.body_name,
            input_force_frame=receipt.input_force_frame,
            application_point=receipt.application_point,
            full_batch_overwrite=True,
            inactive_zero_overwrite=True,
            zero_torque=True,
            world_to_backend_transform_identity_sha256=(
                receipt.world_to_backend_transform_identity_sha256
            ),
            actual_total_mass_kg=receipt.actual_total_mass_kg,
            commanded_force_w=receipt.commanded_force_w,
            commanded_torque_w=receipt.commanded_torque_w,
            applied_force_mask=~receipt.applied_force_mask,
        )

    false_receipt = _RecordingAdapter(receipt_override=wrong_mask)
    with pytest.raises(RuntimeError, match="applied_force_mask"):
        _dispatch(scheduler, result, false_receipt)
    counters = scheduler.consume_counters()
    assert counters["lateral_perturbation_applied_pulse_count"].item() == 0
    assert counters["lateral_perturbation_wrench_write_step_count"].item() == 0
