"""Torch/CPU parity gates for the N=1 counter-rally objective."""

from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]
MDP = (
    ROOT
    / "hope_training/whole_body_tracking/source/whole_body_tracking"
    / "whole_body_tracking/tasks/tracking/mdp"
)


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, MDP / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CPU = _load("_counter_rally_cpu_parity", "counter_rally.py")
TORCH = _load("_counter_rally_torch_parity", "counter_rally_torch.py")


def _binding(profile=None, physics=None):
    profile = profile or CPU.CounterRallyObjectiveProfile()
    physics = physics or CPU.VenueBallPhysics()
    return TORCH.CounterRallyTorchBinding.from_mappings(
        objective_profile=profile.to_mapping(),
        venue_physics=physics.to_mapping(),
        expected_objective_profile_sha256=profile.sha256,
        expected_venue_physics_sha256=physics.sha256,
    )


DEFAULT_BINDING = _binding()


STATES = (
    ((0.8, 0.0, 1.0), (5.0, 0.0, 2.0), (0.0, 0.0, 0.0)),
    ((0.8, 0.0, 0.95), (7.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ((0.8, 0.0, 0.85), (4.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ((0.8, 0.0, 1.0), (5.0, 3.0, 2.0), (0.0, 0.0, 0.0)),
    ((0.8, 0.0, 1.0), (2.5, 0.0, 3.0), (0.0, 0.0, 0.0)),
)


def _torch_rollout(
    dt_s,
    order=range(len(STATES)),
    max_time_s=2.0,
    binding=DEFAULT_BINDING,
):
    rows = [STATES[index] for index in order]
    return TORCH.rollout_counter_rally_torch(
        torch.tensor([row[0] for row in rows], dtype=torch.float64),
        torch.tensor([row[1] for row in rows], dtype=torch.float64),
        torch.tensor([row[2] for row in rows], dtype=torch.float64),
        binding=binding,
        dt_s=dt_s,
        max_time_s=max_time_s,
    )


def _cpu_rollout(index, dt_s, max_time_s=2.0):
    position, velocity, spin = STATES[index]
    return CPU.rollout_counter_rally_eager(
        position_after_paddle_env_m=position,
        velocity_after_paddle_mps=velocity,
        spin_after_paddle_radps=spin,
        profile=CPU.CounterRallyObjectiveProfile(),
        physics=CPU.VenueBallPhysics(),
        dt_s=dt_s,
        max_time_s=max_time_s,
    )


def _task_for_landing(profile, landing_x):
    return CPU.derive_counter_rally_task(
        base_goal_env_xy_m=(0.55, 0.10),
        base_yaw_env_rad=0.0,
        contact_offset_b_yaw_m=(0.25, -0.10, 1.0),
        incoming_direction_b_yaw=(-1.0, 0.0),
        incoming_ball_speed_at_contact_mps=3.0,
        landing_depth_env_x_m=landing_x,
        profile=profile,
    )


def _admissible_task_landing_x(profile, landing_x):
    net_x = profile.table_near_x_env_m + 0.5 * profile.table_length_m
    table_hi = (
        profile.opponent_baseline_x_env_m
        - profile.table_edge_margin_m
    )
    if net_x < landing_x <= table_hi:
        return landing_x
    return 2.5


def _torch_reward(
    fused,
    aim,
    profile,
    *,
    task_profile_shas=None,
    paddle_contact_valid=None,
):
    if task_profile_shas is None:
        task_profile_shas = (profile.sha256,) * aim.shape[0]
    if paddle_contact_valid is None:
        paddle_contact_valid = torch.ones(
            aim.shape[0], dtype=torch.bool
        )
    return TORCH.counter_rally_reward_raw_torch(
        binding=_binding(profile),
        landing_aim_env_xy_m=aim,
        return_direction_env_xy=torch.tensor(
            ((1.0, 0.0),) * aim.shape[0], dtype=aim.dtype
        ),
        target_baseline_speed_mps=torch.full(
            (aim.shape[0],), 3.0, dtype=aim.dtype
        ),
        paddle_contact_valid=paddle_contact_valid,
        task_objective_profile_sha256=task_profile_shas,
        outcome=fused,
    )


@pytest.mark.parametrize("dt_s", (0.001, 0.0005))
def test_torch_fused_rows_match_independent_cpu_oracle(dt_s):
    fused = _torch_rollout(dt_s)
    for index in range(len(STATES)):
        scalar = _cpu_rollout(index, dt_s)
        expected_reason = (
            "none"
            if scalar.rejection_reason is None
            else scalar.rejection_reason
        )
        assert fused.rejection_reasons[index] == expected_reason
        assert bool(fused.net_crossed[index]) == scalar.net_crossed
        assert bool(fused.net_clear[index]) == scalar.net_clear
        assert bool(fused.first_landing_valid[index]) == (
            scalar.first_landing_valid
        )
        assert int(fused.table_bounce_count[index]) == (
            scalar.table_bounce_count
        )
        assert bool(fused.opponent_baseline_crossed[index]) == (
            scalar.opponent_baseline_crossed
        )
        if scalar.first_landing_env_xy_m is not None:
            assert fused.first_landing_env_xy_m[index].tolist() == (
                pytest.approx(scalar.first_landing_env_xy_m, abs=3.0e-10)
            )
        if scalar.baseline_velocity_mps is not None:
            assert fused.baseline_velocity_mps[index].tolist() == (
                pytest.approx(scalar.baseline_velocity_mps, abs=3.0e-10)
            )
            assert float(fused.baseline_time_s[index]) == pytest.approx(
                scalar.baseline_time_s, abs=3.0e-10
            )


def test_fused_batch_is_row_separable_and_permutation_invariant():
    original = _torch_rollout(0.001)
    order = (4, 2, 0, 3, 1)
    permuted = _torch_rollout(0.001, order=order)
    inverse = {source: target for target, source in enumerate(order)}
    for source in range(len(STATES)):
        target = inverse[source]
        assert original.rejection_reasons[source] == (
            permuted.rejection_reasons[target]
        )
        assert torch.equal(
            original.table_bounce_count[source],
            permuted.table_bounce_count[target],
        )
        assert torch.allclose(
            original.first_landing_env_xy_m[source],
            permuted.first_landing_env_xy_m[target],
            atol=0.0,
            rtol=0.0,
            equal_nan=True,
        )
        assert torch.allclose(
            original.baseline_velocity_mps[source],
            permuted.baseline_velocity_mps[target],
            atol=0.0,
            rtol=0.0,
            equal_nan=True,
        )


def test_torch_reward_has_the_same_staged_eligibility_gates_as_cpu():
    fused = _torch_rollout(0.001)
    aim = fused.first_landing_env_xy_m.clone()
    aim[~torch.isfinite(aim)] = torch.tensor((2.5, 0.0), dtype=aim.dtype)
    raw = _torch_reward(
        fused, aim, CPU.CounterRallyObjectiveProfile()
    )
    assert raw.shape == (len(STATES), 5)
    assert torch.all((raw >= 0.0) & (raw <= 1.0))
    # Net miss, own-half and outside-table rows receive no shaping at all.
    assert torch.equal(raw[1:4], torch.zeros_like(raw[1:4]))
    # Second bounce reached a legal first landing, but cannot collect either
    # baseline-qualified direction or speed shaping.
    assert raw[4, 0].item() == 1.0
    assert raw[4, 1].item() == pytest.approx(1.0)
    assert raw[4, 2].item() == 0.0
    assert raw[4, 3].item() == 0.0
    assert raw[4, 4].item() == pytest.approx(0.65)


def test_missing_actual_paddle_contact_cannot_harvest_dummy_rollout_reward():
    profile = CPU.CounterRallyObjectiveProfile()
    fused = _torch_rollout(0.001)
    aim = fused.first_landing_env_xy_m.clone()
    aim[~torch.isfinite(aim)] = torch.tensor(
        (2.5, 0.0), dtype=aim.dtype
    )
    contact = torch.ones(len(STATES), dtype=torch.bool)
    contact[0] = False
    raw = _torch_reward(
        fused,
        aim,
        profile,
        paddle_contact_valid=contact,
    )
    assert torch.equal(raw[0], torch.zeros_like(raw[0]))


def test_torch_outcome_gate_api_exposes_stages_after_identity_check():
    profile = CPU.CounterRallyObjectiveProfile()
    fused = _torch_rollout(0.001)
    aim = fused.first_landing_env_xy_m.clone()
    aim[~torch.isfinite(aim)] = torch.tensor(
        (2.5, 0.0), dtype=aim.dtype
    )
    gates = TORCH.counter_rally_outcome_gates_torch(
        outcome=fused,
        binding=_binding(profile),
        landing_aim_env_xy_m=aim,
        return_direction_env_xy=torch.tensor(
            ((1.0, 0.0),) * len(STATES), dtype=aim.dtype
        ),
        target_baseline_speed_mps=torch.full(
            (len(STATES),), 3.0, dtype=aim.dtype
        ),
        paddle_contact_valid=torch.ones(
            len(STATES), dtype=torch.bool
        ),
        task_objective_profile_sha256=(profile.sha256,) * len(STATES),
    )
    assert gates.objective_profile_sha256 == profile.sha256
    assert gates.legal_first_landing.tolist() == [
        True,
        False,
        False,
        False,
        True,
    ]
    assert gates.baseline_valid.tolist() == [
        True,
        False,
        False,
        False,
        False,
    ]
    assert gates.primary_reasons == (
        "baseline_speed_miss",
        "net_not_clear",
        "net_not_crossed",
        "first_landing_invalid",
        "opponent_baseline_not_crossed",
    )
    assert not bool(gates.accepted.any())


@pytest.mark.parametrize("dt_s", (0.001, 0.0005))
def test_torch_full_assessment_matches_cpu_primary_reason_and_metrics(dt_s):
    profile = CPU.CounterRallyObjectiveProfile()
    fused = _torch_rollout(dt_s)
    aim = fused.first_landing_env_xy_m.clone()
    aim[~torch.isfinite(aim)] = torch.tensor(
        (2.5, 0.0), dtype=aim.dtype
    )
    task_aim = aim.clone()
    for index in range(len(STATES)):
        task_aim[index, 0] = _admissible_task_landing_x(
            profile, float(task_aim[index, 0])
        )
        if task_aim[index, 0] == 2.5:
            task_aim[index, 1] = 0.0
    gates = TORCH.counter_rally_outcome_gates_torch(
        outcome=fused,
        binding=_binding(profile),
        landing_aim_env_xy_m=task_aim,
        return_direction_env_xy=torch.tensor(
            ((1.0, 0.0),) * len(STATES), dtype=aim.dtype
        ),
        target_baseline_speed_mps=torch.full(
            (len(STATES),), 3.0, dtype=aim.dtype
        ),
        paddle_contact_valid=torch.ones(
            len(STATES), dtype=torch.bool
        ),
        task_objective_profile_sha256=(profile.sha256,) * len(STATES),
    )
    for index in range(len(STATES)):
        task = _task_for_landing(
            profile, float(task_aim[index, 0])
        )
        assessment = CPU.assess_counter_rally_outcome(
            task=task,
            outcome=_cpu_rollout(index, dt_s),
            profile=profile,
        )
        primary = (
            assessment.reasons[0]
            if assessment.reasons
            else "accepted"
        )
        assert gates.primary_reasons[index] == primary
        assert bool(gates.accepted[index]) == assessment.accepted
        metric_pairs = (
            (
                gates.landing_error_m[index],
                assessment.landing_error_m,
            ),
            (
                gates.reverse_direction_error_deg[index],
                assessment.reverse_direction_error_deg,
            ),
            (
                gates.baseline_direction_error_deg[index],
                assessment.baseline_direction_error_deg,
            ),
            (
                gates.baseline_speed_mps[index],
                assessment.baseline_speed_mps,
            ),
            (
                gates.baseline_speed_error_mps[index],
                assessment.baseline_speed_error_mps,
            ),
        )
        for actual, expected in metric_pairs:
            if expected is None:
                assert torch.isnan(actual)
            else:
                assert float(actual) == pytest.approx(
                    expected, abs=1.0e-9
                )


@pytest.mark.parametrize(
    "profile",
    (
        CPU.CounterRallyObjectiveProfile(),
        CPU.CounterRallyObjectiveProfile(
            reward_legal_fraction=0.20,
            reward_landing_fraction=0.30,
            reward_reverse_fraction=0.10,
            reward_speed_fraction=0.40,
        ),
    ),
)
@pytest.mark.parametrize("dt_s", (0.001, 0.0005))
def test_torch_reward_matches_cpu_terms_and_explicit_profile_weights(
    profile,
    dt_s,
):
    fused = _torch_rollout(dt_s)
    aim = fused.first_landing_env_xy_m.clone()
    aim[~torch.isfinite(aim)] = torch.tensor((2.5, 0.0), dtype=aim.dtype)
    actual = _torch_reward(fused, aim, profile)
    for index in range(len(STATES)):
        task = _task_for_landing(
            profile,
            _admissible_task_landing_x(
                profile, float(aim[index, 0])
            ),
        )
        expected = CPU.counter_rally_reward_raw(
            task=task,
            outcome=_cpu_rollout(index, dt_s),
            profile=profile,
        )
        assert actual[index].tolist() == pytest.approx(
            (
                expected["legal"],
                expected["landing"],
                expected["reverse"],
                expected["speed"],
                expected["total"],
            ),
            abs=1.0e-9,
        )


def test_torch_and_cpu_reward_profile_sha_mismatch_are_identity_errors():
    profile = CPU.CounterRallyObjectiveProfile()
    fused = _torch_rollout(0.001)
    aim = fused.first_landing_env_xy_m.clone()
    aim[~torch.isfinite(aim)] = torch.tensor((2.5, 0.0), dtype=aim.dtype)
    task_profile_shas = ("0" * 64,) + (profile.sha256,) * (
        len(STATES) - 1
    )
    with pytest.raises(
        TORCH.CounterRallyTorchIdentityError,
        match="rows 0",
    ):
        _torch_reward(
            fused,
            aim,
            profile,
            task_profile_shas=task_profile_shas,
        )

    task = replace(
        _task_for_landing(profile, float(aim[0, 0])),
        objective_profile_sha256="0" * 64,
    )
    with pytest.raises(CPU.CounterRallyIdentityError):
        CPU.counter_rally_reward_raw(
            task=task,
            outcome=_cpu_rollout(0, 0.001),
            profile=profile,
        )


@pytest.mark.parametrize(
    "task_profile_shas",
    (
        (),
        "0" * 64,
        ("A" * 64,) * len(STATES),
        ("abc",) * len(STATES),
    ),
)
def test_torch_objective_identity_shape_and_encoding_are_strict(
    task_profile_shas,
):
    profile = CPU.CounterRallyObjectiveProfile()
    fused = _torch_rollout(0.001)
    aim = fused.first_landing_env_xy_m.clone()
    aim[~torch.isfinite(aim)] = torch.tensor((2.5, 0.0), dtype=aim.dtype)
    with pytest.raises(ValueError):
        _torch_reward(
            fused,
            aim,
            profile,
            task_profile_shas=task_profile_shas,
        )


@pytest.mark.parametrize(
    "weights",
    (
        (-0.1, 0.1, 0.1, 0.9),
        (float("nan"), 0.0, 0.0, 1.0),
        (0.2, 0.2, 0.2, 0.2),
    ),
)
def test_bound_torch_contract_rejects_invalid_reward_fractions(weights):
    profile = CPU.CounterRallyObjectiveProfile()
    mapping = dict(profile.to_mapping())
    (
        mapping["reward_legal_fraction"],
        mapping["reward_landing_fraction"],
        mapping["reward_reverse_fraction"],
        mapping["reward_speed_fraction"],
    ) = weights
    with pytest.raises(ValueError):
        TORCH.CounterRallyTorchBinding.from_mappings(
            objective_profile=mapping,
            venue_physics=CPU.VenueBallPhysics().to_mapping(),
            expected_objective_profile_sha256=(
                CPU._canonical_sha256(mapping)
            ),
            expected_venue_physics_sha256=(
                CPU.VenueBallPhysics().sha256
            ),
        )


def test_bound_torch_contract_rejects_same_sha_scalar_or_physics_tamper():
    profile = CPU.CounterRallyObjectiveProfile()
    physics = CPU.VenueBallPhysics()
    profile_tamper = dict(profile.to_mapping())
    profile_tamper["table_length_m"] = 5.0
    with pytest.raises(
        TORCH.CounterRallyTorchIdentityError,
        match="objective_profile_sha256_mismatch",
    ):
        TORCH.CounterRallyTorchBinding.from_mappings(
            objective_profile=profile_tamper,
            venue_physics=physics.to_mapping(),
            expected_objective_profile_sha256=profile.sha256,
            expected_venue_physics_sha256=physics.sha256,
        )

    physics_tamper = dict(physics.to_mapping())
    physics_tamper["table_e_eff"] = 0.5
    with pytest.raises(
        TORCH.CounterRallyTorchIdentityError,
        match="venue_physics_sha256_mismatch",
    ):
        TORCH.CounterRallyTorchBinding.from_mappings(
            objective_profile=profile.to_mapping(),
            venue_physics=physics_tamper,
            expected_objective_profile_sha256=profile.sha256,
            expected_venue_physics_sha256=physics.sha256,
        )

    with pytest.raises(
        TORCH.CounterRallyTorchIdentityError,
        match="objective_profile_scalar_mismatch",
    ):
        replace(DEFAULT_BINDING, landing_tolerance_m=0.5)


def test_fitted_table_impulse_matches_cpu_formula():
    velocity = torch.tensor(
        ((4.0, 1.0, -2.0), (3.0, -0.5, -1.5)),
        dtype=torch.float64,
    )
    spin = torch.tensor(
        ((0.0, 0.0, 0.0), (2.0, -3.0, 1.0)),
        dtype=torch.float64,
    )
    torch_v, torch_w, descending = TORCH.fitted_table_impulse_torch(
        velocity,
        spin,
        ball_radius_m=0.020,
        ball_mass_kg=0.0034,
        ball_inertia_coeff=2.0 / 3.0,
        table_e_eff=0.9215,
        table_a_t=0.369,
        table_b_t=0.0,
        table_mu_safety=2.0,
    )
    assert bool(descending.all())
    for index in range(2):
        cpu_v, cpu_w = CPU.fitted_table_impulse(
            velocity_before_mps=velocity[index].tolist(),
            spin_before_radps=spin[index].tolist(),
            physics=CPU.VenueBallPhysics(),
        )
        assert torch_v[index].tolist() == pytest.approx(cpu_v, abs=1.0e-12)
        assert torch_w[index].tolist() == pytest.approx(cpu_w, abs=1.0e-12)
