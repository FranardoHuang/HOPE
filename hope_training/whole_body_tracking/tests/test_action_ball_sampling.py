from copy import deepcopy
from dataclasses import replace
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import sys

import pytest


PATH = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
    / "action_ball_sampling.py"
)
SPEC = importlib.util.spec_from_file_location("action_ball_sampling", PATH)
S = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = S
SPEC.loader.exec_module(S)


def _profile(action_uid=101, **overrides):
    values = dict(
        action_uid=action_uid,
        contact_offset_center_b_yaw_m=(0.55, 0.12, 0.82),
        contact_offset_std_lower_initial_m=(0.005, 0.01, 0.01),
        contact_offset_std_lower_max_m=(0.02, 0.12, 0.16),
        contact_offset_std_upper_initial_m=(0.004, 0.02, 0.01),
        contact_offset_std_upper_max_m=(0.02, 0.10, 0.15),
        contact_offset_min_b_yaw_m=(0.45, -0.20, 0.55),
        contact_offset_max_b_yaw_m=(0.65, 0.35, 1.10),
        time_to_contact_center_s=1.20,
        time_to_contact_std_lower_initial_s=0.01,
        time_to_contact_std_lower_max_s=0.15,
        time_to_contact_std_upper_initial_s=0.02,
        time_to_contact_std_upper_max_s=0.30,
        time_to_contact_min_s=1.05,
        time_to_contact_max_s=1.60,
        incoming_direction_center_b_yaw=(-1.0, 0.0, 0.0),
        incoming_direction_tangent_u_b_yaw=(0.0, 1.0, 0.0),
        incoming_direction_tangent_v_b_yaw=(0.0, 0.0, -1.0),
        incoming_direction_tangent_u_neg_initial_deg=0.5,
        incoming_direction_tangent_u_neg_max_deg=8.0,
        incoming_direction_tangent_u_pos_initial_deg=0.6,
        incoming_direction_tangent_u_pos_max_deg=7.0,
        incoming_direction_tangent_v_neg_initial_deg=0.7,
        incoming_direction_tangent_v_neg_max_deg=6.0,
        incoming_direction_tangent_v_pos_initial_deg=0.8,
        incoming_direction_tangent_v_pos_max_deg=5.0,
        incoming_inbound_axis_b_yaw=(-1.0, 0.0, 0.0),
        incoming_inbound_min_cosine=0.8,
        incoming_speed_center_mps=4.0,
        incoming_speed_std_lower_initial_mps=0.05,
        incoming_speed_std_lower_max_mps=1.2,
        incoming_speed_std_upper_initial_mps=0.06,
        incoming_speed_std_upper_max_mps=1.0,
        incoming_speed_min_mps=1.6,
        incoming_speed_max_mps=7.0,
        spin_direction_center_b_yaw=(0.0, 1.0, 0.0),
        spin_direction_tangent_u_b_yaw=(0.0, 0.0, 1.0),
        spin_direction_tangent_v_b_yaw=(1.0, 0.0, 0.0),
        spin_direction_tangent_u_neg_initial_deg=0.0,
        spin_direction_tangent_u_neg_max_deg=35.0,
        spin_direction_tangent_u_pos_initial_deg=0.0,
        spin_direction_tangent_u_pos_max_deg=30.0,
        spin_direction_tangent_v_neg_initial_deg=0.0,
        spin_direction_tangent_v_neg_max_deg=25.0,
        spin_direction_tangent_v_pos_initial_deg=0.0,
        spin_direction_tangent_v_pos_max_deg=20.0,
        spin_magnitude_center_radps=15.0,
        spin_magnitude_std_lower_initial_radps=0.2,
        spin_magnitude_std_lower_max_radps=8.0,
        spin_magnitude_std_upper_initial_radps=0.3,
        spin_magnitude_std_upper_max_radps=9.0,
        spin_magnitude_min_radps=0.0,
        spin_magnitude_max_radps=40.0,
        base_spawn_center_w_m=(-0.10, 0.05, 0.0),
        base_spawn_std_lower_initial_m=(0.005, 0.005, 0.0),
        base_spawn_std_lower_max_m=(0.15, 0.20, 0.0),
        base_spawn_std_upper_initial_m=(0.006, 0.007, 0.0),
        base_spawn_std_upper_max_m=(0.12, 0.18, 0.0),
        base_spawn_min_w_m=(-0.50, -0.40, 0.0),
        base_spawn_max_w_m=(0.30, 0.50, 0.0),
        base_travel_center_b_yaw_m=(0.20, -0.05, 0.0),
        base_travel_std_lower_initial_m=(0.01, 0.01, 0.0),
        base_travel_std_lower_max_m=(0.25, 0.25, 0.0),
        base_travel_std_upper_initial_m=(0.02, 0.01, 0.0),
        base_travel_std_upper_max_m=(0.25, 0.25, 0.0),
        base_travel_min_b_yaw_m=(-0.40, -0.40, 0.0),
        base_travel_max_b_yaw_m=(0.50, 0.40, 0.0),
        landing_aim_center_w_xy_m=(2.55, 0.0),
        landing_aim_std_lower_initial_m=(0.01, 0.01),
        landing_aim_std_lower_max_m=(0.25, 0.35),
        landing_aim_std_upper_initial_m=(0.02, 0.01),
        landing_aim_std_upper_max_m=(0.20, 0.30),
        landing_aim_min_w_xy_m=(2.20, -0.55),
        landing_aim_max_w_xy_m=(2.90, 0.55),
        reference_t_hit_s=0.80,
        reference_t_cycle_s=1.60,
        reference_racket_site_speed_mps=6.0,
        reaction_margin_s=0.05,
        teacher_rate_min=0.80,
        teacher_rate_max=1.20,
        mobility_mode="no_move",
    )
    legacy_pairs = {
        "contact_offset_std_initial_m": (
            "contact_offset_std_lower_initial_m",
            "contact_offset_std_upper_initial_m",
        ),
        "contact_offset_std_max_m": (
            "contact_offset_std_lower_max_m",
            "contact_offset_std_upper_max_m",
        ),
        "incoming_speed_std_initial_mps": (
            "incoming_speed_std_lower_initial_mps",
            "incoming_speed_std_upper_initial_mps",
        ),
        "incoming_speed_std_max_mps": (
            "incoming_speed_std_lower_max_mps",
            "incoming_speed_std_upper_max_mps",
        ),
        "spin_magnitude_std_initial_radps": (
            "spin_magnitude_std_lower_initial_radps",
            "spin_magnitude_std_upper_initial_radps",
        ),
        "spin_magnitude_std_max_radps": (
            "spin_magnitude_std_lower_max_radps",
            "spin_magnitude_std_upper_max_radps",
        ),
        "base_spawn_std_initial_m": (
            "base_spawn_std_lower_initial_m",
            "base_spawn_std_upper_initial_m",
        ),
        "base_spawn_std_max_m": (
            "base_spawn_std_lower_max_m",
            "base_spawn_std_upper_max_m",
        ),
        "base_travel_std_initial_m": (
            "base_travel_std_lower_initial_m",
            "base_travel_std_upper_initial_m",
        ),
        "base_travel_std_max_m": (
            "base_travel_std_lower_max_m",
            "base_travel_std_upper_max_m",
        ),
        "landing_aim_std_initial_m": (
            "landing_aim_std_lower_initial_m",
            "landing_aim_std_upper_initial_m",
        ),
        "landing_aim_std_max_m": (
            "landing_aim_std_lower_max_m",
            "landing_aim_std_upper_max_m",
        ),
    }
    for legacy, destinations in legacy_pairs.items():
        if legacy in overrides:
            value = overrides.pop(legacy)
            for destination in destinations:
                overrides[destination] = value
    if "incoming_direction_cone_deg" in overrides:
        width = overrides.pop("incoming_direction_cone_deg") / math.sqrt(2.0)
        for side in ("u_neg", "u_pos", "v_neg", "v_pos"):
            overrides[
                f"incoming_direction_tangent_{side}_initial_deg"
            ] = width
            overrides[
                f"incoming_direction_tangent_{side}_max_deg"
            ] = width
    if "spin_direction_cone_initial_deg" in overrides:
        width = overrides.pop("spin_direction_cone_initial_deg")
        for side in ("u_neg", "u_pos", "v_neg", "v_pos"):
            overrides[
                f"spin_direction_tangent_{side}_initial_deg"
            ] = width / math.sqrt(2.0)
    if "spin_direction_cone_max_deg" in overrides:
        width = overrides.pop("spin_direction_cone_max_deg")
        for side in ("u_neg", "u_pos", "v_neg", "v_pos"):
            overrides[
                f"spin_direction_tangent_{side}_max_deg"
            ] = width / math.sqrt(2.0)
    values.update(overrides)
    return S.SamplingProfile(**values)


def _levels(**overrides):
    """Test-only convenience: expand old conceptual groups into v3 arms."""

    values = {}
    aliases = {
        "aim": (
            "landing_aim_x_lower",
            "landing_aim_x_upper",
            "landing_aim_y_lower",
            "landing_aim_y_upper",
        ),
        "position": (
            "contact_x_lower",
            "contact_x_upper",
            "contact_y_lower",
            "contact_y_upper",
            "contact_z_lower",
            "contact_z_upper",
        ),
        "speed": ("incoming_speed_lower", "incoming_speed_upper"),
        "spin_magnitude": (
            "spin_magnitude_lower",
            "spin_magnitude_upper",
        ),
        "spin_direction": (
            "spin_direction_u_neg",
            "spin_direction_u_pos",
            "spin_direction_v_neg",
            "spin_direction_v_pos",
        ),
        "base_spawn": (
            "base_spawn_x_lower",
            "base_spawn_x_upper",
            "base_spawn_y_lower",
            "base_spawn_y_upper",
        ),
        "base_travel": (
            "base_travel_x_lower",
            "base_travel_x_upper",
            "base_travel_y_lower",
            "base_travel_y_upper",
        ),
        "time_to_contact": (
            "time_to_contact_lower",
            "time_to_contact_upper",
        ),
    }
    for name, value in overrides.items():
        if name in aliases:
            for arm in aliases[name]:
                values[arm] = value
        else:
            values[name] = value
    return S.DomainLevels(**values)


def _norm(vector):
    return math.sqrt(sum(component * component for component in vector))


def _angle_deg(a, b):
    cosine = sum(x * y for x, y in zip(a, b)) / (_norm(a) * _norm(b))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _integrity(payload):
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _birth(
    sampler,
    *,
    uid=101,
    epoch=0,
    levels=None,
    yaw=0.0,
):
    levels = _levels() if levels is None else levels
    return sampler.reserve_birth(
        action_uid=uid,
        domain_epoch=epoch,
        levels=levels,
        base_yaw_rad=yaw,
    )


def _sample(
    sampler,
    birth,
    *,
    uid=101,
    epoch=0,
    levels=None,
    yaw=0.0,
):
    levels = _levels() if levels is None else levels
    return sampler.sample(
        birth=birth,
        action_uid=uid,
        domain_epoch=epoch,
        levels=levels,
        base_yaw_rad=yaw,
    )


def test_same_seed_birth_and_sample_are_bit_exact_and_task_is_unresolved():
    levels = _levels(
        aim=0.5,
        position=0.5,
        speed=0.75,
        spin_magnitude=1.0,
        spin_direction=0.25,
        base_spawn=0.5,
        base_travel=0.75,
    )
    left = S.ActionBallSampler([_profile()], seed=20260727)
    right = S.ActionBallSampler([_profile()], seed=20260727)
    left_birth = _birth(left, epoch=7, levels=levels, yaw=0.37)
    right_birth = _birth(right, epoch=7, levels=levels, yaw=0.37)
    assert left_birth == right_birth

    left_sample = _sample(
        left, left_birth, epoch=7, levels=levels, yaw=0.37
    )
    right_sample = _sample(
        right, right_birth, epoch=7, levels=levels, yaw=0.37
    )
    assert left_sample == right_sample
    assert left.state_dict() == right.state_dict()
    assert left.draw_count == S.DRAWS_PER_BIRTH + S.DRAWS_PER_SAMPLE
    receipt = left_sample.to_receipt()
    assert receipt["task"] is None
    assert receipt["birth_id"] == left_birth.birth_id
    assert receipt["domain_levels"]["landing_aim_x_lower"] == 0.5
    assert receipt["arm_catalog_sha256"] == S.ARM_CATALOG_SHA256
    assert receipt["ball"]["time_to_contact_s"] == (
        left_sample.time_to_contact_s
    )
    assert receipt["sample_index"] == 0
    assert receipt["solver_input"]["landing_aim_w_xy_m"] == list(
        left_sample.landing_aim_w_xy_m
    )
    assert (
        S._sha256_json(left_sample.identity_payload())
        == left_sample.sample_id
    )
    left_sample.verify_sample_id()
    with pytest.raises(ValueError, match="canonical identity"):
        replace(left_sample, sample_index=1).verify_sample_id()


def test_exact_resume_reproduces_future_wrap_samples_bit_for_bit():
    levels = _levels(
        aim=0.75,
        position=1.0,
        spin_magnitude=0.25,
        base_spawn=0.5,
    )
    profiles = [_profile(11), _profile(29)]
    original = S.ActionBallSampler(profiles, seed=44)
    birth = _birth(original, uid=11, epoch=4, levels=levels)
    _sample(original, birth, uid=11, epoch=4, levels=levels)
    saved = deepcopy(original.state_dict())
    expected = [
        _sample(original, birth, uid=11, epoch=4, levels=levels),
        _sample(original, birth, uid=11, epoch=4, levels=levels),
    ]

    restored = S.ActionBallSampler(profiles, seed=44)
    restored.load_state_dict(saved)
    actual = [
        _sample(restored, birth, uid=11, epoch=4, levels=levels),
        _sample(restored, birth, uid=11, epoch=4, levels=levels),
    ]
    assert actual == expected
    assert [sample.sample_index for sample in actual] == [1, 2]
    for sample in actual:
        sample.verify_sample_id()
    assert restored.state_dict() == original.state_dict()


def test_exact_issued_sample_assertion_accepts_object_and_receipts_purely():
    levels = _levels(
        aim=0.5,
        position=0.75,
        speed=1.0,
        spin_magnitude=0.5,
        spin_direction=0.25,
    )
    sampler = S.ActionBallSampler([_profile()], seed=2026)
    birth = _birth(sampler, epoch=5, levels=levels, yaw=0.1)
    sample = _sample(
        sampler, birth, epoch=5, levels=levels, yaw=0.1
    )
    before = deepcopy(sampler.state_dict())
    assert sampler.assert_issued_birth(birth) is None
    assert (
        sampler.assert_issued_birth(birth.to_identity_receipt())
        is None
    )
    assert sampler.assert_issued_sample(sample) is None
    assert (
        sampler.assert_issued_sample(sample.to_identity_receipt())
        is None
    )
    assert sampler.assert_emitted_sample(sample.to_receipt()) is None
    assert sampler.state_dict() == before

    split_brain_receipt = deepcopy(sample.to_receipt())
    split_brain_receipt["ball"]["contact_w_m"][0] += 0.1
    with pytest.raises(ValueError, match="disagrees"):
        sampler.assert_issued_sample(split_brain_receipt)
    assert sampler.state_dict() == before

    restored = S.ActionBallSampler([_profile()], seed=2026)
    restored.load_state_dict(before)
    restored_before = deepcopy(restored.state_dict())
    restored.assert_issued_sample(sample)
    assert restored.state_dict() == restored_before


def test_public_replay_is_pure_exact_and_mixed_birth_batch_restores():
    sampler = S.ActionBallSampler([_profile()], seed=2028)
    first_levels = _levels(
        position=0.25,
        time_to_contact_lower=0.50,
    )
    second_levels = _levels(
        position=0.75,
        time_to_contact_upper=1.0,
    )
    first_birth = _birth(sampler, epoch=4, levels=first_levels)
    first_sample = _sample(
        sampler, first_birth, epoch=4, levels=first_levels
    )
    second_birth = _birth(sampler, epoch=9, levels=second_levels)
    second_sample = _sample(
        sampler, second_birth, epoch=9, levels=second_levels
    )
    third_sample = _sample(
        sampler, first_birth, epoch=4, levels=first_levels
    )
    expected = (first_sample, second_sample, third_sample)
    before = deepcopy(sampler.state_dict())

    assert sampler.replay_issued_sample(first_birth, 0) == first_sample
    assert (
        sampler.replay_issued_sample(
            second_birth.to_identity_receipt(), 1
        )
        == second_sample
    )
    assert sampler.replay_issued_sample(first_birth, 2) == third_sample
    assert sampler.replay_issued_samples(
        (
            (first_birth, 0),
            (second_birth.to_identity_receipt(), 1),
            (first_birth, 2),
        )
    ) == expected
    assert [
        sample.time_to_contact_s for sample in expected
    ] == [
        sampler.replay_issued_sample(birth, index).time_to_contact_s
        for birth, index in (
            (first_birth, 0),
            (second_birth, 1),
            (first_birth, 2),
        )
    ]
    assert sampler.state_dict() == before

    restored = S.ActionBallSampler([_profile()], seed=2028)
    restored.load_state_dict(before)
    restored_before = deepcopy(restored.state_dict())
    assert restored.replay_issued_samples(
        (
            (first_birth, 0),
            (second_birth, 1),
            (first_birth, 2),
        )
    ) == expected
    assert restored.state_dict() == restored_before


def test_public_replay_rejects_cross_birth_index_and_forged_levels_purely():
    sampler = S.ActionBallSampler([_profile()], seed=2029)
    first_levels = _levels(position=0.2)
    second_levels = _levels(position=0.8)
    first_birth = _birth(sampler, epoch=1, levels=first_levels)
    second_birth = _birth(sampler, epoch=2, levels=second_levels)
    first_sample = _sample(
        sampler, first_birth, epoch=1, levels=first_levels
    )
    second_sample = _sample(
        sampler, second_birth, epoch=2, levels=second_levels
    )
    before = deepcopy(sampler.state_dict())

    with pytest.raises(ValueError, match="different episode birth"):
        sampler.replay_issued_sample(second_birth, first_sample.sample_index)
    with pytest.raises(ValueError, match="different episode birth"):
        sampler.replay_issued_sample(first_birth, second_sample.sample_index)
    with pytest.raises(ValueError, match="high-water"):
        sampler.replay_issued_sample(
            second_birth, sampler.sample_count_for(101)
        )
    with pytest.raises(ValueError, match="sample_index"):
        sampler.replay_issued_sample(second_birth, -1)

    forged_levels = _levels(position=0.9)
    forged = replace(
        first_birth,
        domain_levels=forged_levels,
        levels_sha256=forged_levels.sha256,
    )
    forged_identity = S._birth_identity_payload(
        sampler_contract_sha256=forged.sampler_contract_sha256,
        arm_catalog_sha256=forged.arm_catalog_sha256,
        action_uid=forged.action_uid,
        domain_epoch=forged.domain_epoch,
        levels_sha256=forged.levels_sha256,
        profile_sha256=forged.profile_sha256,
        birth_index=forged.birth_index,
        draw_start=forged.draw_start,
        draw_end=forged.draw_end,
        mobility_mode=forged.mobility_mode,
        base_yaw_rad=forged.base_yaw_rad,
        base_start_w_m=forged.base_start_w_m,
    )
    forged = replace(
        forged, birth_id=S._sha256_json(forged_identity)
    )
    with pytest.raises(ValueError, match="issued transcript"):
        sampler.replay_issued_sample(forged, first_sample.sample_index)
    with pytest.raises(TypeError, match="non-string sequence"):
        sampler.replay_issued_samples("not-a-batch")
    with pytest.raises(TypeError, match="pair"):
        sampler.replay_issued_samples(((first_birth,),))
    assert sampler.state_dict() == before


def test_birth_mapping_assertion_is_strict_and_pure():
    sampler = S.ActionBallSampler([_profile()], seed=602)
    birth = _birth(sampler, epoch=4)
    before = deepcopy(sampler.state_dict())
    sampler.assert_issued_birth(birth.to_state_dict())
    assert sampler.state_dict() == before

    tampered = birth.to_identity_receipt()
    tampered["base_start_w_m"][0] = 999.0
    with pytest.raises(ValueError, match="canonical identity"):
        sampler.assert_issued_birth(tampered)
    assert sampler.state_dict() == before

    unknown = {**birth.to_identity_receipt(), "extra": 1}
    with pytest.raises(ValueError, match="unknown"):
        sampler.assert_issued_birth(unknown)
    assert sampler.state_dict() == before


def test_replay_does_not_rebuild_profile_registry_and_batch_is_pure(
    monkeypatch,
):
    profiles = [_profile(index + 1) for index in range(93)]
    sampler = S.ActionBallSampler(profiles, seed=601)
    samples = []
    for uid in (1, 47, 93):
        levels = _levels(aim=uid / 100.0)
        birth = _birth(sampler, uid=uid, levels=levels)
        samples.append(
            _sample(sampler, birth, uid=uid, levels=levels)
        )
    before = deepcopy(sampler.state_dict())

    def forbidden_init(*args, **kwargs):
        raise AssertionError("assertion rebuilt ActionBallSampler registry")

    monkeypatch.setattr(S.ActionBallSampler, "__init__", forbidden_init)
    sampler.assert_issued_samples(samples)
    sampler.assert_issued_samples(
        [sample.to_identity_receipt() for sample in samples]
    )
    assert sampler.state_dict() == before


@pytest.mark.parametrize(
    ("field_name", "new_value"),
    [
        ("contact_w_m", (999.0, 999.0, 999.0)),
        ("incoming_speed_mps", 6.9),
        ("spin_magnitude_radps", 39.0),
        ("landing_aim_w_xy_m", (2.89, 0.54)),
    ],
)
def test_self_consistent_sample_forgery_fails_exact_replay(
    field_name, new_value
):
    levels = _levels(
        aim=1.0,
        position=1.0,
        speed=1.0,
        spin_magnitude=1.0,
    )
    sampler = S.ActionBallSampler([_profile()], seed=2027)
    birth = _birth(sampler, epoch=3, levels=levels)
    sample = _sample(sampler, birth, epoch=3, levels=levels)
    forged = replace(sample, **{field_name: new_value})
    forged = replace(
        forged,
        sample_id=S._sha256_json(forged.identity_payload()),
    )
    forged.verify_sample_id()
    before = deepcopy(sampler.state_dict())
    with pytest.raises(ValueError, match="deterministic issued replay"):
        sampler.assert_issued_sample(forged)
    with pytest.raises(ValueError, match="deterministic issued replay"):
        sampler.assert_issued_sample(forged.to_identity_receipt())
    assert sampler.state_dict() == before


def test_sample_index_maps_around_interleaved_birth_draw_events():
    sampler = S.ActionBallSampler([_profile()], seed=81)
    levels = _levels(position=0.5)
    first_birth = _birth(sampler, epoch=0, levels=levels)
    first = _sample(sampler, first_birth, epoch=0, levels=levels)
    second_birth = _birth(sampler, epoch=0, levels=levels)
    second = _sample(sampler, second_birth, epoch=0, levels=levels)
    assert (first.sample_index, first.draw_start) == (
        0,
        S.DRAWS_PER_BIRTH,
    )
    assert (second.sample_index, second.draw_start) == (
        1,
        2 * S.DRAWS_PER_BIRTH + S.DRAWS_PER_SAMPLE,
    )
    sampler.assert_issued_sample(first)
    sampler.assert_issued_sample(second)

    shifted = replace(
        second,
        draw_start=second.draw_start - S.DRAWS_PER_BIRTH,
        draw_end=second.draw_end - S.DRAWS_PER_BIRTH,
    )
    shifted = replace(
        shifted,
        sample_id=S._sha256_json(shifted.identity_payload()),
    )
    shifted.verify_sample_id()
    with pytest.raises(ValueError, match="action-tape index"):
        sampler.assert_issued_sample(shifted)


def test_sample_highwater_is_exact_pure_and_survives_restore():
    sampler = S.ActionBallSampler(
        [_profile(11), _profile(29)], seed=91
    )
    assert sampler.sample_highwater_for(11) == (-1, 0)
    first_birth = _birth(sampler, uid=11)
    assert sampler.sample_highwater_for(11) == (-1, 0)
    first = _sample(sampler, first_birth, uid=11)
    assert sampler.sample_highwater_for(11) == (
        first.sample_index,
        first.draw_end,
    )

    # A later birth advances total draw_count but must not forge a larger
    # sample draw high-water.
    _birth(sampler, uid=11)
    assert sampler.draw_count_for(11) > first.draw_end
    before = deepcopy(sampler.state_dict())
    assert sampler.sample_highwater_for(11) == (0, first.draw_end)
    assert sampler.sample_highwater_for(29) == (-1, 0)
    assert sampler.state_dict() == before
    with pytest.raises(ValueError, match="unknown"):
        sampler.sample_highwater_for(999)

    restored = S.ActionBallSampler(
        [_profile(11), _profile(29)], seed=91
    )
    restored.load_state_dict(before)
    restored_before = deepcopy(restored.state_dict())
    assert restored.sample_highwater_for(11) == (0, first.draw_end)
    assert restored.sample_highwater_for(29) == (-1, 0)
    assert restored.state_dict() == restored_before


def test_birth_highwater_is_birth_only_interleaved_and_restorable():
    sampler = S.ActionBallSampler(
        [_profile(11), _profile(29)], seed=92
    )
    assert sampler.birth_highwater_for(11) == (-1, 0)
    first_birth = _birth(sampler, uid=11)
    assert sampler.birth_highwater_for(11) == (
        first_birth.birth_index,
        first_birth.draw_end,
    )
    first_sample = _sample(sampler, first_birth, uid=11)
    assert first_sample.draw_end > first_birth.draw_end
    assert sampler.birth_highwater_for(11) == (0, first_birth.draw_end)

    second_birth = _birth(sampler, uid=11)
    second_highwater = (1, second_birth.draw_end)
    assert sampler.birth_highwater_for(11) == second_highwater
    _sample(sampler, second_birth, uid=11)
    assert sampler.birth_highwater_for(11) == second_highwater
    assert sampler.birth_highwater_for(29) == (-1, 0)
    before = deepcopy(sampler.state_dict())
    assert sampler.birth_highwater_for(11) == second_highwater
    assert sampler.state_dict() == before
    with pytest.raises(ValueError, match="unknown"):
        sampler.birth_highwater_for(999)

    restored = S.ActionBallSampler(
        [_profile(11), _profile(29)], seed=92
    )
    restored.load_state_dict(before)
    restored_before = deepcopy(restored.state_dict())
    assert restored.birth_highwater_for(11) == second_highwater
    assert restored.birth_highwater_for(29) == (-1, 0)
    assert restored.state_dict() == restored_before


def test_action_tapes_are_independent_of_other_action_sampling_history():
    profiles = [_profile(11), _profile(29)]
    with_a_history = S.ActionBallSampler(profiles, seed=88)
    no_a_history = S.ActionBallSampler(profiles, seed=88)

    a_levels = _levels(position=1.0, spin_direction=1.0)
    for epoch in range(8):
        birth_a = _birth(
            with_a_history, uid=11, epoch=epoch, levels=a_levels
        )
        _sample(
            with_a_history,
            birth_a,
            uid=11,
            epoch=epoch,
            levels=a_levels,
        )

    b_levels = _levels(aim=1.0, speed=0.5)
    b_birth_after_a = _birth(
        with_a_history, uid=29, epoch=3, levels=b_levels
    )
    b_after_a = _sample(
        with_a_history,
        b_birth_after_a,
        uid=29,
        epoch=3,
        levels=b_levels,
    )
    b_birth_clean = _birth(
        no_a_history, uid=29, epoch=3, levels=b_levels
    )
    b_clean = _sample(
        no_a_history,
        b_birth_clean,
        uid=29,
        epoch=3,
        levels=b_levels,
    )
    assert b_birth_after_a == b_birth_clean
    assert b_after_a == b_clean
    assert with_a_history.draw_count_for(29) == no_a_history.draw_count_for(29)
    assert with_a_history.draw_count_for(11) > no_a_history.draw_count_for(11)


def test_no_move_wraps_reuse_birth_base_but_resample_ball_and_aim():
    levels = _levels(
        aim=1.0,
        position=1.0,
        speed=1.0,
        base_spawn=1.0,
        base_travel=1.0,
    )
    sampler = S.ActionBallSampler([_profile()], seed=9)
    birth = _birth(sampler, epoch=2, levels=levels)
    first = _sample(sampler, birth, epoch=2, levels=levels)
    second = _sample(sampler, birth, epoch=2, levels=levels)

    assert first.base_start_w_m == birth.base_start_w_m
    assert second.base_start_w_m == birth.base_start_w_m
    assert first.base_goal_w_m == birth.base_start_w_m
    assert second.base_goal_w_m == birth.base_start_w_m
    assert first.base_spawn_latent_w_m != second.base_spawn_latent_w_m
    assert first.base_travel_latent_b_yaw_m != (
        0.0,
        0.0,
        0.0,
    )
    assert first.contact_w_m != second.contact_w_m
    assert first.landing_aim_w_xy_m != second.landing_aim_w_xy_m
    assert first.draw_end - first.draw_start == S.DRAWS_PER_SAMPLE
    assert second.draw_end - second.draw_start == S.DRAWS_PER_SAMPLE


def test_mobility_is_profile_identity_and_cannot_be_overridden_at_sample():
    frozen_profile = _profile(mobility_mode="no_move")
    moving_profile = _profile(mobility_mode="move")
    assert frozen_profile.sha256 != moving_profile.sha256
    frozen = S.ActionBallSampler([frozen_profile], seed=9)
    moving = S.ActionBallSampler([moving_profile], seed=9)
    assert (
        frozen.sampler_contract_sha256
        != moving.sampler_contract_sha256
    )
    assert "mobility_mode" not in inspect.signature(frozen.sample).parameters

    frozen_birth = _birth(frozen)
    moving_birth = _birth(moving)
    frozen_sample = _sample(frozen, frozen_birth)
    moving_sample = _sample(moving, moving_birth)
    assert frozen_sample.base_goal_w_m == frozen_sample.base_start_w_m
    assert moving_sample.base_goal_w_m != moving_sample.base_start_w_m
    assert frozen.draw_count == moving.draw_count

    with pytest.raises(TypeError, match="unexpected keyword"):
        frozen.sample(
            birth=frozen_birth,
            action_uid=101,
            domain_epoch=0,
            levels=_levels(),
            mobility_mode="move",
        )


def test_landing_aim_has_independent_level_hard_bounds_and_receipt():
    profile = _profile(
        landing_aim_std_initial_m=(0.0, 0.0),
        landing_aim_std_max_m=(0.25, 0.35),
    )
    center_sampler = S.ActionBallSampler([profile], seed=321)
    center_levels = _levels(aim=0.0)
    center_birth = _birth(center_sampler, levels=center_levels)
    center_sample = _sample(
        center_sampler, center_birth, levels=center_levels
    )
    assert center_sample.landing_aim_w_xy_m == (
        profile.landing_aim_center_w_xy_m
    )

    wide_sampler = S.ActionBallSampler([profile], seed=321)
    wide_levels = _levels(aim=1.0)
    wide_birth = _birth(wide_sampler, levels=wide_levels)
    aims = [
        _sample(
            wide_sampler, wide_birth, levels=wide_levels
        ).landing_aim_w_xy_m
        for _ in range(200)
    ]
    assert len(set(aims)) > 150
    for aim in aims:
        for value, lower, upper in zip(
            aim,
            profile.landing_aim_min_w_xy_m,
            profile.landing_aim_max_w_xy_m,
        ):
            assert lower <= value <= upper


def test_asymmetric_contact_arms_promote_lower_and_upper_independently():
    profile = _profile(
        contact_offset_std_lower_initial_m=(0.0, 0.0, 0.0),
        contact_offset_std_lower_max_m=(0.0, 0.12, 0.0),
        contact_offset_std_upper_initial_m=(0.0, 0.0, 0.0),
        contact_offset_std_upper_max_m=(0.0, 0.10, 0.0),
    )
    center = profile.contact_offset_center_b_yaw_m[1]

    lower_levels = _levels(contact_y_lower=1.0)
    lower_sampler = S.ActionBallSampler([profile], seed=7001)
    lower_birth = _birth(lower_sampler, levels=lower_levels)
    lower_values = [
        _sample(
            lower_sampler, lower_birth, levels=lower_levels
        ).contact_offset_from_base_goal_b_yaw_m[1]
        for _ in range(256)
    ]
    assert min(lower_values) < center
    assert max(lower_values) == center

    upper_levels = _levels(contact_y_upper=1.0)
    upper_sampler = S.ActionBallSampler([profile], seed=7001)
    upper_birth = _birth(upper_sampler, levels=upper_levels)
    upper_values = [
        _sample(
            upper_sampler, upper_birth, levels=upper_levels
        ).contact_offset_from_base_goal_b_yaw_m[1]
        for _ in range(256)
    ]
    assert min(upper_values) == center
    assert max(upper_values) > center


def test_time_to_contact_is_asymmetric_bounded_and_identity_authoritative():
    profile = _profile(
        time_to_contact_std_lower_initial_s=0.0,
        time_to_contact_std_upper_initial_s=0.0,
    )
    levels = _levels(time_to_contact_lower=1.0)
    sampler = S.ActionBallSampler([profile], seed=7002)
    birth = _birth(sampler, levels=levels)
    samples = [
        _sample(sampler, birth, levels=levels)
        for _ in range(256)
    ]
    values = [sample.time_to_contact_s for sample in samples]
    assert profile.time_to_contact_min_s <= min(values)
    assert min(values) < profile.time_to_contact_center_s
    assert max(values) == profile.time_to_contact_center_s
    assert all(
        sample.to_receipt()["ball"]["time_to_contact_s"]
        == sample.time_to_contact_s
        for sample in samples
    )

    forged = replace(samples[0], time_to_contact_s=1.59)
    forged = replace(
        forged,
        sample_id=S._sha256_json(forged.identity_payload()),
    )
    forged.verify_sample_id()
    with pytest.raises(ValueError, match="deterministic issued replay"):
        sampler.assert_issued_sample(forged)


def test_tangent_direction_uses_fixed_half_side_probability_and_inbound_gate():
    profile = _profile(
        incoming_direction_tangent_u_neg_initial_deg=1.0,
        incoming_direction_tangent_u_neg_max_deg=1.0,
        incoming_direction_tangent_u_pos_initial_deg=10.0,
        incoming_direction_tangent_u_pos_max_deg=10.0,
        incoming_direction_tangent_v_neg_initial_deg=0.0,
        incoming_direction_tangent_v_neg_max_deg=0.0,
        incoming_direction_tangent_v_pos_initial_deg=0.0,
        incoming_direction_tangent_v_pos_max_deg=0.0,
    )
    sampler = S.ActionBallSampler([profile], seed=7003)
    birth = _birth(sampler)
    signed_u = []
    for _ in range(4096):
        direction = _sample(
            sampler, birth
        ).incoming_direction_b_yaw
        signed_u.append(
            sum(
                value * tangent
                for value, tangent in zip(
                    direction,
                    profile.incoming_direction_tangent_u_b_yaw,
                )
            )
        )
        assert sum(
            value * axis
            for value, axis in zip(
                direction, profile.incoming_inbound_axis_b_yaw
            )
        ) >= profile.incoming_inbound_min_cosine
    negative_fraction = sum(value < 0.0 for value in signed_u) / len(
        signed_u
    )
    assert 0.47 <= negative_fraction <= 0.53
    assert abs(min(signed_u)) < max(signed_u)


def test_base_z_is_not_a_curriculum_axis_and_stays_exactly_zero():
    with pytest.raises(ValueError, match="z must be exactly zero"):
        _profile(
            base_spawn_center_w_m=(-0.10, 0.05, 0.01),
            base_spawn_min_w_m=(-0.50, -0.40, 0.0),
            base_spawn_max_w_m=(0.30, 0.50, 0.02),
        )
    levels = _levels(
        base_spawn_x_lower=1.0,
        base_spawn_x_upper=1.0,
        base_spawn_y_lower=1.0,
        base_spawn_y_upper=1.0,
        base_travel_x_lower=1.0,
        base_travel_x_upper=1.0,
        base_travel_y_lower=1.0,
        base_travel_y_upper=1.0,
    )
    sampler = S.ActionBallSampler([_profile(mobility_mode="move")], seed=7004)
    birth = _birth(sampler, levels=levels)
    assert birth.base_start_w_m[2] == 0.0
    for _ in range(64):
        sample = _sample(sampler, birth, levels=levels)
        assert sample.base_spawn_latent_w_m[2] == 0.0
        assert sample.base_travel_latent_b_yaw_m[2] == 0.0
        assert sample.base_goal_w_m[2] == 0.0


def test_arm_catalog_is_explicitly_pinned_and_v2_state_is_rejected():
    sampler = S.ActionBallSampler([_profile()], seed=7005)
    birth = _birth(sampler)
    sample = _sample(sampler, birth)
    assert len(S.ARM_KEYS) == 32
    assert birth.arm_catalog_sha256 == S.ARM_CATALOG_SHA256
    assert sample.arm_catalog_sha256 == S.ARM_CATALOG_SHA256
    state = sampler.state_dict()
    assert state["arm_catalog_sha256"] == S.ARM_CATALOG_SHA256

    old = deepcopy(state)
    old["schema_version"] = 2
    old_payload = {
        key: value
        for key, value in old.items()
        if key != "integrity_sha256"
    }
    old["integrity_sha256"] = _integrity(old_payload)
    before = deepcopy(sampler.state_dict())
    with pytest.raises(ValueError, match="schema_version must be 5"):
        sampler.load_state_dict(old)
    assert sampler.state_dict() == before


def test_incoming_direction_cone_is_fixed_when_speed_level_expands():
    profile = _profile(incoming_direction_cone_deg=3.0)
    sampler = S.ActionBallSampler([profile], seed=123)
    slow_levels = _levels(speed=0.0)
    slow_birth = _birth(sampler, epoch=0, levels=slow_levels)
    slow_angles = [
        _angle_deg(
            _sample(
                sampler, slow_birth, epoch=0, levels=slow_levels
            ).incoming_direction_b_yaw,
            profile.incoming_direction_center_b_yaw,
        )
        for _ in range(100)
    ]
    fast_levels = _levels(speed=1.0)
    fast_birth = _birth(sampler, epoch=1, levels=fast_levels)
    fast_angles = [
        _angle_deg(
            _sample(
                sampler, fast_birth, epoch=1, levels=fast_levels
            ).incoming_direction_b_yaw,
            profile.incoming_direction_center_b_yaw,
        )
        for _ in range(100)
    ]
    assert max(slow_angles) <= 3.0 + 1.0e-9
    assert max(fast_angles) <= 3.0 + 1.0e-9


def test_contact_position_is_relative_to_goal_base_not_birth_spawn():
    profile = _profile(mobility_mode="move")
    sampler = S.ActionBallSampler([profile], seed=500)
    levels = _levels(position=0.8, base_travel=0.7)
    yaw = math.pi / 2.0
    birth = _birth(sampler, epoch=2, levels=levels, yaw=yaw)
    sample = _sample(
        sampler, birth, epoch=2, levels=levels, yaw=yaw
    )
    offset_world = tuple(
        sample.contact_w_m[index] - sample.base_goal_w_m[index]
        for index in range(3)
    )
    x, y, z = sample.contact_offset_from_base_goal_b_yaw_m
    assert offset_world == pytest.approx((-y, x, z))
    spawn_relative = tuple(
        sample.contact_w_m[index] - sample.base_start_w_m[index]
        for index in range(3)
    )
    assert spawn_relative != pytest.approx(offset_world)


def test_spin_is_magnitude_times_unit_direction_cone():
    profile = _profile()
    sampler = S.ActionBallSampler([profile], seed=123)
    levels = _levels(
        spin_magnitude=1.0,
        spin_direction=1.0,
    )
    birth = _birth(sampler, levels=levels, yaw=-0.2)
    for _ in range(80):
        sample = _sample(sampler, birth, levels=levels, yaw=-0.2)
        assert _norm(sample.spin_direction_b_yaw) == pytest.approx(1.0)
        assert _norm(sample.spin_direction_w) == pytest.approx(1.0)
        assert _norm(sample.spin_w_radps) == pytest.approx(
            sample.spin_magnitude_radps
        )
        assert _angle_deg(
            sample.spin_direction_b_yaw,
            profile.spin_direction_center_b_yaw,
        ) <= math.hypot(
            profile.spin_direction_tangent_u_neg_max_deg,
            profile.spin_direction_tangent_v_neg_max_deg,
        ) + 1.0e-9


@pytest.mark.parametrize("count", [1, 5, 93])
def test_arbitrary_n_action_identity_and_epoch(count):
    profiles = [_profile(index + 1) for index in range(count)]
    sampler = S.ActionBallSampler(profiles, seed=88)
    assert sampler.action_uids == tuple(range(1, count + 1))
    samples = []
    for index in range(count):
        levels = _levels(
            aim=(index % 5) / 4.0,
            position=(index % 5) / 4.0,
        )
        birth = _birth(
            sampler,
            uid=index + 1,
            epoch=index,
            levels=levels,
        )
        samples.append(
            _sample(
                sampler,
                birth,
                uid=index + 1,
                epoch=index,
                levels=levels,
            )
        )
    assert len({sample.sample_id for sample in samples}) == count
    assert sampler.birth_count == count
    assert sampler.sample_count == count

    reverse = S.ActionBallSampler(list(reversed(profiles)), seed=88)
    assert (
        sampler.sampler_contract_sha256
        == reverse.sampler_contract_sha256
    )


def test_epoch_changes_action_tape_draws_and_identity():
    levels = _levels(position=0.5)
    zero = S.ActionBallSampler([_profile()], seed=88)
    birth_zero = _birth(zero, epoch=0, levels=levels)
    sample_zero = _sample(zero, birth_zero, epoch=0, levels=levels)
    one = S.ActionBallSampler([_profile()], seed=88)
    birth_one = _birth(one, epoch=1, levels=levels)
    sample_one = _sample(one, birth_one, epoch=1, levels=levels)
    assert birth_zero.birth_id != birth_one.birth_id
    assert sample_zero.sample_id != sample_one.sample_id
    assert sample_zero.contact_w_m != sample_one.contact_w_m


def test_tampered_state_is_rejected_atomically_including_per_action_counts():
    sampler = S.ActionBallSampler([_profile(11), _profile(29)], seed=9)
    birth = _birth(sampler, uid=11)
    _sample(sampler, birth, uid=11)
    before = deepcopy(sampler.state_dict())

    tampered = deepcopy(before)
    tampered["per_action"]["11"]["draw_count"] += 1
    with pytest.raises(ValueError, match="integrity"):
        sampler.load_state_dict(tampered)
    assert sampler.state_dict() == before

    forged = deepcopy(before)
    forged["per_action"]["11"]["draw_count"] += 1
    forged["total_draw_count"] += 1
    payload = {
        key: forged[key]
        for key in forged
        if key != "integrity_sha256"
    }
    forged["integrity_sha256"] = _integrity(payload)
    with pytest.raises(ValueError, match="inconsistent"):
        sampler.load_state_dict(forged)
    assert sampler.state_dict() == before

    unknown = {**before, "surprise": 1}
    with pytest.raises(ValueError, match="unknown"):
        sampler.load_state_dict(unknown)
    assert sampler.state_dict() == before

    transcript_tamper = deepcopy(before)
    transcript_tamper["issued_births"]["11"][0]["birth_id"] = "f" * 64
    with pytest.raises(ValueError, match="integrity"):
        sampler.load_state_dict(transcript_tamper)
    assert sampler.state_dict() == before

    transcript_forgery = deepcopy(before)
    transcript_forgery["issued_births"]["11"].append(
        deepcopy(transcript_forgery["issued_births"]["11"][0])
    )
    transcript_payload = {
        key: transcript_forgery[key]
        for key in transcript_forgery
        if key != "integrity_sha256"
    }
    transcript_forgery["integrity_sha256"] = _integrity(
        transcript_payload
    )
    with pytest.raises(ValueError, match="length is inconsistent"):
        sampler.load_state_dict(transcript_forgery)
    assert sampler.state_dict() == before


def test_sample_assignment_ledger_tampering_is_rejected_atomically():
    sampler = S.ActionBallSampler([_profile()], seed=2030)
    first_birth = _birth(sampler, epoch=1)
    second_birth = _birth(sampler, epoch=2)
    _sample(sampler, first_birth, epoch=1)
    _sample(sampler, second_birth, epoch=2)
    before = deepcopy(sampler.state_dict())
    assert before["issued_sample_birth_indices"]["101"] == [0, 1]

    cross_birth = deepcopy(before)
    cross_birth["issued_sample_birth_indices"]["101"][0] = 1
    cross_payload = {
        key: value
        for key, value in cross_birth.items()
        if key != "integrity_sha256"
    }
    cross_birth["integrity_sha256"] = _integrity(cross_payload)
    with pytest.raises(ValueError, match="assignment hash mismatch"):
        sampler.load_state_dict(cross_birth)
    assert sampler.state_dict() == before

    unknown_birth = deepcopy(before)
    unknown_birth["issued_sample_birth_indices"]["101"][1] = 99
    unknown_payload = {
        key: value
        for key, value in unknown_birth.items()
        if key != "integrity_sha256"
    }
    unknown_birth["integrity_sha256"] = _integrity(unknown_payload)
    with pytest.raises(ValueError, match="unknown birth"):
        sampler.load_state_dict(unknown_birth)
    assert sampler.state_dict() == before

    missing_assignment = deepcopy(before)
    missing_assignment["issued_sample_birth_indices"]["101"].pop()
    missing_payload = {
        key: value
        for key, value in missing_assignment.items()
        if key != "integrity_sha256"
    }
    missing_assignment["integrity_sha256"] = _integrity(
        missing_payload
    )
    with pytest.raises(ValueError, match="length is inconsistent"):
        sampler.load_state_dict(missing_assignment)
    assert sampler.state_dict() == before


def test_compact_assignment_mapping_has_large_margin_to_full_receipts():
    action_count = 100
    samples_per_action = 4096
    mapping = {
        str(uid): [0] * samples_per_action
        for uid in range(1, action_count + 1)
    }
    mapping_bytes = len(
        json.dumps(
            mapping,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    sampler = S.ActionBallSampler([_profile()], seed=2031)
    birth = _birth(sampler)
    sample = _sample(sampler, birth)
    full_receipt_bytes = len(
        json.dumps(
            sample.to_receipt(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    full_receipts_projection = (
        action_count * samples_per_action * full_receipt_bytes
    )
    assert mapping_bytes < 1_000_000
    assert mapping_bytes < 0.01 * full_receipts_projection


def test_forged_or_cross_bound_birth_receipt_fails_before_rng_mutation():
    sampler = S.ActionBallSampler([_profile()], seed=7)
    levels = _levels(position=0.5)
    birth = _birth(sampler, epoch=3, levels=levels)
    before = deepcopy(sampler.state_dict())
    altered_base = (999.0, 999.0, 999.0)
    forged = replace(
        birth,
        base_start_w_m=altered_base,
    )
    with pytest.raises(ValueError, match="issued transcript"):
        _sample(sampler, forged, epoch=3, levels=levels)
    assert sampler.state_dict() == before

    # Red-team case: the attacker also recomputes the public self-hash.  The
    # sampler-issued transcript, not the caller-provided hash, is authoritative.
    forged_payload = S._birth_identity_payload(
        sampler_contract_sha256=forged.sampler_contract_sha256,
        arm_catalog_sha256=forged.arm_catalog_sha256,
        action_uid=forged.action_uid,
        domain_epoch=forged.domain_epoch,
        levels_sha256=forged.levels_sha256,
        profile_sha256=forged.profile_sha256,
        birth_index=forged.birth_index,
        draw_start=forged.draw_start,
        draw_end=forged.draw_end,
        mobility_mode=forged.mobility_mode,
        base_yaw_rad=forged.base_yaw_rad,
        base_start_w_m=altered_base,
    )
    forged = replace(
        forged, birth_id=S._sha256_json(forged_payload)
    )
    with pytest.raises(ValueError, match="issued transcript"):
        _sample(sampler, forged, epoch=3, levels=levels)
    assert sampler.state_dict() == before

    shifted_payload = S._birth_identity_payload(
        sampler_contract_sha256=birth.sampler_contract_sha256,
        arm_catalog_sha256=birth.arm_catalog_sha256,
        action_uid=birth.action_uid,
        domain_epoch=birth.domain_epoch,
        levels_sha256=birth.levels_sha256,
        profile_sha256=birth.profile_sha256,
        birth_index=birth.birth_index,
        draw_start=birth.draw_start + 1,
        draw_end=birth.draw_end + 1,
        mobility_mode=birth.mobility_mode,
        base_yaw_rad=birth.base_yaw_rad,
        base_start_w_m=birth.base_start_w_m,
    )
    shifted = replace(
        birth,
        draw_start=birth.draw_start + 1,
        draw_end=birth.draw_end + 1,
        birth_id=S._sha256_json(shifted_payload),
    )
    with pytest.raises(ValueError, match="issued transcript"):
        _sample(sampler, shifted, epoch=3, levels=levels)
    assert sampler.state_dict() == before

    with pytest.raises(ValueError, match="does not match"):
        _sample(
            sampler,
            birth,
            epoch=3,
            levels=_levels(position=0.75),
        )
    assert sampler.state_dict() == before


def test_forged_checkpoint_transcript_is_replayed_not_self_authorized():
    profile = _profile()
    issuer = S.ActionBallSampler([profile], seed=7)
    levels = _levels(position=0.5)
    _birth(issuer, epoch=3, levels=levels)
    forged_state = deepcopy(issuer.state_dict())
    row = forged_state["issued_births"]["101"][0]
    row["base_start_w_m"] = [999.0, 999.0, 999.0]
    identity = S._birth_identity_payload(
        sampler_contract_sha256=row["sampler_contract_sha256"],
        arm_catalog_sha256=row["arm_catalog_sha256"],
        action_uid=row["action_uid"],
        domain_epoch=row["domain_epoch"],
        levels_sha256=row["levels_sha256"],
        profile_sha256=row["profile_sha256"],
        birth_index=row["birth_index"],
        draw_start=row["draw_start"],
        draw_end=row["draw_end"],
        mobility_mode=row["mobility_mode"],
        base_yaw_rad=row["base_yaw_rad"],
        base_start_w_m=tuple(row["base_start_w_m"]),
    )
    row["birth_id"] = S._sha256_json(identity)
    outer_payload = {
        key: forged_state[key]
        for key in forged_state
        if key != "integrity_sha256"
    }
    forged_state["integrity_sha256"] = _integrity(outer_payload)

    victim = S.ActionBallSampler([profile], seed=7)
    before = deepcopy(victim.state_dict())
    with pytest.raises(ValueError, match="does not replay"):
        victim.load_state_dict(forged_state)
    assert victim.state_dict() == before


def test_sampler_contract_binds_seed_as_well_as_profiles():
    profile = _profile()
    first = S.ActionBallSampler([profile], seed=1)
    second = S.ActionBallSampler([profile], seed=2)
    assert (
        first.sampler_contract_sha256
        != second.sampler_contract_sha256
    )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        (
            {"incoming_direction_center_b_yaw": (0.0, 0.0, 0.0)},
            "non-zero",
        ),
        (
            {"incoming_direction_center_b_yaw": (1.1, 0.0, 0.0)},
            "unit length",
        ),
        (
            {"spin_direction_center_b_yaw": (float("nan"), 1.0, 0.0)},
            "finite",
        ),
        (
            {"contact_offset_min_b_yaw_m": (0.60, -0.20, 0.55)},
            "center",
        ),
        (
            {
                "landing_aim_min_w_xy_m": (2.60, -0.55),
            },
            "center",
        ),
        (
            {
                "contact_offset_std_max_m": (0.09, 0.08, 0.10),
            },
            "x std",
        ),
    ],
)
def test_invalid_profile_fails_closed(override, message):
    with pytest.raises(ValueError, match=message):
        _profile(**override)


def test_zero_std_and_no_move_still_consume_fixed_birth_and_sample_budget():
    zero = _profile(
        contact_offset_std_initial_m=(0.0, 0.0, 0.0),
        contact_offset_std_max_m=(0.0, 0.0, 0.0),
        incoming_speed_std_initial_mps=0.0,
        incoming_speed_std_max_mps=0.0,
        spin_magnitude_std_initial_radps=0.0,
        spin_magnitude_std_max_radps=0.0,
        base_spawn_std_initial_m=(0.0, 0.0, 0.0),
        base_spawn_std_max_m=(0.0, 0.0, 0.0),
        base_travel_std_initial_m=(0.0, 0.0, 0.0),
        base_travel_std_max_m=(0.0, 0.0, 0.0),
        landing_aim_std_initial_m=(0.0, 0.0),
        landing_aim_std_max_m=(0.0, 0.0),
    )
    sampler = S.ActionBallSampler([zero], seed=1)
    birth = _birth(sampler)
    sample = _sample(sampler, birth)
    assert birth.draw_start == 0
    assert birth.draw_end == S.DRAWS_PER_BIRTH
    assert sample.draw_start == S.DRAWS_PER_BIRTH
    assert sample.draw_end == S.DRAWS_PER_BIRTH + S.DRAWS_PER_SAMPLE
    assert sampler.draw_count_for(101) == (
        S.DRAWS_PER_BIRTH + S.DRAWS_PER_SAMPLE
    )
    assert sample.base_goal_w_m == birth.base_start_w_m


def test_levels_and_requests_reject_nan_unknown_and_zero_uid():
    with pytest.raises(ValueError, match="finite"):
        _levels(aim=float("nan"))
    with pytest.raises(ValueError, match="unknown"):
        S.DomainLevels.from_mapping(
            {
                **_levels().as_dict(),
                "unknown": 0.0,
            }
        )
    sampler = S.ActionBallSampler([_profile()], seed=1)
    with pytest.raises(ValueError, match="action_uid"):
        sampler.reserve_birth(
            action_uid=0,
            domain_epoch=0,
            levels=_levels(),
        )
    with pytest.raises(ValueError, match="domain_epoch"):
        sampler.reserve_birth(
            action_uid=101,
            domain_epoch=-1,
            levels=_levels(),
        )
