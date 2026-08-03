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

COUNTER_RALLY_PATH = PATH.with_name("counter_rally.py")
COUNTER_RALLY_SPEC = importlib.util.spec_from_file_location(
    "counter_rally_for_sampling_test", COUNTER_RALLY_PATH
)
assert (
    COUNTER_RALLY_SPEC is not None
    and COUNTER_RALLY_SPEC.loader is not None
)
CR = importlib.util.module_from_spec(COUNTER_RALLY_SPEC)
sys.modules[COUNTER_RALLY_SPEC.name] = CR
COUNTER_RALLY_SPEC.loader.exec_module(CR)


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


def _scalar_request_digest(*, kind, action_uid, domain_epoch, levels):
    return bytes.fromhex(
        _integrity(
            {
                "kind": kind,
                "action_uid": action_uid,
                "domain_epoch": domain_epoch,
                "levels_sha256": levels.sha256,
            }
        )
    )


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


def test_request_digest_cache_is_scalar_exact_and_non_authoritative():
    levels = _levels(position=0.25, speed=0.75)
    equivalent_levels = S.DomainLevels.from_mapping(levels.as_dict())
    sampler = S.ActionBallSampler([_profile()], seed=20260731)
    before = deepcopy(sampler.state_dict())

    expected = _scalar_request_digest(
        kind="swing_sample",
        action_uid=101,
        domain_epoch=7,
        levels=levels,
    )
    first = sampler._request_digest(
        kind="swing_sample",
        action_uid=101,
        domain_epoch=7,
        levels=levels,
    )
    second = sampler._request_digest(
        kind="swing_sample",
        action_uid=101,
        domain_epoch=7,
        levels=equivalent_levels,
    )

    assert first == second == expected
    assert len(sampler._levels_sha256_cache) == 1
    assert list(sampler._request_digest_cache) == [
        ("swing_sample", 101, 7, levels.sha256)
    ]
    assert sampler.state_dict() == before


def test_request_digest_cache_lazily_initializes_for_replay_views():
    levels = _levels(position=0.25, speed=0.75)
    sampler = S.ActionBallSampler([_profile()], seed=20260731)
    before = deepcopy(sampler.state_dict())
    del sampler._request_digest_cache_limit
    del sampler._levels_sha256_cache
    del sampler._request_digest_cache

    actual = sampler._request_digest(
        kind="swing_sample",
        action_uid=101,
        domain_epoch=7,
        levels=levels,
    )

    assert actual == _scalar_request_digest(
        kind="swing_sample",
        action_uid=101,
        domain_epoch=7,
        levels=levels,
    )
    assert sampler._request_digest_cache_limit == 64
    assert len(sampler._levels_sha256_cache) == 1
    assert len(sampler._request_digest_cache) == 1
    assert sampler.state_dict() == before


def test_request_digest_cache_invalidates_epoch_and_exact_level_content():
    center = _levels()
    changed = _levels(contact_y_upper=0.5)
    negative_zero = _levels(contact_y_upper=-0.0)
    sampler = S.ActionBallSampler([_profile()], seed=20260731)

    center_epoch_0 = sampler._request_digest(
        kind="swing_sample",
        action_uid=101,
        domain_epoch=0,
        levels=center,
    )
    center_epoch_1 = sampler._request_digest(
        kind="swing_sample",
        action_uid=101,
        domain_epoch=1,
        levels=center,
    )
    changed_epoch_1 = sampler._request_digest(
        kind="swing_sample",
        action_uid=101,
        domain_epoch=1,
        levels=changed,
    )
    negative_zero_epoch_1 = sampler._request_digest(
        kind="swing_sample",
        action_uid=101,
        domain_epoch=1,
        levels=negative_zero,
    )

    assert center_epoch_0 == _scalar_request_digest(
        kind="swing_sample",
        action_uid=101,
        domain_epoch=0,
        levels=center,
    )
    assert center_epoch_1 == _scalar_request_digest(
        kind="swing_sample",
        action_uid=101,
        domain_epoch=1,
        levels=center,
    )
    assert changed_epoch_1 == _scalar_request_digest(
        kind="swing_sample",
        action_uid=101,
        domain_epoch=1,
        levels=changed,
    )
    assert negative_zero_epoch_1 == _scalar_request_digest(
        kind="swing_sample",
        action_uid=101,
        domain_epoch=1,
        levels=negative_zero,
    )
    assert len(
        {
            center_epoch_0,
            center_epoch_1,
            changed_epoch_1,
            negative_zero_epoch_1,
        }
    ) == 4
    assert len(sampler._levels_sha256_cache) == 3
    assert len(sampler._request_digest_cache) == 4


def test_request_digest_cache_clear_boundary_is_bounded_and_scalar_exact():
    sampler = S.ActionBallSampler([_profile()], seed=20260731)
    sampler._request_digest_cache_limit = 3

    for index in range(7):
        levels = _levels(contact_y_upper=index / 10.0)
        actual = sampler._request_digest(
            kind="swing_sample",
            action_uid=101,
            domain_epoch=index,
            levels=levels,
        )
        assert actual == _scalar_request_digest(
            kind="swing_sample",
            action_uid=101,
            domain_epoch=index,
            levels=levels,
        )
        assert len(sampler._levels_sha256_cache) <= 3
        assert len(sampler._request_digest_cache) <= 3


def test_diagnostic_prevalidated_sample_batch_is_scalar_bit_exact():
    profile = _profile()
    mixture = S.SamplingMixture()
    levels = _levels(
        position=0.75,
        speed=0.5,
        spin_magnitude=0.25,
        spin_direction=0.5,
        aim=0.75,
        base_spawn=0.25,
        base_travel=0.5,
    )
    kwargs = {
        "seed": 20260731,
        "sampling_mixture": mixture,
        "contact_time_step_s": 0.02,
        "diagnostic_unauthorized": True,
    }
    scalar = S.ActionBallSampler([profile], **kwargs)
    batched = S.ActionBallSampler([profile], **kwargs)
    scalar_birth = _birth(
        scalar, epoch=9, levels=levels, yaw=0.17
    )
    batched_birth = _birth(
        batched, epoch=9, levels=levels, yaw=0.17
    )
    assert batched_birth == scalar_birth

    expected = tuple(
        _sample(
            scalar,
            scalar_birth,
            epoch=9,
            levels=levels,
            yaw=0.17,
        )
        for _ in range(7)
    )
    actual = batched.sample_many_prevalidated(
        birth=batched_birth,
        action_uid=101,
        domain_epoch=9,
        levels=levels,
        base_yaw_rad=0.17,
        count=7,
    )

    assert actual == expected
    assert batched.draw_count_for(101) == scalar.draw_count_for(101)
    assert (
        batched.sample_highwater_for(101)
        == scalar.sample_highwater_for(101)
    )
    for sample in actual:
        sample.verify_sample_id()


def test_diagnostic_prevalidated_sample_batch_is_fail_closed():
    profile = _profile()
    levels = _levels(position=0.5)
    formal = S.ActionBallSampler([profile], seed=20260731)
    formal_birth = _birth(formal, epoch=3, levels=levels)
    formal_before = deepcopy(formal.state_dict())
    with pytest.raises(
        RuntimeError,
        match="requires diagnostic_unauthorized",
    ):
        formal.sample_many_prevalidated(
            birth=formal_birth,
            action_uid=101,
            domain_epoch=3,
            levels=levels,
            count=1,
        )
    assert formal.state_dict() == formal_before

    diagnostic = S.ActionBallSampler(
        [profile],
        seed=20260731,
        diagnostic_unauthorized=True,
    )
    birth = _birth(diagnostic, epoch=3, levels=levels)
    draw_before = diagnostic.draw_count_for(101)
    sample_before = diagnostic.sample_highwater_for(101)
    with pytest.raises(
        ValueError,
        match="not the exact live sampler object",
    ):
        diagnostic.sample_many_prevalidated(
            birth=replace(birth),
            action_uid=101,
            domain_epoch=3,
            levels=levels,
            count=1,
        )
    assert diagnostic.draw_count_for(101) == draw_before
    assert diagnostic.sample_highwater_for(101) == sample_before


def test_diagnostic_prevalidated_batch_skips_only_redundant_identity_rehash(
    monkeypatch,
):
    profile = _profile()
    levels = _levels(speed=0.5, aim=0.25)
    sampler = S.ActionBallSampler(
        [profile],
        seed=20260731,
        diagnostic_unauthorized=True,
    )
    birth = _birth(sampler, epoch=4, levels=levels)

    def fail_redundant_rehash(_sample):
        raise AssertionError("diagnostic batch repeated sample identity hash")

    monkeypatch.setattr(
        S.BallBaseSample,
        "verify_sample_id",
        fail_redundant_rehash,
    )
    samples = sampler.sample_many_prevalidated(
        birth=birth,
        action_uid=101,
        domain_epoch=4,
        levels=levels,
        count=3,
    )

    assert len(samples) == 3
    for sample in samples:
        assert sample.sample_id == S._sha256_json(
            sample.identity_payload()
        )


def test_diagnostic_retirement_is_bounded_and_preserves_exact_random_tape():
    profile = _profile()
    formal = S.ActionBallSampler([profile], seed=20260730)
    diagnostic = S.ActionBallSampler(
        [profile],
        seed=20260730,
        diagnostic_unauthorized=True,
    )
    for epoch in range(8):
        formal_birth = _birth(formal, epoch=epoch)
        diagnostic_birth = _birth(diagnostic, epoch=epoch)
        assert diagnostic_birth == formal_birth
        assert (
            _sample(diagnostic, diagnostic_birth, epoch=epoch)
            == _sample(formal, formal_birth, epoch=epoch)
        )
        diagnostic.forget_diagnostic_births((diagnostic_birth,))
        assert diagnostic.birth_highwater_for(101) == (
            formal.birth_highwater_for(101)
        )
        assert diagnostic.sample_highwater_for(101) == (
            formal.sample_highwater_for(101)
        )
        assert diagnostic.draw_count_for(101) == formal.draw_count_for(101)
    assert diagnostic._issued_births_by_action[101] == {}
    assert diagnostic._issued_sample_birth_indices_by_action[101] == []
    with pytest.raises(
        RuntimeError,
        match="requires diagnostic_unauthorized",
    ):
        formal.forget_diagnostic_births(())

    live_count = 4096
    sampler = S.ActionBallSampler(
        [profile],
        seed=20260731,
        diagnostic_unauthorized=True,
    )
    live = {}
    for env_id in range(live_count):
        birth = _birth(sampler, epoch=1)
        _sample(sampler, birth, epoch=1)
        live[env_id] = birth
    previous_birth_highwater = sampler.birth_highwater_for(101)
    previous_sample_highwater = sampler.sample_highwater_for(101)

    # Repeated asynchronous reset subsets must replace, not accumulate,
    # sampler authority.  The batch size is deliberately not a divisor of
    # 4096 so each round touches a different live subset.
    for generation in range(2, 6):
        env_ids = tuple(
            (generation * 263 + offset) % live_count
            for offset in range(257)
        )
        sampler.forget_diagnostic_births(
            tuple(live[env_id] for env_id in env_ids)
        )
        for env_id in env_ids:
            birth = _birth(sampler, epoch=generation)
            _sample(sampler, birth, epoch=generation)
            live[env_id] = birth
        assert len(sampler._issued_births_by_action[101]) == live_count
        assert sampler._issued_sample_birth_indices_by_action[101] == []
        assert sampler._compaction_segments_by_action[101] == []
        assert sampler.birth_highwater_for(101) > previous_birth_highwater
        assert sampler.sample_highwater_for(101) > previous_sample_highwater
        previous_birth_highwater = sampler.birth_highwater_for(101)
        previous_sample_highwater = sampler.sample_highwater_for(101)


def test_counter_rally_reserves_draw_17_and_derives_landing_y_from_reverse_ray():
    profile = _profile(
        counter_rally_objective=CR.CounterRallyObjectiveProfile()
    )
    levels = _levels(
        aim=1.0,
        incoming_direction_u_neg=0.75,
        incoming_direction_u_pos=0.75,
        incoming_direction_v_neg=0.75,
        incoming_direction_v_pos=0.75,
        position=0.5,
    )
    sampler = S.ActionBallSampler([profile], seed=20260729)
    birth = _birth(sampler, epoch=4, levels=levels, yaw=0.11)
    before = sampler.draw_count
    sample = _sample(
        sampler,
        birth,
        epoch=4,
        levels=levels,
        yaw=0.11,
    )
    assert sampler.draw_count - before == S.DRAWS_PER_SAMPLE
    assert sample.draw_end - sample.draw_start == 18
    incoming_xy_norm = math.hypot(*sample.incoming_direction_w[:2])
    return_x = -sample.incoming_direction_w[0] / incoming_xy_norm
    return_y = -sample.incoming_direction_w[1] / incoming_xy_norm
    scale = (
        sample.landing_aim_w_xy_m[0] - sample.contact_w_m[0]
    ) / return_x
    assert sample.landing_aim_w_xy_m[1] == pytest.approx(
        sample.contact_w_m[1] + scale * return_y
    )
    assert sample.profile_sha256 == profile.sha256
    assert profile.as_dict()["counter_rally_objective"] == (
        profile.counter_rally_objective.to_mapping()
    )
    task = CR.derive_counter_rally_task(
        base_goal_env_xy_m=sample.base_goal_w_m[:2],
        base_yaw_env_rad=sample.base_yaw_rad,
        contact_offset_b_yaw_m=(
            sample.contact_offset_from_base_goal_b_yaw_m
        ),
        incoming_direction_b_yaw=sample.incoming_direction_b_yaw[:2],
        incoming_ball_speed_at_contact_mps=sample.incoming_speed_mps,
        landing_depth_env_x_m=sample.landing_aim_w_xy_m[0],
        profile=profile.counter_rally_objective,
    )
    assert sample.landing_aim_w_xy_m == pytest.approx(
        task.landing_aim_env_xy_m
    )


def test_counter_rally_landing_y_arms_never_enter_the_frontier_schedule():
    profile = _profile(
        counter_rally_objective=CR.CounterRallyObjectiveProfile()
    )
    eligible = S._eligible_swing_frontier_arms(
        profile,
        _levels(aim=1.0),
        S.SamplingMixture(),
    )
    assert "landing_aim_x_lower" in eligible
    assert "landing_aim_x_upper" in eligible
    assert "landing_aim_y_lower" not in eligible
    assert "landing_aim_y_upper" not in eligible


@pytest.mark.parametrize(
    ("incoming_direction", "landing_x", "expected_reason"),
    (
        ((-0.80, -0.60, 0.0), 2.5, "reverse_ray_not_opponent_bound"),
        ((-1.00, 0.00, 0.0), 1.2, "landing_depth_not_opponent_half"),
        ((-1.00, 0.00, 0.0), 3.3, "landing_depth_outside_table"),
        ((-0.86, -0.51, 0.0), 2.5, "reverse_ray_misses_table"),
    ),
)
def test_counter_rally_geometry_returns_named_solver_rejection_without_redraw(
    incoming_direction,
    landing_x,
    expected_reason,
):
    aim, reason = S._counter_rally_reverse_ray_geometry(
        contact_w_m=(0.8, 0.0, 1.0),
        incoming_direction_w=incoming_direction,
        landing_x_w_m=landing_x,
        objective=CR.CounterRallyObjectiveProfile(),
    )
    assert aim[0] == landing_x
    assert reason == expected_reason
    with pytest.raises(CR.CounterRallyRejected) as caught:
        CR.derive_counter_rally_task(
            base_goal_env_xy_m=(0.55, 0.10),
            base_yaw_env_rad=0.0,
            contact_offset_b_yaw_m=(0.25, -0.10, 1.0),
            incoming_direction_b_yaw=incoming_direction[:2],
            incoming_ball_speed_at_contact_mps=3.0,
            landing_depth_env_x_m=landing_x,
            profile=CR.CounterRallyObjectiveProfile(),
        )
    assert caught.value.reason == expected_reason


def test_counter_rally_shared_geometry_canonicalizes_behind_contact():
    objective = CR.CounterRallyObjectiveProfile()
    aim, reason = S._counter_rally_reverse_ray_geometry(
        contact_w_m=(3.0, 0.0, 1.0),
        incoming_direction_w=(-1.0, 0.0, 0.0),
        landing_x_w_m=2.5,
        objective=objective,
    )
    assert aim == (2.5, 0.0)
    assert reason == "landing_behind_contact"
    with pytest.raises(CR.CounterRallyRejected) as caught:
        CR.derive_counter_rally_task(
            base_goal_env_xy_m=(2.75, 0.10),
            base_yaw_env_rad=0.0,
            contact_offset_b_yaw_m=(0.25, -0.10, 1.0),
            incoming_direction_b_yaw=(-1.0, 0.0),
            incoming_ball_speed_at_contact_mps=3.0,
            landing_depth_env_x_m=2.5,
            profile=objective,
        )
    assert caught.value.reason == reason


def test_counter_rally_support_rejects_unsupported_yaw_before_any_draw():
    profile = _profile(
        counter_rally_objective=CR.CounterRallyObjectiveProfile()
    )
    sampler = S.ActionBallSampler([profile], seed=20260729)
    before = sampler.state_dict()
    with pytest.raises(ValueError, match="leaves the opponent cone"):
        sampler.reserve_birth(
            action_uid=profile.action_uid,
            domain_epoch=0,
            levels=S.DomainLevels(),
            base_yaw_rad=math.radians(40.0),
        )
    assert sampler.state_dict() == before
    assert sampler.draw_count == 0


def test_counter_rally_profile_requires_all_landing_x_support_on_opponent_half():
    with pytest.raises(
        ValueError,
        match="landing-x support must lie on the bounded opponent half",
    ):
        _profile(
            counter_rally_objective=CR.CounterRallyObjectiveProfile(),
            landing_aim_min_w_xy_m=(1.2, -0.55),
        )


def test_counter_rally_table_y_miss_keeps_proposal_and_exact_resume_transcript():
    profile = _profile(
        counter_rally_objective=CR.CounterRallyObjectiveProfile()
    )
    levels = _levels(
        aim=1.0,
        position=1.0,
        incoming_direction_u_neg=1.0,
        incoming_direction_u_pos=1.0,
        incoming_direction_v_neg=1.0,
        incoming_direction_v_pos=1.0,
    )
    sampler = S.ActionBallSampler([profile], seed=2)
    birth = _birth(sampler, levels=levels, yaw=0.25)
    before = sampler.draw_count
    sample = _sample(
        sampler,
        birth,
        levels=levels,
        yaw=0.25,
    )
    assert sampler.draw_count - before == S.DRAWS_PER_SAMPLE
    assert sampler.sample_count_for(profile.action_uid) == 1
    sample.verify_sample_id()
    _, reason = S._counter_rally_reverse_ray_geometry(
        contact_w_m=sample.contact_w_m,
        incoming_direction_w=sample.incoming_direction_w,
        landing_x_w_m=sample.landing_aim_w_xy_m[0],
        objective=profile.counter_rally_objective,
    )
    assert reason == "reverse_ray_misses_table"

    saved = sampler.state_dict()
    restored = S.ActionBallSampler([profile], seed=2)
    restored.load_state_dict(saved)
    assert restored.state_dict() == saved


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
    assert frozen_sample.executed_planar_base_travel_distance_m == 0.0
    assert math.isclose(
        moving_sample.executed_planar_base_travel_distance_m,
        math.hypot(
            moving_sample.base_travel_latent_b_yaw_m[0],
            moving_sample.base_travel_latent_b_yaw_m[1],
        ),
    )
    assert (
        "executed_planar_base_travel_distance_m"
        not in frozen_sample.identity_payload()
    )
    assert (
        "executed_planar_base_travel_distance_m"
        not in moving_sample.identity_payload()
    )
    frozen_sample.verify_sample_id()
    moving_sample.verify_sample_id()
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
    # 单次 uniform 覆盖整个区间(Franco 07-28:不选侧);上侧零宽 => 全部 <= center。
    assert max(lower_values) <= center

    upper_levels = _levels(contact_y_upper=1.0)
    upper_sampler = S.ActionBallSampler([profile], seed=7001)
    upper_birth = _birth(upper_sampler, levels=upper_levels)
    upper_values = [
        _sample(
            upper_sampler, upper_birth, levels=upper_levels
        ).contact_offset_from_base_goal_b_yaw_m[1]
        for _ in range(256)
    ]
    assert min(upper_values) >= center
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
    assert max(values) <= profile.time_to_contact_center_s
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


def test_tangent_direction_is_uniform_over_asymmetric_interval_and_inbound_gate():
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
    # 单次 uniform 覆盖 [-1°, +10°](Franco 07-28:不选侧),负侧质量 ≈ 1/11。
    assert 0.05 <= negative_fraction <= 0.14
    assert abs(min(signed_u)) < max(signed_u)


def test_base_z_is_not_a_curriculum_axis_and_stays_exactly_zero():
    with pytest.raises(
        ValueError, match="one constant, not a curriculum axis"
    ):
        _profile(
            base_spawn_center_w_m=(-0.10, 0.05, 0.01),
            base_spawn_min_w_m=(-0.50, -0.40, 0.0),
            base_spawn_max_w_m=(0.30, 0.50, 0.02),
        )
    # 常数非零 z(= canonical-ready root Z 由 runtime 注入)是合法的。
    constant_z = _profile(
        base_spawn_center_w_m=(-0.10, 0.05, 0.78),
        base_spawn_min_w_m=(-0.50, -0.40, 0.78),
        base_spawn_max_w_m=(0.30, 0.50, 0.78),
    )
    assert constant_z.base_spawn_center_w_m[2] == 0.78
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


def test_explicit_joint_mixture_is_exact_20_60_20_and_center_is_not_a_point():
    mixture = S.SamplingMixture()
    assert mixture.schedule == (
        "interior",
        "center",
        "interior",
        "frontier",
        "interior",
    )
    levels = _levels(**{name: 1.0 for name in S.ARM_KEYS})
    sampler = S.ActionBallSampler(
        [_profile()], seed=7101, sampling_mixture=mixture
    )
    birth = _birth(sampler, levels=levels)
    samples = [
        _sample(sampler, birth, levels=levels) for _ in range(100)
    ]
    counts = {
        name: sum(sample.sampling_stratum == name for sample in samples)
        for name in ("center", "interior", "frontier")
    }
    assert counts == {"center": 20, "interior": 60, "frontier": 20}

    center_samples = [
        sample for sample in samples if sample.sampling_stratum == "center"
    ]
    assert all(
        sample.sampling_levels == _levels() for sample in center_samples
    )
    # Level zero is the profile's narrow initial support, not a measure-zero
    # point at the center.
    assert len(
        {sample.time_to_contact_s for sample in center_samples}
    ) > 10
    assert any(
        sample.time_to_contact_s
        != _profile().time_to_contact_center_s
        for sample in center_samples
    )
    for sample in samples:
        receipt = sample.to_receipt()
        assert "in_new_band" not in receipt["sampling"]
        assert (
            receipt["sampling"]["stratum"]
            == sample.sampling_stratum
        )
        sampler.verify_sampling_membership(receipt)
        sampler.assert_issued_sample(receipt)


def test_production_ttc_is_native_tick_sampled_replayable_and_two_sided():
    profile = _profile()
    mixture = S.SamplingMixture()
    levels = S.DomainLevels()
    policy_dt_s = 0.02
    sampler = S.ActionBallSampler(
        [profile],
        seed=7111,
        sampling_mixture=mixture,
        contact_time_step_s=policy_dt_s,
    )
    birth = _birth(sampler, epoch=3, levels=levels)
    samples = [
        _sample(
            sampler,
            birth,
            epoch=3,
            levels=levels,
        )
        for _ in range(9)
    ]

    assert all(
        sample.time_to_contact_s
        == sample.time_to_contact_tick * policy_dt_s
        for sample in samples
    )
    lower = samples[3]
    upper = samples[8]
    assert lower.frontier_arm == "time_to_contact_lower"
    assert upper.frontier_arm == "time_to_contact_upper"
    grid = sampler._contact_time_grid_by_action[profile.action_uid]
    assert lower.time_to_contact_tick < grid.center_tick
    assert upper.time_to_contact_tick > grid.center_tick

    strict_floor_s = (
        profile.reference_t_hit_s / profile.teacher_rate_min
        + profile.reaction_margin_s
    )
    assert lower.time_to_contact_s > strict_floor_s
    assert (
        upper.time_to_contact_s
        - profile.reference_t_hit_s / profile.teacher_rate_max
        <= 1.0
    )
    assert sampler.replay_issued_sample(
        birth, lower.sample_index
    ) == lower
    sampler.verify_sampling_membership(lower)
    sampler.verify_sampling_membership(upper)

    restored = S.ActionBallSampler(
        [profile],
        seed=7111,
        sampling_mixture=mixture,
        contact_time_step_s=policy_dt_s,
    )
    restored.load_state_dict(sampler.state_dict())
    assert _sample(
        restored,
        birth,
        epoch=3,
        levels=levels,
    ) == _sample(
        sampler,
        birth,
        epoch=3,
        levels=levels,
    )

    tampered = lower.to_identity_receipt()
    tampered["time_to_contact_tick"] += 1
    with pytest.raises(ValueError):
        sampler.assert_issued_sample(tampered)


def test_frontier_receipt_recomputes_outer_band_and_forced_arm_has_no_starvation():
    profile = _profile()
    levels = _levels(**{name: 1.0 for name in S.ARM_KEYS})
    mixture = S.SamplingMixture()
    sampler = S.ActionBallSampler(
        [profile], seed=7102, sampling_mixture=mixture
    )
    birth = _birth(sampler, levels=levels)
    eligible = S._eligible_swing_frontier_arms(profile, levels)
    assert len(eligible) == 24
    # One frontier slot per five samples: exactly one full round of all
    # eligible per-swing action/axis/side arms.
    samples = [
        _sample(sampler, birth, levels=levels)
        for _ in range(5 * len(eligible))
    ]
    frontier = [
        sample for sample in samples if sample.sampling_stratum == "frontier"
    ]
    assert [sample.frontier_arm for sample in frontier] == list(eligible)
    assert set(sample.frontier_arm for sample in frontier) == set(eligible)
    for sample in frontier:
        sampler.verify_sampling_membership(sample.to_identity_receipt())
        negative, positive, side, _ = S._frontier_width_pair(
            profile, sample.sampling_levels, sample.frontier_arm
        )
        delta = S._frontier_coordinate_delta(
            sample, profile, sample.frontier_arm
        )
        width = negative if side == "negative" else positive
        normalized = -delta / width if side == "negative" else delta / width
        assert (
            normalized + 1.0e-10
            >= 1.0 - mixture.frontier_band_fraction
        )
        assert normalized <= 1.0 + 1.0e-10

    forged = frontier[0]
    # Move the selected coordinate back to the center without relying on a
    # caller-supplied membership flag, then make the outer identity
    # self-consistent.  Geometry recomputation must still reject it.
    assert forged.frontier_arm == "time_to_contact_lower"
    forged = replace(
        forged,
        time_to_contact_s=profile.time_to_contact_center_s,
    )
    forged = replace(
        forged,
        sample_id=S._sha256_json(forged.identity_payload()),
    )
    forged.verify_sample_id()
    with pytest.raises(ValueError, match="recomputed frontier band"):
        sampler.verify_sampling_membership(forged)

    unrelated_forge = replace(
        frontier[0],
        incoming_speed_mps=1000.0,
    )
    unrelated_forge = replace(
        unrelated_forge,
        sample_id=S._sha256_json(
            unrelated_forge.identity_payload()
        ),
    )
    unrelated_forge.verify_sample_id()
    with pytest.raises(ValueError, match="deterministic issued replay"):
        sampler.verify_sampling_membership(unrelated_forge)


def test_move_frontier_round_robin_adds_all_four_travel_sides():
    profile = _profile(mobility_mode="move")
    levels = _levels(**{name: 1.0 for name in S.ARM_KEYS})
    sampler = S.ActionBallSampler(
        [profile],
        seed=7105,
        sampling_mixture=S.SamplingMixture(),
    )
    birth = _birth(sampler, levels=levels)
    eligible = S._eligible_swing_frontier_arms(profile, levels)
    assert len(eligible) == 28
    travel = {
        "base_travel_x_lower",
        "base_travel_x_upper",
        "base_travel_y_lower",
        "base_travel_y_upper",
    }
    assert travel.issubset(set(eligible))
    samples = [
        _sample(sampler, birth, levels=levels)
        for _ in range(5 * len(eligible))
    ]
    frontier_arms = [
        sample.frontier_arm
        for sample in samples
        if sample.sampling_stratum == "frontier"
    ]
    assert frontier_arms == list(eligible)
    assert travel.issubset(set(frontier_arms))


def test_joint_mixture_state_roundtrip_preserves_midcycle_strata_and_rng():
    mixture = S.SamplingMixture()
    levels = _levels(**{name: 0.75 for name in S.ARM_KEYS})
    profiles = [_profile(11), _profile(29)]
    original = S.ActionBallSampler(
        profiles, seed=7103, sampling_mixture=mixture
    )
    birth = _birth(original, uid=11, epoch=3, levels=levels)
    for _ in range(7):
        _sample(original, birth, uid=11, epoch=3, levels=levels)
    saved = deepcopy(original.state_dict())
    expected = [
        _sample(original, birth, uid=11, epoch=3, levels=levels)
        for _ in range(13)
    ]

    restored = S.ActionBallSampler(
        profiles, seed=7103, sampling_mixture=mixture
    )
    restored.load_state_dict(saved)
    actual = [
        _sample(restored, birth, uid=11, epoch=3, levels=levels)
        for _ in range(13)
    ]
    assert actual == expected
    assert [
        (sample.sampling_stratum, sample.frontier_arm)
        for sample in actual
    ] == [
        (sample.sampling_stratum, sample.frontier_arm)
        for sample in expected
    ]
    assert restored.state_dict() == original.state_dict()

    legacy = S.ActionBallSampler(profiles, seed=7103)
    with pytest.raises(ValueError, match="state contract"):
        legacy.load_state_dict(saved)


def test_joint_mixture_cursor_is_independent_per_action_and_batch_history():
    mixture = S.SamplingMixture()
    levels = _levels(**{name: 0.5 for name in S.ARM_KEYS})
    profiles = [_profile(11), _profile(29)]
    with_other_history = S.ActionBallSampler(
        profiles, seed=7106, sampling_mixture=mixture
    )
    clean_action = S.ActionBallSampler(
        profiles, seed=7106, sampling_mixture=mixture
    )
    first_birth = _birth(
        with_other_history, uid=11, epoch=1, levels=levels
    )
    for _ in range(37):
        _sample(
            with_other_history,
            first_birth,
            uid=11,
            epoch=1,
            levels=levels,
        )

    mixed_birth = _birth(
        with_other_history, uid=29, epoch=2, levels=levels
    )
    clean_birth = _birth(
        clean_action, uid=29, epoch=2, levels=levels
    )
    mixed_samples = [
        _sample(
            with_other_history,
            mixed_birth,
            uid=29,
            epoch=2,
            levels=levels,
        )
        for _ in range(15)
    ]
    clean_samples = [
        _sample(
            clean_action,
            clean_birth,
            uid=29,
            epoch=2,
            levels=levels,
        )
        for _ in range(15)
    ]
    assert mixed_birth == clean_birth
    assert mixed_samples == clean_samples
    assert [
        (sample.sampling_stratum, sample.frontier_arm)
        for sample in mixed_samples
    ] == [
        (sample.sampling_stratum, sample.frontier_arm)
        for sample in clean_samples
    ]


def test_sampling_mixture_is_configurable_but_all_strata_must_be_live():
    mixture = S.SamplingMixture(
        center_slots=2,
        interior_slots=1,
        frontier_slots=1,
        interior_level_scale=0.5,
        frontier_band_fraction=0.1,
    )
    assert len(mixture.schedule) == 4
    assert mixture.schedule.count("center") == 2
    assert mixture.schedule.count("interior") == 1
    assert mixture.schedule.count("frontier") == 1
    assert S.SamplingMixture.from_mapping(mixture.as_dict()) == mixture
    with pytest.raises(ValueError, match="center_slots"):
        S.SamplingMixture(center_slots=0)
    with pytest.raises(ValueError, match="must be in"):
        S.SamplingMixture(frontier_band_fraction=0.0)
    with pytest.raises(ValueError, match="must be in"):
        S.SamplingMixture(
            interior_level_scale=0.0,
            frontier_band_fraction=1.0,
        )


def test_zero_width_frontier_fails_before_rng_or_receipt_state_mutates():
    zero = _profile(
        contact_offset_std_initial_m=(0.0, 0.0, 0.0),
        contact_offset_std_max_m=(0.0, 0.0, 0.0),
        time_to_contact_std_lower_initial_s=0.0,
        time_to_contact_std_lower_max_s=0.0,
        time_to_contact_std_upper_initial_s=0.0,
        time_to_contact_std_upper_max_s=0.0,
        incoming_speed_std_initial_mps=0.0,
        incoming_speed_std_max_mps=0.0,
        incoming_direction_cone_deg=0.0,
        spin_magnitude_std_initial_radps=0.0,
        spin_magnitude_std_max_radps=0.0,
        spin_direction_cone_initial_deg=0.0,
        spin_direction_cone_max_deg=0.0,
        base_spawn_std_initial_m=(0.0, 0.0, 0.0),
        base_spawn_std_max_m=(0.0, 0.0, 0.0),
        base_travel_std_initial_m=(0.0, 0.0, 0.0),
        base_travel_std_max_m=(0.0, 0.0, 0.0),
        landing_aim_std_initial_m=(0.0, 0.0),
        landing_aim_std_max_m=(0.0, 0.0),
    )
    sampler = S.ActionBallSampler(
        [zero], seed=7107, sampling_mixture=S.SamplingMixture()
    )
    levels = _levels(**{name: 1.0 for name in S.ARM_KEYS})
    birth = _birth(sampler, levels=levels)
    for _ in range(3):
        _sample(sampler, birth, levels=levels)
    assert sampler.sample_count == 3
    before = deepcopy(sampler.state_dict())
    with pytest.raises(ValueError, match="no non-zero per-swing arm"):
        _sample(sampler, birth, levels=levels)
    assert sampler.state_dict() == before


def test_level_zero_frontier_uses_initial_birth_and_swing_outer_bands():
    profile = _profile()
    mixture = S.SamplingMixture()
    levels = S.DomainLevels()
    sampler = S.ActionBallSampler(
        [profile],
        seed=7108,
        sampling_mixture=mixture,
        contact_time_step_s=0.02,
    )

    births = [_birth(sampler, levels=levels) for _ in range(4)]
    frontier_birth = births[-1]
    assert frontier_birth.sampling_stratum == "frontier"
    assert frontier_birth.frontier_arm == "base_spawn_x_lower"
    assert frontier_birth.sampling_levels == levels
    assert frontier_birth.base_start_w_m[0] <= (
        profile.base_spawn_center_w_m[0]
        - (1.0 - mixture.frontier_band_fraction)
        * profile.base_spawn_std_lower_initial_m[0]
    )
    sampler.verify_birth_sampling_membership(frontier_birth)

    samples = [
        _sample(sampler, frontier_birth, levels=levels)
        for _ in range(4)
    ]
    frontier_sample = samples[-1]
    assert frontier_sample.sampling_stratum == "frontier"
    assert frontier_sample.frontier_arm == "time_to_contact_lower"
    assert frontier_sample.sampling_levels == levels
    grid = sampler._contact_time_grid_by_action[profile.action_uid]
    assert (
        frontier_sample.time_to_contact_tick
        < grid.center_tick
    )
    sampler.verify_sampling_membership(frontier_sample)
    assert sampler.birth_count == 4
    assert sampler.sample_count == 4


def test_promoted_frontier_preempts_initial_support_fallback():
    profile = _profile()
    mixture = S.SamplingMixture()

    birth_levels = _levels(base_spawn_x_lower=1.0)
    assert S._eligible_birth_frontier_arms(
        profile, birth_levels, mixture
    ) == ("base_spawn_x_lower",)

    swing_levels = _levels(contact_x_lower=1.0)
    assert S._eligible_swing_frontier_arms(
        profile, swing_levels, mixture
    ) == ("contact_x_lower",)


def test_legacy_sampler_remains_opt_in_free_and_marks_domain_receipt():
    levels = _levels(position=0.5)
    sampler = S.ActionBallSampler([_profile()], seed=7104)
    birth = _birth(sampler, levels=levels)
    sample = _sample(sampler, birth, levels=levels)
    assert sampler.sampling_mixture is None
    assert sample.sampling_mixture is None
    assert sample.sampling_stratum == "domain"
    assert sample.sampling_levels == levels
    assert sample.frontier_arm is None
    sampler.verify_sampling_membership(sample.to_receipt())


def test_base_birth_mixture_is_exact_20_60_20_and_forces_all_four_sides():
    mixture = S.SamplingMixture()
    levels = _levels(**{name: 1.0 for name in S.ARM_KEYS})
    sampler = S.ActionBallSampler(
        [_profile()], seed=7201, sampling_mixture=mixture
    )

    # Reserving a birth is the proposal-accounting boundary.  These receipts
    # are deliberately discarded as though every downstream solver rejected;
    # the pre-rejection stratum denominator must still be exact.
    births = [_birth(sampler, levels=levels) for _ in range(100)]
    counts = {
        name: sum(birth.sampling_stratum == name for birth in births)
        for name in ("center", "interior", "frontier")
    }
    assert counts == {"center": 20, "interior": 60, "frontier": 20}
    assert sampler.birth_count == 100

    expected_arms = (
        "base_spawn_x_lower",
        "base_spawn_x_upper",
        "base_spawn_y_lower",
        "base_spawn_y_upper",
    )
    frontier = [
        birth for birth in births
        if birth.sampling_stratum == "frontier"
    ]
    assert [birth.frontier_arm for birth in frontier[:4]] == list(
        expected_arms
    )
    assert {
        birth.frontier_arm for birth in frontier
    } == set(expected_arms)
    for birth in births:
        sampler.verify_birth_sampling_membership(
            birth.to_identity_receipt()
        )
        receipt = birth.to_receipt()
        assert receipt["sampling"]["stratum"] == birth.sampling_stratum
        assert (
            receipt["sampling"]["effective_levels"]
            == birth.sampling_levels.as_dict()
        )


def test_inactive_zero_width_birth_scope_stays_at_center():
    profile = _profile(
        base_spawn_std_lower_initial_m=(0.0, 0.0, 0.0),
        base_spawn_std_lower_max_m=(0.0, 0.0, 0.0),
        base_spawn_std_upper_initial_m=(0.0, 0.0, 0.0),
        base_spawn_std_upper_max_m=(0.0, 0.0, 0.0),
    )
    sampler = S.ActionBallSampler(
        [profile], seed=72011, sampling_mixture=S.SamplingMixture()
    )

    # The 20/60/20 schedule requests twenty frontier proposals in this
    # batch.  An intentionally inactive birth scope cannot have a physical
    # frontier, so those quotas remain at the exact center while the cursor
    # still advances.
    births = [_birth(sampler, levels=S.DomainLevels()) for _ in range(100)]

    assert [birth.sampling_stratum for birth in births].count(
        "frontier"
    ) == 0
    assert [birth.sampling_stratum for birth in births].count(
        "interior"
    ) == 60
    assert [birth.sampling_stratum for birth in births].count(
        "center"
    ) == 40
    assert all(
        birth.base_start_w_m == profile.base_spawn_center_w_m
        for birth in births
    )
    assert sampler.birth_count == 100


def test_swing_receipt_binds_actual_birth_stratum_not_unused_spawn_latent():
    mixture = S.SamplingMixture()
    levels = _levels(**{name: 1.0 for name in S.ARM_KEYS})
    sampler = S.ActionBallSampler(
        [_profile()], seed=7202, sampling_mixture=mixture
    )
    births = [_birth(sampler, levels=levels) for _ in range(4)]
    frontier_birth = births[-1]
    assert frontier_birth.sampling_stratum == "frontier"

    sample = _sample(sampler, frontier_birth, levels=levels)
    assert sample.birth_index == frontier_birth.birth_index
    assert (
        sample.birth_sampling_stratum
        == frontier_birth.sampling_stratum
    )
    assert sample.birth_sampling_levels == frontier_birth.sampling_levels
    assert sample.birth_frontier_arm == frontier_birth.frontier_arm
    assert sample.base_spawn_latent_w_m == frontier_birth.base_start_w_m
    identity = sample.to_identity_receipt()
    assert identity["birth_index"] == frontier_birth.birth_index
    assert (
        identity["birth_sampling_levels"]
        == frontier_birth.sampling_levels.as_dict()
    )
    sampler.assert_issued_sample(identity)

    before = deepcopy(sampler.state_dict())
    forged = replace(
        sample,
        birth_sampling_levels=S.DomainLevels(),
    )
    forged = replace(
        forged,
        sample_id=S._sha256_json(forged.identity_payload()),
    )
    with pytest.raises(ValueError, match="birth sampling metadata"):
        sampler.assert_issued_sample(forged)
    assert sampler.state_dict() == before


def test_joint_rho_scales_full_physical_width_and_cannot_overlap_frontier():
    profile = _profile()
    mixture = S.SamplingMixture()
    levels = _levels(**{name: 1.0 for name in S.ARM_KEYS})

    _, birth_interior, _ = S._sampling_plan(
        profile=profile,
        levels=levels,
        mixture=mixture,
        proposal_index=0,
        scope="birth",
    )
    _, swing_interior, _ = S._sampling_plan(
        profile=profile,
        levels=levels,
        mixture=mixture,
        proposal_index=0,
        scope="swing",
    )
    for arm, effective in (
        ("base_spawn_x_lower", birth_interior),
        ("time_to_contact_lower", swing_interior),
        ("contact_y_upper", swing_interior),
    ):
        initial = S._arm_physical_width(
            profile, S.DomainLevels(), arm
        )
        full = S._arm_physical_width(profile, levels, arm)
        actual = S._arm_physical_width(profile, effective, arm)
        assert actual == pytest.approx(
            max(initial, mixture.interior_level_scale * full),
            abs=1.0e-12,
        )

    _, birth_frontier_levels, birth_arm = S._sampling_plan(
        profile=profile,
        levels=levels,
        mixture=mixture,
        proposal_index=3,
        scope="birth",
    )
    _, swing_frontier_levels, swing_arm = S._sampling_plan(
        profile=profile,
        levels=levels,
        mixture=mixture,
        proposal_index=3,
        scope="swing",
    )
    assert birth_arm == "base_spawn_x_lower"
    assert swing_arm == "time_to_contact_lower"
    for arm, frontier_levels, interior_levels in (
        (birth_arm, birth_frontier_levels, birth_interior),
        (swing_arm, swing_frontier_levels, swing_interior),
    ):
        full = S._arm_physical_width(profile, frontier_levels, arm)
        interior = S._arm_physical_width(
            profile, interior_levels, arm
        )
        frontier_start = (
            1.0 - mixture.frontier_band_fraction
        ) * full
        assert interior <= frontier_start + 1.0e-12

    with pytest.raises(ValueError, match="cannot overlap"):
        S.SamplingMixture(
            interior_level_scale=0.81,
            frontier_band_fraction=0.2,
        )


def test_birth_mixture_resume_and_action_cursors_are_independent_bit_exact():
    mixture = S.SamplingMixture()
    levels = _levels(**{name: 0.9 for name in S.ARM_KEYS})
    profiles = [_profile(11), _profile(29)]
    original = S.ActionBallSampler(
        profiles, seed=7203, sampling_mixture=mixture
    )
    births = {11: [], 29: []}
    for uid in (11, 29, 11, 11, 29, 11, 29):
        birth = _birth(
            original, uid=uid, epoch=4, levels=levels
        )
        births[uid].append(birth)
        if len(births[uid]) % 2 == 0:
            _sample(
                original,
                birth,
                uid=uid,
                epoch=4,
                levels=levels,
            )
    saved = deepcopy(original.state_dict())

    expected_births = []
    expected_samples = []
    latest = {uid: births[uid][-1] for uid in births}
    for uid in (29, 11, 29, 11, 11, 29, 29, 11):
        birth = _birth(
            original, uid=uid, epoch=4, levels=levels
        )
        latest[uid] = birth
        expected_births.append(birth)
        expected_samples.append(
            _sample(
                original,
                birth,
                uid=uid,
                epoch=4,
                levels=levels,
            )
        )

    restored = S.ActionBallSampler(
        profiles, seed=7203, sampling_mixture=mixture
    )
    restored.load_state_dict(saved)
    actual_births = []
    actual_samples = []
    for uid in (29, 11, 29, 11, 11, 29, 29, 11):
        birth = _birth(
            restored, uid=uid, epoch=4, levels=levels
        )
        actual_births.append(birth)
        actual_samples.append(
            _sample(
                restored,
                birth,
                uid=uid,
                epoch=4,
                levels=levels,
            )
        )
    assert actual_births == expected_births
    assert actual_samples == expected_samples
    assert restored.state_dict() == original.state_dict()

    mixed = S.ActionBallSampler(
        profiles, seed=7204, sampling_mixture=mixture
    )
    clean = S.ActionBallSampler(
        profiles, seed=7204, sampling_mixture=mixture
    )
    for _ in range(37):
        _birth(mixed, uid=11, levels=levels)
    assert [
        _birth(mixed, uid=29, levels=levels) for _ in range(12)
    ] == [
        _birth(clean, uid=29, levels=levels) for _ in range(12)
    ]


def test_birth_mixture_compaction_keeps_absolute_cursor_and_replays_suffix():
    mixture = S.SamplingMixture()
    levels = _levels(**{name: 1.0 for name in S.ARM_KEYS})
    sampler = S.ActionBallSampler(
        [_profile()], seed=7208, sampling_mixture=mixture
    )
    births = []
    for _ in range(8):
        birth = _birth(sampler, levels=levels)
        births.append(birth)
        _sample(sampler, birth, levels=levels)
    sampler.compact_retired_prefix(
        [
            S.SamplerRetirePrefixBarrier(
                action_uid=101,
                retire_birth_through_inclusive=2,
                retire_sample_through_inclusive=2,
                expected_birth_highwater=(
                    sampler.birth_highwater_for(101)
                ),
                expected_sample_highwater=(
                    sampler.sample_highwater_for(101)
                ),
                expected_assignment_head_sha256=(
                    sampler.assignment_head_for(101)
                ),
            )
        ]
    )
    saved = deepcopy(sampler.state_dict())
    restored = S.ActionBallSampler(
        [_profile()], seed=7208, sampling_mixture=mixture
    )
    restored.load_state_dict(saved)
    assert restored.state_dict() == saved
    assert restored.retired_prefix_for(101) == (3, 3)
    for birth in births[3:]:
        restored.verify_birth_sampling_membership(birth)
    expected = [_birth(sampler, levels=levels) for _ in range(7)]
    actual = [_birth(restored, levels=levels) for _ in range(7)]
    assert actual == expected


def test_birth_mixture_tamper_is_atomic_and_inactive_birth_is_center():
    mixture = S.SamplingMixture()
    levels = _levels(**{name: 1.0 for name in S.ARM_KEYS})
    sampler = S.ActionBallSampler(
        [_profile()], seed=7205, sampling_mixture=mixture
    )
    for _ in range(5):
        _birth(sampler, levels=levels)
    before = deepcopy(sampler.state_dict())
    tampered = deepcopy(before)
    tampered["issued_births"]["101"][3][
        "sampling_stratum"
    ] = "center"
    payload = {
        key: value
        for key, value in tampered.items()
        if key != "integrity_sha256"
    }
    tampered["integrity_sha256"] = _integrity(payload)
    with pytest.raises(ValueError, match="sampling plan mismatch"):
        sampler.load_state_dict(tampered)
    assert sampler.state_dict() == before

    zero_spawn = _profile(
        base_spawn_std_initial_m=(0.0, 0.0, 0.0),
        base_spawn_std_max_m=(0.0, 0.0, 0.0),
    )
    inactive = S.ActionBallSampler(
        [zero_spawn], seed=7206, sampling_mixture=mixture
    )
    for _ in range(3):
        _birth(inactive, levels=levels)
    frontier_quota = _birth(inactive, levels=levels)
    assert frontier_quota.sampling_stratum == "center"
    assert frontier_quota.frontier_arm is None
    assert frontier_quota.base_start_w_m == zero_spawn.base_spawn_center_w_m
    assert inactive.birth_count == 4


def test_legacy_birth_and_sample_identity_shapes_remain_mixture_free():
    sampler = S.ActionBallSampler([_profile()], seed=7207)
    levels = _levels(base_spawn=0.6, position=0.4)
    birth = _birth(sampler, levels=levels)
    sample = _sample(sampler, birth, levels=levels)
    assert set(birth.to_state_dict()) == set(S._BIRTH_STATE_KEYS)
    assert "sampling" not in birth.to_receipt()
    identity = sample.to_identity_receipt()
    assert set(identity) == {"sample_id", *S._SAMPLE_IDENTITY_KEYS}
    assert "birth_index" not in identity
    assert "birth_sampling_stratum" not in identity
    sampler.assert_issued_birth(birth.to_identity_receipt())
    sampler.assert_issued_sample(identity)


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


def _frozen_proposal(
    index,
    *,
    seed=7001,
    selected_arm="contact_y_upper",
    levels=None,
):
    mixture = S.SamplingMixture()
    stratum = mixture.stratum_for(index)
    return S.sample_frozen_evaluation_proposal(
        _profile(),
        evaluation_seed=seed,
        external_sample_index=index,
        external_birth_index=index,
        domain_epoch=9,
        domain_levels=(
            _levels(
                position=1.0,
                speed=1.0,
                time_to_contact=1.0,
                base_spawn=1.0,
            )
            if levels is None
            else levels
        ),
        rho=0.6,
        sampling_stratum=stratum,
        selected_arm=(
            selected_arm if stratum == "frontier" else None
        ),
        base_yaw_rad=0.17,
        policy_dt_s=0.02,
    )


def test_frozen_evaluation_sampler_is_random_access_and_training_isolated():
    live = S.ActionBallSampler(
        (_profile(),),
        seed=99,
        sampling_mixture=S.SamplingMixture(),
        contact_time_step_s=0.02,
    )
    before = deepcopy(live.state_dict())
    first = _frozen_proposal(8, seed=8008)
    second = _frozen_proposal(8, seed=8008)

    assert first == second
    assert live.state_dict() == before
    assert first.external_sample_index == 8
    assert first.external_birth_index == 8
    assert first.birth.draw_start == 0
    assert first.birth.draw_end == S.DRAWS_PER_BIRTH
    assert first.sample.draw_start == S.DRAWS_PER_BIRTH
    assert first.sample.draw_end == (
        S.DRAWS_PER_BIRTH + S.DRAWS_PER_SAMPLE
    )
    assert first.sample.birth_id == first.birth.birth_id
    first.verify()
    assert (
        first.proposal_sampler_contract_sha256
        == S.FROZEN_EVALUATION_PROPOSAL_SAMPLER_CONTRACT_SHA256
        == S.frozen_evaluation_proposal_sampler_contract()["sha256"]
    )
    assert _frozen_proposal(8, seed=8009) != first
    assert _frozen_proposal(13, seed=8008) != first


def test_frozen_evaluation_exact_quota_and_selected_frontier_arm():
    rows = [
        _frozen_proposal(index, seed=9000 + index)
        for index in range(5)
    ]
    assert [row.sampling_stratum for row in rows].count("center") == 1
    assert [row.sampling_stratum for row in rows].count("interior") == 3
    assert [row.sampling_stratum for row in rows].count("frontier") == 1
    frontier = next(
        row for row in rows if row.sampling_stratum == "frontier"
    )
    assert frontier.selected_arm == "contact_y_upper"
    assert frontier.sample.sampling_stratum == "frontier"
    assert frontier.sample.frontier_arm == "contact_y_upper"
    assert (
        frontier.sample.sampling_levels.contact_y_upper
        == frontier.domain_levels.contact_y_upper
    )


def test_frozen_evaluation_base_frontier_is_owned_by_birth_only():
    row = _frozen_proposal(
        3,
        seed=9191,
        selected_arm="base_spawn_y_upper",
    )
    assert row.sampling_stratum == "frontier"
    assert row.birth.sampling_stratum == "frontier"
    assert row.birth.frontier_arm == "base_spawn_y_upper"
    assert (
        row.birth.sampling_levels.base_spawn_y_upper
        == row.domain_levels.base_spawn_y_upper
    )
    assert row.sample.frontier_arm is None
    assert (
        row.sample.sampling_levels.base_spawn_y_upper == 0.0
    )
    row.verify()


def test_frozen_evaluation_all_32_frontier_arms_change_one_arm_only():
    profile = _profile(mobility_mode="move")
    levels = S.DomainLevels(
        **{arm: 1.0 for arm in S.ARM_KEYS}
    )
    base_arms = {
        "base_spawn_x_lower",
        "base_spawn_x_upper",
        "base_spawn_y_lower",
        "base_spawn_y_upper",
    }
    for ordinal, selected_arm in enumerate(S.ARM_KEYS):
        row = S.sample_frozen_evaluation_proposal(
            profile,
            evaluation_seed=12000 + ordinal,
            external_sample_index=3,
            external_birth_index=3,
            domain_epoch=11,
            domain_levels=levels,
            rho=0.6,
            sampling_stratum="frontier",
            selected_arm=selected_arm,
            base_yaw_rad=0.0,
            policy_dt_s=0.02,
        )
        selected_is_birth = selected_arm in base_arms
        assert (
            row.birth_component_stratum,
            row.ball_task_component_stratum,
        ) == (
            ("frontier", "center")
            if selected_is_birth
            else ("center", "frontier")
        )
        for arm in S.ARM_KEYS:
            birth_level = getattr(row.birth.sampling_levels, arm)
            swing_level = getattr(row.sample.sampling_levels, arm)
            expected = 1.0 if arm == selected_arm else 0.0
            assert birth_level + swing_level == expected
            if arm in base_arms:
                assert swing_level == 0.0
            else:
                assert birth_level == 0.0
        row.verify()


def test_frozen_evaluation_refuses_schedule_and_receipt_drift():
    with pytest.raises(ValueError, match="allocation schedule"):
        S.sample_frozen_evaluation_proposal(
            _profile(),
            evaluation_seed=1,
            external_sample_index=3,
            external_birth_index=3,
            domain_epoch=0,
            domain_levels=_levels(position=1.0),
            rho=0.6,
            sampling_stratum="interior",
            selected_arm=None,
            base_yaw_rad=0.0,
            policy_dt_s=0.02,
        )
    row = _frozen_proposal(3)
    with pytest.raises(ValueError, match="receipt SHA"):
        replace(
            row,
            proposal_receipt_sha256="0" * 64,
        ).verify()
    with pytest.raises(ValueError, match="policy_dt_s"):
        S.sample_frozen_evaluation_proposal(
            _profile(),
            evaluation_seed=1,
            external_sample_index=1,
            external_birth_index=1,
            domain_epoch=0,
            domain_levels=_levels(position=1.0),
            rho=0.6,
            sampling_stratum="center",
            selected_arm=None,
            base_yaw_rad=0.0,
            policy_dt_s=0.0,
        )
