"""Host-only integration tests for manifest-to-sampler profile adaptation."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = (
    ROOT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
)


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(
        name, MODULE_DIR / filename
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


M = _load("action_ball_manifest", "action_ball_manifest.py")
S = _load("action_ball_sampling", "action_ball_sampling.py")
C = _load("action_ball_curriculum", "action_ball_curriculum.py")
A = _load(
    "action_ball_profile_adapter",
    "action_ball_profile_adapter.py",
)


def _curriculum():
    return {
        "min_proposals": 256,
        "min_safe_closed": 256,
        "target_failure_rate": 0.10,
        "failure_band_half_width": 0.025,
        "min_solver_admit_rate": 0.95,
        "min_install_rate": 0.95,
        "min_start_rate": 0.95,
        "min_close_rate": 0.95,
        "max_other_unsafe_rate": 0.02,
        "confidence_z": 1.96,
        "max_center_failures": 8,
    }


def _ball_profile(index):
    return {
        "contact_offset_center_b_yaw_m": [
            0.55,
            -0.12 + 0.001 * index,
            0.82,
        ],
        "contact_offset_std_lower_initial_m": [0.005, 0.01, 0.01],
        "contact_offset_std_lower_max_m": [0.02, 0.12, 0.16],
        "contact_offset_std_upper_initial_m": [0.004, 0.02, 0.01],
        "contact_offset_std_upper_max_m": [0.02, 0.10, 0.15],
        "contact_offset_min_b_yaw_m": [0.45, -0.30, 0.55],
        "contact_offset_max_b_yaw_m": [0.65, 0.35, 1.10],
        "time_to_contact_center_s": 1.20,
        "time_to_contact_std_lower_initial_s": 0.01,
        "time_to_contact_std_lower_max_s": 0.15,
        "time_to_contact_std_upper_initial_s": 0.02,
        "time_to_contact_std_upper_max_s": 0.30,
        "time_to_contact_min_s": 1.05,
        "time_to_contact_max_s": 1.60,
        "incoming_direction_center_b_yaw": [-1.0, 0.0, 0.0],
        "incoming_direction_tangent_u_b_yaw": [0.0, 1.0, 0.0],
        "incoming_direction_tangent_v_b_yaw": [0.0, 0.0, -1.0],
        "incoming_direction_tangent_u_neg_initial_deg": 0.5,
        "incoming_direction_tangent_u_neg_max_deg": 10.0,
        "incoming_direction_tangent_u_pos_initial_deg": 0.6,
        "incoming_direction_tangent_u_pos_max_deg": 8.0,
        "incoming_direction_tangent_v_neg_initial_deg": 0.7,
        "incoming_direction_tangent_v_neg_max_deg": 6.0,
        "incoming_direction_tangent_v_pos_initial_deg": 0.8,
        "incoming_direction_tangent_v_pos_max_deg": 7.0,
        "incoming_inbound_axis_b_yaw": [-1.0, 0.0, 0.0],
        "incoming_inbound_min_cosine": 0.20,
        "incoming_speed_center_mps": 4.0,
        "incoming_speed_std_lower_initial_mps": 0.05,
        "incoming_speed_std_lower_max_mps": 1.2,
        "incoming_speed_std_upper_initial_mps": 0.06,
        "incoming_speed_std_upper_max_mps": 1.0,
        "incoming_speed_min_mps": 1.6,
        "incoming_speed_max_mps": 7.0,
        "spin_direction_center_b_yaw": [0.0, 1.0, 0.0],
        "spin_direction_tangent_u_b_yaw": [0.0, 0.0, 1.0],
        "spin_direction_tangent_v_b_yaw": [1.0, 0.0, 0.0],
        "spin_direction_tangent_u_neg_initial_deg": 0.0,
        "spin_direction_tangent_u_neg_max_deg": 35.0,
        "spin_direction_tangent_u_pos_initial_deg": 0.0,
        "spin_direction_tangent_u_pos_max_deg": 30.0,
        "spin_direction_tangent_v_neg_initial_deg": 0.0,
        "spin_direction_tangent_v_neg_max_deg": 25.0,
        "spin_direction_tangent_v_pos_initial_deg": 0.0,
        "spin_direction_tangent_v_pos_max_deg": 20.0,
        "spin_magnitude_center_radps": 15.0,
        "spin_magnitude_std_lower_initial_radps": 0.2,
        "spin_magnitude_std_lower_max_radps": 8.0,
        "spin_magnitude_std_upper_initial_radps": 0.3,
        "spin_magnitude_std_upper_max_radps": 9.0,
        "spin_magnitude_min_radps": 0.0,
        "spin_magnitude_max_radps": 40.0,
        "base_spawn_center_w_xy_m": [-0.10, 0.05],
        "base_spawn_std_lower_initial_m": [0.005, 0.005],
        "base_spawn_std_lower_max_m": [0.10, 0.20],
        "base_spawn_std_upper_initial_m": [0.006, 0.007],
        "base_spawn_std_upper_max_m": [0.15, 0.15],
        "base_spawn_min_w_xy_m": [-0.35, -0.30],
        "base_spawn_max_w_xy_m": [0.20, 0.40],
        "base_travel_center_b_yaw_xy_m": [0.04, -0.02],
        "base_travel_std_lower_initial_m": [0.01, 0.01],
        "base_travel_std_lower_max_m": [0.25, 0.20],
        "base_travel_std_upper_initial_m": [0.02, 0.01],
        "base_travel_std_upper_max_m": [0.30, 0.20],
        "base_travel_min_b_yaw_xy_m": [-0.50, -0.40],
        "base_travel_max_b_yaw_xy_m": [0.50, 0.40],
    }


def _action(index):
    action_id = f"action_{index:03d}"
    family = "forehand" if index % 2 == 0 else "backhand"
    motion_sha = hashlib.sha256(f"motion-{index}".encode()).hexdigest()
    return {
        "action_id": action_id,
        "action_uid": M.derive_action_ball_action_uid(
            action_id, family, motion_sha
        ),
        "motion_path": f"motions/action_{index:03d}.npz",
        "motion_sha256": motion_sha,
        "strike_phase": 0.50,
        "reference_t_hit_s": 0.80,
        "reference_t_cycle_s": 1.60,
        "reference_racket_site_speed_mps": 6.0,
        "reaction_margin_s": 0.05,
        "teacher_rate_min": 0.80,
        "teacher_rate_max": 1.20,
        "family": family,
        "mount_normal_sign": -1,
        "ball_profile": _ball_profile(index),
    }


def _document(action_count=5, mobility_mode="no_move"):
    actions = [_action(index) for index in range(action_count)]
    return {
        "schema_version": 3,
        "manifest_id": f"adapter_n{action_count}_{mobility_mode}_v3",
        "mobility_mode": mobility_mode,
        "action_order": [action["action_id"] for action in actions],
        "prototype": {
            "path": "configs/stroke_prototypes_v2.json",
            "sha256": hashlib.sha256(b"prototype").hexdigest(),
            "scope": "full",
        },
        "solver_profile_sha256": hashlib.sha256(b"solver").hexdigest(),
        "physics_profile_sha256": hashlib.sha256(b"physics").hexdigest(),
        "landing_aim": {
            "center_w_xy_m": [2.55, 0.10],
            "std_lower_initial_m": [0.01, 0.02],
            "std_lower_max_m": [0.25, 0.45],
            "std_upper_initial_m": [0.02, 0.01],
            "std_upper_max_m": [0.20, 0.40],
            "min_w_xy_m": [2.10, -0.60],
            "max_w_xy_m": [3.00, 0.60],
        },
        "actions": actions,
        "curriculum": _curriculum(),
        "holdout": {
            "seed": 20260727,
            "samples_per_action": 512,
            "split_id": "adapter_holdout_v1",
        },
        "notes": "",
    }


def _manifest(tmp_path, action_count=5, mobility_mode="no_move"):
    path = tmp_path / f"{mobility_mode}-{action_count}.json"
    path.write_text(
        json.dumps(_document(action_count, mobility_mode)),
        encoding="utf-8",
    )
    return M.load_action_ball_manifest(path).manifest


@pytest.mark.parametrize("action_count", [1, 5, 93])
def test_adapter_preserves_arbitrary_n_manifest_order(
    tmp_path, action_count
):
    manifest = _manifest(tmp_path, action_count)
    bundle = A.adapt_action_ball_manifest(manifest)

    assert bundle.action_order == manifest.action_order
    assert bundle.action_uids == tuple(
        action.action_uid for action in manifest.actions
    )
    assert tuple(profile.action_uid for profile in bundle.profiles) == (
        bundle.action_uids
    )
    assert len(bundle.profiles) == action_count
    assert A.build_sampling_profiles(manifest) == bundle.profiles


def test_every_manifest_field_maps_to_sampling_profile_without_frame_guessing(
    tmp_path,
):
    manifest = _manifest(tmp_path, 1)
    profile = A.build_sampling_profile(
        manifest, action_id="action_000"
    )
    ball = manifest.actions[0].ball_profile
    aim = manifest.landing_aim

    assert profile.contact_offset_center_b_yaw_m == (
        ball.contact_offset_center_b_yaw_m
    )
    for name in (
        "contact_offset_std_lower_initial_m",
        "contact_offset_std_lower_max_m",
        "contact_offset_std_upper_initial_m",
        "contact_offset_std_upper_max_m",
        "contact_offset_min_b_yaw_m",
        "contact_offset_max_b_yaw_m",
        "time_to_contact_center_s",
        "time_to_contact_std_lower_initial_s",
        "time_to_contact_std_lower_max_s",
        "time_to_contact_std_upper_initial_s",
        "time_to_contact_std_upper_max_s",
        "time_to_contact_min_s",
        "time_to_contact_max_s",
        "incoming_direction_center_b_yaw",
        "incoming_direction_tangent_u_b_yaw",
        "incoming_direction_tangent_v_b_yaw",
        "incoming_direction_tangent_u_neg_initial_deg",
        "incoming_direction_tangent_u_neg_max_deg",
        "incoming_direction_tangent_u_pos_initial_deg",
        "incoming_direction_tangent_u_pos_max_deg",
        "incoming_direction_tangent_v_neg_initial_deg",
        "incoming_direction_tangent_v_neg_max_deg",
        "incoming_direction_tangent_v_pos_initial_deg",
        "incoming_direction_tangent_v_pos_max_deg",
        "incoming_inbound_axis_b_yaw",
        "incoming_inbound_min_cosine",
        "incoming_speed_center_mps",
        "incoming_speed_std_lower_initial_mps",
        "incoming_speed_std_lower_max_mps",
        "incoming_speed_std_upper_initial_mps",
        "incoming_speed_std_upper_max_mps",
        "incoming_speed_min_mps",
        "incoming_speed_max_mps",
        "spin_direction_center_b_yaw",
        "spin_direction_tangent_u_b_yaw",
        "spin_direction_tangent_v_b_yaw",
        "spin_direction_tangent_u_neg_initial_deg",
        "spin_direction_tangent_u_neg_max_deg",
        "spin_direction_tangent_u_pos_initial_deg",
        "spin_direction_tangent_u_pos_max_deg",
        "spin_direction_tangent_v_neg_initial_deg",
        "spin_direction_tangent_v_neg_max_deg",
        "spin_direction_tangent_v_pos_initial_deg",
        "spin_direction_tangent_v_pos_max_deg",
        "spin_magnitude_center_radps",
        "spin_magnitude_std_lower_initial_radps",
        "spin_magnitude_std_lower_max_radps",
        "spin_magnitude_std_upper_initial_radps",
        "spin_magnitude_std_upper_max_radps",
        "spin_magnitude_min_radps",
        "spin_magnitude_max_radps",
    ):
        assert getattr(profile, name) == getattr(ball, name)
    assert profile.landing_aim_center_w_xy_m == aim.center_w_xy_m
    assert profile.landing_aim_std_lower_initial_m == aim.std_lower_initial_m
    assert profile.landing_aim_std_lower_max_m == aim.std_lower_max_m
    assert profile.landing_aim_std_upper_initial_m == aim.std_upper_initial_m
    assert profile.landing_aim_std_upper_max_m == aim.std_upper_max_m
    assert profile.landing_aim_min_w_xy_m == aim.min_w_xy_m
    assert profile.landing_aim_max_w_xy_m == aim.max_w_xy_m
    action = manifest.actions[0]
    for name in (
        "reference_t_hit_s",
        "reference_t_cycle_s",
        "reference_racket_site_speed_mps",
        "reaction_margin_s",
        "teacher_rate_min",
        "teacher_rate_max",
    ):
        assert getattr(profile, name) == getattr(action, name)


def test_all_manifest_xy_domains_receive_an_explicit_zero_z(tmp_path):
    profile = A.build_sampling_profiles(_manifest(tmp_path, 1))[0]
    expected_xy = {
        "base_spawn_center_w_m": (-0.10, 0.05),
        "base_spawn_std_lower_initial_m": (0.005, 0.005),
        "base_spawn_std_lower_max_m": (0.10, 0.20),
        "base_spawn_std_upper_initial_m": (0.006, 0.007),
        "base_spawn_std_upper_max_m": (0.15, 0.15),
        "base_spawn_min_w_m": (-0.35, -0.30),
        "base_spawn_max_w_m": (0.20, 0.40),
        "base_travel_center_b_yaw_m": (0.04, -0.02),
        "base_travel_std_lower_initial_m": (0.01, 0.01),
        "base_travel_std_lower_max_m": (0.25, 0.20),
        "base_travel_std_upper_initial_m": (0.02, 0.01),
        "base_travel_std_upper_max_m": (0.30, 0.20),
        "base_travel_min_b_yaw_m": (-0.50, -0.40),
        "base_travel_max_b_yaw_m": (0.50, 0.40),
    }
    for name, xy in expected_xy.items():
        value = getattr(profile, name)
        assert value[:2] == xy
        assert value[2] == 0.0
        assert type(value[2]) is float


def test_mobility_is_manifest_bound_while_latent_travel_is_comparable(
    tmp_path,
):
    no_move_manifest = _manifest(tmp_path, 1, "no_move")
    move_manifest = _manifest(tmp_path, 1, "move")
    no_move = A.adapt_action_ball_manifest(no_move_manifest)
    move = A.adapt_action_ball_manifest(move_manifest)
    no_move_profile = no_move.profiles[0]
    move_profile = move.profiles[0]

    assert no_move_profile.mobility_mode == "no_move"
    assert move_profile.mobility_mode == "move"
    assert no_move.manifest_canonical_sha256 != (
        move.manifest_canonical_sha256
    )
    assert no_move_profile.sha256 != move_profile.sha256
    assert no_move.contract_sha256 != move.contract_sha256
    no_move_values = no_move_profile.as_dict()
    move_values = move_profile.as_dict()
    del no_move_values["mobility_mode"]
    del move_values["mobility_mode"]
    assert no_move_values == move_values
    assert "mobility_mode" not in inspect.signature(
        S.ActionBallSampler.sample
    ).parameters
    assert "mobility_mode" not in inspect.signature(
        S.ActionBallSampler.reserve_birth
    ).parameters


def test_bundle_is_checkpoint_safe_and_action_lookup_is_strict(tmp_path):
    manifest = _manifest(tmp_path, 5)
    bundle = A.adapt_action_ball_manifest(manifest)
    contract = bundle.to_contract()

    assert bundle.manifest_canonical_sha256 == (
        M.canonical_manifest_sha256(manifest)
    )
    assert contract["manifest_canonical_sha256"] == (
        bundle.manifest_canonical_sha256
    )
    assert [row["action_id"] for row in contract["profiles"]] == list(
        manifest.action_order
    )
    assert [row["sampling_profile_sha256"] for row in contract["profiles"]] == (
        list(bundle.profile_sha256)
    )
    assert len(bundle.contract_sha256) == 64
    assert bundle.profile_for_action_id("action_003") == bundle.profiles[3]
    with pytest.raises(ValueError, match="unknown action_id"):
        bundle.profile_for_action_id("missing")
    with pytest.raises(TypeError, match="must be a string"):
        bundle.profile_for_action_id(1)
    with pytest.raises(ValueError, match="64 lowercase"):
        replace(bundle, manifest_canonical_sha256="A" * 64)
    with pytest.raises(ValueError, match="columns must align"):
        replace(bundle, action_uids=bundle.action_uids[:-1])


@pytest.mark.parametrize("action_count", [1, 5, 93])
def test_schema_v3_profile_adaptation_is_lossless_for_arbitrary_n(
    tmp_path, action_count
):
    manifest = _manifest(tmp_path, action_count)
    bundle = A.adapt_action_ball_manifest(manifest)
    assert len(bundle.profiles) == action_count
    assert all(profile.reference_t_hit_s == 0.8 for profile in bundle.profiles)
    assert all(
        profile.incoming_speed_min_mps
        == 0.4 * profile.incoming_speed_center_mps
        for profile in bundle.profiles
    )


def test_curriculum_config_adapter_is_exact_and_has_no_runtime_defaults(
    tmp_path,
):
    manifest = _manifest(tmp_path, 1)
    config = A.build_curriculum_config(manifest)
    assert isinstance(config, C.BallCurriculumConfig)
    assert config.as_dict() == manifest.curriculum.to_mapping()
    assert config.min_proposals == 256
    assert config.min_safe_closed == 256
    assert config.min_solver_admit_rate == 0.95
    assert config.min_install_rate == 0.95
    assert config.min_start_rate == 0.95
    assert config.min_close_rate == 0.95
    assert config.max_other_unsafe_rate == 0.02
    assert config.max_center_failures == 8


def test_adapter_revalidates_directly_constructed_dataclass_bypasses(
    tmp_path,
):
    manifest = _manifest(tmp_path, 1)
    action = manifest.actions[0]
    malformed_profile = replace(
        action.ball_profile,
        base_spawn_std_lower_initial_m=(0.1, 0.2, 0.3),
    )
    malformed_action = replace(action, ball_profile=malformed_profile)
    malformed_manifest = replace(
        manifest, actions=(malformed_action,)
    )
    with pytest.raises(ValueError, match="exactly 2 numbers"):
        A.adapt_action_ball_manifest(malformed_manifest)

    wrong_unit = replace(
        action.ball_profile,
        incoming_direction_center_b_yaw=(-2.0, 0.0, 0.0),
    )
    wrong_action = replace(action, ball_profile=wrong_unit)
    with pytest.raises(ValueError, match="unit length"):
        A.adapt_action_ball_manifest(
            replace(manifest, actions=(wrong_action,))
        )


def test_adapted_profiles_drive_sampler_aim_axis_and_no_move_birth(tmp_path):
    manifest = _manifest(tmp_path, 1, "no_move")
    profile = A.build_sampling_profiles(manifest)[0]
    sampler = S.ActionBallSampler((profile,), seed=20260727)
    levels = S.DomainLevels(
        landing_aim_x_lower=1.0,
        landing_aim_x_upper=1.0,
        landing_aim_y_lower=1.0,
        landing_aim_y_upper=1.0,
        base_spawn_x_lower=1.0,
        base_spawn_x_upper=1.0,
        base_spawn_y_lower=1.0,
        base_spawn_y_upper=1.0,
        base_travel_x_lower=1.0,
        base_travel_x_upper=1.0,
        base_travel_y_lower=1.0,
        base_travel_y_upper=1.0,
    )
    birth = sampler.reserve_birth(
        action_uid=profile.action_uid,
        domain_epoch=0,
        levels=levels,
    )
    sample = sampler.sample(
        birth=birth,
        action_uid=profile.action_uid,
        domain_epoch=0,
        levels=levels,
    )

    assert sample.domain_levels.landing_aim_x_lower == 1.0
    assert 2.10 <= sample.landing_aim_w_xy_m[0] <= 3.00
    assert -0.60 <= sample.landing_aim_w_xy_m[1] <= 0.60
    assert birth.base_start_w_m[2] == 0.0
    assert sample.base_goal_w_m == birth.base_start_w_m
    assert sample.base_travel_latent_b_yaw_m != (0.0, 0.0, 0.0)
