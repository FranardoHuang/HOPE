"""Dependency-light contracts for the strict native MuJoCo A211/C211 ABI."""

from __future__ import annotations

from collections import OrderedDict
import hashlib
from pathlib import Path

import pytest

# This host's macOS Conda build aborts in a later Torch linear kernel when
# NumPy/OpenBLAS initializes first in the same pytest process.  Import Torch
# first when present; the ABI itself remains Torch-independent.
pytest.importorskip("torch")
import numpy as np

from hope_training.whole_body_tracking.mujoco_native import action_ball_211_abi as abi


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_mirrored_isaac_source_pins_reopen_exact_current_authorities():
    root = Path(abi.__file__).resolve().parents[3]
    source_root = (
        root
        / "hope_training/whole_body_tracking/source/whole_body_tracking/"
        "whole_body_tracking/tasks/tracking"
    )
    assert _file_sha256(source_root / "action_ball_a211_trainability.py") == (
        abi.A211_SOURCE_SHA256
    )
    assert _file_sha256(source_root / "action_ball_c211_trainability.py") == (
        abi.C211_SOURCE_SHA256
    )
    leaf_root = root / "hope_training/whole_body_tracking/cfg/task"
    assert _file_sha256(
        leaf_root / "HOPEPingPongActionBallA211VendorV2N1Learnability.yaml"
    ) == abi.A211_TASK_LEAF_SHA256
    assert _file_sha256(
        leaf_root / "HOPEPingPongActionBallC211VendorV2N1Learnability.yaml"
    ) == abi.C211_TASK_LEAF_SHA256


def _groups(lane: abi.ObservationLane, validity=(0.0, 1.0)):
    rows = len(validity)
    result = OrderedDict()
    for index, field in enumerate(lane.fields):
        result[field.name] = np.full((rows, field.width), index + 1.0, dtype=np.float32)
    result["task_valid"] = np.asarray(validity, dtype=np.float32).reshape(rows, 1)
    return result


def _authorities(character="a"):
    return abi.ObservationAuthorities(
        plant_observation_sha256=character * 64,
        measured_mimic_sha256="b" * 64,
        task_question_sha256="c" * 64,
    )


def test_a211_c211_exact_width_order_source_and_normalizer_identities():
    expected_actor_prefix = (
        ("actual_base_pose_lin_vel_world", 12),
        ("base_ang_vel_body", 3),
        ("joint_pos", 31),
        ("joint_vel", 31),
        ("actions", 31),
        ("racket_site_achieved_now_heading", 9),
        ("teacher_joint_pos", 31),
        ("teacher_joint_vel", 31),
        ("racket_site_teacher_now_heading", 9),
        ("racket_site_teacher_at_reference_hit_heading", 9),
    )
    expected_critic_prefix = (
        ("command", 62),
        ("motion_anchor_pos_b", 3),
        ("motion_anchor_ori_b", 6),
        ("body_pos", 42),
        ("body_ori", 84),
        ("base_lin_vel", 3),
        ("base_ang_vel", 3),
        ("joint_pos", 31),
        ("joint_vel", 31),
        ("actions", 31),
        ("racket_site_teacher_at_reference_hit_heading", 9),
    )
    for profile in (abi.A211_PROFILE, abi.C211_PROFILE):
        assert profile.actor.width == 211
        assert profile.critic.width == 319
        assert profile.actor.layout[:10] == expected_actor_prefix
        assert profile.critic.layout[:11] == expected_critic_prefix
        critic_purposes = {field.name: field.purpose for field in profile.critic.fields}
        critic_authorities = {
            field.name: field.authority for field in profile.critic.fields
        }
        assert critic_purposes["command"] == "privileged_mimic"
        assert critic_purposes["motion_anchor_pos_b"] == "plant_mimic_relation"
        assert critic_authorities["motion_anchor_pos_b"] == "plant+mimic"
        assert critic_purposes["body_pos"] == "plant_state"
        assert critic_authorities["body_pos"] == "plant"
        assert profile.actor.layout[-4:] == (
            ("desired_base_xy_world", 2),
            ("time_to_contact", 1),
            ("time_to_teacher_start", 1),
            ("task_valid", 1),
        )
        assert profile.critic.layout[-4:] == profile.actor.layout[-4:]
        assert profile.actor.task_mask_indices == tuple(range(197, 210))
        assert profile.critic.task_mask_indices == tuple(range(305, 318))
        assert profile.actor.task_valid_index == 210
        assert profile.critic.task_valid_index == 318
        assert profile.actor_normalizer_identity.endswith("actor_norm_v2")
        assert profile.critic_normalizer_identity.endswith("critic_norm_v1")
    assert abi.A211_PROFILE.actor.layout[10:13] == (
        ("task_desired_contact_position_heading", 3),
        ("task_desired_contact_velocity_heading", 3),
        ("task_desired_contact_face_heading", 3),
    )
    assert abi.C211_PROFILE.actor.layout[10:13] == (
        ("incoming_ball_contact_position_heading", 3),
        ("incoming_ball_contact_velocity_heading", 3),
        ("incoming_ball_contact_spin_heading", 3),
    )
    assert (
        abi.A211_PROFILE.observation_contract_sha256
        != abi.C211_PROFILE.observation_contract_sha256
    )
    assert (
        abi.A211_PROFILE.actor_normalizer_identity
        != abi.C211_PROFILE.actor_normalizer_identity
    )
    assert (
        abi.A211_PROFILE.critic_normalizer_identity
        != abi.C211_PROFILE.critic_normalizer_identity
    )


@pytest.mark.parametrize("profile", (abi.A211_PROFILE, abi.C211_PROFILE))
def test_wait_mask_is_derived_and_preserves_all_non_task_groups(profile):
    actor_groups = _groups(profile.actor)
    critic_groups = _groups(profile.critic)
    actor_unmasked = np.concatenate(list(actor_groups.values()), axis=1)
    critic_unmasked = np.concatenate(list(critic_groups.values()), axis=1)

    actor, critic = abi.flatten_profile_groups(
        profile,
        actor_groups=actor_groups,
        critic_groups=critic_groups,
        task_valid=np.asarray([0, 1], dtype=np.int64),
        authorities=_authorities(),
    )

    assert actor.shape == (2, 211)
    assert critic.shape == (2, 319)
    assert np.all(actor[0, profile.actor.task_mask_indices] == 0.0)
    assert np.all(critic[0, profile.critic.task_mask_indices] == 0.0)
    assert np.array_equal(
        actor[1, profile.actor.task_mask_indices],
        actor_unmasked[1, profile.actor.task_mask_indices],
    )
    assert np.array_equal(
        critic[1, profile.critic.task_mask_indices],
        critic_unmasked[1, profile.critic.task_mask_indices],
    )
    actor_keep = sorted(
        set(range(profile.actor.width)) - set(profile.actor.task_mask_indices)
    )
    critic_keep = sorted(
        set(range(profile.critic.width)) - set(profile.critic.task_mask_indices)
    )
    assert np.array_equal(actor[0, actor_keep], actor_unmasked[0, actor_keep])
    assert np.array_equal(critic[0, critic_keep], critic_unmasked[0, critic_keep])
    assert actor[0, profile.actor.task_valid_index] == 0.0
    assert critic[0, profile.critic.task_valid_index] == 0.0
    assert actor[1, profile.actor.task_valid_index] == 1.0
    assert critic[1, profile.critic.task_valid_index] == 1.0


@pytest.mark.parametrize("profile", (abi.A211_PROFILE, abi.C211_PROFILE))
def test_missing_reordered_wrong_width_and_nonfinite_groups_never_pad(profile):
    valid = np.asarray([1], dtype=np.int64)
    groups = _groups(profile.actor, validity=(1.0,))

    missing = OrderedDict(groups)
    missing.pop(next(iter(missing)))
    with pytest.raises(abi.ActionBall211ABIError, match="ordered fields differ"):
        abi.flatten_lane_groups(profile.actor, missing, task_valid=valid)

    reordered = OrderedDict(reversed(tuple(groups.items())))
    with pytest.raises(abi.ActionBall211ABIError, match="ordered fields differ"):
        abi.flatten_lane_groups(profile.actor, reordered, task_valid=valid)

    wrong = OrderedDict(groups)
    name = profile.actor.fields[0].name
    wrong[name] = np.zeros((1, profile.actor.fields[0].width + 1), dtype=np.float32)
    with pytest.raises(abi.ActionBall211ABIError, match="must have shape"):
        abi.flatten_lane_groups(profile.actor, wrong, task_valid=valid)

    nonfinite = OrderedDict(groups)
    nonfinite[name] = nonfinite[name].copy()
    nonfinite[name][0, 0] = np.nan
    with pytest.raises(abi.ActionBall211ABIError, match="non-finite"):
        abi.flatten_lane_groups(profile.actor, nonfinite, task_valid=valid)


def test_task_valid_is_atomic_across_sideband_actor_and_critic():
    profile = abi.C211_PROFILE
    actor_groups = _groups(profile.actor)
    critic_groups = _groups(profile.critic)
    actor_groups["task_valid"][0, 0] = 1.0
    with pytest.raises(abi.ActionBall211ABIError, match="atomic sideband"):
        abi.flatten_profile_groups(
            profile,
            actor_groups=actor_groups,
            critic_groups=critic_groups,
            task_valid=np.asarray([0, 1]),
            authorities=_authorities(),
        )


def test_absent_plant_or_mimic_fails_before_any_tensor_materialization():
    profile = abi.A211_PROFILE
    receipt = abi.construction_receipt(profile, num_envs=4096)
    assert receipt["actor_shape"] == [4096, 211]
    assert receipt["critic_shape"] == [4096, 319]
    assert receipt["runtime_tensor_materialized"] is False
    assert receipt["runtime_ready"] is False
    assert receipt["blockers"] == [
        "a211_plant_observation_authority_unavailable",
        "a211_full_body_measured_mimic_authority_unavailable",
        "a211_task_question_authority_unavailable",
        "a211_runtime_providers_not_reopened_by_abi_construction_only",
    ]
    assert receipt["native_real_vecenv_adapter_available"] is True
    with pytest.raises(abi.ActionBall211AuthorityBlocked) as caught:
        abi.require_runtime_authorities(
            profile,
            plant_observation_sha256=None,
            measured_mimic_sha256=None,
            task_question_sha256=None,
        )
    assert caught.value.blockers == tuple(receipt["blockers"][:-1])


def test_even_all_digest_strings_cannot_claim_runtime_before_provider_reopen():
    receipt = abi.construction_receipt(
        abi.C211_PROFILE,
        num_envs=1,
        plant_observation_sha256="a" * 64,
        measured_mimic_sha256="b" * 64,
        task_question_sha256="c" * 64,
    )
    assert receipt["source_digests_syntactically_bound"] is True
    assert receipt["runtime_ready"] is False
    assert receipt["runtime_tensor_materialized"] is False
    assert receipt["native_real_vecenv_adapter_available"] is True
    assert receipt["blockers"] == [
        "c211_runtime_providers_not_reopened_by_abi_construction_only"
    ]


def test_flatten_requires_typed_authorities_not_same_width_placeholders():
    profile = abi.C211_PROFILE
    with pytest.raises(abi.ActionBall211AuthorityBlocked):
        abi.flatten_profile_groups(
            profile,
            actor_groups=_groups(profile.actor),
            critic_groups=_groups(profile.critic),
            task_valid=np.asarray([0, 1]),
            authorities=None,
        )


def test_authority_sha_cannot_be_relabelled_from_a_plain_plant_contract():
    with pytest.raises(abi.ActionBall211ABIError, match="provider contract"):
        abi.ObservationAuthorities(
            plant_observation_sha256="a" * 64,
            measured_mimic_sha256="b" * 64,
            task_question_sha256="c" * 64,
            plant_observation_kind="plain_training_contract_file_v1",
        )
