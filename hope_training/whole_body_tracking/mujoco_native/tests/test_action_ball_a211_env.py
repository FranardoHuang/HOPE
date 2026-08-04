"""A211 desired-contact authority/reward tests independent of MuJoCo install."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from hope_training.whole_body_tracking.mujoco_native import action_ball_211_abi as abi
from hope_training.whole_body_tracking.mujoco_native import action_ball_a211_env as a211
from hope_training.whole_body_tracking.mujoco_native import n1_reward_event_kernel
from hope_training.whole_body_tracking.mujoco_native.tests import (
    test_action_ball_c211_env as support,
)


IMMUTABLE_TAPE = (
    "configs/action_ball_n1_measured_20260803/"
    "fresh_tape_seed0_20260803_take061_robust20n_r4_splitready/"
    "immutable_n1_tape.v1.22052606032f.json"
)
IMMUTABLE_TAPE_SHA256 = (
    "22052606032f74257ce98b5b6be8e8a4f8175848655ce604f50adf4751409e66"
)
MEASURED_MOTION = (
    "assets/motions/chingmu73_measured_v4_20260803/hope_Take_061_unit04_BH.npz"
)
MEASURED_MOTION_SHA256 = (
    "aab1953b9a857d0a7663a92d85fe4de5bd1d991d22249aa3d4d22ce7ef9fdd8e"
)


@pytest.fixture(scope="module")
def authorities():
    task = a211.A211TaskAuthority.load(
        IMMUTABLE_TAPE, expected_file_sha256=IMMUTABLE_TAPE_SHA256
    )
    mimic = a211.MeasuredA211MimicAuthority.load(
        MEASURED_MOTION,
        expected_file_sha256=MEASURED_MOTION_SHA256,
        task=task,
    )
    return task, mimic


def test_current_lm_111_authority_loads_complete_a211_contact_tuple():
    task = a211.A211TaskAuthority.load(
        IMMUTABLE_TAPE, expected_file_sha256=IMMUTABLE_TAPE_SHA256
    )
    assert task.target_recipe == "current_lm"
    assert task.receipt["target_validity_mask"] == [True, True, True]
    assert len(task.desired_contact_position_w_m) == 3
    assert len(task.desired_contact_velocity_w_mps) == 3
    assert math.isclose(
        sum(value * value for value in task.desired_contact_signed_face_w),
        1.0,
        abs_tol=1.0e-5,
    )
    assert task.receipt["task_tuple"]["desired_contact_signed_face_w"] == list(
        task.desired_contact_signed_face_w
    )
    counter = task.receipt["task_tuple"]["counter_rally_task"]
    assert counter["canonical_sha256"] == (
        task.counter_rally_task_canonical_sha256
    )
    assert counter["objective_profile_sha256"] == (
        "7f490a9163fd5f45a2b4538cf711a03ce8d0a01288688897c4d7220d35a505ce"
    )
    assert task.receipt["diagnostic_unauthorized"] is True


def test_a211_authority_rejects_c_outcome_only_recipe():
    with pytest.raises(a211.A211EnvError, match="current_lm.*111"):
        a211.A211TaskAuthority.load(
            IMMUTABLE_TAPE,
            expected_file_sha256=IMMUTABLE_TAPE_SHA256,
            target_recipe="outcome_dense_only",
        )


def _terms(*, valid: bool, t: float, position=(0.0, 0.0, 0.0)):
    return a211.a211_desired_contact_reward_terms(
        task_valid=valid,
        time_to_contact_s=t,
        achieved_position_w_m=position,
        achieved_velocity_w_mps=(0.0, 0.0, 0.0),
        achieved_signed_face_w=(1.0, 0.0, 0.0),
        desired_position_w_m=(0.0, 0.0, 0.0),
        desired_velocity_w_mps=(0.0, 0.0, 0.0),
        desired_signed_face_w=(1.0, 0.0, 0.0),
    )


def test_wait_masks_every_a211_target_channel_to_exact_zero():
    row = _terms(valid=False, t=0.0)
    assert row["post_policy_dt_reward"] == 0.0
    assert row["any_channel_eligible"] is False
    assert all(value["eligible"] is False for value in row["channels"].values())


def test_a211_target_peak_and_split_windows_match_resolved_isaac_leaf():
    peak = _terms(valid=True, t=0.0)
    assert peak["post_policy_dt_reward"] == pytest.approx(0.71875, abs=1.0e-15)
    edge_pos = _terms(valid=True, t=0.02)
    assert all(value["eligible"] for value in edge_pos["channels"].values())
    outside_pos = _terms(valid=True, t=0.04)
    assert outside_pos["channels"]["position"]["eligible"] is False
    assert outside_pos["channels"]["velocity"]["eligible"] is True
    assert outside_pos["channels"]["face"]["eligible"] is True
    edge_wide = _terms(valid=True, t=-0.10)
    assert edge_wide["channels"]["velocity"]["eligible"] is True
    outside = _terms(valid=True, t=0.1000001)
    assert outside["post_policy_dt_reward"] == 0.0


def test_position_channel_tracks_swing_through_not_static_contact_point():
    row = a211.a211_desired_contact_reward_terms(
        task_valid=True,
        time_to_contact_s=0.02,
        achieved_position_w_m=(-0.02, 0.0, 0.0),
        achieved_velocity_w_mps=(1.0, 0.0, 0.0),
        achieved_signed_face_w=(1.0, 0.0, 0.0),
        desired_position_w_m=(0.0, 0.0, 0.0),
        desired_velocity_w_mps=(1.0, 0.0, 0.0),
        desired_signed_face_w=(1.0, 0.0, 0.0),
    )
    assert row["desired_position_now_w_m"] == [-0.02, 0.0, 0.0]
    assert row["channels"]["position"]["error"] == 0.0


def test_wrong_signed_face_keeps_nonzero_cauchy_tail():
    row = a211.a211_desired_contact_reward_terms(
        task_valid=True,
        time_to_contact_s=0.0,
        achieved_position_w_m=(0.0, 0.0, 0.0),
        achieved_velocity_w_mps=(0.0, 0.0, 0.0),
        achieved_signed_face_w=(-1.0, 0.0, 0.0),
        desired_position_w_m=(0.0, 0.0, 0.0),
        desired_velocity_w_mps=(0.0, 0.0, 0.0),
        desired_signed_face_w=(1.0, 0.0, 0.0),
    )
    face = row["channels"]["face"]
    assert face["error"] == pytest.approx(math.pi)
    assert 0.0 < face["coarse_kernel"] < 0.1


def test_velocity_and_face_are_not_proximity_gated_by_far_position():
    row = a211.a211_desired_contact_reward_terms(
        task_valid=True,
        time_to_contact_s=0.04,
        achieved_position_w_m=(100.0, -100.0, 50.0),
        achieved_velocity_w_mps=(0.0, 0.0, 0.0),
        achieved_signed_face_w=(1.0, 0.0, 0.0),
        desired_position_w_m=(0.0, 0.0, 0.0),
        desired_velocity_w_mps=(0.0, 0.0, 0.0),
        desired_signed_face_w=(1.0, 0.0, 0.0),
    )
    assert row["channels"]["position"]["eligible"] is False
    assert row["channels"]["velocity"]["post_policy_dt_reward"] == pytest.approx(
        0.24725
    )
    assert row["channels"]["face"]["post_policy_dt_reward"] == pytest.approx(
        0.138
    )


def _footwork(
    *,
    valid=True,
    t=0.5,
    base=(0.0, 0.0, 1.0),
    base_target=(0.0, 0.0, 1.0),
    racket=(1.0, 0.0, 1.0),
    contact=(0.0, 0.0, 1.0),
    previous=1.0,
    reset=False,
):
    return a211.a211_prestrike_footwork_reward_terms(
        task_valid=valid,
        time_to_contact_s=t,
        achieved_base_position_w_m=base,
        desired_base_position_w_m=base_target,
        achieved_racket_position_w_m=racket,
        desired_contact_position_w_m=contact,
        previous_racket_distance_m=previous,
        reset_progress_baseline=reset,
    )


def test_a211_prestrike_disables_constant_base_pay_and_keeps_progress():
    row = _footwork(racket=(0.9, 0.0, 1.0), previous=1.0)
    assert row["pre_strike_eligible"] is True
    assert row["base_kernel"] == pytest.approx(1.0)
    assert row["base_position_reward"] == 0.0
    assert row["racket_progress_raw_m"] == pytest.approx(0.1)
    assert row["racket_progress_reward"] == pytest.approx(0.02)
    assert row["post_policy_dt_reward"] == pytest.approx(0.02)

    one_sigma = _footwork(base=(0.2, 0.0, 1.0))
    assert one_sigma["base_kernel"] == pytest.approx(math.exp(-1.0))


@pytest.mark.parametrize(
    ("valid", "time_to_contact"),
    ((False, 0.5), (True, 0.0), (True, -0.02)),
)
def test_wait_exact_strike_and_poststrike_mask_a211_footwork(
    valid, time_to_contact
):
    row = _footwork(
        valid=valid,
        t=time_to_contact,
        racket=(0.5, 0.0, 1.0),
        previous=1.0,
    )
    assert row["pre_strike_eligible"] is False
    assert row["base_position_reward"] == 0.0
    assert row["racket_progress_reward"] == 0.0
    # Baseline bookkeeping still sees the live row while reward eligibility is off.
    assert row["current_racket_distance_m"] == pytest.approx(0.5)


def test_a211_progress_uses_bare_contact_cap_and_reset_suppression():
    capped = _footwork(
        racket=(0.0, 0.0, 1.0),
        contact=(10.0, 0.0, 1.0),
        previous=0.0,
    )
    assert capped["current_racket_potential_m"] == pytest.approx(4.65)
    assert capped["racket_progress_raw_m"] == pytest.approx(-4.65)
    assert capped["racket_progress_reward"] == pytest.approx(-0.93)

    suppressed = _footwork(
        racket=(0.5, 0.0, 1.0), previous=1.0, reset=True
    )
    assert suppressed["reset_progress_baseline"] is True
    assert suppressed["racket_progress_raw_m"] == 0.0
    assert suppressed["racket_progress_reward"] == 0.0


def _landing_terms(
    *,
    legal: bool,
    landing_valid: bool = True,
    kernel: float = 1.0,
    counter_total: float = 0.0,
    classification: str | None = None,
):
    return {
        "landing_valid": landing_valid,
        "net_crossed": True,
        "net_clear": True,
        "legal_opponent_table": legal,
        "kernel": kernel,
        "net_z_w_m": 1.0,
        "classification": classification
        or ("legal_opponent_table" if legal else "zero_ineligible_or_nonopponent"),
        "counter_rally_reward_components": [
            1.0 if legal else 0.0,
            1.0 if legal else 0.0,
            1.0 if legal else 0.0,
            1.0 if legal else 0.0,
            counter_total,
        ],
    }


def test_wait_masks_all_a211_achieved_outcome_terms_to_exact_zero():
    row = a211.a211_achieved_outcome_reward_terms(
        task_valid=False,
        selected_contact_observed_now=True,
        landing_terms=_landing_terms(legal=True),
        net_target_center_z_w_m=1.0,
    )
    assert row["classification"] == "task_invalid"
    assert row["capture_reward"] == 0.0
    assert row["pass_net_reward"] == 0.0
    assert row["landing_dense_reward"] == 0.0
    assert row["legal_landing_reward"] == 0.0
    assert row["post_policy_dt_reward"] == 0.0


def test_a211_off_table_has_neither_dense_guidance_nor_legal_prize():
    row = a211.a211_achieved_outcome_reward_terms(
        task_valid=True,
        selected_contact_observed_now=False,
        landing_terms=_landing_terms(
            legal=False,
            landing_valid=False,
            kernel=0.0,
            classification="opponent_side_off_table",
        ),
        net_target_center_z_w_m=1.0,
    )
    assert row["capture_reward"] == 0.0
    assert row["pass_net_reward"] == 0.0
    assert row["landing_dense_reward"] == 0.0
    assert row["legal_landing_reward"] == 0.0
    assert row["landing_terms"]["opponent_side_off_table_reward"] == 0.0
    assert row["post_policy_dt_reward"] == 0.0


def test_a211_valid_table_landing_keeps_dense_guidance_when_rally_rejected():
    row = a211.a211_achieved_outcome_reward_terms(
        task_valid=True,
        selected_contact_observed_now=False,
        landing_terms=_landing_terms(
            legal=False,
            landing_valid=True,
            kernel=0.5,
            classification="zero_ineligible_or_nonopponent",
        ),
        net_target_center_z_w_m=1.0,
        counter_rally_required=True,
    )
    assert row["pass_net_reward"] == 0.0
    assert row["landing_dense_reward"] == pytest.approx(0.2)
    assert row["counter_rally_outcome_reward"] == 0.0
    assert row["post_policy_dt_reward"] == pytest.approx(0.2)


def test_a211_actual_contact_and_counter_rally_match_resolved_isaac_leaf():
    row = a211.a211_achieved_outcome_reward_terms(
        task_valid=True,
        selected_contact_observed_now=True,
        landing_terms=_landing_terms(legal=True, counter_total=1.0),
        net_target_center_z_w_m=1.0,
    )
    assert row["capture_reward"] == pytest.approx(0.5)
    assert row["pass_net_reward"] == 0.0
    assert row["landing_dense_reward"] == pytest.approx(0.4)
    assert row["legal_landing_reward"] == pytest.approx(14.0)
    assert row["counter_rally_outcome_reward"] == pytest.approx(14.0)
    assert row["post_policy_dt_reward"] == pytest.approx(14.9)


def test_live_a211_groups_use_desired_contact_and_wait_masks_exactly(
    tmp_path, monkeypatch, authorities
):
    task, mimic = authorities
    core, tape, question = support._fake_sources(tmp_path, task, mimic)
    support._patch_selected_rubber(monkeypatch)
    producer = a211.A211ObservationProducer(
        cores=(core,),
        questions=(question,),
        robot_tape=tape,
        task=task,
        mimic=mimic,
        reset_wait_steps=1,
        policy_dt_s=0.02,
    )
    actor_wait, critic_wait = producer.tensors((False,))
    assert np.array_equal(
        actor_wait.numpy()[0, list(abi.A211_PROFILE.actor.task_mask_indices)],
        np.zeros(13, dtype=np.float32),
    )
    assert np.array_equal(
        critic_wait.numpy()[0, list(abi.A211_PROFILE.critic.task_mask_indices)],
        np.zeros(13, dtype=np.float32),
    )
    joint_span = abi.A211_PROFILE.actor.offsets["joint_pos"]
    assert np.linalg.norm(actor_wait.numpy()[0, joint_span]) > 0.0

    core.data.time = 0.02
    actor_active, critic_active = producer.tensors((True,))
    root = np.asarray(core.data.xpos[0], dtype=np.float64)
    expected = {
        "task_desired_contact_position_heading": (
            np.asarray(task.desired_contact_position_w_m) - root
        ),
        "task_desired_contact_velocity_heading": np.asarray(
            task.desired_contact_velocity_w_mps
        ),
        "task_desired_contact_face_heading": np.asarray(
            task.desired_contact_signed_face_w
        ),
    }
    for name, row in expected.items():
        actor_span = abi.A211_PROFILE.actor.offsets[name]
        critic_span = abi.A211_PROFILE.critic.offsets[name]
        assert np.allclose(actor_active.numpy()[0, actor_span], row, atol=1.0e-7)
        assert np.allclose(critic_active.numpy()[0, critic_span], row, atol=1.0e-7)


def test_a211_task_face_uses_raw_y_while_mimic_keeps_signed_physical_face(
    tmp_path, monkeypatch, authorities
):
    task, mimic = authorities
    core, tape, question = support._fake_sources(tmp_path, task, mimic)
    core._selected_rubber_action_lineage = {"mount_normal_sign": -1}
    monkeypatch.setattr(
        a211.shared.selected_rubber_classifier,
        "validate_classifier_binding",
        lambda _value: {"content_sha256": "4" * 64},
    )
    monkeypatch.setattr(
        a211.shared.selected_rubber_classifier,
        "validate_action_lineage",
        lambda _value, *, classifier_binding: {
            "mount_normal_sign": -1,
            "content_sha256": "9" * 64,
            "classifier_binding_sha256": classifier_binding["content_sha256"],
        },
    )
    producer = a211.A211ObservationProducer(
        cores=(core,),
        questions=(question,),
        robot_tape=tape,
        task=task,
        mimic=mimic,
        reset_wait_steps=1,
        policy_dt_s=0.02,
    )
    live = producer._live(0)
    assert live["racket_raw_y_axis"] == pytest.approx((0.0, 1.0, 0.0))
    assert live["racket_normal"] == pytest.approx((0.0, -1.0, 0.0))
    producer.begin_reward_transition(
        (True,), np.zeros((1, 31), dtype=np.float64)
    )
    producer._capture_reward_row(0, 3)
    row = producer.finish_reward_transition()[0]
    assert row["achieved_contact_sample"]["signed_face_w"] == pytest.approx(
        (0.0, 1.0, 0.0)
    )


def test_a211_rejects_native_question_built_for_c_recipe_before_tensor_use(
    tmp_path, monkeypatch, authorities
):
    task, mimic = authorities
    core, tape, question = support._fake_sources(tmp_path, task, mimic)
    question.authority["target_recipe"] = "outcome_dense_only"
    support._patch_selected_rubber(monkeypatch)
    with pytest.raises(a211.A211EnvError, match="differs from A211 task authority"):
        a211.A211ObservationProducer(
            cores=(core,),
            questions=(question,),
            robot_tape=tape,
            task=task,
            mimic=mimic,
            reset_wait_steps=1,
            policy_dt_s=0.02,
        )


def test_integrated_a211_transition_replaces_c_proximity_with_window_target(
    tmp_path, monkeypatch, authorities
):
    task, mimic = authorities
    core, tape, question = support._fake_sources(tmp_path, task, mimic)
    support._patch_selected_rubber(monkeypatch)
    base = support._FakeFixedCenterBase(core, tape, question)
    env = a211.MujocoA211DiagnosticVecEnv(
        base_env=base,
        task_authority=task,
        mimic_authority=mimic,
    )
    monkeypatch.setattr(
        env,
        "_event_evidence",
        lambda _index, _facts: (
            n1_reward_event_kernel.ContactEvidence(False, None, False),
            n1_reward_event_kernel.OutgoingFlightEvidence(
                False, None, None, None, None
            ),
        ),
    )
    actions = torch.zeros((1, 31), dtype=torch.float32)
    _obs, wait_reward, _done, wait_extras = env.step(actions)
    wait_row = wait_extras["reward_terms"][0]
    assert wait_row["desired_contact_reward"] == 0.0
    assert wait_row["landing_reward"] == 0.0
    assert wait_reward.item() == pytest.approx(
        wait_row["isaac_synonymous_prior_reward"], abs=1.0e-6
    )

    base.forced_policy_tick = 93
    _obs, reward, _done, extras = env.step(actions)
    row = extras["reward_terms"][0]
    assert row["nominal_strike_sampled_now"] is True
    assert row["c211_single_tick_proximity_reward_removed"] > 0.0
    assert row["desired_contact_terms"]["any_channel_eligible"] is True
    assert row["desired_contact_reward"] > 0.0
    assert reward.item() == pytest.approx(row["total_reward"], abs=1.0e-6)
    assert row["total_reward"] == pytest.approx(
        sum(row["additive_reward_components"].values()),
        abs=1.0e-12,
    )
    assert row["nonadditive_alias_map"]["strike_reward"] == (
        "desired_contact_reward"
    )
    assert extras["a211_desired_contact_reward_available"] is True
    assert "c211_achieved_outcome_reward_available" not in extras
    receipt = env.diagnostic_training_receipt()
    assert receipt["reward_parity_status"] == "partial_fail_closed"
    assert receipt["complete_isaac_reward_parity_claimed"] is False
    assert receipt["formal_blockers"] == list(a211.A211_FORMAL_BLOCKERS)
    assert receipt["reward_contract"]["landing"]["counter_rally_enabled"] is True


def test_a211_rejects_missing_authoritative_continuous_wait_schedule(
    tmp_path, monkeypatch, authorities
):
    task, mimic = authorities
    core, tape, question = support._fake_sources(tmp_path, task, mimic)
    support._patch_selected_rubber(monkeypatch)
    base = support._FakeFixedCenterBase(core, tape, question)
    base.allow_action_ball_legacy_fixed_wait_test_double = False
    with pytest.raises(a211.A211EnvError, match="authoritative continuous WAIT"):
        a211.MujocoA211DiagnosticVecEnv(
            base_env=base,
            task_authority=task,
            mimic_authority=mimic,
        )
