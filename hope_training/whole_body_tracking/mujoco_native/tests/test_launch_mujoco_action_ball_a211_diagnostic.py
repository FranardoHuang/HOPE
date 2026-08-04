"""Claim and protocol tests for the native A211 execution launcher."""

from __future__ import annotations

from types import SimpleNamespace

from hope_training.whole_body_tracking.mujoco_native import action_ball_211_abi as abi
from hope_training.whole_body_tracking.mujoco_native.scripts import (
    launch_mujoco_action_ball_a211_diagnostic as launch,
)


def test_a211_execution_plan_binds_partial_reward_and_one_env(monkeypatch, tmp_path):
    args = launch._parser().parse_args(
        [
            "--profile",
            "A211",
            "--num-envs",
            "1",
            "--output-dir",
            str(tmp_path / "new-run"),
        ]
    )
    monkeypatch.setattr(
        launch.action_ball_a211_env.A211TaskAuthority,
        "load",
        lambda *_args, **_kwargs: SimpleNamespace(time_to_contact_s=1.84),
    )
    monkeypatch.setattr(launch.shared_launch, "_execution_authorities", lambda _args: {})
    monkeypatch.setattr(
        launch.shared_launch,
        "_source_lineage",
        lambda _profile: {"repo_relative_path": "source.py", "sha256": "a" * 64},
    )
    monkeypatch.setattr(launch, "_runtime_module_sha256s", lambda: {})
    plan = launch._execution_plan(args)
    assert plan["profile"] == "A211"
    assert plan["workload"]["num_envs"] == 1
    assert plan["workload"]["actor_width"] == 211
    assert plan["workload"]["critic_width"] == 319
    assert plan["claims"]["a211_desired_contact_window_reward_implemented"] is True
    assert plan["claims"]["a211_actual_contact_achieved_landing_implemented"] is True
    assert plan["claims"]["hidden_wait_ball_parked"] is True
    assert plan["claims"]["ball_only_atomic_sealed_launch_on_reveal"] is True
    assert plan["claims"]["robot_state_continuous_across_reveal"] is True
    assert plan["claims"]["complete_isaac_reward_parity_claimed"] is False
    assert plan["diagnostic_unauthorized"] is True


def test_a211_trainer_uses_fresh_a_profile_normalizers():
    kwargs = abi.A211_PROFILE.trainer_config_kwargs()
    assert kwargs["observation_dim"] == 211
    assert kwargs["critic_observation_dim"] == 319
    assert kwargs["actor_normalizer_identity"] == "action_ball_a211_actor_norm_v2"
    assert kwargs["critic_normalizer_identity"] == "action_ball_a211_critic_norm_v1"
    assert kwargs != abi.C211_PROFILE.trainer_config_kwargs()


def test_a211_launcher_reopens_exact_mirrored_isaac_source():
    lineage = launch.shared_launch._source_lineage(abi.A211_PROFILE)
    assert lineage["sha256"] == abi.A211_SOURCE_SHA256
    assert lineage["repo_relative_path"].endswith(
        "action_ball_a211_trainability.py"
    )


def test_a211_runtime_module_seal_includes_independent_adapter_and_runner():
    rows = launch._runtime_module_sha256s()
    assert "a211_env" in rows
    assert "a211_runner" in rows
    assert "native_single_env" in rows
    assert "table_termination" in rows
    assert all(len(value) == 64 for value in rows.values())


def test_audited_update_requires_exact_three_and_eleven_tick_windows():
    class FakeEnv:
        num_envs = 2

        @staticmethod
        def reset_reward_audit():
            return None

        @staticmethod
        def reward_audit_receipt():
            return {
                "transition_step_count": 500,
                "row_count": 1000,
                "prior_terms": {"upright_exp": {"sample_count": 1000}},
                "desired_contact_position_window_row_count": 6,
                "desired_contact_velocity_window_row_count": 22,
                "desired_contact_face_window_row_count": 22,
                "desired_contact_any_window_row_count": 22,
            }

    class FakeTrainer:
        config = SimpleNamespace(rollout_steps=500)

        @staticmethod
        def run_update():
            return {"kind": "update", "rollout_steps": 500}

    update, audit = launch._run_audited_update(FakeEnv(), FakeTrainer())
    assert update == {"kind": "update", "rollout_steps": 500}
    assert audit["desired_contact_position_window_row_count"] == 6


def test_audited_update_allows_early_policy_to_miss_all_target_windows():
    class FakeEnv:
        num_envs = 1

        @staticmethod
        def reset_reward_audit():
            return None

        @staticmethod
        def reward_audit_receipt():
            return {
                "transition_step_count": 574,
                "row_count": 574,
                "prior_terms": {"upright_exp": {"sample_count": 574}},
                "desired_contact_position_window_row_count": 0,
                "desired_contact_velocity_window_row_count": 0,
                "desired_contact_face_window_row_count": 0,
                "desired_contact_any_window_row_count": 0,
            }

    class FakeTrainer:
        config = SimpleNamespace(rollout_steps=500)

        @staticmethod
        def run_update():
            return {"kind": "update", "rollout_steps": 574}

    update, audit = launch._run_audited_update(FakeEnv(), FakeTrainer())
    assert update["rollout_steps"] == 574
    assert audit["desired_contact_any_window_row_count"] == 0
