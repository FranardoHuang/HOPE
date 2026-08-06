"""Claim-boundary tests for the native A211/C211 construction launcher."""

from __future__ import annotations

import json
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hope_training.whole_body_tracking.mujoco_native.scripts import (
    launch_mujoco_action_ball_211_diagnostic as launch,
)


class _CanaryEnv:
    num_envs = 1
    num_actions = 31
    num_observations = 211

    def __init__(self, *, reveal_tick=25, hard_tick=None, forbidden_tick=None):
        self.forbidden_tick = forbidden_tick
        self._wait_schedule = SimpleNamespace(min_wait_ticks=25, max_wait_ticks=25)
        self.producer = SimpleNamespace(
            robot_tape=SimpleNamespace(history_fill_action=np.zeros(31))
        )
        binding = SimpleNamespace(
            decode_action=lambda action: (
                np.asarray(action, dtype=np.float64),
                np.asarray(action, dtype=np.float64),
                0,
            )
        )
        plant = SimpleNamespace(
            delay=SimpleNamespace(state=lambda: np.zeros((1, 31)))
        )
        self._native = SimpleNamespace(
            cores=(SimpleNamespace(binding=binding, plant=plant),)
        )
        self.reveal_tick = reveal_tick
        self.hard_tick = hard_tick
        self.tick = 0
        self.boundary = True

    @staticmethod
    def fresh_actor_bootstrap_contract():
        return {"output_layer_bias": [0.0] * 31}

    def reset(self, *, seed=None):
        del seed
        self.tick = 0
        self.boundary = True
        return torch.zeros((1, 211)), {"task_valid": [False]}

    def is_reset_boundary(self):
        return self.boundary

    def step(self, actions):
        self.boundary = False
        self.tick += 1
        hard = self.hard_tick == self.tick
        projected = not torch.equal(actions, torch.zeros_like(actions))
        mask = (bool(projected),) + (False,) * 30
        next_valid = self.tick >= self.reveal_tick
        forbidden = int(
            self.forbidden_tick is not None and self.tick >= self.forbidden_tick
        )
        return (
            torch.zeros((1, 211)),
            torch.zeros(1),
            torch.as_tensor([hard], dtype=torch.bool),
            {
                "diagnostic_qdes_projection_masks": (mask,),
                "diagnostic_event_ledgers": (
                    {
                        "plant_counters": {"effort_clip_joint_events": 0},
                        "joint_actual_forbidden_observed_ticks": forbidden,
                        "promotion_blocking_evidence": {
                            "promotion_blocked": forbidden > 0,
                            "reasons": (
                                [launch.trainer.PROMOTION_BLOCKING_REASON]
                                if forbidden
                                else []
                            ),
                        },
                    },
                ),
                "diagnostic_exact_hard_terminations": torch.as_tensor(
                    [hard], dtype=torch.bool
                ),
                "diagnostic_exact_hard_termination_reasons": [
                    "base_height" if hard else None
                ],
                "task_valid_transition": [self.tick > self.reveal_tick],
                "task_valid_next": [next_valid],
            },
        )


def _canary_trainer():
    actor = torch.nn.Sequential(torch.nn.Linear(211, 31))
    with torch.no_grad():
        actor[-1].weight.zero_()
        actor[-1].bias.zero_()
    model = torch.nn.Module()
    model.actor = actor
    model.register_parameter(
        "log_std", torch.nn.Parameter(torch.full((31,), np.log(0.02)))
    )
    return SimpleNamespace(model=model)


def test_fresh_wait_bootstrap_canary_reports_projection_and_effort_receipt():
    receipt = launch._fresh_wait_bootstrap_canary(
        _CanaryEnv(), _canary_trainer(), profile="C211"
    )
    assert receipt["passed"] is True
    assert receipt["deterministic_max_wait"]["completed_wait_ticks"] == 25
    assert receipt["deterministic_max_wait"]["projected_joint_event_count"] == 0
    noisy = receipt["stochastic_fresh_wait"]
    assert noisy["completed_wait_ticks"] == 25
    assert noisy["per_joint_projection_counts"][0] == 25
    assert noisy["projected_joint_event_count"] == 25
    assert noisy["effort_clip_joint_event_count"] == 0
    assert receipt["installed_action_std"] == pytest.approx([0.02] * 31)


def test_fresh_wait_bootstrap_canary_rejects_early_reveal():
    with pytest.raises(launch.LaunchBlocked, match="reveal timing differs"):
        launch._fresh_wait_bootstrap_canary(
            _CanaryEnv(reveal_tick=10), _canary_trainer(), profile="C211"
        )


def test_fresh_wait_bootstrap_canary_rejects_hard_termination():
    with pytest.raises(launch.LaunchBlocked, match="canary failed"):
        launch._fresh_wait_bootstrap_canary(
            _CanaryEnv(hard_tick=3), _canary_trainer(), profile="C211"
        )


def test_fresh_wait_bootstrap_canary_publishes_its_promotion_conclusion():
    receipt = launch._fresh_wait_bootstrap_canary(
        _CanaryEnv(), _canary_trainer(), profile="C211"
    )
    assert receipt["promotion_blocked"] is False
    for phase in ("deterministic_max_wait", "stochastic_fresh_wait"):
        evidence = receipt[phase]["promotion_blocking_evidence"]
        assert evidence["promotion_blocked"] is False
        assert evidence["checked_sample_count"] == 25


def test_fresh_wait_bootstrap_canary_rejects_a_non_terminal_hard_edge():
    """The soften moved this fault off the Done bit; the canary must still refuse.

    人话:改软之前,起手姿态贴关节硬边会算进 hard_termination_count,canary 直接不过。
    改软之后它不再进 hard,如果没人读结论位,这条 canary 就会静默放行一个"不能上机"的起手。
    """

    with pytest.raises(launch.LaunchBlocked, match="canary failed"):
        launch._fresh_wait_bootstrap_canary(
            _CanaryEnv(forbidden_tick=4), _canary_trainer(), profile="C211"
        )


def test_promotion_blocking_summary_is_fail_closed_and_warns(capsys):
    clean_update = {
        "promotion_blocking_evidence": {"promotion_blocked": False},
    }
    summary = launch._promotion_blocking_summary(
        profile="C211",
        canary={"promotion_blocked": False},
        update_receipts=[clean_update],
        checkpoint_save_receipt={"promotion_blocked": False},
    )
    assert summary["promotion_blocked"] is False
    assert summary["blocked_sources"] == []
    assert capsys.readouterr().err == ""

    # 缺字段与 True 同义:一个说不出结论的来源就是"卡住"。
    blind = launch._promotion_blocking_summary(
        profile="C211",
        canary={},
        update_receipts=[{}],
        checkpoint_save_receipt={},
    )
    assert blind["promotion_blocked"] is True
    assert blind["blocked_sources"] == [
        "fresh_wait_bootstrap_canary",
        "update_receipt[0]",
        "checkpoint_save_receipt",
    ]
    captured = capsys.readouterr().err
    assert "WARN" in captured
    assert "promotion_blocked=True" in captured

    reported = launch._promotion_blocking_summary(
        profile="C211",
        canary={"promotion_blocked": False},
        update_receipts=[
            clean_update,
            {"promotion_blocking_evidence": {"promotion_blocked": True}},
        ],
        checkpoint_save_receipt={"promotion_blocked": False},
    )
    assert reported["promotion_blocked"] is True
    assert reported["blocked_sources"] == ["update_receipt[1]"]
    assert "WARN" in capsys.readouterr().err


@pytest.mark.parametrize("profile", ("A211", "C211"))
def test_4096_plan_constructs_only_the_strict_shape_contract(
    profile: str, capsys
) -> None:
    assert launch.main(["--profile", profile, "--num-envs", "4096"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["mode"] == "plan"
    assert result["profile"] == profile
    assert result["construction"]["actor_shape"] == [4096, 211]
    assert result["construction"]["critic_shape"] == [4096, 319]
    assert result["construction"]["runtime_tensor_materialized"] is False
    assert result["construction"]["runtime_ready"] is False
    assert result["claims"]["matched_4096_runtime_measured"] is False
    assert result["claims"]["two_update_smoke_executed"] is False
    assert result["claims"]["placeholder_or_zero_padded_columns_used"] is False
    assert result["claims"]["safe_ready_formal_pass_claimed"] is False
    assert result["claims"]["c211_achieved_outcome_reward_implemented"] is (
        profile == "C211"
    )
    assert result["claims"]["c211_partial_isaac_synonymous_reward_implemented"] is (
        profile == "C211"
    )
    assert result["claims"]["complete_isaac_reward_parity_claimed"] is False
    assert (
        result["claims"]["true_c211_achieved_outcome_reward_available"] is False
    )
    assert result["claims"]["true_c211_training_lane_ready"] is False
    assert result["safe_ready_authority_status"] == (
        "split_ready_physical_birth_diagnostic_only_cross_engine_unmeasured"
    )
    assert any("plant_observation" in value for value in result["runtime_blockers"])
    assert any(
        "full_body_measured_mimic" in value for value in result["runtime_blockers"]
    )
    assert result["diagnostic_unauthorized"] is True
    assert result["formal_authorized"] is False


def test_two_update_request_fails_before_environment_without_authorities(capsys):
    code = launch.main(
        [
            "--profile",
            "C211",
            "--num-envs",
            "4096",
            "--execute-two-updates",
            "--confirm-diagnostic-unauthorized",
        ]
    )
    assert code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "execution output directory is required" in captured.err


def test_two_update_protocol_is_update1_save_then_cold_matched_update2():
    args = launch._parser().parse_args(["--profile", "C211"])
    assert args.pre_checkpoint_updates == 1
    assert args.reset_wait_steps is None
    assert (args.reset_wait_min_steps, args.reset_wait_max_steps) == (5, 25)
    assert args.reset_wait_seed == 20260804
    assert args.episode_horizon_steps == 500
    assert args.required_active_steps == 200
    assert args.execute_two_updates is False


def test_execution_plan_receipt_is_partial_and_names_fail_closed_terms(
    monkeypatch, tmp_path
):
    args = launch._parser().parse_args(
        [
            "--profile",
            "C211",
            "--num-envs",
            "1",
            "--output-dir",
            str(tmp_path / "new-run"),
        ]
    )
    monkeypatch.setattr(
        launch.action_ball_c211_env.C211TaskAuthority,
        "load",
        lambda *_args, **_kwargs: SimpleNamespace(time_to_contact_s=0.96),
    )
    monkeypatch.setattr(launch, "_execution_authorities", lambda _args: {})
    monkeypatch.setattr(
        launch,
        "_source_lineage",
        lambda _profile: {"repo_relative_path": "source.py", "sha256": "a" * 64},
    )
    monkeypatch.setattr(launch, "_runtime_module_sha256s", lambda: {})

    plan = launch._execution_plan(args)
    claims = plan["claims"]
    assert plan["kind"] == launch.EXECUTION_PLAN_KIND
    assert claims["reward_contract_identity"] == (
        launch.action_ball_c211_env.C211_REWARD_CONTRACT_IDENTITY
    )
    assert claims["reward_parity_status"] == "partial_fail_closed"
    assert claims["full_body_mimic_reward_consumed"] is True
    assert claims["measured_paddle_prior_reward_consumed"] is True
    assert claims["hidden_wait_ball_parked"] is True
    assert claims["ball_only_atomic_sealed_launch_on_reveal"] is True
    assert claims["robot_state_continuous_across_reveal"] is True
    assert claims["complete_isaac_reward_parity_claimed"] is False
    unavailable = {row["term"] for row in claims["unavailable_isaac_reward_terms"]}
    assert {"foot_soft_landing", "undesired_contacts", "joint_torques"} <= unavailable


def test_audited_update_requires_complete_raw_term_coverage():
    class FakeEnv:
        num_envs = 2

        def __init__(self):
            self.reset_count = 0

        def reset_reward_audit(self):
            self.reset_count += 1

        @staticmethod
        def reward_audit_receipt():
            return {
                "transition_step_count": 3,
                "row_count": 6,
                "prior_terms": {"upright_exp": {"sample_count": 6}},
            }

    class FakeTrainer:
        config = SimpleNamespace(rollout_steps=3)

        @staticmethod
        def run_update():
            return {"kind": "update", "rollout_steps": 3}

    env = FakeEnv()
    update, audit = launch._run_audited_update(env, FakeTrainer())
    assert env.reset_count == 1
    assert update == {"kind": "update", "rollout_steps": 3}
    assert audit["prior_terms"]["upright_exp"]["sample_count"] == 6


def test_precheckpoint_count_cannot_silently_expand_to_three_updates(
    tmp_path, capsys
):
    code = launch.main(
        [
            "--profile",
            "C211",
            "--execute-two-updates",
            "--confirm-diagnostic-unauthorized",
            "--output-dir",
            str(tmp_path / "never-created"),
            "--pre-checkpoint-updates",
            "2",
        ]
    )
    assert code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "exactly one pre-checkpoint update" in captured.err
    assert not (tmp_path / "never-created").exists()


def test_a211_execution_dispatches_to_real_adapter_and_fails_closed_without_authorities(
    tmp_path, capsys
):
    code = launch.main(
        [
            "--profile",
            "A211",
            "--execute-two-updates",
            "--confirm-diagnostic-unauthorized",
            "--output-dir",
            str(tmp_path / "never-created"),
        ]
    )
    assert code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "MUJOCO-A211-BLOCKED" in captured.err
    assert not (tmp_path / "never-created").exists()
    assert not (tmp_path / "never-created").exists()


def test_parent_child_runtime_source_set_is_complete_and_content_bound():
    rows = launch._runtime_module_sha256s()
    assert set(rows) == {
        "runner",
        "abi",
        "c211_env",
        "trainer",
        "checkpoint",
        "fixed_center_runner",
        "fixed_center_recipe",
        "native_vec_env",
        "native_single_env",
        "table_termination",
        "task_wait_schedule",
        "reward_event_kernel",
        "physical_ball_scene",
        "virtual_ball",
    }
    assert all(
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= set("0123456789abcdef")
        for value in rows.values()
    )


def test_execute_refuses_wait_schedule_horizon_drift_before_authorities(
    tmp_path, capsys
):
    immutable = (
        launch.REPO_ROOT
        / "configs/action_ball_n1_measured_20260803/"
        "fresh_tape_seed0_20260803_take061_robust20n_r4_splitready/"
        "immutable_n1_tape.v1.22052606032f.json"
    )
    code = launch.main(
        [
            "--profile",
            "C211",
            "--execute-two-updates",
            "--confirm-diagnostic-unauthorized",
            "--output-dir",
            str(tmp_path / "never-created"),
            "--immutable-tape",
            str(immutable),
            "--expected-immutable-tape-sha256",
            "22052606032f74257ce98b5b6be8e8a4f8175848655ce604f50adf4751409e66",
            "--episode-horizon-steps",
            "499",
        ]
    )
    assert code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "frozen seeded WAIT schedule" in captured.err
    assert not (tmp_path / "never-created").exists()


def test_nonpositive_num_envs_is_rejected(capsys):
    assert launch.main(["--profile", "A211", "--num-envs", "0"]) == 2
    assert "positive plain integer" in capsys.readouterr().err


def test_bound_task_question_removes_only_the_task_authority_blocker(tmp_path, capsys):
    question = tmp_path / "immutable_tape.json"
    payload = {
        "kind": "action_ball_n1_immutable_single_question_tape",
        "schema_version": 1,
        "diagnostic_unauthorized": True,
        "row_count": 1,
        "canonical_sha256": "a" * 64,
        "question_sha256": "b" * 64,
        "question_layout": [
            {"name": "base_goal_w_m", "width": 3},
            {"name": "ball_contact_w_m", "width": 3},
            {"name": "time_to_contact_s", "width": 1},
            {"name": "incoming_velocity_w_mps", "width": 3},
            {"name": "incoming_spin_w_radps", "width": 3},
            {"name": "landing_aim_w_xy_m", "width": 2},
        ],
        "target_layout": [
            {"name": "desired_racket_site_w_m", "width": 3},
            {"name": "desired_racket_face_center_velocity_w_mps", "width": 3},
            {"name": "desired_racket_face_normal_w", "width": 3},
        ],
        "targets": {"current_lm": {}},
    }
    question.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    digest = hashlib.sha256(question.read_bytes()).hexdigest()
    assert (
        launch.main(
            [
                "--profile",
                "C211",
                "--task-question-authority",
                str(question),
                "--expected-task-question-sha256",
                digest,
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["task_question_authority"]["sha256"] == digest
    assert not any("task_question" in item for item in result["runtime_blockers"])
    assert any("plant_observation" in item for item in result["runtime_blockers"])
    assert any(
        "full_body_measured_mimic" in item for item in result["runtime_blockers"]
    )
