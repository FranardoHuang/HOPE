"""Host-only checks for the ActionBall pre-optimizer Reward fence."""

from __future__ import annotations

import copy
import json
import sys
import types

import pytest

from test_training_launch_claim import (
    _load_contract_module,
    _load_runner_module,
)


def _prepared(status="PASS"):
    receipt = {
        "event": "hope_reward_episode_segmented_closure_update",
        "schema_version": 1,
        "status": status,
        "evidence_source": "live_isaac_reward_manager",
        "capture_mode": "reward_manager_reset_pre_clear_hook",
        "task_kind": "action_ball",
        "ppo_update": 7,
        "recipe_sha256": "a" * 64,
        "step_dt_s": 0.02,
        "num_envs": 1,
        "segment_key_fields": ["env_id", "reset_generation"],
        "all_reward_manager_term_names": ["death_penalty"],
        "completed_episode_count": 0,
        "completed_episode_segments": [],
        "reset_batches": [],
        "open_episode_count": 1,
        "open_episode_segments": [{"env_id": 0}],
        "dashboard_normalization": {
            "status": "NOT_OBSERVED_NO_RESET",
            "reset_batch_count": 0,
        },
        "e2_eligible": False,
        "checks": {
            "status": "PASS",
            "environment_step_count": 3,
            "all_step_reward_buf_equals_all_term_sums": "PASS",
            "all_episode_sums_equal_captured_term_sums": "PASS",
            "all_reset_episode_sums_cleared": "PASS",
            "exact_environment_step_coverage": "PASS",
        },
    }
    return {
        "ppo_update": 7,
        "activation": {
            "recipe_sha256": "a" * 64,
            "step_dt_s": 0.02,
            "num_envs": 1,
            "environment_step_count": 3,
        },
        "per_action": {},
        "safety": {},
        "action_ball_conservation": receipt,
        "status": "frozen_validated_before_optimizer",
    }


def _runner_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    return _load_runner_module(monkeypatch, _load_contract_module())


def test_action_ball_optimizer_fence_rejects_missing_fail_or_mutated_receipt(
    monkeypatch,
):
    runner_mod = _runner_module(monkeypatch)
    require = (
        runner_mod.MotionOnPolicyRunner
        ._require_action_ball_conservation_pass
    )

    valid = _prepared()
    assert require(valid, step=7) is valid["action_ball_conservation"]
    for invalid in (
        {},
        _prepared("FAIL_CLOSED"),
        {
            **_prepared(),
            "activation": {"recipe_sha256": "b" * 64},
        },
    ):
        with pytest.raises(RuntimeError, match="optimizer is fenced"):
            require(invalid, step=7)

    bad_check = copy.deepcopy(valid)
    bad_check["action_ball_conservation"]["checks"][
        "all_episode_sums_equal_captured_term_sums"
    ] = "FAIL_CLOSED"
    with pytest.raises(RuntimeError, match="optimizer is fenced"):
        require(bad_check, step=7)


def test_persisted_preoptimizer_artifact_contains_the_public_pass_receipt(
    monkeypatch, tmp_path
):
    runner_mod = _runner_module(monkeypatch)
    runner = runner_mod.MotionOnPolicyRunner.__new__(
        runner_mod.MotionOnPolicyRunner
    )
    runner.log_dir = str(tmp_path)
    runner._joint_safety_fsync_directory = lambda _directory: None

    artifact = runner._persist_reward_evidence_update(
        _prepared(),
        step=7,
        rank=0,
        task_kind="action_ball",
    )
    payload = json.loads(
        (tmp_path / artifact["path"]).read_text(encoding="utf-8")
    )
    assert payload["status"] == "prepared_before_optimizer"
    assert payload["action_ball_conservation"]["status"] == "PASS"
    assert artifact["action_ball_conservation_status"] == "PASS"
