from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "configs/phase1_fresh_c_v1v2_base_decel_clean_main_effect_queue_20260715.yaml"
OLD_QUEUE = ROOT / "configs/phase1_fresh_c_v1v2_base_decel_measurement_rerun_queue_20260715.yaml"
SOURCE = "2c2d70d6d0ccf7b0757aac4dd8e575c2e077607e"


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _overrides(job: dict) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in job["recipe"]["delta"]:
        key, value = raw.split("=", 1)
        result[key.lstrip("+")] = value
    return result


def test_clean_pair_is_single_seed_pod2_only_and_exact_source() -> None:
    queue = _load(QUEUE)
    assert queue["launch_authorized"] is True
    assert queue["dispatch_pods"] == ["pod2"]
    assert [job["seed"] for job in queue["jobs"]] == [3, 3]
    assert [job["source"]["commit"] for job in queue["jobs"]] == [SOURCE, SOURCE]
    assert [job["resource"]["required_slot"] for job in queue["jobs"]] == [
        "pod2/gpu1",
        "pod2/gpu2",
    ]
    assert len({job["run_dir"] for job in queue["jobs"]}) == 2
    assert all("clean_main_effect_20260715" in job["run_dir"] for job in queue["jobs"])
    assert all(job["status"] == "ready" for job in queue["jobs"])


def test_clean_pair_disables_endogenous_post_swing_path_in_both_arms() -> None:
    queue = _load(QUEUE)
    activation = queue["decision_contract"]["activation"]
    disabled = activation["post_swing_replay_disabled"]
    assert disabled["setting"] == 0.0
    assert disabled["all_five_counters_must_equal_zero_every_update"] is True
    assert disabled["any_nonzero_counter_invalidates_milestone"] is True
    assert len(disabled["tensorboard_metrics"]) == 5

    for job in queue["jobs"]:
        overrides = _overrides(job)
        assert overrides["task.motion.post_swing_start_prob"] == "0.0"
        assert overrides["task.rewards.free_wrist_vel_mimic"] == "true"
        assert overrides["task.rewards.motion_scale_in_window"] == "0.25"


def test_only_scientific_difference_is_base_deceleration_weight() -> None:
    queue = _load(QUEUE)
    control, treatment = queue["jobs"]
    assert control["recipe"]["base"] == treatment["recipe"]["base"]
    assert control["motion"] == treatment["motion"]
    assert control["bank"] == treatment["bank"]
    assert control["exam"] == treatment["exam"]
    assert control["budget"] == treatment["budget"]

    c = _overrides(control)
    t = _overrides(treatment)
    assert set(c) == set(t)
    changed = {key for key in c if c[key] != t[key]}
    assert changed == {"task.rewards.base_decel_weight"}
    assert c["task.rewards.base_decel_weight"] == "0.0"
    assert t["task.rewards.base_decel_weight"] == "1.0"


def test_reused_probe_is_only_a_source_scene_gate_not_behavior_evidence() -> None:
    queue = _load(QUEUE)
    probe = queue["strict_full_scene_probe_evidence"]
    old = _load(OLD_QUEUE)["strict_full_scene_probe_evidence"]
    for key in (
        "attempt_id",
        "result_path",
        "result_file_sha256",
        "result_content_sha256",
        "claim_content_sha256",
        "checkpoint_sha256",
        "hard_contract_sha256",
    ):
        assert probe[key] == old[key]
    assert probe["terminal_status"] == "passed"
    assert probe["reuse_scope"] == "source_scene_boot_and_checkpoint_wiring_only"
    assert probe["scientific_recipe_identity_reused"] is False
    assert probe["post_swing_setting_covered_by_probe"] is False


def test_no_old_run_or_behavior_decision_is_reused() -> None:
    queue = _load(QUEUE)
    old = _load(OLD_QUEUE)
    assert {job["id"] for job in queue["jobs"]}.isdisjoint(
        {job["id"] for job in old["jobs"]}
    )
    assert {job["run_dir"] for job in queue["jobs"]}.isdisjoint(
        {job["run_dir"] for job in old["jobs"]}
    )
    contract = queue["decision_contract"]
    assert contract["second_seed_authorized"] is False
    assert contract["judge_authorized"] is False
    assert contract["promotion_authorized"] is False
    assert contract["activation_must_precede_behavior_comparison"] is True
