from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_phase1_demo_hotstart_queue.py"
QUEUE = ROOT / "configs" / "phase1_pod2_demo_hotstart_portfolio_20260716.yaml"


def _module():
    spec = importlib.util.spec_from_file_location("demo_hotstart_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


D = _module()


def _raw() -> dict:
    return yaml.safe_load(QUEUE.read_text(encoding="utf-8"))


def _write(tmp_path: Path, value: dict) -> Path:
    path = tmp_path / "queue.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def _values(job: dict) -> dict[str, str]:
    return {
        D.Q._override_key(raw, job["id"]): raw.partition("=")[2]
        for raw in [*job["recipe"]["base"], *job["recipe"]["delta"]]
    }


def _activated(tmp_path: Path) -> tuple[dict, Path]:
    raw = copy.deepcopy(_raw())
    raw["launch_authorized"] = True
    raw["activation_contract"]["state"] = "activated"
    raw["activation_contract"]["receipt_file_sha256"] = "a" * 64
    for parent in raw["parents"].values():
        parent["checkpoint_sha256"] = "b" * 64
        parent["hard_contract_sha256"] = "c" * 64
        parent["training_launch_claim_sha256"] = "d" * 64
    for job in raw["jobs"]:
        job["status"] = "ready"
        job["blocker"] = None
    path = _write(tmp_path, raw)
    return D.load_queue(path), path


def test_pending_queue_is_six_blocked_pod2_rows_and_plan_is_nonlaunching():
    queue = D.load_queue(QUEUE)
    plan = D.cmd_plan(queue)
    assert queue["dispatch_pods"] == ["pod2"]
    assert queue["launch_authorized"] is False
    assert len(queue["jobs"]) == 6
    assert plan["assignments"] == []
    assert len(plan["blocked"]) == 6
    assert all(job["status"] == "blocked" for job in queue["jobs"])
    assert all(job["runtime_binding"] is True for job in queue["jobs"])
    assert all(job["source"]["commit"] == D.EXPECTED_SOURCE for job in queue["jobs"])


def test_six_recipes_match_the_frozen_demo_portfolio():
    queue = D.load_queue(QUEUE)
    jobs = {job["id"]: job for job in queue["jobs"]}
    actual = {}
    for job_id, job in jobs.items():
        value = _values(job)
        actual[job_id] = (
            job["warm_start"]["parent"],
            value["task.rewards.free_wrist_vel_mimic"],
            value["task.rewards.motion_scale_in_window"],
            value["task.rewards.joint_velocity_limit_hinge_weight"],
            value["task.rewards.racket_face_conditional_guidance_weight"],
            value["task.rewards.foot_orientation_weight"],
            value["task.rewards.free_non_striking_arm_mimic"],
        )
        assert value["task.env.episode_length_s"] == "10.0"
        assert (
            value["task.rewards.racket_position_weight"],
            value["task.rewards.racket_velocity_weight"],
            value["task.rewards.racket_normal_weight"],
        ) == ("14.0", "10.0", "5.0")
        assert job["budget"] == {
            "num_envs": 4096, "max_iterations": 5001, "save_interval": 100
        }
        assert job["milestones"] == [3700, 4000, 4500, 5500, 7500]
    assert actual == {
        "demo_qdot_v1v2_face_w0p4": ("qdot", "true", "0.25", "-5.0", "-0.4", "-0.3", "false"),
        "demo_qdot_v1v2_face_w0p2": ("qdot", "true", "0.25", "-5.0", "-0.2", "-0.3", "false"),
        "demo_v1v2_qdot_w5_face_w0p4": ("v1v2", "true", "0.25", "-5.0", "-0.4", "-0.3", "false"),
        "demo_v1v2_qdot_w2p5_face_w0p4_free_arm": ("v1v2", "true", "0.25", "-2.5", "-0.4", "-0.3", "true"),
        "demo_control_qdot_w5_face_w0p4": ("control", "false", "1.0", "-5.0", "-0.4", "-0.3", "false"),
        "demo_control_full_stack_free_arm_foot_w0p6": ("control", "true", "0.25", "-5.0", "-0.4", "-0.6", "true"),
    }


def test_activated_queue_round_robins_and_claim_binds_inexact_parent_receipt(tmp_path):
    queue, _path = _activated(tmp_path)
    plan = D.cmd_plan(queue)
    assert [row["resource"] for row in plan["assignments"]] == D.EXPECTED_SLOTS
    first = queue["jobs"][0]
    slot = next(slot for slot in D.Q.slots(queue) if slot.name == "pod2/gpu0")
    claim, argv = D._demo_claim(queue, first, slot)
    content = claim["content"]
    assert content["formal_exact_eligible"] is False
    assert content["demo_warm_start"]["checkpoint_sha256"] == "b" * 64
    assert content["demo_warm_start"]["hard_contract_sha256"] == "c" * 64
    assert content["demo_warm_start"]["training_launch_claim_sha256"] == "d" * 64
    assert content["activation_receipt"]["file_sha256"] == "a" * 64
    assert "checkpoint_tolerant=false" in argv
    assert "checkpoint_allow_missing_contract=false" in argv
    assert "checkpoint_allow_contract_mismatch=true" in argv
    assert any(item.endswith("model_3500.pt") and item.startswith("checkpoint_path=") for item in argv)
    assert argv[-1] == f"++training_launch_claim_sha256={claim['content_sha256']}"


def test_generic_fresh_queue_guard_remains_unchanged():
    with pytest.raises(D.Q.QueueError, match="supports fresh runs only"):
        D.Q.load_queue(QUEUE)


def test_activation_and_exactness_flags_fail_closed(tmp_path):
    raw = _raw()
    raw["jobs"][0]["warm_start"]["descendant_exact_eligible"] = True
    with pytest.raises(D.DemoQueueError, match="exact-ineligible"):
        D.load_queue(_write(tmp_path, raw))

    raw = _raw()
    raw["launch_authorized"] = True
    with pytest.raises(D.DemoQueueError, match="exactly follow activation"):
        D.load_queue(_write(tmp_path, raw))


def test_parent_attestation_is_separate_one_pod_no_retry_dry_run():
    queue = D.load_queue(QUEUE)
    result = D.cmd_parent_attest(queue, execute=False, confirm=None)
    assert result["automatic_activation"] is False
    assert result["automatic_retry"] is False
    command = " ".join(result["ssh_argv"])
    assert "162.43.172.181" in command
    assert "162.43.172.171" not in command
    assert "pkill" not in command
    assert "killall" not in command
