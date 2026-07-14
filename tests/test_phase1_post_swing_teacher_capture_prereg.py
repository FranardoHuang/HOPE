import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/phase1_post_swing_teacher_capture_prereg_20260715.json"


def test_post_swing_teacher_capture_is_one_shot_and_not_training_authority():
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    assert plan["capture_source"]["commit"] == "cd65ed1d1e314d7f31c2df4cd1a57131ff28a902"
    assert plan["capture_source"]["ignored_runtime_asset"]["file_count"] == 46
    assert plan["capture_source"]["ignored_runtime_asset"]["symlinks_forbidden"] is True
    assert plan["status"] == "preregistered_capture_not_started"
    assert plan["simulation_only"] is True
    assert plan["capture_contract"] == {
        "pod": "pod2",
        "gpu": 1,
        "cuda_visible_devices": "1",
        "runtime_device": "cuda:0",
        "num_envs": 4096,
        "target_count": 4096,
        "max_inference_steps": 20000,
        "wrap_teleport": False,
        "post_swing_start_prob": 0.25,
        "root_linear_velocity_limit_mps": 2.0,
        "root_angular_velocity_limit_radps": 4.0,
        "output_directory": "/workspace/codexschema/phase1_post_swing_teacher_20260715/capture/control_model500_v1",
        "output_must_be_absent_before_one_shot": True,
        "capture_is_inference_only": True,
        "ppo_updates": 0,
        "natural_wrap_only": True,
        "timeout_or_failure_reset_states_forbidden": True,
    }
    assert plan["teacher_checkpoint"]["fresh_lineage"] == 1
    assert plan["teacher_checkpoint"]["nonfinite_floating_elements"] == 0
    assert plan["teacher_checkpoint"]["hard_contract"]["schema_version"] == 3
    assert plan["teacher_checkpoint"]["sha256"] == (
        "22f78f882397c48d1c8763186748935517d669cf4f36205baf69d07dc9e08a6a"
    )
    assert len(plan["ordered_motion_inputs"]) == 2
    assert len({row["sha256"] for row in plan["ordered_motion_inputs"]}) == 2
    assert all(len(row["sha256"]) == 64 for row in plan["ordered_motion_inputs"])
    authorization = plan["authorization"]
    assert authorization["capture_authorized"] is True
    assert authorization["first_reset_probe_authorized"] is False
    assert authorization["scientific_training_authorized"] is False
    assert authorization["second_seed_authorized"] is False
    assert authorization["judge_authorized"] is False
    assert authorization["hardware_authorized"] is False
    failure = plan["failure_policy"]
    assert failure["same_namespace_retry_forbidden"] is True
    assert failure["automatic_retry_forbidden"] is True
    assert failure["pod1_and_pod2_gpu0_forbidden"] is True


def test_post_swing_teacher_capture_source_bindings_match_main_bytes():
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    import hashlib

    for row in plan["capture_source"]["files"].values():
        path = ROOT / row["path"]
        raw = path.read_bytes()
        assert len(raw) == row["bytes"]
        assert hashlib.sha256(raw).hexdigest() == row["sha256"]


def test_recipe_derivation_cannot_keep_training_ownership_keys():
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    derivation = plan["runtime_recipe_derivation"]
    removed = set(derivation["remove_keys"])
    assert {
        "max_iterations",
        "algo.runner.save_interval",
        "run_name",
        "training_queue_claim_path",
        "training_run_binding_path",
        "training_launch_claim_sha256",
    } <= removed
    assert set(derivation["add_keys"]) == {
        "checkpoint",
        "task.motion.post_swing_capture_output_dir",
        "task.motion.post_swing_capture_target_count",
        "post_swing_capture_max_steps",
    }
    assert derivation["runtime_hard_contract_must_equal_teacher_checkpoint_hard_contract_before_first_state"] is True
