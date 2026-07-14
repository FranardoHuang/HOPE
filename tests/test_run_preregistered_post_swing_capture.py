import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_preregistered_post_swing_capture.py"
SPEC = importlib.util.spec_from_file_location("post_swing_capture_runner", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def _plan(tmp_path):
    return {
        "schema_version": 2,
        "status": "preregistered_capture_not_started",
        "simulation_only": True,
        "capture_source": {"checkout": str(tmp_path / "source")},
        "teacher_checkpoint": {"path": "/checkpoint/model_500.pt"},
        "ordered_motion_inputs": [
            {"path": "/motions/f.npz"},
            {"path": "/motions/b.npz"},
        ],
        "question_bank": {"path": "/bank/train.npz"},
        "capture_contract": {
            "pod": "pod2",
            "gpu": 1,
            "cuda_visible_devices": "1",
            "runtime_device": "cuda:0",
            "num_envs": 4096,
            "target_count": 4096,
            "max_inference_steps": 20000,
            "seed": 3,
            "wrap_teleport": False,
            "post_swing_start_prob": 0.25,
            "root_linear_velocity_limit_mps": 2.0,
            "root_angular_velocity_limit_radps": 4.0,
            "capture_is_inference_only": True,
            "ppo_updates": 0,
            "natural_wrap_only": True,
            "output_must_be_absent_before_one_shot": True,
            "output_directory": "/workspace/codexschema/phase1_post_swing_teacher_20260715/capture/v2",
            "launch_root": "/workspace/codexschema/phase1_post_swing_teacher_20260715/launch/v2",
        },
        "runtime_recipe_derivation": {
            "remove_keys": [
                "logger", "video", "checkpoint_path", "checkpoint_tolerant",
                "checkpoint_allow_missing_contract", "checkpoint_allow_contract_mismatch",
                "max_iterations", "algo.runner.save_interval", "run_name",
                "training_queue_claim_path", "training_run_binding_path", "training_launch_claim_sha256",
            ],
            "seed_must_be_applied_by_play": True,
        },
        "authorization": {
            "capture_authorized": True,
            "attestation_authorized_only_after_complete_capture": True,
            "first_reset_probe_authorized": False,
            "scientific_training_authorized": False,
            "second_seed_authorized": False,
            "judge_authorized": False,
            "hardware_authorized": False,
        },
        "failure_policy": {
            "same_namespace_retry_forbidden": True,
            "automatic_retry_forbidden": True,
            "pod1_and_pod2_gpu0_forbidden": True,
        },
    }


def _binding(extra=None):
    args = [
        "/python", "/source/scripts/train.py", "task=HOPEPingPongVirtualBall", "algo=ppo",
        "headless=true", "device=cuda:0", "num_envs=4096", "seed=3",
        "task.motion.wrap_teleport=false", "task.motion.post_swing_start_prob=0.25",
        "motion_file=/motions/f.npz", "motion_file_2=/motions/b.npz",
        "++task.racket.question_bank=/bank/train.npz", "logger=tensorboard", "video=false",
        "checkpoint_path=null", "checkpoint_tolerant=false",
        "checkpoint_allow_missing_contract=false", "checkpoint_allow_contract_mismatch=false",
        "max_iterations=1001", "algo.runner.save_interval=100", "run_name=old",
        "++training_queue_claim_path=/claim", "++training_run_binding_path=/binding",
        "++training_launch_claim_sha256=" + "a" * 64,
    ]
    args.extend(extra or [])
    return {"content": {"training_argv": args}}


def test_derivation_removes_every_train_only_key_and_keeps_seed(tmp_path):
    argv = RUNNER._derive_argv(_plan(tmp_path), _binding())
    normalized = {RUNNER._normal_key(value): value.split("=", 1)[1] for value in argv[2:]}
    for key in (
        "logger", "video", "checkpoint_path", "checkpoint_tolerant",
        "checkpoint_allow_missing_contract", "checkpoint_allow_contract_mismatch",
        "max_iterations", "algo.runner.save_interval", "run_name",
        "training_queue_claim_path", "training_run_binding_path", "training_launch_claim_sha256",
    ):
        assert key not in normalized
    assert normalized["seed"] == "3"
    assert normalized["checkpoint"] == "/checkpoint/model_500.pt"
    assert normalized["task.motion.post_swing_capture_target_count"] == "4096"
    assert argv[1].endswith("hope_training/whole_body_tracking/scripts/play.py")


def test_derivation_fails_on_conflicting_duplicate(tmp_path):
    with pytest.raises(RUNNER.CaptureContractError, match="conflicting duplicate"):
        RUNNER._derive_argv(_plan(tmp_path), _binding(["seed=4"]))


def test_plan_rejects_missing_seed_parity_or_train_only_removal(tmp_path):
    plan = _plan(tmp_path)
    plan["runtime_recipe_derivation"]["seed_must_be_applied_by_play"] = False
    with pytest.raises(RUNNER.CaptureContractError, match="seed parity"):
        RUNNER._validate_plan(plan)
    plan = _plan(tmp_path)
    plan["runtime_recipe_derivation"]["remove_keys"].remove("checkpoint_tolerant")
    with pytest.raises(RUNNER.CaptureContractError, match="retains train-only"):
        RUNNER._validate_plan(plan)


def test_exclusive_writer_refuses_clobber(tmp_path):
    path = tmp_path / "receipt.json"
    RUNNER._exclusive_write(path, b"one")
    with pytest.raises(FileExistsError):
        RUNNER._exclusive_write(path, b"two")
    assert path.read_bytes() == b"one"
