from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT
    / "configs/phase1_non_striking_arm_imitation_a01_v1r1_continuation_20260714.json"
)
RUNNER = ROOT / "scripts/run_phase1_non_striking_arm_imitation_a01_v1r1.py"


def load_module():
    spec = importlib.util.spec_from_file_location("non_striking_arm_a01_v1r1", RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


M = load_module()


def manifest():
    return M.load_manifest(MANIFEST)


def prior_control(data: dict):
    return M.load_prior_control(data, runtime_paths=False, repo_root=ROOT)[:2]


def compact_hard_contract(prior_manifest: dict, prior_module, cell_id: str) -> dict:
    shared = prior_manifest["shared_training_contract"]
    bank = prior_manifest["inputs"]["schema3_train_bank"]
    return {
        "schema_version": 3,
        "actor_obs_contract": shared["actor_observation_contract"],
        "actor_obs_total_dim": shared["actor_observation_dim"],
        "face_command_pairing": shared["face_command_pairing"],
        "mount_normal_sign_per_clip": shared["mount_normal_sign_per_clip"],
        "strike_phase_per_clip": shared["strike_phase_per_clip"],
        "motion_kinematics_exact": True,
        "motion_allow_legacy_link_origin_velocity": False,
        "motion_event_timing": {"mode": "disabled"},
        "motion_imitation_body_names": copy.deepcopy(
            prior_module.cell_map(prior_manifest)[cell_id]["body_names"]
        ),
        "joint_names": [f"j{i}" for i in range(31)],
        "action_joint_ids": list(range(31)),
        "joint_friction_coefficients": [0.0] * 31,
        "motion_clips": [
            {"sha256": prior_manifest["inputs"]["forehand_motion"]["sha256"]},
            {"sha256": prior_manifest["inputs"]["backhand_motion"]["sha256"]},
        ],
        "question_bank": {
            "sha256": bank["sha256"],
            "schema_version": 3,
            "split": "train",
            "source_family_sha256": bank["source_family_sha256"],
            "exact": True,
        },
    }


def write_json(tmp_path: Path, name: str, value: dict) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def test_manifest_and_plan_are_exact_one_time_a1_only():
    data = manifest()
    plan = M.build_plan(data, MANIFEST, RUNNER)
    assert data["prior_control"]["launcher_sha256"] == (
        "716279ec68ea1b1e22cc32e634e38cd9e81d4fc969b059d21ec7a1f8e081489f"
    )
    assert data["a0_existing_evidence"]["pid"] == 1811464
    assert data["a0_existing_evidence"]["hard_contract_sha256"] == (
        "14ef410be5bdcc341901b3678d5331a59af89382e07939ad2049210bf68c29f1"
    )
    assert plan["only_new_cell"] == "A1"
    assert plan["a0_restart_forbidden"] is True
    assert plan["writes_or_launches_performed"] is False
    command = plan["a1_command"]
    assert "run_name=phase1_non_striking_arm_A1_left_arm_free_seed17" in command
    assert not any("phase1_non_striking_arm_A0" in item for item in command)


@pytest.mark.parametrize(
    "mutator,match",
    [
        (lambda d: d["a0_existing_evidence"].__setitem__("pid", 1811465), "A0"),
        (
            lambda d: d["prior_control"].__setitem__(
                "exact_failure_message", "some other failure"
            ),
            "prior control",
        ),
        (
            lambda d: d["schema3_bank_metadata"].__setitem__(
                "physics_contract_sha256", "0" * 64
            ),
            "bank metadata",
        ),
        (
            lambda d: d["continuation_invariants"].__setitem__(
                "a0_launch_is_never_reissued", False
            ),
            "invariants",
        ),
    ],
)
def test_manifest_rejects_evidence_or_safety_drift(tmp_path, mutator, match):
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    mutator(data)
    with pytest.raises(M.ContinuationError, match=match):
        M.load_manifest(write_json(tmp_path, "manifest.json", data))


def test_corrected_verifier_accepts_compact_bank_but_frozen_v1_rejects_exactly(tmp_path):
    data = manifest()
    prior_module, prior_manifest = prior_control(data)
    contract_path = write_json(
        tmp_path, "training_contract.json",
        compact_hard_contract(prior_manifest, prior_module, "A0"),
    )
    bank = prior_manifest["inputs"]["schema3_train_bank"]
    metadata = {
        "sha256": bank["sha256"],
        "source_family_sha256": bank["source_family_sha256"],
        "physics_contract_sha256": bank["physics_contract_sha256"],
    }
    _, contract = M.verify_hard_contract_v1r1(
        contract_path, prior_manifest, "A0", metadata, prior_module
    )
    assert set(contract["question_bank"]) == {
        "sha256", "schema_version", "split", "source_family_sha256", "exact"
    }
    failure = M.reproduce_exact_v1_false_rejection(
        prior_module, contract_path, prior_manifest,
        data["prior_control"]["exact_failure_message"],
    )
    assert failure["classification"] == "outer_verifier_false_rejection_only"
    assert failure["exact_stderr_line"] == data["prior_control"]["exact_failure_line"]


def test_corrected_verifier_rejects_direct_physics_leaf_or_bank_drift(tmp_path):
    data = manifest()
    prior_module, prior_manifest = prior_control(data)
    bank = prior_manifest["inputs"]["schema3_train_bank"]
    metadata = {
        "sha256": bank["sha256"],
        "source_family_sha256": bank["source_family_sha256"],
        "physics_contract_sha256": bank["physics_contract_sha256"],
    }
    direct_leaf = compact_hard_contract(prior_manifest, prior_module, "A0")
    direct_leaf["question_bank"]["physics_contract_sha256"] = bank[
        "physics_contract_sha256"
    ]
    with pytest.raises(M.ContinuationError, match="question_bank shape"):
        M.verify_hard_contract_v1r1(
            write_json(tmp_path, "direct.json", direct_leaf),
            prior_manifest, "A0", metadata, prior_module,
        )
    drift = compact_hard_contract(prior_manifest, prior_module, "A0")
    drift["question_bank"]["sha256"] = "0" * 64
    with pytest.raises(M.ContinuationError, match="compact train-bank"):
        M.verify_hard_contract_v1r1(
            write_json(tmp_path, "drift.json", drift),
            prior_manifest, "A0", metadata, prior_module,
        )


def make_bank(tmp_path: Path):
    physics = "1" * 64
    family = {"artifact_kind": "test_family", "physics_contract_sha256": physics}
    family_sha = M.canonical_sha256(family)
    meta = {
        "schema_version": 3,
        "split": "train",
        "source_family_sha256": family_sha,
        "source_family_contract": family,
        "physics_contract_sha256": physics,
    }
    path = tmp_path / "bank.npz"
    raw = json.dumps(meta, sort_keys=True, separators=(",", ":")).encode("utf-8")
    np.savez(path, meta_json=np.frombuffer(raw, dtype=np.uint8))
    expected = {
        "sha256": M.sha256_file(path),
        "schema_version": 3,
        "split": "train",
        "source_family_sha256": family_sha,
        "physics_contract_sha256": physics,
    }
    return path, expected


def test_bank_metadata_is_independently_parsed_and_binds_physics(tmp_path):
    path, expected = make_bank(tmp_path)
    result = M.verify_schema3_bank_metadata(path, expected)
    assert result["physics_contract_sha256"] == expected["physics_contract_sha256"]
    wrong = {**expected, "physics_contract_sha256": "2" * 64}
    with pytest.raises(M.ContinuationError, match="metadata physics_contract_sha256"):
        M.verify_schema3_bank_metadata(path, wrong)
    path.write_bytes(path.read_bytes() + b"drift")
    with pytest.raises(M.ContinuationError, match="file SHA"):
        M.verify_schema3_bank_metadata(path, expected)


def test_attestation_and_a1_launch_contract_bind_both_control_generations(tmp_path):
    data = manifest()
    prior_module, prior_manifest, prior_manifest_path, prior_launcher_path = (
        M.load_prior_control(data, runtime_paths=False, repo_root=ROOT)
    )
    a0 = {
        "hard_contract_sha256": data["a0_existing_evidence"]["hard_contract_sha256"],
        "old_failure_reproduction": {
            "exact_message": data["prior_control"]["exact_failure_message"]
        },
    }
    attestation = M.build_recovery_attestation(
        data, MANIFEST, RUNNER, prior_manifest_path, prior_launcher_path, a0
    )
    assert attestation["content_sha256"] == M.canonical_sha256(attestation["content"])
    attestation_path = write_json(tmp_path, "attestation.json", attestation)
    preflight = {
        "verified_inputs": {"inputs": "exact"},
        "ignored_asset": {"asset": "exact"},
        "training_module_path": "/exact/module.py",
    }
    launch = M.expected_a1_launch_contract(
        data, MANIFEST, RUNNER, prior_manifest, prior_module, preflight,
        attestation_path, M.sha256_file(attestation_path),
        {"gpu": 0, "compute_pids": [1811464], "trainer_pids": [1811464]},
    )
    assert launch["prior_manifest_sha256"] == data["prior_control"]["manifest_sha256"]
    assert launch["prior_launcher_sha256"] == data["prior_control"]["launcher_sha256"]
    assert launch["prior_a0_launch_contract_sha256"] == data["a0_existing_evidence"]["launch_contract_sha256"]
    assert launch["prior_a0_launch_state_sha256"] == data["a0_existing_evidence"]["launch_state_sha256"]
    assert launch["prior_a0_hard_contract_sha256"] == data["a0_existing_evidence"]["hard_contract_sha256"]
    assert launch["cell_id"] == "A1"
    assert launch["a0_restart_performed"] is False


def test_a1_preexistence_is_absorbing_and_fail_closed(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    logs = checkout / "wbt/logs/rsl_rl/agibot_a3_hope_virtualball"
    logs.mkdir(parents=True)
    prior_manifest = {
        "source": {"training_checkout": str(checkout), "wbt_relative_path": "wbt"}
    }
    arm = tmp_path / "a1-claim"
    arm.mkdir()
    monkeypatch.setitem(M.EXPECTED_A1, "arm_dir", str(arm))
    monkeypatch.setitem(M.EXPECTED_A1, "run_name", "impossible_unique_v1r1_test_run")
    with pytest.raises(M.ContinuationError, match="claim already exists"):
        M.require_a1_absent(prior_manifest, object())


def test_launch_requires_root_and_exact_confirmation_before_runtime_access(monkeypatch):
    data = manifest()
    monkeypatch.setattr(M.os, "geteuid", lambda: 501)
    with pytest.raises(M.ContinuationError, match="requires root"):
        M.launch_a1(data, MANIFEST, RUNNER, data["a1_continuation"]["root_launch_confirmation"])
    monkeypatch.setattr(M.os, "geteuid", lambda: 0)
    with pytest.raises(M.ContinuationError, match="confirmation token"):
        M.launch_a1(data, MANIFEST, RUNNER, "wrong")


def test_source_has_no_broad_signal_robot_command_or_a0_launch():
    text = RUNNER.read_text(encoding="utf-8")
    for forbidden in ("pkill", "killall", "os.kill", "signal.", "scripts/run_deploy"):
        assert forbidden not in text
    assert 'subprocess.run(["ros2"' not in text
    assert 'subprocess.Popen(["ros2"' not in text
    assert "build_command(prior_manifest, \"A0\")" in text  # identity verification only
    assert "build_command(prior_manifest, \"A1\")" in text  # sole launched command
    assert "a0_launch_is_never_reissued" in text
