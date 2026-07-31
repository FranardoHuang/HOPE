from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import pytest
import yaml


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/launch_a3_vendor_identity_smoke.py"
)
SPEC = importlib.util.spec_from_file_location("a3_vendor_identity_smoke", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
L = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(L)

REWARD_SHA = L.EXPECTED_REWARD_RECIPE_SHA256
POLICY_SHA = "b" * 64


def _canonical(value) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _spec(tmp_path: Path, *, stage: str = "recipe") -> dict:
    checkout = tmp_path / "checkout"
    checkout.mkdir(exist_ok=True)
    isaac_python = tmp_path / "isaac-python"
    isaac_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    isaac_python.chmod(0o755)
    runs = tmp_path / "runs"
    runs.mkdir(exist_ok=True)
    namespace = runs / f"a3vendor-identity-{stage}-gpu0-r1"
    return {
        "schema_version": 1,
        "kind": L.SPEC_KIND,
        "source": {
            "checkout": str(checkout),
            "commit_sha": "c" * 40,
            "isaac_python": str(isaac_python),
        },
        "motion": dict(L.MOTION_PIN),
        "manifest": dict(L.MANIFEST_PIN),
        "expected_effective_reward_recipe_sha256": REWARD_SHA,
        "policy_contract_sha256": None if stage == "recipe" else POLICY_SHA,
        "seed": 0,
        "stage": stage,
        "num_envs": 1,
        "max_iterations": 1 if stage == "recipe" else 2,
        "save_interval": 1,
        "gpu": {
            "index": 0,
            "uuid": "GPU-00000000-0000-0000-0000-000000000000",
            "owner": "Franco",
            "lock_path": "/tmp/hope_lean_queue_gpu0.lock",
            "require_empty": True,
        },
        "namespace": str(namespace),
        "log_path": str(namespace / "run.log"),
    }


@pytest.mark.parametrize(
    ("stage", "iterations", "policy_sha"),
    (("recipe", 1, None), ("smoke", 2, POLICY_SHA)),
)
def test_only_exact_two_stage_protocol_is_accepted(
    tmp_path: Path, stage: str, iterations: int, policy_sha: str | None
) -> None:
    normalized = L._validate_spec_document(_spec(tmp_path, stage=stage))
    assert normalized["stage"] == stage
    assert normalized["num_envs"] == 1
    assert normalized["max_iterations"] == iterations
    assert normalized["save_interval"] == 1
    assert normalized["policy_contract_sha256"] == policy_sha


def test_stage_budget_seed_policy_and_long_fail_closed(tmp_path: Path) -> None:
    bad_budget = _spec(tmp_path, stage="smoke")
    bad_budget["max_iterations"] = 3
    with pytest.raises(L.LaunchRefused, match="exactly 1 env / 2 PPO"):
        L._validate_spec_document(bad_budget)

    bad_seed = _spec(tmp_path, stage="smoke")
    bad_seed["seed"] = 1
    with pytest.raises(L.LaunchRefused, match="seed must be exactly 0"):
        L._validate_spec_document(bad_seed)

    missing_policy = _spec(tmp_path, stage="smoke")
    missing_policy["policy_contract_sha256"] = None
    with pytest.raises(L.LaunchRefused, match="policy_contract_sha256"):
        L._validate_spec_document(missing_policy)

    sentinel = _spec(tmp_path, stage="smoke")
    sentinel["policy_contract_sha256"] = L.RECIPE_SENTINEL_POLICY_SHA256
    with pytest.raises(L.LaunchRefused, match="sentinel"):
        L._validate_spec_document(sentinel)

    recipe_with_policy = _spec(tmp_path, stage="recipe")
    recipe_with_policy["policy_contract_sha256"] = POLICY_SHA
    with pytest.raises(L.LaunchRefused, match="must be null"):
        L._validate_spec_document(recipe_with_policy)

    long = _spec(tmp_path, stage="smoke")
    long["stage"] = "long"
    with pytest.raises(L.LaunchRefused, match="recipe.*smoke"):
        L._validate_spec_document(long)


def test_fixed_task_inputs_and_no_override_surface(tmp_path: Path) -> None:
    wrong_reward = _spec(tmp_path)
    wrong_reward["expected_effective_reward_recipe_sha256"] = "a" * 64
    with pytest.raises(L.LaunchRefused, match="fixed.*reward receipt"):
        L._validate_spec_document(wrong_reward)

    changed_motion = _spec(tmp_path)
    changed_motion["motion"] = {
        "path": "other.npz",
        "sha256": L.MOTION_PIN["sha256"],
    }
    with pytest.raises(L.LaunchRefused, match="fixed tracked.*motion"):
        L._validate_spec_document(changed_motion)

    changed_manifest = _spec(tmp_path)
    changed_manifest["manifest"] = {
        "path": L.MANIFEST_PIN["path"],
        "sha256": "d" * 64,
    }
    with pytest.raises(L.LaunchRefused, match="fixed tracked.*manifest"):
        L._validate_spec_document(changed_manifest)

    task_override = _spec(tmp_path)
    task_override["task"] = "HOPEPingPongActionBall"
    with pytest.raises(L.LaunchRefused, match="keys differ"):
        L._validate_spec_document(task_override)

    dynamic_override = _spec(tmp_path)
    dynamic_override["dynamic_ready"] = {"path": "old.json"}
    with pytest.raises(L.LaunchRefused, match="keys differ"):
        L._validate_spec_document(dynamic_override)


def test_fixed_reward_sha_is_justified_by_vendor_leaf_inheritance() -> None:
    checkout = Path(__file__).resolve().parents[3]
    task = yaml.safe_load((checkout / L.TASK_SOURCE).read_text(encoding="utf-8"))
    assert set(task) == {"defaults", "name", "actions", "push"}
    assert "rewards" not in task
    assert L.EXPECTED_REWARD_RECIPE_SHA256 == (
        "c2f13419a22fd12d1ab93d936516f8e990dad1b5b51a03f4e93c4d02e4e26c11"
    )


def test_namespace_and_no_clobber_are_identity_specific(tmp_path: Path) -> None:
    wrong_name = _spec(tmp_path)
    namespace = Path(wrong_name["namespace"]).with_name("generic-smoke")
    wrong_name["namespace"] = str(namespace)
    wrong_name["log_path"] = str(namespace / "run.log")
    with pytest.raises(L.LaunchRefused, match="a3vendor-identity"):
        L._validate_spec_document(wrong_name)

    spent = _spec(tmp_path)
    Path(spent["namespace"]).mkdir()
    with pytest.raises(L.LaunchRefused, match="permanently spent"):
        L._validate_spec_document(spent)

    bad_gpu = _spec(tmp_path)
    bad_gpu["gpu"]["uuid"] = "0"
    with pytest.raises(L.LaunchRefused, match="explicit GPU UUID"):
        L._validate_spec_document(bad_gpu)


def _scientific_inputs() -> dict:
    return {
        "motion": dict(L.MOTION_PIN),
        "manifest": {
            **dict(L.MANIFEST_PIN),
            "schema_version": 3,
            "action_order": [L.ACTION_ID],
        },
    }


@pytest.mark.parametrize("stage", ["recipe", "smoke"])
def test_training_argv_is_vendor_shared_ready_and_never_dynamic(
    tmp_path: Path, stage: str
) -> None:
    spec = L._validate_spec_document(_spec(tmp_path, stage=stage))
    argv = L._build_training_argv(spec, _scientific_inputs())
    assert f"task={L.TASK_PROFILE_ID}" in argv
    assert "action_ball_shared_ready_bootstrap=true" in argv
    assert "task.racket.action_ball_diagnostic_unauthorized=true" in argv
    assert "algo.policy.init_noise_std=0.02" in argv
    assert f"seed={L.SEED}" in argv
    assert "num_envs=1" in argv
    assert not any("task=HOPEPingPongActionBall" == item for item in argv)
    forbidden = (
        "action_ball_dynamic_ready",
        "checkpoint_path=",
        "stable_ready_plant",
        "task.push",
        "randomize_pd_gains",
        "kp_gain_range",
        "kd_gain_range",
        "control_step_action_delay_",
    )
    assert not any(fragment in item for item in argv for fragment in forbidden)
    recipe_args = [
        item for item in argv if item.startswith("action_ball_policy_recipe_output_path=")
    ]
    if stage == "recipe":
        assert recipe_args == [
            "action_ball_policy_recipe_output_path="
            + str(Path(spec["namespace"]) / L.RECIPE_FILENAME)
        ]
        assert (
            "task.racket.action_ball_policy_contract_sha256="
            + L.RECIPE_SENTINEL_POLICY_SHA256
        ) in argv
    else:
        assert recipe_args == []
        assert (
            "task.racket.action_ball_policy_contract_sha256=" + POLICY_SHA
        ) in argv


def test_output_contract_distinguishes_recipe_and_two_update_smoke(
    tmp_path: Path,
) -> None:
    recipe = L._validate_spec_document(_spec(tmp_path, stage="recipe"))
    recipe_output = L._rsl_output_contract(recipe)
    assert recipe_output["ppo_update_count"] == 0
    assert recipe_output["namespace_recipe"].endswith(L.RECIPE_FILENAME)
    assert recipe_output["training_contract"] is None
    assert recipe_output["checkpoints"] == []
    assert recipe_output["required_runtime_log_events"] == []

    smoke = L._validate_spec_document(_spec(tmp_path, stage="smoke"))
    smoke_output = L._rsl_output_contract(smoke)
    assert smoke_output["ppo_update_count"] == 2
    assert smoke_output["training_contract"] == "params/training_contract.json"
    assert smoke_output["checkpoints"] == ["model_0.pt", "model_1.pt"]
    assert smoke_output["required_runtime_log_events"] == [
        "HOPE_CONTROL_STEP_ACTION_DELAY_RUNTIME_JSON=",
        "HOPE_RSL_RL_RUNTIME_ABI_JSON=",
        "HOPE_POLICY_STD_UPDATE_JSON=",
    ]
    assert L.EXPERIMENT_NAME in smoke_output["rsl_experiment_root"]


def test_scientific_inputs_require_exact_n1_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = {
        "schema_version": 3,
        "action_order": [L.ACTION_ID],
        "mobility_mode": "no_move",
        "actions": [
            {
                "action_id": L.ACTION_ID,
                "motion_path": L.MOTION_PIN["path"],
                "motion_sha256": L.MOTION_PIN["sha256"],
            }
        ],
    }
    monkeypatch.setattr(
        L._S,
        "_verify_tracked_file",
        lambda *args, **kwargs: (dict(L.MOTION_PIN), tmp_path / "motion.npz"),
    )
    monkeypatch.setattr(
        L._S,
        "_load_tracked_json",
        lambda *args, **kwargs: (
            {
                **dict(L.MANIFEST_PIN),
                "schema_version": 3,
                "action_order": [L.ACTION_ID],
            },
            manifest,
        ),
    )
    result = L._validate_scientific_inputs(
        tmp_path, "c" * 40, L.MOTION_PIN, L.MANIFEST_PIN
    )
    assert result == _scientific_inputs()

    manifest["action_order"] = ["bh_block"]
    with pytest.raises(L.LaunchRefused, match="exact N=1"):
        L._validate_scientific_inputs(
            tmp_path, "c" * 40, L.MOTION_PIN, L.MANIFEST_PIN
        )


def test_dirty_source_refuses_before_any_runtime_or_gpu_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.name", "Test"], check=True
    )
    tracked = checkout / "tracked.txt"
    tracked.write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(checkout), "add", "tracked.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "commit", "-qm", "fixture"], check=True
    )
    commit = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tracked.write_text("dirty\n", encoding="utf-8")

    document = _spec(tmp_path, stage="recipe")
    document["source"]["checkout"] = str(checkout)
    document["source"]["commit_sha"] = commit
    spec_path = tmp_path / "dirty-spec.json"
    spec_path.write_bytes(_canonical(document))
    runtime_called = False

    def unexpected(*args, **kwargs):
        nonlocal runtime_called
        runtime_called = True
        raise AssertionError("runtime validation must not run after dirty source")

    monkeypatch.setattr(L, "_validate_runtime_sources", unexpected)
    with pytest.raises(L.LaunchRefused, match="clean|dirty|status"):
        L.build_plan(spec_path)
    assert runtime_called is False


def test_launch_refuses_occupied_gpu_before_namespace_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = L._validate_spec_document(_spec(tmp_path, stage="recipe"))
    plan = {
        "launch_claim_sha256": "e" * 64,
        "canonical_payload": {
            "spec": spec,
            "runtime_assets": {"pinned": True},
        },
    }
    lock_path = tmp_path / "gpu.lock"
    lock_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(L._S, "_verify_clean_source", lambda *args: {})
    monkeypatch.setattr(L._S, "_validate_runtime_asset_claim", lambda value: value)
    monkeypatch.setattr(L._S, "_open_gpu_lock", lambda path: os.open(lock_path, os.O_RDWR))
    monkeypatch.setattr(
        L._S,
        "_verify_gpu_empty",
        lambda *args: (_ for _ in ()).throw(L.LaunchRefused("GPU occupied")),
    )
    with pytest.raises(L.LaunchRefused, match="GPU occupied"):
        L.launch(plan, confirm_claim="e" * 64)
    assert not Path(spec["namespace"]).exists()


def test_internal_second_gpu_check_closes_plan_launch_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_spec = _spec(tmp_path, stage="smoke")
    namespace = Path(raw_spec["namespace"])
    namespace.mkdir()
    spec = L._validate_spec_document(raw_spec, namespace_claimed=True)
    lock_path = tmp_path / "race-gpu.lock"
    lock_path.write_text("", encoding="utf-8")
    spec["gpu"]["lock_path"] = str(lock_path)
    scientific = _scientific_inputs()
    argv = L._build_training_argv(spec, scientific)
    output = L._rsl_output_contract(spec)
    payload = {
        "kind": L.CLAIM_KIND,
        "spec": spec,
        "source": {"clean": True},
        "runtime_sources": {"launcher": {"path": L.LAUNCHER_SOURCE}},
        "runtime_assets": {"pinned": True},
        "scientific_inputs": scientific,
        "training_argv": argv,
        "output_contract": output,
    }
    claim_sha = L.canonical_sha256(payload)
    plan = {
        "schema_version": 1,
        "kind": L.CLAIM_KIND,
        "launch_claim_sha256": claim_sha,
        "canonical_payload": payload,
    }
    claim_path = namespace / "launch_claim.json"
    claim_path.write_bytes(_canonical(plan))
    monkeypatch.setattr(
        L,
        "_validate_spec_document",
        lambda document, namespace_claimed=False: spec,
    )
    monkeypatch.setattr(L._S, "_verify_clean_source", lambda *args: {"clean": True})
    monkeypatch.setattr(
        L, "_validate_runtime_sources", lambda *args: payload["runtime_sources"]
    )
    monkeypatch.setattr(L._S, "_validate_runtime_asset_claim", lambda value: value)
    monkeypatch.setattr(
        L, "_validate_scientific_inputs", lambda *args: scientific
    )
    monkeypatch.setattr(L, "_build_training_argv", lambda *args: argv)
    monkeypatch.setattr(L, "_rsl_output_contract", lambda *args: output)
    monkeypatch.setattr(
        L._S,
        "_verify_gpu_empty",
        lambda *args: (_ for _ in ()).throw(
            L.LaunchRefused("GPU occupied after namespace claim")
        ),
    )
    descriptor = os.open(lock_path, os.O_RDWR)
    try:
        with pytest.raises(L.LaunchRefused, match="after namespace claim"):
            L._internal_exec(claim_path, claim_sha, descriptor)
    finally:
        os.close(descriptor)
    assert not (namespace / "pre_exec_gpu_admission.json").exists()


def test_concrete_pod_template_has_no_hydra_override_channel(tmp_path: Path) -> None:
    args = argparse.Namespace(
        stage="smoke",
        checkout="/workspace/franco/a3vendor_commit",
        commit_sha="f" * 40,
        isaac_python="/workspace/hope_isaac_venv/bin/python",
        policy_contract_sha256=POLICY_SHA,
        gpu_index=2,
        gpu_uuid="GPU-473a79f3-8736-6c7f-c3db-290c6be385b8",
        owner="Franco",
        namespace="/workspace/franco/a3vendor-identity-smoke-gpu2-r1",
    )
    document = L._template_document(args)
    assert set(document) == set(L._SPEC_KEYS)
    assert document["stage"] == "smoke"
    assert document["num_envs"] == 1
    assert document["max_iterations"] == 2
    assert document["seed"] == 0
    assert document["gpu"]["lock_path"] == "/tmp/hope_lean_queue_gpu2.lock"
    assert "overrides" not in document


def test_runtime_source_set_pins_identity_delay_and_std_implementations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = Path(__file__).resolve().parents[3]
    observed: list[str] = []

    def verify(checkout_arg, commit, pin, *, name, **kwargs):
        observed.append(pin["path"])
        return pin, checkout_arg / pin["path"]

    monkeypatch.setattr(L._S, "_verify_tracked_file", verify)
    result = L._validate_runtime_sources(checkout, "f" * 40)
    assert L.LAUNCHER_SOURCE in observed
    assert L.TRAIN_SOURCE in observed
    assert L.TASK_SOURCE in observed
    assert L.ROBOT_SOURCE in observed
    assert L.TRAINING_CONTRACT_SOURCE in observed
    assert L.ACTION_SOURCE in observed
    assert L.RUNNER_SOURCE in observed
    assert L.KIT_LAUNCHER_SOURCE in observed
    assert len(result) == len(observed)
