"""CPU-only safety/argv tests for the isolated VendorV2 N1 launcher."""

from __future__ import annotations

import copy
import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/launch_n1_measured_vendor_v2_diagnostic.py"
SPEC = importlib.util.spec_from_file_location("launch_measured_vendor_v2", SCRIPT)
launcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = launcher
SPEC.loader.exec_module(launcher)

MATERIALIZER_SCRIPT = SCRIPT.with_name("materialize_measured_action_ball_n1_bundle.py")
MATERIALIZER_SPEC = importlib.util.spec_from_file_location(
    "materialize_measured_vendor_v2_roundtrip", MATERIALIZER_SCRIPT
)
materializer = importlib.util.module_from_spec(MATERIALIZER_SPEC)
sys.modules[MATERIALIZER_SPEC.name] = materializer
MATERIALIZER_SPEC.loader.exec_module(materializer)


def _canonical_write(path: Path, value) -> str:
    raw = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _materializations(tmp_path: Path):
    reward_payload = {"schema_version": 1, "terms": []}
    reward_sha = launcher.canonical_sha256(reward_payload)
    reward_path = tmp_path / "materialized" / "reward.json"
    reward_file_sha = _canonical_write(
        reward_path, {**reward_payload, "sha256": reward_sha}
    )
    policy_recipe = {"policy_initialization": {}}
    policy_sha = launcher.canonical_sha256(policy_recipe)
    policy_path = tmp_path / "materialized" / "policy.json"
    policy_file_sha = _canonical_write(
        policy_path,
        {
            "schema_version": 1,
            "kind": "action_ball_shared_ready_policy_recipe_materialization_v1",
            "action_count": 1,
            "action_order": [launcher.ACTION_ID],
            "policy_contract_sha256": policy_sha,
            "action_ball_ppo_runner_recipe": {
                "schema_version": 1,
                "sha256": policy_sha,
                "recipe": policy_recipe,
            },
            "policy_bootstrap": {},
        },
    )
    return {
        "reward": {"path": str(reward_path), "sha256": reward_file_sha},
        "reward_sha": reward_sha,
        "policy": {"path": str(policy_path), "sha256": policy_file_sha},
        "policy_sha": policy_sha,
    }


def _spec(
    tmp_path: Path, recipe: str = "current_lm", stage: str = "smoke"
):
    checkout = tmp_path / "checkout"
    checkout.mkdir(parents=True, exist_ok=True)
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\n")
    python.chmod(0o755)
    namespace_parent = tmp_path / launcher.EXPERIMENT_NAME
    namespace_parent.mkdir(parents=True, exist_ok=True)
    namespace = namespace_parent / ("n1_%s" % recipe)
    budget = launcher.BUDGETS[stage]
    materialized = _materializations(tmp_path)
    reward_pin = None if stage == "materialize" else materialized["reward"]
    policy_pin = (
        None if stage in ("materialize", "recipe") else materialized["policy"]
    )
    return {
        "schema_version": launcher.SCHEMA_VERSION,
        "kind": launcher.SPEC_KIND,
        "source": {
            "checkout": str(checkout),
            "commit_sha": "a" * 40,
            "isaac_python": str(python),
        },
        "action_id": launcher.ACTION_ID,
        "bundle": {"path": "configs/bundle.json", "sha256": "b" * 64},
        "target_recipe": recipe,
        "target_validity_mask": list(launcher.RECIPES[recipe]),
        "reward_materialization": reward_pin,
        "policy_materialization": policy_pin,
        "policy_contract_sha256": (
            None if policy_pin is None else materialized["policy_sha"]
        ),
        "expected_effective_reward_recipe_sha256": (
            None if reward_pin is None else materialized["reward_sha"]
        ),
        "seed": 0,
        "stage": stage,
        "num_envs": budget[0],
        "max_iterations": budget[1],
        "save_interval": budget[2],
        "gpu": {
            "index": 2,
            "uuid": "GPU-12345678",
            "owner": "Franco",
            "lock_path": "/tmp/hope_lean_queue_gpu2.lock",
            "require_empty": True,
        },
        "namespace": str(namespace),
        "log_path": str(namespace / "run.log"),
    }


def _bundle():
    return {
        "motion": {"path": "assets/motion.npz", "sha256": "1" * 64},
        "immutable_tape": {"path": "configs/tape.npz", "sha256": "2" * 64},
        "core": {
            "manifest": {"path": "configs/manifest.json", "sha256": "3" * 64},
            "dynamic_ready": {
                "artifact": {"path": "configs/ready.json", "sha256": "4" * 64},
                "nominal_hold_receipt": {
                    "path": "configs/hold.json",
                    "sha256": "5" * 64,
                },
            },
        },
    }


def _real_dynamic_ready_pair():
    checkout = Path(__file__).resolve().parents[3]
    core_path = checkout / (
        "configs/action_ball_n1_measured_20260803/"
        "fresh_core_seed0_20260803_take061_robust20n_r1/"
        "take_061_unit04_bh.full.bundle.v2.ff031a2caf05.json"
    )
    core = json.loads(core_path.read_text(encoding="utf-8"))
    dynamic = core["dynamic_ready"]
    artifact = json.loads(
        (checkout / dynamic["artifact"]["path"]).read_text(encoding="utf-8")
    )
    receipt = json.loads(
        (checkout / dynamic["nominal_hold_receipt"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    return checkout, core, dynamic, artifact, receipt


def _reseal_dynamic_ready(candidate):
    candidate.pop("content_sha256", None)
    candidate["content_sha256"] = launcher._B._canonical_ascii_sha256(candidate)


def _reseal_nominal_hold(receipt):
    receipt.pop("content_sha256", None)
    receipt["content_sha256"] = launcher.canonical_sha256(receipt)


def _validate_in_memory_dynamic_pair(
    monkeypatch: pytest.MonkeyPatch, core, dynamic, candidate, receipt
):
    values = iter(
        (
            (dynamic["artifact"], candidate),
            (dynamic["nominal_hold_receipt"], receipt),
        )
    )
    monkeypatch.setattr(
        launcher._B,
        "_load_tracked_json",
        lambda *args, **kwargs: next(values),
    )
    monkeypatch.setattr(
        launcher,
        "_load_training_contract_module",
        lambda checkout: types.SimpleNamespace(
            load_action_ball_dynamic_ready_runtime_binding=lambda **kwargs: {
                "schema_version": 2,
                "kind": "action_ball_dynamic_ready_runtime_binding_v2",
                "action_order": [launcher.ACTION_ID],
                "motion_sha256_per_action": [core["motion"]["sha256"]],
            }
        ),
    )
    return launcher._validate_measured_dynamic_ready_v2(
        Path("/unused"),
        "a" * 40,
        dynamic,
        action_id=launcher.ACTION_ID,
        motion_sha256=core["motion"]["sha256"],
    )


def test_real_schema_v2_dynamic_ready_and_hold_pair_is_accepted():
    checkout, core, dynamic, _candidate, _receipt = _real_dynamic_ready_pair()
    commit = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    result = launcher._validate_measured_dynamic_ready_v2(
        checkout,
        commit,
        dynamic,
        action_id=launcher.ACTION_ID,
        motion_sha256=core["motion"]["sha256"],
    )
    assert result == dynamic


def test_real_fresh_current_lm_bundle_crosses_solver_contact_and_dynamic_gates():
    checkout = Path(__file__).resolve().parents[3]
    relative = (
        "configs/action_ball_n1_measured_20260803/"
        "fresh_final_seed0_20260803_take061_robust20n_r3/"
        "take_061_unit04_bh.current_lm.measured_bundle.v1.70bacece099b.json"
    )
    path = checkout / relative
    commit = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    result = launcher._validate_bundle(
        checkout,
        commit,
        {"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
        action_id=launcher.ACTION_ID,
        recipe="current_lm",
        seed=0,
    )
    assert result["core"]["full_solver_admission_preflight"] == {
        "schema_version": 1,
        "kind": "full_fixed_action_exact_solver_admission_preflight_v1",
        "proposal_count": 512,
        "admitted_count": 421,
        "rejected_count": 91,
        "admit_rate": 421 / 512,
        "diagnostic_status": "PASS",
    }


@pytest.mark.parametrize(
    "mutation, expected_error",
    (
        ("unknown_candidate_field", "keys differ"),
        ("legacy_schema", "schema-v2"),
        ("action", "schema-v2"),
        ("motion", "schema-v2"),
        ("receipt_cross_pin", "nominal-hold receipt"),
    ),
)
def test_schema_v2_dynamic_ready_mutations_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    expected_error: str,
):
    _checkout, core, dynamic, candidate, receipt = _real_dynamic_ready_pair()
    dynamic = copy.deepcopy(dynamic)
    candidate = copy.deepcopy(candidate)
    receipt = copy.deepcopy(receipt)
    if mutation == "unknown_candidate_field":
        candidate["unexpected"] = True
        _reseal_dynamic_ready(candidate)
    elif mutation == "legacy_schema":
        candidate["schema_version"] = 1
        candidate["kind"] = "agibot_a3_action_dynamic_ready_candidate_v1"
        _reseal_dynamic_ready(candidate)
    elif mutation == "action":
        candidate["action_id"] = "take_060_unit00_bh"
        _reseal_dynamic_ready(candidate)
    elif mutation == "motion":
        candidate["sources"]["stable_motion"]["sha256"] = "0" * 64
        _reseal_dynamic_ready(candidate)
    elif mutation == "receipt_cross_pin":
        receipt["artifact"]["content_sha256"] = "0" * 64
        _reseal_nominal_hold(receipt)
    else:  # pragma: no cover
        raise AssertionError(mutation)
    with pytest.raises(launcher.LaunchRefused, match=expected_error):
        _validate_in_memory_dynamic_pair(
            monkeypatch, core, dynamic, candidate, receipt
        )


def test_spec_freezes_action_mask_budget_delay_wave_and_human_owner(tmp_path: Path):
    normalized = launcher._validate_spec(_spec(tmp_path, "teacher_pos_face_no_velocity"))
    assert normalized["action_id"] == launcher.ACTION_ID
    assert normalized["target_validity_mask"] == [True, False, True]
    assert (
        normalized["num_envs"],
        normalized["max_iterations"],
        normalized["save_interval"],
    ) == (1, 2, 1)

    wrong_mask = _spec(tmp_path, "analytic_no_velocity")
    wrong_mask["target_validity_mask"] = [True, True, True]
    with pytest.raises(launcher.LaunchRefused, match="validity"):
        launcher._validate_spec(wrong_mask)

    wrong_action = _spec(tmp_path, "outcome_dense_only")
    wrong_action["action_id"] = "take_060_unit00_bh"
    with pytest.raises(launcher.LaunchRefused, match="code-owned"):
        launcher._validate_spec(wrong_action)

    wrong_root = _spec(tmp_path / "wrong_root", "current_lm")
    namespace = Path(wrong_root["namespace"])
    bad_parent = namespace.parents[1] / "agibot_a3_action_ball_vendor_v1"
    bad_parent.mkdir(parents=True)
    wrong_root["namespace"] = str(bad_parent / namespace.name)
    wrong_root["log_path"] = str(Path(wrong_root["namespace"]) / "run.log")
    with pytest.raises(launcher.LaunchRefused, match="dedicated VendorV2"):
        launcher._validate_spec(wrong_root)


def test_materialize_then_recipe_then_training_identity_chain_is_fail_closed(
    tmp_path: Path,
):
    materialize = launcher._validate_spec(
        _spec(tmp_path / "materialize", stage="materialize")
    )
    assert materialize["policy_contract_sha256"] is None
    assert materialize["expected_effective_reward_recipe_sha256"] is None
    assert (materialize["num_envs"], materialize["max_iterations"]) == (1, 0)

    recipe = launcher._validate_spec(
        _spec(tmp_path / "recipe", stage="recipe")
    )
    assert recipe["reward_materialization"] is not None
    assert recipe["policy_contract_sha256"] is None
    assert (recipe["num_envs"], recipe["max_iterations"]) == (1, 0)

    smoke = _spec(tmp_path / "smoke", stage="smoke")
    smoke["policy_contract_sha256"] = "f" * 64
    with pytest.raises(launcher.LaunchRefused, match="materialized recipe"):
        launcher._validate_spec(smoke)

    bad_materialize = _spec(tmp_path / "bad", stage="materialize")
    bad_materialize["expected_effective_reward_recipe_sha256"] = "a" * 64
    with pytest.raises(launcher.LaunchRefused, match="must not predeclare"):
        launcher._validate_spec(bad_materialize)

    wrong_identity_arm = _spec(
        tmp_path / "wrong_arm",
        recipe="analytic_full",
        stage="materialize",
    )
    with pytest.raises(launcher.LaunchRefused, match="current_lm identity"):
        launcher._validate_spec(wrong_identity_arm)

    wrong_seed = _spec(tmp_path / "wrong_seed", stage="smoke")
    wrong_seed["seed"] = 1
    with pytest.raises(launcher.LaunchRefused, match="seed 0"):
        launcher._validate_spec(wrong_seed)


def test_finalize_bundle_schema_roundtrips_into_launcher_successor_contract():
    assert set(launcher.BUNDLE_KEYS) == set(materializer.FINAL_BUNDLE_KEYS)
    assert "immutable_tape_build_report" in launcher.BUNDLE_KEYS
    assert "immutable_tape_receipt" not in launcher.BUNDLE_KEYS


def test_training_argv_is_fresh_delay0_fixed_tape_virtual_ball_and_same_abi(tmp_path: Path):
    spec = launcher._validate_spec(_spec(tmp_path, "outcome_dense_only"))
    bundle = _bundle()
    argv = launcher._training_argv(spec, bundle)
    joined = "\n".join(argv)
    assert "task=%s" % launcher.TASK_PROFILE_ID in argv
    assert "task.actor_obs_contract=%s" % launcher.ACTOR_CONTRACT in argv
    assert "task.racket.action_ball_target_source=immutable_tape" in argv
    assert "task.racket.action_ball_target_recipe=outcome_dense_only" in argv
    assert "task.racket.action_ball_target_validity_mask=[false,false,false]" in argv
    assert "task.actions.control_step_action_delay_min=0" in argv
    assert "task.actions.control_step_action_delay_max=0" in argv
    assert "task.physical_ball=false" in argv
    assert "task.racket.adaptive_sigma=false" in argv
    assert argv.count(launcher.POLICY_NOISE_STD_OVERRIDE) == 1
    assert not any("checkpoint" in value or "resume" in value for value in argv)
    assert "DIAGNOSTIC_UNAUTHORIZED" in joined


def test_zero_ppo_reward_then_policy_recipe_argv_are_distinct(tmp_path: Path):
    reward_spec = launcher._validate_spec(
        _spec(tmp_path / "reward", stage="materialize")
    )
    reward_argv = launcher._training_argv(reward_spec, _bundle())
    assert "num_envs=1" in reward_argv
    assert "max_iterations=0" in reward_argv
    assert (
        "+n1_vendor_sigma_profile=" + launcher.REWARD_MATERIALIZATION_PROFILE
    ) in reward_argv
    assert any(
        value.startswith("+action_ball_effective_reward_recipe_output_path=")
        for value in reward_argv
    )
    assert not any(
        value.startswith("expected_effective_reward_recipe_sha256=")
        or value.startswith("action_ball_policy_recipe_output_path=")
        for value in reward_argv
    )
    assert (
        "task.racket.action_ball_policy_contract_sha256="
        + launcher.RECIPE_SENTINEL_POLICY_SHA256
    ) in reward_argv

    policy_spec = launcher._validate_spec(
        _spec(tmp_path / "policy", stage="recipe")
    )
    policy_argv = launcher._training_argv(policy_spec, _bundle())
    assert "max_iterations=0" in policy_argv
    assert any(
        value.startswith("expected_effective_reward_recipe_sha256=")
        for value in policy_argv
    )
    assert any(
        value.startswith("action_ball_policy_recipe_output_path=")
        for value in policy_argv
    )
    assert not any(
        value.startswith("+action_ball_effective_reward_recipe_output_path=")
        for value in policy_argv
    )
    assert launcher._output_contract(reward_spec)["ppo_update_count"] == 0
    assert launcher._output_contract(policy_spec)["ppo_update_count"] == 0


def test_policy_materialization_binds_dynamic_ready_and_log_std(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    binding_sha = "a" * 64
    bootstrap = {
        "schema_version": 3,
        "action_count": 1,
        "action_order": [launcher.ACTION_ID],
        "ready_source": {"identity": {"binding_sha256": binding_sha}},
        "initialization": {
            "noise_std_type": "log",
            "init_noise_std": 0.02,
            "required_realized_init_noise_std": 0.02,
        },
    }
    runner = {"policy_initialization": bootstrap}
    policy_sha = launcher.canonical_sha256(runner)
    path = tmp_path / "policy.json"
    file_sha = _canonical_write(
        path,
        {
            "schema_version": 1,
            "kind": "action_ball_shared_ready_policy_recipe_materialization_v1",
            "action_count": 1,
            "action_order": [launcher.ACTION_ID],
            "policy_contract_sha256": policy_sha,
            "action_ball_ppo_runner_recipe": {
                "schema_version": 1,
                "sha256": policy_sha,
                "recipe": runner,
            },
            "policy_bootstrap": bootstrap,
        },
    )
    contract = types.SimpleNamespace(
        load_action_ball_dynamic_ready_runtime_binding=lambda **kwargs: {
            "binding_sha256": binding_sha
        },
        validate_action_ball_policy_bootstrap=lambda *args, **kwargs: None,
        action_ball_policy_bootstrap_scientific_identity=(
            lambda value, repo_root: value
        ),
    )
    monkeypatch.setattr(
        launcher, "_load_training_contract_module", lambda checkout: contract
    )
    result = launcher._validate_policy_materialization(
        {"path": str(path), "sha256": file_sha},
        checkout=tmp_path,
        bundle=_bundle(),
    )
    assert result["policy_contract_sha256"] == policy_sha
    assert result["dynamic_ready_binding_sha256"] == binding_sha
    assert result["noise_std_type"] == "log"

    contract.load_action_ball_dynamic_ready_runtime_binding = lambda **kwargs: {
        "binding_sha256": "b" * 64
    }
    with pytest.raises(launcher.LaunchRefused, match="exact log-std"):
        launcher._validate_policy_materialization(
            {"path": str(path), "sha256": file_sha},
            checkout=tmp_path,
            bundle=_bundle(),
        )


def test_every_arm_uses_same_actor_contract_and_only_mask_recipe_change(tmp_path: Path):
    contracts = set()
    for recipe in launcher.RECIPES:
        spec = launcher._validate_spec(_spec(tmp_path / recipe, recipe))
        bundle = _bundle()
        argv = launcher._training_argv(spec, bundle)
        contracts.add(next(value for value in argv if value.startswith("task.actor_obs_contract=")))
    assert contracts == {"task.actor_obs_contract=%s" % launcher.ACTOR_CONTRACT}


def test_template_and_training_argv_preserve_venv_symlink_entry(tmp_path: Path):
    real_python = tmp_path / "real-python"
    real_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    real_python.chmod(0o755)
    venv_bin = tmp_path / "hope_isaac_venv" / "bin"
    venv_bin.mkdir(parents=True)
    venv_python = venv_bin / "python"
    venv_python.symlink_to(real_python)
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    namespace_parent = tmp_path / launcher.EXPERIMENT_NAME
    namespace_parent.mkdir()
    output = tmp_path / "materialize.json"

    launcher._write_template(
        types.SimpleNamespace(
            stage="materialize",
            namespace=str(namespace_parent / "fresh_materialize_r1"),
            reward_materialization_path=None,
            reward_materialization_sha256=None,
            policy_materialization_path=None,
            policy_materialization_sha256=None,
            checkout=str(checkout),
            commit_sha="a" * 40,
            isaac_python=str(venv_python),
            action_id=launcher.ACTION_ID,
            bundle_path="configs/bundle.json",
            bundle_sha256="b" * 64,
            target_recipe="current_lm",
            seed=0,
            gpu_index=0,
            gpu_uuid="GPU-12345678",
            owner="Franco",
            output=str(output),
        )
    )

    raw = json.loads(output.read_text(encoding="utf-8"))
    assert raw["source"]["isaac_python"] == str(venv_python)
    assert raw["source"]["isaac_python"] != str(venv_python.resolve())
    spec = launcher._validate_spec(raw)
    assert spec["source"]["isaac_python"] == str(venv_python)
    assert launcher._training_argv(spec, _bundle())[0] == str(venv_python)
