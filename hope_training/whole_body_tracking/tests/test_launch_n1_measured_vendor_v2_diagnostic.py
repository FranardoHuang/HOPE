"""CPU-only safety/argv tests for the isolated VendorV2 N1 launcher."""

from __future__ import annotations

import ast
import copy
import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

import pytest
import yaml


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
        "fresh_core_seed0_20260803_take061_robust20n_r8_splitready/"
        "take_061_unit04_bh.full.bundle.v2.ddeed84329be.json"
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


@pytest.mark.parametrize(
    "recipe, basename, validity",
    (
        (
            "current_lm",
            "take_061_unit04_bh.current_lm.measured_bundle.v1.a223d4c99f29.json",
            [True, True, True],
        ),
        (
            "analytic_no_velocity",
            "take_061_unit04_bh.analytic_no_velocity.measured_bundle.v1.d3c2632cbd67.json",
            [True, False, True],
        ),
        (
            "outcome_dense_only",
            "take_061_unit04_bh.outcome_dense_only.measured_bundle.v1.589db83947b7.json",
            [False, False, False],
        ),
    ),
)
def test_real_fresh_split_ready_bundles_cross_all_launch_gates(
    recipe, basename, validity
):
    checkout = Path(__file__).resolve().parents[3]
    relative = (
        "configs/action_ball_n1_measured_20260803/"
        "fresh_final_seed0_20260803_take061_robust20n_r4_splitready/"
        + basename
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
        recipe=recipe,
        seed=0,
    )
    assert result["target_validity"] == {
        "order": ["position", "velocity", "face"],
        "mask": validity,
    }
    assert result["runtime_contract"]["target_source"] == "immutable_tape"
    assert result["runtime_contract"]["reset_inverse_solve"] is False
    assert result["core"]["dynamic_ready"] == {
        "artifact": {
            "path": (
                "configs/action_ball_n1_measured_20260803/"
                "evidence_holdpass_robust20n_20260803/"
                "take061.measured_teacher.yaw_aligned_full_seed.robust20n."
                "dynamic_ready.v2.json"
            ),
            "sha256": (
                "ab6b7e41ff129f91238835c533c8d589e68cc21f7e6184d639e95d8938d38069"
            ),
        },
        "nominal_hold_receipt": {
            "path": (
                "configs/action_ball_n1_measured_20260803/"
                "evidence_holdpass_robust20n_20260803/"
                "take061.robust20n.nominal_hold.v1.json"
            ),
            "sha256": (
                "c8b92a28203cbf9b9a4f6dee784d6cc08f3f279672d8a9fc886aa6d92b5bb19b"
            ),
        },
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


def test_cuda_launch_blocking_is_boolean_claim_owned_and_default_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "0")
    baseline = launcher._validate_spec(_spec(tmp_path, "current_lm"))
    assert baseline[launcher.CUDA_LAUNCH_BLOCKING_SPEC_KEY] is False
    assert launcher._cuda_launch_blocking_environment(baseline) == {}

    requested = _spec(tmp_path / "requested", "current_lm")
    requested[launcher.CUDA_LAUNCH_BLOCKING_SPEC_KEY] = True
    normalized = launcher._validate_spec(requested)
    assert normalized[launcher.CUDA_LAUNCH_BLOCKING_SPEC_KEY] is True
    assert launcher._cuda_launch_blocking_environment(normalized) == {
        "CUDA_LAUNCH_BLOCKING": "1"
    }
    assert launcher.canonical_sha256(normalized) != launcher.canonical_sha256(
        baseline
    )

    arbitrary = _spec(tmp_path / "arbitrary", "current_lm")
    arbitrary["environment"] = {"CUDA_LAUNCH_BLOCKING": "1"}
    with pytest.raises(launcher.LaunchRefused, match="keys differ"):
        launcher._validate_spec(arbitrary)


@pytest.mark.parametrize("value", (None, 0, 1, "1", [], {}))
def test_cuda_launch_blocking_rejects_non_boolean(tmp_path: Path, value):
    document = _spec(tmp_path, "current_lm")
    document[launcher.CUDA_LAUNCH_BLOCKING_SPEC_KEY] = value
    with pytest.raises(launcher.LaunchRefused, match="must be a boolean"):
        launcher._validate_spec(document)


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
    assert (
        "task.motion.action_ball_diagnostic_split_ready_teacher=true"
        in argv
    )
    assert "task.push.enable=false" in argv
    assert {
        value[len("~task.push.") :]
        for value in argv
        if value.startswith("~task.push.")
    } == set(launcher.DISABLED_PUSH_DORMANT_FIELDS)
    assert "task.physical_ball=false" in argv
    assert not any(
        value.lstrip("+").startswith("task.racket.physical_ball")
        for value in argv
    )
    assert "task.racket.adaptive_sigma=false" in argv
    assert argv.count(launcher.POLICY_NOISE_STD_OVERRIDE) == 1
    assert "algo.policy.noise_std_type=log" in argv
    assert "+algo.policy.noise_std_type=log" not in argv
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


def test_additive_hydra_overrides_only_name_absent_root_keys(tmp_path: Path):
    materialize = launcher._validate_spec(
        _spec(tmp_path / "materialize", stage="materialize")
    )
    recipe = launcher._validate_spec(_spec(tmp_path / "recipe", stage="recipe"))
    smoke = launcher._validate_spec(_spec(tmp_path / "smoke", stage="smoke"))
    expected_by_stage = {
        "materialize": {
        "+n1_vendor_sigma_profile",
        "+action_ball_effective_reward_recipe_output_path",
        },
        "recipe": set(),
        "smoke": set(),
    }
    for spec in (materialize, recipe, smoke):
        additions = {
            value.split("=", 1)[0]
            for value in launcher._training_argv(spec, _bundle())
            if value.startswith("+")
        }
        assert additions == expected_by_stage[spec["stage"]]
    train_yaml = (
        SCRIPT.parents[1] / "cfg" / "train.yaml"
    ).read_text(encoding="utf-8")
    assert "\nn1_vendor_sigma_profile:" not in train_yaml
    assert "\naction_ball_effective_reward_recipe_output_path:" not in train_yaml
    task_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for folder in ("task", "base")
        for path in sorted((SCRIPT.parents[1] / "cfg" / folder).glob("*.yaml"))
    )
    assert "\n  physical_ball:" not in task_sources
    assert "\n  physical_ball_impulse:" not in task_sources


def test_every_training_override_matches_composed_config_ownership(tmp_path: Path):
    cfg_root = SCRIPT.parents[1] / "cfg"

    def owned_paths(path: Path, prefix: str, seen=None):
        if seen is None:
            seen = set()
        identity = (path.resolve(), prefix)
        if identity in seen:
            return set()
        seen.add(identity)
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        result = set()

        def visit(value, parent):
            if not isinstance(value, dict):
                return
            for key, child in value.items():
                if key == "defaults":
                    continue
                owned = "%s.%s" % (parent, key) if parent else str(key)
                result.add(owned)
                visit(child, owned)

        visit(document, prefix)
        for default in document.get("defaults", []):
            if not isinstance(default, str) or default == "_self_":
                continue
            reference = default.split("@", 1)[0]
            inherited = (
                cfg_root / (reference[1:] + ".yaml")
                if reference.startswith("/")
                else path.parent / (reference + ".yaml")
            )
            if inherited.exists():
                result.update(owned_paths(inherited, prefix, seen))
        return result

    # Group selectors choose these exact two leaves; train.yaml's default task/algo
    # entries are intentionally not part of the selected composition.
    train_document = yaml.safe_load(
        (cfg_root / "train.yaml").read_text(encoding="utf-8")
    )
    train_document.pop("defaults")
    root_copy = tmp_path / "train_without_defaults.yaml"
    root_copy.write_text(yaml.safe_dump(train_document), encoding="utf-8")
    ownership = owned_paths(root_copy, "")
    ownership.update(
        owned_paths(
            cfg_root / "task" / (launcher.TASK_PROFILE_ID + ".yaml"), "task"
        )
    )
    ownership.update(owned_paths(cfg_root / "algo" / "ppo.yaml", "algo"))

    for stage in ("materialize", "recipe", "smoke"):
        spec = launcher._validate_spec(_spec(tmp_path / stage, stage=stage))
        argv = launcher._training_argv(spec, _bundle())
        assert argv[2:4] == ["task=%s" % launcher.TASK_PROFILE_ID, "algo=ppo"]
        for override in argv[4:]:
            additive = override.startswith("+")
            deletion = override.startswith("~")
            key = override.lstrip("+~").split("=", 1)[0]
            if deletion:
                assert "=" not in override
            assert (key not in ownership) if additive else (key in ownership), override


def test_no_push_argv_deletes_every_inherited_dormant_field(tmp_path: Path):
    cfg_root = SCRIPT.parents[1] / "cfg"

    def merge(base, overlay):
        result = copy.deepcopy(base)
        for key, value in overlay.items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        return result

    def compose_task(path: Path):
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        result = {}
        for default in document.get("defaults", []):
            if not isinstance(default, str) or default == "_self_":
                continue
            reference = default.split("@", 1)[0]
            inherited = (
                cfg_root / (reference[1:] + ".yaml")
                if reference.startswith("/")
                else path.parent / (reference + ".yaml")
            )
            if inherited.exists():
                result = merge(result, compose_task(inherited))
        return merge(result, {k: v for k, v in document.items() if k != "defaults"})

    task = compose_task(
        cfg_root / "task" / (launcher.TASK_PROFILE_ID + ".yaml")
    )
    assert task["push"]["enable"] is False
    assert set(task["push"]) == {
        "enable",
        *launcher.DISABLED_PUSH_DORMANT_FIELDS,
    }
    spec = launcher._validate_spec(_spec(tmp_path, stage="materialize"))
    argv = launcher._training_argv(spec, _bundle())
    for field in launcher.DISABLED_PUSH_DORMANT_FIELDS:
        assert "~task.push.%s" % field in argv
        task["push"].pop(field)
    # This is the exact clean disable shape accepted by
    # train._apply_push_robot_task_override: no loaded value except enable=false.
    assert task["push"] == {"enable": False}


def test_physical_ball_disable_uses_only_consumed_top_level_switch(tmp_path: Path):
    train_path = SCRIPT.with_name("train.py")
    train_tree = ast.parse(train_path.read_text(encoding="utf-8"))
    racket_keys = ast.literal_eval(
        next(
            node.value
            for node in train_tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "_RACKET_KEYS"
                for target in node.targets
            )
        )
    )
    assert "physical_ball" not in racket_keys
    assert "physical_ball_impulse" not in racket_keys

    task_yaml = yaml.safe_load(
        (
            SCRIPT.parents[1]
            / "cfg"
            / "task"
            / (launcher.TASK_PROFILE_ID + ".yaml")
        ).read_text(encoding="utf-8")
    )
    assert task_yaml["physical_ball"] is False
    for stage in ("materialize", "recipe", "smoke"):
        spec = launcher._validate_spec(_spec(tmp_path / stage, stage=stage))
        argv = launcher._training_argv(spec, _bundle())
        assert argv.count("task.physical_ball=false") == 1
        assert not any(
            value.lstrip("+").startswith("task.racket.physical_ball")
            for value in argv
        )

    command_source = (
        SCRIPT.parents[1]
        / "source"
        / "whole_body_tracking"
        / "whole_body_tracking"
        / "tasks"
        / "tracking"
        / "mdp"
        / "hope_commands.py"
    ).read_text(encoding="utf-8")
    env_source = (
        SCRIPT.parents[1]
        / "source"
        / "whole_body_tracking"
        / "whole_body_tracking"
        / "tasks"
        / "tracking"
        / "config"
        / "agibot_a3"
        / "hope_env_cfg.py"
    ).read_text(encoding="utf-8")
    assert "physical_ball: bool = False" in command_source
    assert "\n    physical_ball_impulse:" not in command_source
    assert 'getattr(cfg, "physical_ball_impulse", False)' in command_source
    assert "physical_ball: bool = False" in env_source


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
