"""CPU-only fail-closed tests for the independent C225 launcher."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/launch_action_ball_c225_diagnostic.py"
)
SPEC = importlib.util.spec_from_file_location("launch_c225_diagnostic", SCRIPT)
launcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = launcher
SPEC.loader.exec_module(launcher)


def _write(path: Path, value) -> str:
    raw = launcher._B._canonical_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _sealed(value):
    return {**value, "content_sha256": launcher.canonical_sha256(value)}


def _lineage(checkout: Path) -> dict:
    action_id = launcher.ACTION_ID
    action_uid = launcher.ACTION_UID
    teacher_id = launcher.TEACHER_ID
    pins = {}

    motion_path = checkout / "motion.npz"
    motion_path.write_bytes(b"c225-motion\n")
    pins["motion"] = {
        "path": motion_path.name,
        "sha256": hashlib.sha256(motion_path.read_bytes()).hexdigest(),
    }

    tape_unsigned = {
        "schema_version": 1,
        "kind": "action_ball_n1_immutable_single_question_tape",
        "diagnostic_unauthorized": True,
        "row_count": 1,
        "question": {
            "action_uid": action_uid,
            "motion_sha256": pins["motion"]["sha256"],
            "ball_contact_w_m": [0.5, 0.0, 1.0],
            "incoming_velocity_w_mps": [-3.0, 0.1, 0.2],
            "incoming_spin_w_radps": [0.0, 10.0, 0.0],
        },
        "targets": {
            "outcome_dense_only": {
                "recipe": "outcome_dense_only",
                "validity_mask": [False, False, False],
            }
        },
        "reset_semantics": {"online_lm_calls": 0, "physical_rng_draws": 0},
    }
    tape = {
        **tape_unsigned,
        "canonical_sha256": launcher.canonical_sha256(tape_unsigned),
    }
    tape_path = checkout / "immutable_tape.json"
    pins["immutable_tape"] = {
        "path": tape_path.name,
        "sha256": _write(tape_path, tape),
    }

    manifest = {
        "schema_version": 3,
        "action_order": [action_id],
        "mobility_mode": "no_move",
        "actions": [
            {
                "action_id": action_id,
                "action_uid": action_uid,
                "motion_path": pins["motion"]["path"],
                "motion_sha256": pins["motion"]["sha256"],
            }
        ],
    }
    manifest_path = checkout / "action_manifest.json"
    pins["action_manifest"] = {
        "path": manifest_path.name,
        "sha256": _write(manifest_path, manifest),
    }

    dynamic_unsigned = {
        "schema_version": 1,
        "kind": "agibot_a3_action_dynamic_ready_candidate_v2",
        "action_id": action_id,
        "motion_sha256": pins["motion"]["sha256"],
    }
    dynamic_path = checkout / "dynamic_ready.json"
    pins["dynamic_ready_artifact"] = {
        "path": dynamic_path.name,
        "sha256": _write(dynamic_path, _sealed(dynamic_unsigned)),
    }
    nominal_unsigned = {
        "schema_version": 1,
        "kind": "agibot_a3_action_dynamic_ready_nominal_receipt_v1",
        "action_id": action_id,
        "motion_sha256": pins["motion"]["sha256"],
        "verdict": "PASS",
    }
    nominal_path = checkout / "dynamic_ready_nominal.json"
    pins["dynamic_ready_nominal_receipt"] = {
        "path": nominal_path.name,
        "sha256": _write(nominal_path, _sealed(nominal_unsigned)),
    }

    bundle_unsigned = {
        "schema_version": 1,
        "kind": launcher.C225_BUNDLE_KIND,
        "diagnostic_unauthorized": True,
        "action_id": action_id,
        "action_uid": action_uid,
        "teacher_id": teacher_id,
        "actor_contract": launcher.ACTOR_CONTRACT,
        "actor_width": launcher.ACTOR_WIDTH,
        "critic_contract": launcher.CRITIC_CONTRACT,
        "critic_width": launcher.CRITIC_WIDTH,
        "actor_normalizer_identity": launcher.ACTOR_NORMALIZER_IDENTITY,
        "critic_normalizer_identity": launcher.CRITIC_NORMALIZER_IDENTITY,
        "target_recipe": launcher.TARGET_RECIPE,
        "target_validity_mask": list(launcher.TARGET_VALIDITY_MASK),
        "incoming_ball_fields": list(launcher.INCOMING_BALL_FIELDS),
        "reset_inverse_solve": False,
        "online_solver_calls": 0,
        "online_lm_calls": 0,
        "physical_rng_draws": 0,
        "motion": pins["motion"],
        "immutable_tape": pins["immutable_tape"],
        "action_manifest": pins["action_manifest"],
        "dynamic_ready_artifact": pins["dynamic_ready_artifact"],
        "dynamic_ready_nominal_receipt": pins[
            "dynamic_ready_nominal_receipt"
        ],
    }
    bundle_path = checkout / "c225_bundle.json"
    pins["bundle"] = {
        "path": bundle_path.name,
        "sha256": _write(bundle_path, _sealed(bundle_unsigned)),
    }
    return {
        "schema_version": 1,
        "kind": launcher.LINEAGE_KIND,
        "actor_contract": launcher.ACTOR_CONTRACT,
        "actor_width": launcher.ACTOR_WIDTH,
        "critic_contract": launcher.CRITIC_CONTRACT,
        "critic_width": launcher.CRITIC_WIDTH,
        "actor_normalizer_identity": launcher.ACTOR_NORMALIZER_IDENTITY,
        "critic_normalizer_identity": launcher.CRITIC_NORMALIZER_IDENTITY,
        "task_profile": launcher.TASK_PROFILE_ID,
        "gym_task": launcher.GYM_TASK_ID,
        "target_semantics": launcher.TARGET_SEMANTICS,
        "target_recipe": launcher.TARGET_RECIPE,
        "target_validity_mask": list(launcher.TARGET_VALIDITY_MASK),
        "incoming_ball_fields": list(launcher.INCOMING_BALL_FIELDS),
        "reset_inverse_solve": False,
        "online_solver_calls": 0,
        "online_lm_calls": 0,
        "physical_rng_draws": 0,
        "action_id": action_id,
        "action_uid": action_uid,
        "teacher_id": teacher_id,
        "seed": 0,
        **pins,
    }


def _result(
    path: Path,
    *,
    stage: str,
    materialization,
    policy=None,
    oracle=None,
    predecessor=None,
    completion=None,
    output_contract=None,
) -> dict:
    budget = launcher.BUDGETS[stage]
    if completion is None:
        completion = {
            "completion_exit_code": "0",
            "terminal_kind": "clean_completion",
            "terminal_exit_code": "0",
        }
    if output_contract is None:
        output_contract = {
            "ppo_update_count": budget[1],
            "finite_model_save_interval": budget[2],
        }
    unsigned = {
        "schema_version": 1,
        "kind": launcher.RESULT_KIND,
        "diagnostic_unauthorized": True,
        "accepted": True,
        "launch_claim_sha256": "1" * 64,
        "stage": stage,
        "namespace": "/tmp/c225-fixture-" + stage,
        "completion": completion,
        "gpu_admission": {"phase": "fixture"},
        "output_contract": output_contract,
        "reward_materialization": materialization,
        "policy_recipe_materialization": policy,
        "oracle32_receipt": oracle,
        "predecessor_result": predecessor,
    }
    return {"path": str(path), "sha256": _write(path, _sealed(unsigned))}


def _chain(tmp_path: Path, lineage_sha: str):
    recipe = launcher._recipe_contract()
    planned = launcher._planned_materialization(
        recipe=recipe, lineage={"lineage_sha256": lineage_sha}
    )
    reward_artifact = tmp_path / "c225.effective_reward.json"
    reward_artifact.write_text("fixture\n", encoding="utf-8")
    materialization_unsigned = {
        key: value for key, value in planned.items() if key != "content_sha256"
    }
    materialization_unsigned.update(
        {
            "runtime_effective_reward_artifact": {
                "path": str(reward_artifact),
                "sha256": hashlib.sha256(reward_artifact.read_bytes()).hexdigest(),
            },
            "runtime_effective_reward_sha256": "3" * 64,
            "runtime_effective_reward_term_count": 12,
            "runtime_soft_weights": {
                "death_penalty": -30.0,
                "joint_limit": -0.5,
                "qdes_limit_barrier": -0.5,
                "qdes_projection_penalty": -0.5,
            },
        }
    )
    materialization = _sealed(materialization_unsigned)
    materialize = _result(
        tmp_path / "materialize.result.json",
        stage="materialize",
        materialization=materialization,
    )
    policy_artifact = tmp_path / "c225.policy.json"
    policy_artifact.write_text("fixture\n", encoding="utf-8")
    policy = _sealed(
        {
            "schema_version": 1,
            "kind": launcher.POLICY_MATERIALIZATION_KIND,
            "diagnostic_unauthorized": True,
            "recipe_id": launcher.RECIPE_ID,
            "lineage_sha256": lineage_sha,
            "recipe_contract_sha256": recipe["recipe_contract_sha256"],
            "runtime_policy_recipe_artifact": {
                "path": str(policy_artifact),
                "sha256": hashlib.sha256(policy_artifact.read_bytes()).hexdigest(),
            },
            "runtime_policy_recipe_sha256": "4" * 64,
            "dynamic_ready_binding_sha256": "5" * 64,
            "noise_std_type": "log",
            "configured_and_realized_init_noise_std": 0.02,
        }
    )
    recipe_result = _result(
        tmp_path / "recipe.result.json",
        stage="recipe",
        materialization=materialization,
        policy=policy,
    )
    raw_oracle = tmp_path / "raw-oracle.json"
    raw_oracle.write_text("fixture\n", encoding="utf-8")
    oracle = _sealed(
        {
            "schema_version": 1,
            "kind": launcher.ORACLE32_KIND,
            "diagnostic_unauthorized": True,
            "verdict": "PASS",
            "episodes": 32,
            "recipe_id": launcher.RECIPE_ID,
            "lineage_sha256": lineage_sha,
            "recipe_contract_sha256": recipe["recipe_contract_sha256"],
            "reward_contract_sha256": materialization["reward_contract_sha256"],
            "runtime_effective_reward_sha256": "3" * 64,
            "runtime_policy_recipe_sha256": "4" * 64,
            "actor_contract": launcher.ACTOR_CONTRACT,
            "actor_width": 225,
            "critic_contract": launcher.CRITIC_CONTRACT,
            "critic_width": 318,
            "target_recipe": launcher.TARGET_RECIPE,
            "target_validity_mask": [False, False, False],
            "incoming_ball_fields": list(launcher.INCOMING_BALL_FIELDS),
            "reset_inverse_solve": False,
            "online_solver_calls": 0,
            "online_lm_calls": 0,
            "physical_rng_draws": 0,
            "seed": 0,
            "raw_oracle_artifact": {
                "path": str(raw_oracle),
                "sha256": hashlib.sha256(raw_oracle.read_bytes()).hexdigest(),
            },
        }
    )
    oracle_result = _result(
        tmp_path / "oracle32.result.json",
        stage="oracle32",
        materialization=materialization,
        policy=policy,
        oracle=oracle,
    )
    scale_result = _result(
        tmp_path / "scale4096.result.json",
        stage="scale4096",
        materialization=materialization,
        policy=policy,
        oracle=oracle,
        completion={
            "completion_exit_code": "0",
            "terminal_kind": "clean_completion",
            "terminal_exit_code": "0",
        },
        output_contract={
            "ppo_update_count": 5,
            "finite_model_save_interval": 1,
        },
    )
    return materialize, recipe_result, oracle_result, scale_result


def _case(tmp_path: Path, *, stage: str, allow_colocation: bool = False):
    checkout = tmp_path / "checkout"
    checkout.mkdir(parents=True)
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    python.chmod(0o755)
    lineage = _lineage(checkout)
    lineage_path = checkout / "c225_lineage.json"
    lineage_sha = _write(lineage_path, lineage)
    materialize, recipe, oracle, scale = _chain(tmp_path, lineage_sha)
    root = tmp_path / launcher.EXPERIMENT_NAME
    root.mkdir()
    namespace = root / ("c225-" + stage)
    budget = launcher.BUDGETS[stage]
    spec = {
        "schema_version": launcher.SCHEMA_VERSION,
        "kind": launcher.SPEC_KIND,
        "source": {
            "checkout": str(checkout),
            "commit_sha": "a" * 40,
            "isaac_python": str(python),
        },
        "recipe_id": launcher.RECIPE_ID,
        "lineage": {"path": lineage_path.name, "sha256": lineage_sha},
        "materialization_result": None if stage == "materialize" else materialize,
        "recipe_result": None if stage in ("materialize", "recipe") else recipe,
        "oracle32_result": oracle if stage in ("scale4096", "long4096") else None,
        "predecessor_result": scale if stage == "long4096" else None,
        "stage": stage,
        "num_envs": budget[0],
        "max_iterations": budget[1],
        "save_interval": budget[2],
        "gpu": {
            "index": 2,
            "uuid": "GPU-12345678",
            "owner": "Franco",
            "lock_path": "/tmp/hope_lean_queue_gpu2.lock",
            "require_empty": not allow_colocation,
        },
        "namespace": str(namespace),
        "log_path": str(namespace / "run.log"),
    }
    if allow_colocation:
        spec[launcher.COLOCATION_SPEC_KEY] = True
    spec_path = tmp_path / (stage + ".spec.json")
    _write(spec_path, spec)
    return spec_path, spec, lineage


def _runtime_source_fixture(checkout: Path):
    output = {}
    for path, label in launcher.RUNTIME_SOURCE_PATHS:
        target = checkout / path
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_text("fixture: %s\n" % label, encoding="utf-8")
        output[label] = {
            "path": path,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        }
    return output


def _patch_plan_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        launcher._B,
        "_verify_clean_source",
        lambda checkout, commit: {
            "checkout": str(checkout),
            "commit_sha": commit,
            "clean": True,
        },
    )
    monkeypatch.setattr(
        launcher,
        "_runtime_sources",
        lambda checkout, commit: _runtime_source_fixture(checkout),
    )
    monkeypatch.setattr(
        launcher._B,
        "_validate_runtime_asset_environment",
        lambda: {"kind": "test_runtime_assets"},
    )
    monkeypatch.setattr(
        launcher._B, "_validate_runtime_asset_claim", lambda value: None
    )

    def verify(checkout, commit, pin, *, name):
        path = checkout / pin["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == pin["sha256"]
        return dict(pin), path

    monkeypatch.setattr(launcher._B, "_verify_tracked_file", verify)


def test_code_owned_recipe_and_five_stage_chain_are_exact():
    recipe = launcher._recipe_contract()
    assert tuple(launcher.BUDGETS) == (
        "materialize",
        "recipe",
        "oracle32",
        "scale4096",
        "long4096",
    )
    assert launcher.BUDGETS == {
        "materialize": (1, 0, 1),
        "recipe": (1, 0, 1),
        "oracle32": (1, 0, 1),
        "scale4096": (4096, 5, 1),
        "long4096": (4096, 1000, 100),
    }
    assert recipe["recipe_id"] == "C0-corrected-phase-fixedlr"
    assert recipe["actor_width"] == 225
    assert recipe["critic_width"] == 318
    assert recipe["fresh_normalizers_required"] is True
    assert recipe["foreign_checkpoint_reuse_prohibited"] is True


def test_long_plan_seals_true_c225_question_and_fresh_state(tmp_path, monkeypatch):
    _patch_plan_environment(monkeypatch)
    spec_path, _spec, _lineage_doc = _case(tmp_path, stage="long4096")
    payload = launcher.build_plan(spec_path)["canonical_payload"]
    assert payload["fresh_only"] is True
    assert payload["ppo_updates_authorized"] == 0
    assert payload["output_contract"]["requested_ppo_update_count"] == 1000
    assert payload["output_contract"]["runtime_gate"] == "C225_ORACLE_NOT_IMPLEMENTED"
    assert payload["reset_inverse_solve"] is False
    assert payload["bundle"]["question_contract"] == {
        "target_source": "immutable_tape",
        "target_recipe": "outcome_dense_only",
        "target_validity_mask": [False, False, False],
        "target_observation_noise": False,
        "incoming_ball_fields": [
            "incoming_ball_contact_position_heading",
            "incoming_ball_contact_velocity_heading",
            "incoming_ball_contact_spin_heading",
        ],
        "desired_contact_fields_observed": False,
        "reset_inverse_solve": False,
        "online_solver_calls": 0,
        "online_lm_calls": 0,
        "physical_rng_draws": 0,
    }
    assert payload["bundle"]["normalizers"] == launcher._normalizer_contract()
    assert payload["bundle"]["checkpoint_contract"]["input"] is None
    assert payload["bundle"]["checkpoint_contract"]["state"] == "fresh_empty"
    assert payload["materialization_inputs"]["predecessor_result"]["completion"][
        "terminal_kind"
    ] == "clean_completion"


def test_training_argv_pins_c225_000_and_never_accepts_foreign_checkpoint(
    tmp_path, monkeypatch
):
    _patch_plan_environment(monkeypatch)
    spec_path, _spec, _lineage_doc = _case(tmp_path, stage="recipe")
    argv = launcher.build_plan(spec_path)["canonical_payload"]["training_argv"]
    joined = "\n".join(argv)
    assert "task=HOPEPingPongActionBallC225VendorV2N1Learnability" in joined
    assert "task.actor_obs_contract=action_ball_c225" in joined
    assert "task.racket.action_ball_target_recipe=outcome_dense_only" in joined
    assert "task.racket.action_ball_target_source=immutable_tape" in joined
    assert "task.racket.action_ball_target_validity_mask=[false,false,false]" in joined
    assert "task.racket.action_ball_target_observation_noise=false" in joined
    assert "task.racket.adaptive_sigma=false" in joined
    assert "task.racket.adaptive_sigma_monotonic=false" in joined
    assert "task.racket.adaptive_sigma_normal=false" in joined
    assert "task.racket.target_noise_white=0.0" in joined
    assert "task.racket.target_noise_ar1_sigma=0.0" in joined
    assert "task.racket.action_ball_immutable_tape_path=" in joined
    assert "task.racket.action_ball_immutable_tape_sha256=" in joined
    assert "task.racket.action_ball_diagnostic_unauthorized=true" in joined
    assert "algo.runner.empirical_normalization=true" in joined
    assert "resume=" not in joined.lower()
    assert "checkpoint=" not in joined.lower()
    assert "action_ball_a225" not in joined.lower()
    assert "l194" not in joined.lower()
    assert "current_lm" not in joined.lower()
    assert "online_solver" not in joined.lower()


@pytest.mark.parametrize(
    "key,bad",
    (
        ("actor_contract", "action_ball_a225"),
        ("actor_width", 194),
        ("critic_width", 225),
        ("actor_normalizer_identity", "action_ball_a225_actor_norm_v1"),
        ("target_recipe", "current_lm"),
        ("target_validity_mask", [True, True, True]),
        ("incoming_ball_fields", ["desired_contact_position"]),
        ("reset_inverse_solve", True),
        ("online_solver_calls", 1),
        ("online_lm_calls", 1),
        ("physical_rng_draws", 1),
        ("action_id", "take_061_unit05_bh"),
        ("action_uid", 1),
    ),
)
def test_lineage_rejects_foreign_or_noncausal_contract(
    tmp_path, monkeypatch, key, bad
):
    _patch_plan_environment(monkeypatch)
    spec_path, spec, lineage = _case(tmp_path, stage="materialize")
    lineage[key] = bad
    lineage_path = Path(spec["source"]["checkout"]) / spec["lineage"]["path"]
    spec["lineage"]["sha256"] = _write(lineage_path, lineage)
    _write(spec_path, spec)
    with pytest.raises(launcher.LaunchRefused):
        launcher.build_plan(spec_path)


def test_c_lineage_cannot_relabel_legacy_bundle_as_true_c225(tmp_path, monkeypatch):
    _patch_plan_environment(monkeypatch)
    spec_path, spec, lineage = _case(tmp_path, stage="materialize")
    checkout = Path(spec["source"]["checkout"])
    bundle_path = checkout / lineage["bundle"]["path"]
    bundle = json.loads(bundle_path.read_text())
    bundle["actor_contract"] = "action_ball_a225"
    bundle["actor_width"] = 194
    unsigned = dict(bundle)
    unsigned.pop("content_sha256")
    bundle["content_sha256"] = launcher.canonical_sha256(unsigned)
    lineage["bundle"]["sha256"] = _write(bundle_path, bundle)
    lineage_path = checkout / spec["lineage"]["path"]
    spec["lineage"]["sha256"] = _write(lineage_path, lineage)
    _write(spec_path, spec)
    with pytest.raises(launcher.LaunchRefused):
        launcher.build_plan(spec_path)


@pytest.mark.parametrize("stage", tuple(launcher.BUDGETS))
def test_stage_receipt_requirements_are_exact(tmp_path, stage):
    spec_path, spec, _lineage_doc = _case(tmp_path, stage=stage)
    required_key = {
        "materialize": "materialization_result",
        "recipe": "recipe_result",
        "oracle32": "oracle32_result",
        "scale4096": "predecessor_result",
        "long4096": "predecessor_result",
    }[stage]
    if stage == "materialize":
        spec[required_key] = {"path": "/tmp/foreign", "sha256": "0" * 64}
    elif stage == "recipe":
        spec[required_key] = spec["materialization_result"]
    elif stage == "oracle32":
        spec[required_key] = spec["recipe_result"]
    elif stage == "scale4096":
        spec[required_key] = spec["oracle32_result"]
    else:
        spec[required_key] = None
    _write(spec_path, spec)
    with pytest.raises(launcher.LaunchRefused):
        launcher._validate_spec(spec)


def test_cross_lineage_or_broken_scale_terminal_is_rejected(tmp_path, monkeypatch):
    _patch_plan_environment(monkeypatch)
    spec_path, spec, _lineage_doc = _case(tmp_path, stage="long4096")
    predecessor = json.loads(Path(spec["predecessor_result"]["path"]).read_text())
    predecessor["completion"]["terminal_kind"] = "launch_accepted"
    predecessor_unsigned = dict(predecessor)
    predecessor_unsigned.pop("content_sha256")
    predecessor["content_sha256"] = launcher.canonical_sha256(predecessor_unsigned)
    spec["predecessor_result"]["sha256"] = _write(
        Path(spec["predecessor_result"]["path"]), predecessor
    )
    _write(spec_path, spec)
    with pytest.raises(launcher.LaunchRefused, match="natural-exit receipt"):
        launcher.build_plan(spec_path)


@pytest.mark.parametrize("stage", ("materialize", "recipe", "oracle32"))
def test_upstream_result_requires_natural_exit_and_exact_budget(
    tmp_path, monkeypatch, stage
):
    _patch_plan_environment(monkeypatch)
    consumer = {
        "materialize": "recipe",
        "recipe": "oracle32",
        "oracle32": "scale4096",
    }[stage]
    spec_path, spec, _lineage_doc = _case(tmp_path, stage=consumer)
    key = {
        "materialize": "materialization_result",
        "recipe": "recipe_result",
        "oracle32": "oracle32_result",
    }[stage]
    result_path = Path(spec[key]["path"])
    result = json.loads(result_path.read_text())
    result["completion"]["terminal_exit_code"] = "125"
    unsigned = dict(result)
    unsigned.pop("content_sha256")
    result["content_sha256"] = launcher.canonical_sha256(unsigned)
    spec[key]["sha256"] = _write(result_path, result)
    _write(spec_path, spec)
    with pytest.raises(launcher.LaunchRefused, match="natural-exit receipt"):
        launcher.build_plan(spec_path)


@pytest.mark.parametrize(
    "consumer,key,artifact_key",
    (
        ("recipe", "materialization_result", "runtime_effective_reward_artifact"),
        ("oracle32", "recipe_result", "runtime_policy_recipe_artifact"),
        ("scale4096", "oracle32_result", "raw_oracle_artifact"),
    ),
)
def test_consumed_receipt_requires_live_matching_artifact(
    tmp_path, monkeypatch, consumer, key, artifact_key
):
    _patch_plan_environment(monkeypatch)
    spec_path, spec, _lineage_doc = _case(tmp_path, stage=consumer)
    result = json.loads(Path(spec[key]["path"]).read_text())
    receipt_key = {
        "materialization_result": "reward_materialization",
        "recipe_result": "policy_recipe_materialization",
        "oracle32_result": "oracle32_receipt",
    }[key]
    Path(result[receipt_key][artifact_key]["path"]).unlink()
    with pytest.raises(launcher.LaunchRefused):
        launcher.build_plan(spec_path)


def test_blocked_oracle_and_ppo_stages_refuse_before_lock_or_namespace(
    tmp_path, monkeypatch
):
    _patch_plan_environment(monkeypatch)
    for stage in launcher.BLOCKED_RUNTIME_STAGES:
        stage_path = tmp_path / stage
        stage_path.mkdir()
        spec_path, _spec, _lineage_doc = _case(stage_path, stage=stage)
        plan = launcher.build_plan(spec_path)
        monkeypatch.setattr(
            launcher._B,
            "_verify_clean_source",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("blocked stage reached source/lock path")
            ),
        )
        with pytest.raises(
            launcher.LaunchRefused, match="C225_ORACLE_NOT_IMPLEMENTED"
        ):
            launcher.execute(plan, confirm_claim=plan["launch_claim_sha256"])
        assert not Path(plan["canonical_payload"]["spec"]["namespace"]).exists()
        _patch_plan_environment(monkeypatch)


def test_internal_exec_replay_of_blocked_stage_refuses_before_source_or_gpu(
    tmp_path, monkeypatch
):
    _patch_plan_environment(monkeypatch)
    spec_path, _spec, _lineage_doc = _case(tmp_path, stage="oracle32")
    plan = launcher.build_plan(spec_path)
    claim_path = tmp_path / "blocked.claim.json"
    _write(claim_path, plan)
    monkeypatch.setattr(
        launcher,
        "_revalidate_claim_payload",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("blocked internal exec revalidated source")
        ),
    )
    with pytest.raises(
        launcher.LaunchRefused, match="C225_ORACLE_NOT_IMPLEMENTED"
    ):
        launcher._internal_exec(claim_path, plan["launch_claim_sha256"], -1)


def test_materialize_claim_is_vendor_admission_revalidatable_and_no_clobber(
    tmp_path, monkeypatch
):
    _patch_plan_environment(monkeypatch)
    spec_path, spec, _lineage_doc = _case(tmp_path, stage="materialize")
    plan = launcher.build_plan(spec_path)
    namespace = launcher._B._claim_namespace(plan)
    validated = launcher._ADMISSION._validate_namespace_claim(
        namespace,
        plan["launch_claim_sha256"],
        checkout=Path(spec["source"]["checkout"]),
        commit=spec["source"]["commit_sha"],
        gpu_index=2,
        gpu_uuid="GPU-12345678",
        require_colocation_opt_in=False,
    )
    assert validated["spec"]["stage"] == "materialize"
    assert validated["training_argv"] == plan["canonical_payload"]["training_argv"]
    original_claim = (namespace / "launch_claim.json").read_bytes()
    with pytest.raises(launcher.LaunchRefused):
        launcher._B._claim_namespace(plan)
    assert (namespace / "launch_claim.json").read_bytes() == original_claim


def test_claim_revalidation_detects_question_or_fresh_state_mutation(
    tmp_path, monkeypatch
):
    _patch_plan_environment(monkeypatch)
    spec_path, _spec, _lineage_doc = _case(tmp_path, stage="materialize")
    plan = launcher.build_plan(spec_path)
    namespace = launcher._B._claim_namespace(plan)
    payload = copy.deepcopy(plan["canonical_payload"])
    payload["bundle"]["question_contract"]["online_solver_calls"] = 1
    with pytest.raises(launcher.LaunchRefused):
        launcher._revalidate_claim_payload(payload)
    payload = copy.deepcopy(plan["canonical_payload"])
    payload["bundle"]["checkpoint_contract"]["input"] = str(namespace / "model.pt")
    with pytest.raises(launcher.LaunchRefused):
        launcher._revalidate_claim_payload(payload)


def test_default_empty_gpu_and_explicit_colocation_are_sealed(tmp_path):
    spec_path, spec, _lineage_doc = _case(tmp_path / "empty", stage="materialize")
    normalized = launcher._validate_spec(spec)
    assert normalized["gpu"]["require_empty"] is True
    assert normalized[launcher.COLOCATION_SPEC_KEY] is False

    _path, colocated, _lineage = _case(
        tmp_path / "colocated", stage="materialize", allow_colocation=True
    )
    normalized = launcher._validate_spec(colocated)
    assert normalized["gpu"]["require_empty"] is False
    assert normalized[launcher.COLOCATION_SPEC_KEY] is True


def test_confirm_digest_mismatch_blocks_materialize_before_source_or_lock(
    tmp_path, monkeypatch
):
    _patch_plan_environment(monkeypatch)
    spec_path, _spec, _lineage_doc = _case(tmp_path, stage="materialize")
    plan = launcher.build_plan(spec_path)
    monkeypatch.setattr(
        launcher._B,
        "_verify_clean_source",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("source touched")),
    )
    with pytest.raises(launcher.LaunchRefused, match="confirm-claim"):
        launcher.execute(plan, confirm_claim="0" * 64)
    assert not Path(plan["canonical_payload"]["spec"]["namespace"]).exists()


def test_mutated_plan_payload_blocks_before_source_lock_or_namespace(
    tmp_path, monkeypatch
):
    _patch_plan_environment(monkeypatch)
    spec_path, _spec, _lineage_doc = _case(tmp_path, stage="materialize")
    plan = launcher.build_plan(spec_path)
    plan["canonical_payload"]["bundle"]["question_contract"][
        "online_solver_calls"
    ] = 1
    monkeypatch.setattr(
        launcher._B,
        "_verify_clean_source",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("source touched")),
    )
    with pytest.raises(launcher.LaunchRefused, match="payload seal"):
        launcher.execute(plan, confirm_claim=plan["launch_claim_sha256"])
    assert not Path(plan["canonical_payload"]["spec"]["namespace"]).exists()


def test_prelaunch_gpu_refusal_does_not_claim_namespace(tmp_path, monkeypatch):
    _patch_plan_environment(monkeypatch)
    spec_path, _spec, _lineage_doc = _case(tmp_path, stage="materialize")
    plan = launcher.build_plan(spec_path)
    monkeypatch.setattr(launcher, "_open_gpu_shared_lock", lambda path: 91)
    monkeypatch.setattr(launcher, "_lock_gpu_admission", lambda fd: None)
    monkeypatch.setattr(launcher, "_unlock_gpu_admission", lambda fd: None)
    monkeypatch.setattr(launcher.os, "close", lambda fd: None)
    monkeypatch.setattr(
        launcher,
        "_verify_gpu_admission",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            launcher.LaunchRefused("fixture GPU admission refused")
        ),
    )
    with pytest.raises(launcher.LaunchRefused, match="GPU admission refused"):
        launcher.execute(plan, confirm_claim=plan["launch_claim_sha256"])
    assert not Path(plan["canonical_payload"]["spec"]["namespace"]).exists()


@pytest.mark.parametrize(
    "text",
    (
        "completion_exit_code=0\nterminal_kind=clean_completion\nterminal_exit_code=1\n",
        "completion_exit_code=0\nterminal_kind=stale_timeout\nterminal_exit_code=0\n",
        "completion_exit_code=0\nterminal_kind=clean_completion\n",
        "completion_exit_code=0\ncompletion_exit_code=0\n"
        "terminal_kind=clean_completion\nterminal_exit_code=0\n",
    ),
)
def test_completion_state_rejects_nonexact_or_duplicate_rows(tmp_path, text):
    path = tmp_path / "completion.state"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(launcher.LaunchRefused):
        launcher._validate_completion_state(path)


def test_launcher_never_sets_or_repurposes_home():
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"HOME"' not in source
    assert '"CODEX_HOME"' not in source
