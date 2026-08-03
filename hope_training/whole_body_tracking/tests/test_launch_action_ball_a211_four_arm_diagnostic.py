"""CPU-only fail-closed tests for the executable A211 four-arm launcher."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/launch_action_ball_a211_four_arm_diagnostic.py"
SPEC = importlib.util.spec_from_file_location("launch_a211_four_arm", SCRIPT)
launcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = launcher
SPEC.loader.exec_module(launcher)


class _FakeTensor:
    def __init__(self, values):
        self.values = list(values)

    def numel(self):
        return len(self.values)


class _FakeFinite:
    def __init__(self, tensor):
        self.tensor = tensor

    def all(self):
        return self

    def item(self):
        return all(math.isfinite(value) for value in self.tensor.values)


class _FakeTorch:
    checkpoint = None
    load_error = None

    @staticmethod
    def is_tensor(value):
        return isinstance(value, _FakeTensor)

    @staticmethod
    def isfinite(value):
        return _FakeFinite(value)

    @classmethod
    def load(cls, stream, *, map_location, weights_only):
        assert map_location == "cpu"
        assert weights_only is True
        assert stream.read()
        if cls.load_error is not None:
            raise cls.load_error
        return cls.checkpoint


def _write(path: Path, value) -> str:
    raw = launcher._B._canonical_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _sealed(value):
    return {**value, "content_sha256": launcher.canonical_sha256(value)}


def _terminal_acceptance_fixture():
    return _sealed(
        {
            "schema_version": 1,
            "kind": launcher.SCALE4096_TERMINAL_ACCEPTANCE_KIND,
            "diagnostic_unauthorized": True,
            "launch_claim_sha256": "1" * 64,
            "run_log": {"path": "/fixture/run.log", "size_bytes": 1, "sha256": "2" * 64},
            "checkpoint": {
                "path": "/fixture/model_5.pt",
                "size_bytes": 1,
                "sha256": "3" * 64,
                "filename_iteration": 5,
                "embedded_iteration": 5,
                "map_location": "cpu",
                "load_mode": "torch_weights_only",
                "tensor_groups": {},
                "all_tensors_finite": True,
            },
            "safety_counters": {
                "observed_ppo_updates": 5,
                "actual_hard_edge_event_count": 0,
                "actual_hard_terminal_count": 0,
                "hard_termination_count": 0,
                "table_contact_count": 0,
                "nonfinite_count": 0,
            },
        }
    )


def _live_safety(action_id: str, motion_sha: str, ticks: int) -> dict:
    names = ["joint_%02d" % index for index in range(31)]
    joint = {
        "schema_version": 1,
        "complete": True,
        "joint_order": names,
        "current_actual_hard_edge_joint_count": 0,
        "current_actual_hard_edge_joint_names": [],
        "substep_actual_hard_edge_joint_count": 0,
        "substep_actual_hard_edge_joint_names": [],
        "final_minimum_hard_gap_rad": 0.05,
        "preterminal_joint_pos_rad": [0.0] * 31,
        "preterminal_joint_vel_radps": [0.0] * 31,
        "final_joint_pos_rad": [0.0] * 31,
        "final_joint_vel_radps": [0.0] * 31,
        "hard_lower_rad": [-1.0] * 31,
        "hard_upper_rad": [1.0] * 31,
    }
    unsigned = {
        "schema_version": 1,
        "kind": launcher.FRAME0_LIVE_RECEIPT_KIND,
        "verdict": "PASS",
        "action_id": action_id,
        "motion_sha256": motion_sha,
        "teacher_reference_unchanged": True,
        "teacher_physical_birth_separated": False,
        "candidate_physical_birth_written": True,
        "candidate_hold_qdes_and_delay_history_installed": True,
        "plant_contract_match": True,
        "active_terminations": list(launcher.HARD_TERMINATION_UNION),
        "requested_duration_s": ticks * launcher.POLICY_DT_S,
        "completed_duration_s": ticks * launcher.POLICY_DT_S,
        "completed_policy_steps": ticks,
        "completed_physics_steps": ticks * 4,
        "terminal_reasons": [],
        "generic_terminated": False,
        "generic_truncated": False,
        "minimum_root_z_m": 0.9,
        "maximum_root_tilt_rad": 0.1,
        "both_feet_contact_fraction": 1.0,
        "joint_safety_telemetry": joint,
        "screenshots": [
            {"label": label, "sha256": ("%x" % (index + 1)) * 64}
            for index, label in enumerate((
                "raw_env_reset", "physical_ready_after_reset_write",
                "after_step_1", "after_step_10", "final",
            ))
        ],
    }
    return _sealed(unsigned)


def _required_effective_terms():
    return [
        {
            "name": name,
            "callable": requirement["callable"],
            "weight": 1.0,
            "params": dict(requirement["params"]),
        }
        for name, requirement in sorted(launcher.REQUIRED_EFFECTIVE_TERMS.items())
    ]


def _lineage(checkout: Path) -> dict:
    pins = {}
    for key in (
        "bundle",
        "immutable_tape",
        "action_manifest",
        "dynamic_ready_artifact",
        "dynamic_ready_nominal_receipt",
    ):
        raw = ("a211-%s\n" % key).encode()
        path = checkout / (key + ".bin")
        path.write_bytes(raw)
        pins[key] = {"path": path.name, "sha256": hashlib.sha256(raw).hexdigest()}
    root_pos = np.asarray(
        [
            [[-0.125, 0.375, 0.8125], [1.0, 2.0, 3.0]],
            [[9.0, 8.0, 7.0], [6.0, 5.0, 4.0]],
        ],
        dtype=np.float32,
    )
    root_quat = np.asarray(
        [
            [[0.5, 0.5, -0.5, 0.5], [1.0, 0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    joint_pos = (
        np.arange(62, dtype=np.float32).reshape(2, 31) / np.float32(17.0)
    )
    motion_path = checkout / "motion.npz"
    np.savez(
        motion_path,
        body_names=np.asarray(["pelvis_link", "torso_link"]),
        body_pos_w=root_pos,
        body_quat_w=root_quat,
        joint_pos=joint_pos,
    )
    pins["motion"] = {
        "path": motion_path.name,
        "sha256": hashlib.sha256(motion_path.read_bytes()).hexdigest(),
    }
    frame0_artifact = _sealed(
        {
            "schema_version": 1,
            "kind": launcher.FRAME0_EXACT_ARTIFACT_KIND,
            "diagnostic_unauthorized": True,
            "source_kind": launcher.FRAME0_EXACT_SOURCE_KIND,
            "action_id": "take_061_unit04_bh",
            "motion_sha256": pins["motion"]["sha256"],
            "task_close_ticks": 200,
            "policy_dt_s": launcher.POLICY_DT_S,
            "wait_schedule_canonical_sha256": launcher.WAIT_SCHEDULE[
                "canonical_sha256"
            ],
            "frame0": {
                "root_pos_w_m": root_pos[0, 0].tolist(),
                "root_quat_wxyz": root_quat[0, 0].tolist(),
                "root_lin_vel_w_mps": [0.0, 0.0, 0.0],
                "root_ang_vel_w_radps": [0.0, 0.0, 0.0],
                "joint_pos_rad": joint_pos[0].tolist(),
                "joint_vel_radps": [0.0] * 31,
            },
        }
    )
    frame0_artifact_path = checkout / "frame0_exact_artifact.json"
    frame0_artifact_sha = _write(frame0_artifact_path, frame0_artifact)
    live = _live_safety("take_061_unit04_bh", pins["motion"]["sha256"], 200)
    live_file_sha = hashlib.sha256(launcher._B._canonical_bytes(live)).hexdigest()
    frame0_receipt = _sealed(
        {
            "schema_version": 1,
            "kind": launcher.FRAME0_EXACT_RECEIPT_KIND,
            "diagnostic_unauthorized": True,
            "source_kind": launcher.FRAME0_EXACT_SOURCE_KIND,
            "verdict": "PASS",
            "action_id": "take_061_unit04_bh",
            "motion_sha256": pins["motion"]["sha256"],
            "artifact_file_sha256": frame0_artifact_sha,
            "artifact_content_sha256": frame0_artifact["content_sha256"],
            "artifact_source_commit": "a" * 40,
            "probe_source_commit": "b" * 40,
            "plant_template_file_sha256": "1" * 64,
            "plant_template_content_sha256": "2" * 64,
            "probe_input_file_sha256": "3" * 64,
            "probe_input_content_sha256": "4" * 64,
            "live_safety_evidence_file_sha256": live_file_sha,
            "live_safety_evidence_content_sha256": live["content_sha256"],
            "live_safety_evidence": live,
            "task_close_ticks": 200,
            "policy_dt_s": launcher.POLICY_DT_S,
            "wait_schedule_canonical_sha256": launcher.WAIT_SCHEDULE[
                "canonical_sha256"
            ],
        }
    )
    frame0_receipt_path = checkout / "frame0_exact_receipt.json"
    pins["frame0_exact_artifact"] = {
        "path": frame0_artifact_path.name,
        "sha256": frame0_artifact_sha,
    }
    pins["frame0_exact_receipt"] = {
        "path": frame0_receipt_path.name,
        "sha256": _write(frame0_receipt_path, frame0_receipt),
    }
    return {
        "schema_version": 2,
        "kind": launcher.LINEAGE_KIND,
        "actor_contract": launcher.ACTOR_CONTRACT,
        "actor_width": 211,
        "critic_contract": launcher.CRITIC_CONTRACT,
        "critic_width": 319,
        "trainability_contract": launcher.TRAINABILITY_CONTRACT,
        "actor_layout_identity": launcher._actor_layout_identity(),
        "task_profile": launcher.TASK_PROFILE_ID,
        "gym_task": launcher.GYM_TASK_ID,
        "target_semantics": launcher.TARGET_SEMANTICS,
        "curriculum_scope": launcher._curriculum_scope_contract(),
        "action_id": "take_061_unit04_bh",
        "teacher_id": "Take_061_unit04_BH",
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
    terminal_acceptance=None,
) -> dict:
    result_namespace = path.parent / (path.stem + ".namespace")
    result_namespace.mkdir(parents=True, exist_ok=True)
    unsigned = {
        "schema_version": 1,
        "kind": launcher.RESULT_KIND,
        "diagnostic_unauthorized": True,
        "accepted": True,
        "launch_claim_sha256": "1" * 64,
        "stage": stage,
        "namespace": str(result_namespace),
        "completion": (
            {"terminal_kind": "clean_completion"}
            if completion is None
            else completion
        ),
        "gpu_admission": {"phase": "post_completion"},
        "output_contract": (
            {"fixture": True} if output_contract is None else output_contract
        ),
        "arm_materialization": materialization,
        "policy_recipe_materialization": policy,
        "oracle32_receipt": oracle,
        "predecessor_result": predecessor,
    }
    if terminal_acceptance is not None:
        unsigned["terminal_acceptance"] = terminal_acceptance
    digest = _write(path, _sealed(unsigned))
    return {"path": str(path), "sha256": digest}


def _generated_chain(tmp_path: Path, arm_id: str, lineage_sha: str):
    arm = launcher._arm_contract(arm_id)
    planned = launcher._planned_materialization(
        arm=arm, lineage={"lineage_sha256": lineage_sha}
    )
    reward_artifact = tmp_path / (arm_id + ".effective_reward.json")
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
            "runtime_effective_reward_term_count": 10,
            "runtime_soft_weights": {
                "death_penalty": arm["soft_weights"]["death_penalty"],
                "joint_limit": arm["soft_weights"]["joint_limit"],
                "qdes_limit_barrier": arm["soft_weights"]["qdes_limit"],
                "qdes_projection_penalty": arm["soft_weights"]["qdes_projection"],
            },
        }
    )
    materialization = _sealed(materialization_unsigned)
    materialize = _result(
        tmp_path / (arm_id + ".materialize.json"),
        stage="materialize",
        materialization=materialization,
    )
    policy_artifact = tmp_path / (arm_id + ".policy_recipe.json")
    policy_artifact.write_text("fixture-policy\n", encoding="utf-8")
    policy = _sealed(
        {
            "schema_version": 1,
            "kind": launcher.POLICY_MATERIALIZATION_KIND,
            "diagnostic_unauthorized": True,
            "arm_id": arm_id,
            "lineage_sha256": lineage_sha,
            "arm_contract_sha256": arm["arm_contract_sha256"],
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
    recipe = _result(
        tmp_path / (arm_id + ".recipe.json"),
        stage="recipe",
        materialization=materialization,
        policy=policy,
    )
    oracle = _sealed(
        {
            "schema_version": 1,
            "kind": launcher.ORACLE32_KIND,
            "diagnostic_unauthorized": True,
            "verdict": "PASS",
            "episodes": 32,
            "arm_id": arm_id,
            "lineage_sha256": lineage_sha,
            "arm_contract_sha256": arm["arm_contract_sha256"],
            "reward_contract_sha256": materialization["reward_contract_sha256"],
            "runtime_effective_reward_sha256": "3" * 64,
            "policy_contract_sha256": "4" * 64,
            "runtime_policy_recipe_sha256": "4" * 64,
            "actor_contract": launcher.ACTOR_CONTRACT,
            "actor_width": 211,
            "critic_contract": launcher.CRITIC_CONTRACT,
            "critic_width": 319,
            "trainability_contract": launcher.TRAINABILITY_CONTRACT,
            "seed": 0,
            "raw_oracle_sha256": "2" * 64,
        }
    )
    oracle_result = _result(
        tmp_path / (arm_id + ".oracle32.json"),
        stage="oracle32",
        materialization=materialization,
        policy=policy,
        oracle=oracle,
    )
    smoke_result = _result(
        tmp_path / (arm_id + ".smoke.json"),
        stage="smoke",
        materialization=materialization,
        policy=policy,
        oracle=oracle,
    )
    probe_result = _result(
        tmp_path / (arm_id + ".probe512.json"),
        stage="probe512",
        materialization=materialization,
        policy=policy,
        oracle=oracle,
        predecessor={"stage": "smoke"},
    )
    scale_result = _result(
        tmp_path / (arm_id + ".scale4096.json"),
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
        terminal_acceptance=_terminal_acceptance_fixture(),
    )
    return (
        materialize,
        recipe,
        oracle_result,
        smoke_result,
        probe_result,
        scale_result,
    )


def _case(tmp_path: Path, *, arm_id: str, stage: str, allow_colocation: bool = False):
    checkout = tmp_path / "checkout"
    checkout.mkdir(parents=True)
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    python.chmod(0o755)
    lineage = _lineage(checkout)
    lineage_path = checkout / "a211_lineage.json"
    lineage_sha = _write(lineage_path, lineage)
    generated = _generated_chain(tmp_path, arm_id, lineage_sha)
    (
        materialize,
        recipe_result,
        oracle_result,
        smoke_result,
        probe_result,
        scale_result,
    ) = generated
    root = tmp_path / launcher.EXPERIMENT_NAME
    root.mkdir()
    namespace = root / (arm_id + "-" + stage)
    budget = launcher.BUDGETS[stage]
    spec = {
        "schema_version": launcher.SCHEMA_VERSION,
        "kind": launcher.SPEC_KIND,
        "source": {
            "checkout": str(checkout),
            "commit_sha": "a" * 40,
            "isaac_python": str(python),
        },
        "arm_id": arm_id,
        "lineage": {"path": lineage_path.name, "sha256": lineage_sha},
        "arm_materialization": None if stage == "materialize" else materialize,
        "policy_recipe_materialization": (
            None if stage in ("materialize", "recipe") else recipe_result
        ),
        "oracle32_receipt": (
            oracle_result
            if stage
            in ("smoke", "probe512", "long512", "scale4096", "long4096")
            else None
        ),
        "predecessor_result": (
            smoke_result
            if stage == "probe512"
            else probe_result
            if stage == "long512"
            else scale_result
            if stage == "long4096"
            else None
        ),
        "stage": stage,
        "num_envs": budget[0],
        "max_iterations": budget[1],
        "save_interval": budget[2],
        "wait_contract": launcher._wait_contract(),
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
    spec_path = tmp_path / (arm_id + "-" + stage + ".spec.json")
    _write(spec_path, spec)
    return spec_path, spec, lineage


def _patch_plan_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        launcher._B,
        "_verify_clean_source",
        lambda checkout, commit: {"checkout": str(checkout), "commit_sha": commit, "clean": True},
    )
    monkeypatch.setattr(launcher, "_runtime_sources", lambda checkout, commit: {})
    monkeypatch.setattr(
        launcher._B, "_validate_runtime_asset_environment", lambda: {"kind": "test_runtime_assets"}
    )

    def verify(checkout, commit, pin, *, name):
        path = checkout / pin["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == pin["sha256"]
        return dict(pin), path

    monkeypatch.setattr(launcher._B, "_verify_tracked_file", verify)
    monkeypatch.setattr(
        launcher, "_verify_frame0_artifact_source_commit", lambda *_args: None
    )
    monkeypatch.setattr(
        launcher, "_verify_frame0_probe_source_commit", lambda *_args: None
    )
    monkeypatch.setattr(
        launcher, "_verify_commit_ancestor", lambda *_args, **_kwargs: None
    )

    def runtime_policy(*, path, checkout, lineage, arm):
        return _sealed(
            {
                "schema_version": 1,
                "kind": launcher.POLICY_MATERIALIZATION_KIND,
                "diagnostic_unauthorized": True,
                "arm_id": arm["arm_id"],
                "lineage_sha256": lineage["lineage_sha256"],
                "arm_contract_sha256": arm["arm_contract_sha256"],
                "runtime_policy_recipe_artifact": {
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                },
                "runtime_policy_recipe_sha256": "4" * 64,
                "dynamic_ready_binding_sha256": "5" * 64,
                "noise_std_type": "log",
                "configured_and_realized_init_noise_std": 0.02,
            }
        )

    monkeypatch.setattr(launcher, "_runtime_policy_materialization", runtime_policy)
    monkeypatch.setattr(
        launcher,
        "_audit_scale4096_terminal",
        lambda **_kwargs: copy.deepcopy(_terminal_acceptance_fixture()),
    )


def _raw_oracle_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    qdes_manager_weight: float = -1.0,
    qdes_objective_weight: float | None = None,
    drift_raw_reward_sha: bool = False,
    hard_identity_mutation: dict | None = None,
) -> tuple[Path, dict]:
    _patch_plan_environment(monkeypatch)
    spec_path, _spec, _lineage_doc = _case(
        tmp_path, arm_id=launcher.ARM_IDS[0], stage="oracle32"
    )
    claim = launcher.build_plan(spec_path)["canonical_payload"]
    arm = claim["bundle"]["arm"]
    if qdes_objective_weight is None:
        qdes_objective_weight = arm["soft_weights"]["qdes_projection"]
    names_and_weights = {
        "death_penalty": arm["soft_weights"]["death_penalty"],
        "joint_limit": arm["soft_weights"]["joint_limit"],
        "qdes_limit_barrier": arm["soft_weights"]["qdes_limit"],
        "qdes_projection_penalty": arm["soft_weights"]["qdes_projection"],
    }
    terms = _required_effective_terms()
    for name, weight in sorted(names_and_weights.items()):
        terms.append(
            {
                "name": name,
                "callable": "fixture." + name,
                "weight": qdes_manager_weight if name == "qdes_projection_penalty" else weight,
                "params": (
                    {
                        "action_name": "joint_pos",
                        "shape_rate": 4.0,
                        "objective_weight": qdes_objective_weight,
                    }
                    if name == "qdes_projection_penalty"
                    else {}
                ),
            }
        )
    terms.sort(key=lambda term: term["name"])
    semantic = {"schema_version": 1, "terms": terms}
    reward_document = {
        **semantic,
        "sha256": launcher.canonical_sha256(semantic),
    }
    materialization = claim["materialization_inputs"]["arm_materialization"]
    reward_path = Path(materialization["runtime_effective_reward_artifact"]["path"])
    reward_file_sha = _write(reward_path, reward_document)
    materialization.update(
        {
            "runtime_effective_reward_artifact": {
                "path": str(reward_path),
                "sha256": reward_file_sha,
            },
            "runtime_effective_reward_sha256": reward_document["sha256"],
            "runtime_effective_reward_term_count": len(terms),
            "runtime_soft_weights": names_and_weights,
        }
    )

    hard_reward = copy.deepcopy(reward_document)
    if drift_raw_reward_sha:
        hard_reward["terms"].append(
            {
                "name": "unrelated_identity_drift",
                "callable": "fixture.unrelated_identity_drift",
                "weight": 0.0,
                "params": {},
            }
        )
        hard_reward["sha256"] = launcher.canonical_sha256(
            {"schema_version": 1, "terms": hard_reward["terms"]}
        )
    policy_materialization = claim["materialization_inputs"][
        "policy_recipe_materialization"
    ]
    ppo = arm["ppo"]
    hard_document = {
        "schema_version": 3,
        "target_mode": "action_ball",
        "actor_obs_contract": launcher.ACTOR_CONTRACT,
        "actor_obs_total_dim": launcher.ACTOR_WIDTH,
        "actor_obs_term_names": [
            name for name, _width in launcher.ACTOR_ORDERED_LAYOUT
        ],
        "actor_obs_term_dims": [
            width for _name, width in launcher.ACTOR_ORDERED_LAYOUT
        ],
        "critic_obs_contract": launcher.CRITIC_CONTRACT,
        "critic_obs_total_dim": launcher.CRITIC_WIDTH,
        "action_ball_211_trainability_contract": launcher.TRAINABILITY_CONTRACT,
        "action_ball_task_wait_contract": launcher._wait_contract(),
        "actor_obs_normalizer_identity": launcher.ACTOR_NORMALIZER_IDENTITY,
        "critic_obs_normalizer_identity": launcher.CRITIC_NORMALIZER_IDENTITY,
        "fresh_normalizers_required": True,
        "symmetric_critic_fallback_forbidden": True,
        "effective_reward_recipe": hard_reward,
        "action_ball_ppo_runner_recipe": {
            "sha256": policy_materialization["runtime_policy_recipe_sha256"],
            "recipe": {
                "algorithm": {
                    "schedule": ppo["schedule"],
                    "learning_rate": ppo["learning_rate"],
                    "desired_kl": 0.01,
                    "clip_param": 0.2,
                    "num_learning_epochs": 5,
                    "num_mini_batches": 4,
                    "entropy_coef": arm["entropy_coef"],
                },
                "policy": {
                    "actor_hidden_dims": arm["actor_hidden_dims"],
                    "critic_hidden_dims": arm["critic_hidden_dims"],
                    "init_noise_std": arm["init_noise_std"],
                    "noise_std_type": arm["noise_std_type"],
                },
            },
        },
    }
    if hard_identity_mutation:
        hard_document.update(hard_identity_mutation)
    checkout = Path(claim["spec"]["source"]["checkout"])
    runtime_dir = (
        checkout
        / launcher._B.WBT_RELATIVE
        / "logs/rsl_rl"
        / launcher.EXPERIMENT_NAME
        / (
            "fixture_"
            + Path(claim["spec"]["namespace"]).name
            + "-DIAGNOSTIC_UNAUTHORIZED"
        )
    )
    hard_path = runtime_dir / "params/training_contract.json"
    hard_sha = _write(hard_path, hard_document)
    claim["runtime_sources"] = {
        "training entrypoint": {"sha256": "6" * 64},
        "A211 task profile": {"sha256": "7" * 64},
    }
    lineage = claim["bundle"]["lineage"]
    bindings = {
        "source_sha256": "6" * 64,
        "task_sha256": "7" * 64,
        "hard_contract_sha256": hard_sha,
        "reward_sha256": hard_reward["sha256"],
        "policy_sha256": policy_materialization["runtime_policy_recipe_sha256"],
        "policy_contract_sha256": policy_materialization[
            "runtime_policy_recipe_sha256"
        ],
        "dynamic_ready_sha256": policy_materialization[
            "dynamic_ready_binding_sha256"
        ],
        "dynamic_ready_artifact_sha256": lineage["dynamic_ready_artifact"][
            "sha256"
        ],
        "dynamic_ready_nominal_hold_sha256": lineage[
            "dynamic_ready_nominal_receipt"
        ]["sha256"],
        "manifest_sha256": lineage["action_manifest"]["sha256"],
        "motion_sha256": lineage["motion"]["sha256"],
        "tape_file_sha256": lineage["immutable_tape"]["sha256"],
        "tape_canonical_sha256": "8" * 64,
        "tape_base_question_sha256": "9" * 64,
        "tape_target_producer_sha256": "a" * 64,
        "tape_target_column_sha256": "b" * 64,
    }
    oracle_path = tmp_path / "raw-oracle32.json"
    _write(
        oracle_path,
        {
            "schema_version": 2,
            "kind": "action_ball_teacher_qdes_dynamic_oracle_v2",
            "diagnostic_unauthorized": True,
            "bindings": bindings,
            "completion": {
                "exact_strike_observed_nonterminal": 32,
                "pre_strike_or_same_step_unknown": 0,
                "control_steps": 32,
            },
            "phase_by_termination": {},
            "exact_strike": {},
            "capture_rejection": {},
            "measurement_contract": {},
            "safety_exposure": {
                "termination": {},
                "projection": {},
                "soft_limit": {},
                "reference_guard": {},
            },
            "teacher_qdes": {},
            "episodes": 32,
        },
    )

    class TrainingContract:
        @staticmethod
        def validate_schema3_contract_structure(document):
            return None

        @staticmethod
        def validate_action_ball_training_authorization(document):
            return True

    monkeypatch.setattr(
        launcher._OLD,
        "_load_training_contract_module",
        lambda checkout: TrainingContract,
    )
    monkeypatch.setattr(
        launcher._OLD,
        "_oracle32_acceptance_failures",
        lambda **kwargs: [],
    )
    return oracle_path, claim


def _flatten_strings(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _flatten_strings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _flatten_strings(child)
    elif isinstance(value, str):
        yield value


def test_four_code_owned_arms_are_exact():
    expected = {
        launcher.ARM_IDS[0]: (-30.0, -0.5, "metrics_only", "fixed", 1e-4),
        launcher.ARM_IDS[1]: (-300.0, -5.0, "metrics_only", "fixed", 1e-4),
        launcher.ARM_IDS[2]: (-30.0, -0.5, "phase_gated", "fixed", 1e-4),
        launcher.ARM_IDS[3]: (-30.0, -0.5, "phase_gated", "adaptive", 1e-3),
    }
    assert tuple(launcher.ARMS) == launcher.ARM_IDS
    for arm_id, values in expected.items():
        arm = launcher._arm_contract(arm_id)
        assert (arm["soft_weights"]["death_penalty"], arm["soft_weights"]["qdes_limit"], arm["reference_guard_mode"], arm["ppo"]["schedule"], arm["ppo"]["learning_rate"]) == values
        assert arm["actor_hidden_dims"] == arm["critic_hidden_dims"] == [512, 256, 128]
        assert arm["init_noise_std"] == 0.02
        assert arm["entropy_coef"] == 0.01


def test_a211_v2_actor_layout_binds_localizer_and_body_gyro_exact_slices():
    identity = launcher._actor_layout_identity()
    assert identity["schema_version"] == 2
    assert identity["total_dim"] == 211
    assert identity["ordered_terms"][0] == {
        "name": "actual_base_pose_lin_vel_world",
        "width": 12,
        "slice": [0, 12],
    }
    assert identity["ordered_terms"][1] == {
        "name": "base_ang_vel_body",
        "width": 3,
        "slice": [12, 15],
    }
    assert identity["sensor_sources"]["actual_base_pose_lin_vel_world"][
        "angular_velocity_included"
    ] is False
    assert identity["sensor_sources"]["base_ang_vel_body"]["producer"] == (
        "mdp.action_ball_base_ang_vel_body"
    )
    assert launcher.canonical_sha256(
        {key: value for key, value in identity.items() if key != "content_sha256"}
    ) == identity["content_sha256"]


def test_a211_v2_lineage_rejects_same_width_pre_imu_layout(tmp_path, monkeypatch):
    _patch_plan_environment(monkeypatch)
    spec_path, spec, lineage = _case(
        tmp_path, arm_id=launcher.ARM_IDS[0], stage="materialize"
    )
    old = copy.deepcopy(lineage["actor_layout_identity"])
    old["ordered_terms"][0]["name"] = "stage1_base_state_world"
    old["ordered_terms"][0]["width"] = 15
    old["ordered_terms"][0]["slice"] = [0, 15]
    old["ordered_terms"].pop(1)
    old.pop("content_sha256")
    old["content_sha256"] = launcher.canonical_sha256(old)
    lineage["actor_layout_identity"] = old
    lineage_path = Path(spec["source"]["checkout"]) / spec["lineage"]["path"]
    spec["lineage"]["sha256"] = _write(lineage_path, lineage)
    _write(spec_path, spec)
    with pytest.raises(launcher.LaunchRefused):
        launcher.build_plan(spec_path)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda live: live.__setitem__("completed_policy_steps", 199),
        lambda live: live["joint_safety_telemetry"].__setitem__(
            "current_actual_hard_edge_joint_count", 1
        ),
    ),
)
def test_frame0_receipt_rejects_resealed_live_safety_tamper(tmp_path, mutation):
    lineage = _lineage(tmp_path)
    artifact = json.loads(
        (tmp_path / lineage["frame0_exact_artifact"]["path"]).read_text()
    )
    receipt_path = tmp_path / lineage["frame0_exact_receipt"]["path"]
    receipt = json.loads(receipt_path.read_text())
    live = receipt["live_safety_evidence"]
    live.pop("content_sha256")
    mutation(live)
    live["content_sha256"] = launcher.canonical_sha256(live)
    receipt["live_safety_evidence_content_sha256"] = live["content_sha256"]
    receipt["live_safety_evidence_file_sha256"] = hashlib.sha256(
        launcher._B._canonical_bytes(live)
    ).hexdigest()
    receipt.pop("content_sha256")
    receipt["content_sha256"] = launcher.canonical_sha256(receipt)
    with pytest.raises(launcher.LaunchRefused, match="live safety evidence"):
        launcher._validate_frame0_live_safety_evidence(receipt, artifact)


@pytest.mark.parametrize("stage,budget", list(launcher.BUDGETS.items()))
def test_stage_budgets_are_code_owned(stage, budget):
    assert launcher.BUDGETS[stage] == budget


def test_plan_claim_is_a211_fresh_and_denies_retired_lineage(tmp_path, monkeypatch):
    _patch_plan_environment(monkeypatch)
    spec_path, _spec, _lineage_doc = _case(tmp_path, arm_id=launcher.ARM_IDS[2], stage="long4096")
    payload = launcher.build_plan(spec_path)["canonical_payload"]
    assert payload["fresh_only"] is True
    assert payload["single_gpu"] is True
    assert payload["max_compute_pids_on_physical_gpu"] == 2
    assert payload["minimum_free_memory_mib"] == 8192
    assert payload["gpu_default_empty"] is True
    assert payload["vendor_v2_colocation_opt_in"] is False
    assert payload["bundle"]["lineage"]["actor_contract"] == launcher.ACTOR_CONTRACT
    assert payload["bundle"]["normalizers"] == launcher._normalizer_contract()
    assert payload["bundle"]["curriculum_scope"] == {
        "current_tape_scope": "diagnostic_n1_early_fixed_band_only",
        "permanent_single_question_curriculum": False,
        "final_curriculum_source": "pregenerated_cached_band_question_bank",
        "reset_question_selection": "index_pregenerated_bank_row",
        "online_inverse_solve_calls": 0,
    }
    assert payload["bundle"]["continuation_stop_gate"]["iter500_quantitative_threshold_status"] == "UNSET"
    assert payload["bundle"]["continuation_stop_gate"]["scale4096_required_for_long4096"] is True
    assert payload["materialization_inputs"]["predecessor_result"][
        "terminal_attestation"
    ]["completion"]["terminal_kind"] == "clean_completion"
    assert payload["output_contract"]["speed_benchmark_eligible"] is True
    flattened = "\n".join(_flatten_strings(payload)).lower()
    for retired in ("target_recipe", "target_validity_mask", "l194", "checkpoint"):
        assert retired not in flattened


@pytest.mark.parametrize(
    "field,retired",
    (
        ("actor_contract", "action_ball_a225"),
        ("actor_contract", "action_ball_a210"),
        ("actor_width", 225),
        ("actor_width", 210),
        ("critic_contract", "action_ball_a225_critic_v1"),
        ("critic_contract", "action_ball_a210_critic_v1"),
        ("critic_width", 318),
        ("trainability_contract", "action_ball_a225_fixed_question_learnability_v1"),
        ("trainability_contract", "action_ball_a210_fixed_question_learnability_v1"),
        ("task_profile", "HOPEPingPongActionBallA225VendorV2N1Learnability"),
        ("gym_task", "HOPE-PingPong-ActionBall-A210Learnability-AgibotA3-v0"),
    ),
)
def test_structurally_resealed_retired_lineage_is_rejected(
    tmp_path, monkeypatch, field, retired
):
    _patch_plan_environment(monkeypatch)
    spec_path, spec, lineage = _case(
        tmp_path, arm_id=launcher.ARM_IDS[0], stage="materialize"
    )
    lineage[field] = retired
    lineage_path = Path(spec["source"]["checkout"]) / spec["lineage"]["path"]
    spec["lineage"]["sha256"] = _write(lineage_path, lineage)
    _write(spec_path, spec)
    with pytest.raises(launcher.LaunchRefused):
        launcher.build_plan(spec_path)


@pytest.mark.parametrize(
    "mutation",
    (
        {"actor_obs_contract": "action_ball_a225"},
        {"actor_obs_total_dim": 225},
        {"actor_obs_term_names": ["stage1_base_state_world"] + [
            name for name, _width in launcher.ACTOR_ORDERED_LAYOUT[2:]
        ]},
        {"critic_obs_contract": "action_ball_a210_critic_v1"},
        {"critic_obs_total_dim": 318},
        {
            "action_ball_211_trainability_contract":
                "action_ball_a225_fixed_question_learnability_v1"
        },
        {"actor_obs_normalizer_identity": "action_ball_a210_actor_norm_v1"},
        {"critic_obs_normalizer_identity": "action_ball_a225_critic_norm_v1"},
    ),
)
def test_structurally_resealed_retired_hard_contract_is_rejected(
    tmp_path, monkeypatch, mutation
):
    raw, claim = _raw_oracle_fixture(
        tmp_path, monkeypatch, hard_identity_mutation=mutation
    )
    with pytest.raises(launcher.LaunchRefused):
        launcher._validate_raw_oracle32(raw, claim=claim)


def test_retired_vocabulary_scan_treats_hashes_as_opaque():
    launcher._assert_no_retired_contract(
        {"spec_file_sha256": "0" * 12 + "c225" + "0" * 48},
        name="opaque digest",
    )
    with pytest.raises(launcher.LaunchRefused, match="retired ABI/arm token"):
        launcher._assert_no_retired_contract(
            {"obs_mode": "action_ball_c225"}, name="semantic value"
        )


def test_training_argv_pins_a211_lineage_bootstrap_and_optimizer(tmp_path, monkeypatch):
    _patch_plan_environment(monkeypatch)
    spec_path, _spec, _lineage_doc = _case(tmp_path, arm_id=launcher.ARM_IDS[3], stage="probe512")
    argv = launcher.build_plan(spec_path)["canonical_payload"]["training_argv"]
    for exact in (
        "task=HOPEPingPongActionBallA211VendorV2N1Learnability",
        "task.actor_obs_contract=action_ball_a211",
        "algo.policy.actor_hidden_dims=[512,256,128]",
        "algo.policy.critic_hidden_dims=[512,256,128]",
        "algo.algorithm.schedule=adaptive",
        "algo.algorithm.learning_rate=0.001",
        "+task.racket.reference_guard_mode=phase_gated",
        "action_ball_dynamic_ready_bootstrap=true",
    ):
        assert exact in argv
    joined = "\n".join(argv)
    assert "action_ball_policy_contract_sha256=" in joined
    assert "expected_effective_reward_recipe_sha256=" + "3" * 64 in argv
    assert "action_ball_manifest_sha256=" in joined
    assert "target_recipe" not in joined and "validity_mask" not in joined


def test_materialize_stage_publishes_and_binds_runtime_effective_reward(tmp_path, monkeypatch):
    _patch_plan_environment(monkeypatch)
    spec_path, _spec, _lineage_doc = _case(
        tmp_path, arm_id=launcher.ARM_IDS[0], stage="materialize"
    )
    payload = launcher.build_plan(spec_path)["canonical_payload"]
    assert "+n1_vendor_sigma_profile=" + launcher.REWARD_MATERIALIZATION_PROFILE in payload[
        "training_argv"
    ]
    assert any(
        value.startswith("+action_ball_effective_reward_recipe_output_path=")
        for value in payload["training_argv"]
    )
    assert payload["output_contract"]["boot_marker"] == (
        "ACTION_BALL_EFFECTIVE_REWARD_RECIPE_MATERIALIZED_JSON"
    )

    arm = payload["bundle"]["arm"]
    names_and_weights = {
        "death_penalty": arm["soft_weights"]["death_penalty"],
        "joint_limit": arm["soft_weights"]["joint_limit"],
        "qdes_limit_barrier": arm["soft_weights"]["qdes_limit"],
        "qdes_projection_penalty": arm["soft_weights"]["qdes_projection"],
    }
    terms = _required_effective_terms()
    for name, weight in sorted(names_and_weights.items()):
        terms.append(
            {
                "name": name,
                "callable": "fixture." + name,
                "weight": -1.0 if name == "qdes_projection_penalty" else weight,
                "params": (
                    {"objective_weight": weight}
                    if name == "qdes_projection_penalty"
                    else {}
                ),
            }
        )
    terms.sort(key=lambda term: term["name"])
    semantic = {"schema_version": 1, "terms": terms}
    document = {**semantic, "sha256": launcher.canonical_sha256(semantic)}
    output = Path(payload["output_contract"]["effective_reward_recipe"])
    output.parent.mkdir(parents=True)
    _write(output, document)
    runtime = launcher._runtime_reward_materialization(
        path=output,
        planned=payload["materialization_inputs"]["arm_materialization"],
        arm=arm,
    )
    assert runtime["runtime_effective_reward_sha256"] == document["sha256"]
    assert runtime["runtime_soft_weights"] == names_and_weights

    next(term for term in document["terms"] if term["name"] == "death_penalty")["weight"] -= 1.0
    semantic = {"schema_version": 1, "terms": document["terms"]}
    document["sha256"] = launcher.canonical_sha256(semantic)
    output.unlink()
    _write(output, document)
    with pytest.raises(launcher.LaunchRefused, match="soft weights differ"):
        launcher._runtime_reward_materialization(
            path=output,
            planned=payload["materialization_inputs"]["arm_materialization"],
            arm=arm,
        )


def test_materialize_stage_refuses_missing_required_effective_term(tmp_path, monkeypatch):
    _patch_plan_environment(monkeypatch)
    spec_path, _spec, _lineage_doc = _case(
        tmp_path, arm_id=launcher.ARM_IDS[0], stage="materialize"
    )
    payload = launcher.build_plan(spec_path)["canonical_payload"]
    arm = payload["bundle"]["arm"]
    terms = _required_effective_terms()
    terms = [term for term in terms if term["name"] != "virtual_landing"]
    for name, weight in sorted(
        {
            "death_penalty": arm["soft_weights"]["death_penalty"],
            "joint_limit": arm["soft_weights"]["joint_limit"],
            "qdes_limit_barrier": arm["soft_weights"]["qdes_limit"],
            "qdes_projection_penalty": arm["soft_weights"]["qdes_projection"],
        }.items()
    ):
        terms.append(
            {
                "name": name,
                "callable": "fixture." + name,
                "weight": -1.0 if name == "qdes_projection_penalty" else weight,
                "params": {"objective_weight": weight}
                if name == "qdes_projection_penalty"
                else {},
            }
        )
    terms.sort(key=lambda term: term["name"])
    semantic = {"schema_version": 1, "terms": terms}
    document = {**semantic, "sha256": launcher.canonical_sha256(semantic)}
    output = Path(payload["output_contract"]["effective_reward_recipe"])
    output.parent.mkdir(parents=True)
    _write(output, document)
    with pytest.raises(launcher.LaunchRefused, match="required effective term is absent: virtual_landing"):
        launcher._runtime_reward_materialization(
            path=output,
            planned=payload["materialization_inputs"]["arm_materialization"],
            arm=arm,
        )


def test_raw_oracle_accepts_projection_manager_minus_one_with_arm_objective(
    tmp_path, monkeypatch
):
    raw, claim = _raw_oracle_fixture(tmp_path, monkeypatch)
    receipt = launcher._validate_raw_oracle32(raw, claim=claim)
    assert receipt["runtime_effective_reward_sha256"] == claim[
        "materialization_inputs"
    ]["arm_materialization"]["runtime_effective_reward_sha256"]


@pytest.mark.parametrize(
    "mutation",
    (
        {"qdes_objective_weight": -0.25},
        {"qdes_manager_weight": -0.5},
    ),
)
def test_raw_oracle_rejects_projection_objective_or_manager_weight_drift(
    tmp_path, monkeypatch, mutation
):
    raw, claim = _raw_oracle_fixture(tmp_path, monkeypatch, **mutation)
    with pytest.raises(launcher.LaunchRefused, match="soft weights differ"):
        launcher._validate_raw_oracle32(raw, claim=claim)


def test_raw_oracle_rejects_reward_sha_drift_from_revalidated_materialization(
    tmp_path, monkeypatch
):
    raw, claim = _raw_oracle_fixture(
        tmp_path, monkeypatch, drift_raw_reward_sha=True
    )
    with pytest.raises(launcher.LaunchRefused, match="lineage bindings differ"):
        launcher._validate_raw_oracle32(raw, claim=claim)


def test_recipe_stage_materializes_policy_before_oracle32(tmp_path, monkeypatch):
    _patch_plan_environment(monkeypatch)
    recipe_path, _spec, _ = _case(
        tmp_path / "recipe", arm_id=launcher.ARM_IDS[3], stage="recipe"
    )
    recipe = launcher.build_plan(recipe_path)["canonical_payload"]
    assert recipe["policy_recipe_materialization_only"] is True
    assert recipe["materialization_inputs"]["policy_recipe_materialization"] is None
    assert recipe["output_contract"]["boot_marker"] == (
        "ACTION_BALL_POLICY_RECIPE_MATERIALIZED"
    )
    assert (
        "task.racket.action_ball_policy_contract_sha256="
        + launcher.RECIPE_SENTINEL_POLICY_SHA256
    ) in recipe["training_argv"]
    assert any(
        value.startswith("action_ball_policy_recipe_output_path=")
        for value in recipe["training_argv"]
    )
    assert "policy_contract_sha256" not in recipe[
        "materialization_inputs"
    ]["arm_materialization"]

    oracle_path, _spec, _ = _case(
        tmp_path / "oracle", arm_id=launcher.ARM_IDS[3], stage="oracle32"
    )
    oracle = launcher.build_plan(oracle_path)["canonical_payload"]
    assert (
        "task.racket.action_ball_policy_contract_sha256=" + "4" * 64
    ) in oracle["training_argv"]
    assert not any(
        value.endswith("=" + launcher.RECIPE_SENTINEL_POLICY_SHA256)
        and "policy_contract_sha256" in value
        for value in oracle["training_argv"]
    )


def test_recipe_accepts_exact_legacy_reward_only_result_without_trusting_its_planned_policy(
    tmp_path, monkeypatch
):
    _patch_plan_environment(monkeypatch)
    spec_path, spec, _ = _case(
        tmp_path, arm_id=launcher.ARM_IDS[0], stage="recipe"
    )
    result_path = Path(spec["arm_materialization"]["path"])
    result = json.loads(result_path.read_text())
    materialization = dict(result["arm_materialization"])
    materialization.pop("content_sha256")
    materialization["policy_contract_sha256"] = "9" * 64
    result["arm_materialization"] = _sealed(materialization)
    result.pop("policy_recipe_materialization")
    result.pop("content_sha256")
    spec["arm_materialization"]["sha256"] = _write(result_path, _sealed(result))
    _write(spec_path, spec)
    payload = launcher.build_plan(spec_path)["canonical_payload"]
    normalized = payload["materialization_inputs"]["arm_materialization"]
    assert "policy_contract_sha256" not in normalized
    assert (
        "task.racket.action_ball_policy_contract_sha256="
        + launcher.RECIPE_SENTINEL_POLICY_SHA256
    ) in payload["training_argv"]


def test_runtime_policy_recipe_is_exact_arm_owned_and_no_observed_sha_is_baked(
    tmp_path, monkeypatch
):
    arm = launcher._arm_contract(launcher.ARM_IDS[3])
    lineage = {
        "lineage_sha256": "1" * 64,
        "motion": {"path": "motion", "sha256": "2" * 64},
        "dynamic_ready_artifact": {"path": "dynamic", "sha256": "3" * 64},
        "dynamic_ready_nominal_receipt": {
            "path": "nominal",
            "sha256": "4" * 64,
        },
    }
    runner = {
        "schema_version": 2,
        "runner": {
            "empirical_normalization": True,
            "init_at_random_ep_len": False,
        },
        "policy": {
            "actor_hidden_dims": arm["actor_hidden_dims"],
            "critic_hidden_dims": arm["critic_hidden_dims"],
            "init_noise_std": arm["init_noise_std"],
            "noise_std_type": arm["noise_std_type"],
        },
        "algorithm": {"entropy_coef": arm["entropy_coef"], **arm["ppo"]},
        "policy_initialization": {"fixture": True},
    }
    document = {
        "schema_version": 1,
        "kind": "action_ball_shared_ready_policy_recipe_materialization_v1",
        "action_count": 1,
        "action_order": ["take_061_unit04_bh"],
        "policy_contract_sha256": launcher.canonical_sha256(runner),
        "action_ball_ppo_runner_recipe": {
            "schema_version": 1,
            "sha256": launcher.canonical_sha256(runner),
            "recipe": runner,
        },
        "policy_bootstrap": {"fixture": True},
    }
    path = tmp_path / "policy.json"
    _write(path, document)

    def validate(value, *, checkout, bundle):
        return {
            "artifact": dict(value),
            "policy_contract_sha256": document["policy_contract_sha256"],
            "dynamic_ready_binding_sha256": "5" * 64,
            "noise_std_type": "log",
            "configured_and_realized_init_noise_std": 0.02,
        }

    monkeypatch.setattr(launcher._OLD, "_validate_policy_materialization", validate)
    receipt = launcher._runtime_policy_materialization(
        path=path, checkout=tmp_path, lineage=lineage, arm=arm
    )
    assert receipt["runtime_policy_recipe_sha256"] == launcher.canonical_sha256(
        runner
    )
    assert "3a3" not in SCRIPT.read_text(encoding="utf-8")
    assert "f344" not in SCRIPT.read_text(encoding="utf-8")

    document["action_ball_ppo_runner_recipe"]["recipe"]["algorithm"][
        "learning_rate"
    ] = 0.5
    path.unlink()
    _write(path, document)
    with pytest.raises(launcher.LaunchRefused, match="selected A211 PPO arm"):
        launcher._runtime_policy_materialization(
            path=path, checkout=tmp_path, lineage=lineage, arm=arm
        )


def test_full_stage_chain_is_enforced(tmp_path, monkeypatch):
    _patch_plan_environment(monkeypatch)
    for stage in (
        "materialize",
        "recipe",
        "oracle32",
        "scale4096",
        "long4096",
        "smoke",
        "probe512",
        "long512",
    ):
        spec_path, _spec, _lineage = _case(tmp_path / stage, arm_id=launcher.ARM_IDS[0], stage=stage)
        assert launcher.build_plan(spec_path)["canonical_payload"]["spec"]["stage"] == stage
    spec_path, spec, _ = _case(
        tmp_path / "missing-policy", arm_id=launcher.ARM_IDS[0], stage="oracle32"
    )
    spec["policy_recipe_materialization"] = None
    _write(spec_path, spec)
    with pytest.raises(launcher.LaunchRefused, match="policy recipe receipt"):
        launcher.build_plan(spec_path)
    spec_path, spec, _ = _case(tmp_path / "missing", arm_id=launcher.ARM_IDS[0], stage="probe512")
    spec["predecessor_result"] = None
    _write(spec_path, spec)
    with pytest.raises(launcher.LaunchRefused, match="completed smoke"):
        launcher.build_plan(spec_path)

    spec_path, spec, _ = _case(
        tmp_path / "long4096-missing-scale",
        arm_id=launcher.ARM_IDS[0],
        stage="long4096",
    )
    spec["predecessor_result"] = None
    _write(spec_path, spec)
    with pytest.raises(launcher.LaunchRefused, match="completed scale4096"):
        launcher.build_plan(spec_path)


def test_cross_arm_or_oracle_content_drift_is_rejected(tmp_path, monkeypatch):
    _patch_plan_environment(monkeypatch)
    spec_path, spec, _ = _case(tmp_path, arm_id=launcher.ARM_IDS[0], stage="smoke")
    oracle_path = Path(spec["oracle32_receipt"]["path"])
    outer = json.loads(oracle_path.read_text())
    outer["oracle32_receipt"]["arm_id"] = launcher.ARM_IDS[1]
    receipt = dict(outer["oracle32_receipt"])
    receipt.pop("content_sha256")
    outer["oracle32_receipt"] = _sealed(receipt)
    outer.pop("content_sha256")
    outer = _sealed(outer)
    spec["oracle32_receipt"]["sha256"] = _write(oracle_path, outer)
    _write(spec_path, spec)
    with pytest.raises(launcher.LaunchRefused, match="binding differs"):
        launcher.build_plan(spec_path)


def test_policy_recipe_artifact_sha_drift_is_rejected(tmp_path, monkeypatch):
    _patch_plan_environment(monkeypatch)
    spec_path, spec, _ = _case(
        tmp_path, arm_id=launcher.ARM_IDS[0], stage="oracle32"
    )
    recipe_result = json.loads(
        Path(spec["policy_recipe_materialization"]["path"]).read_text()
    )
    artifact = Path(
        recipe_result["policy_recipe_materialization"][
            "runtime_policy_recipe_artifact"
        ]["path"]
    )
    artifact.write_text("drifted-policy\n", encoding="utf-8")
    with pytest.raises(
        launcher.LaunchRefused, match="runtime policy materialization binding"
    ):
        launcher.build_plan(spec_path)


def test_default_empty_gpu_and_explicit_colocation_claim_scope(tmp_path, monkeypatch):
    _patch_plan_environment(monkeypatch)
    default_path, _, _ = _case(tmp_path / "default", arm_id=launcher.ARM_IDS[0], stage="materialize")
    default = launcher.build_plan(default_path)["canonical_payload"]
    assert default["spec"]["gpu"]["require_empty"] is True
    assert default["output_contract"]["speed_benchmark_eligible"] is True

    opted_path, _, _ = _case(tmp_path / "opted", arm_id=launcher.ARM_IDS[0], stage="materialize", allow_colocation=True)
    opted = launcher.build_plan(opted_path)["canonical_payload"]
    assert opted["spec"]["gpu"]["require_empty"] is False
    assert opted["vendor_v2_colocation_opt_in"] is True
    assert opted["output_contract"]["speed_benchmark_eligible"] is False
    assert opted["output_contract"]["colocation_result_scope"] == "training_diagnostic_only"


def test_colocation_gpu_validation_is_cross_bound_and_fail_closed(tmp_path, monkeypatch):
    _, raw_spec, _ = _case(tmp_path, arm_id=launcher.ARM_IDS[0], stage="materialize", allow_colocation=True)
    spec = launcher._validate_spec(raw_spec)
    query = lambda index, uuid: {
        "total_memory_mib": 24576,
        "free_memory_mib": launcher._A.MIN_VENDOR_V2_FREE_MEMORY_MIB,
        "processes": [{"pid": 99}],
        "nvidia_smi_path": "/usr/bin/nvidia-smi",
        "nvidia_smi_sha256": "3" * 64,
    }
    monkeypatch.setattr(launcher, "_query_gpu_processes", query)
    monkeypatch.setattr(launcher, "_live_reservations", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        launcher,
        "_validate_runtime_gpu_process",
        lambda *args, **kwargs: (_ for _ in ()).throw(launcher.LaunchRefused("unknown GPU co-resident")),
    )
    with pytest.raises(launcher.LaunchRefused, match="unknown GPU"):
        launcher._verify_gpu_admission(spec, phase="pre_launch", current_namespace=None)

    monkeypatch.setattr(launcher, "_query_gpu_processes", lambda index, uuid: {**query(index, uuid), "free_memory_mib": launcher._A.MIN_VENDOR_V2_FREE_MEMORY_MIB - 1, "processes": []})
    with pytest.raises(launcher.LaunchRefused, match="below conservative headroom"):
        launcher._verify_gpu_admission(spec, phase="pre_launch", current_namespace=None)


def test_scale4096_executes_as_completion_stage_and_emits_natural_exit_result(
    tmp_path, monkeypatch
):
    _patch_plan_environment(monkeypatch)
    spec_path, _, _ = _case(tmp_path, arm_id=launcher.ARM_IDS[1], stage="scale4096")
    plan = launcher.build_plan(spec_path)
    monkeypatch.setattr(launcher._B, "_validate_runtime_asset_claim", lambda value: value)
    lock_file = tmp_path / "gpu.lock"
    monkeypatch.setattr(
        launcher,
        "_open_gpu_shared_lock",
        lambda path: os.open(lock_file, os.O_RDWR | os.O_CREAT, 0o600),
    )
    monkeypatch.setattr(launcher, "_lock_gpu_admission", lambda fd: None)
    monkeypatch.setattr(launcher, "_unlock_gpu_admission", lambda fd: None)
    phases = []

    def admission(spec, *, phase, current_namespace, require_current_compute=False, **kwargs):
        phases.append((phase, require_current_compute))
        return {"phase": phase}

    monkeypatch.setattr(launcher, "_verify_gpu_admission", admission)
    monkeypatch.setattr(
        launcher, "_reservation_document", lambda spec, digest: {"claim": digest}
    )

    def run(*args, **kwargs):
        state = Path(kwargs["env"]["KIT_BOOT_STATE_FILE"])
        state.write_text(
            "completion_exit_code=0\n"
            "terminal_kind=clean_completion\n"
            "terminal_exit_code=0\n",
            encoding="utf-8",
        )
        assert kwargs["env"]["KIT_WAIT_FOR_COMPLETION"] == "1"
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(launcher.subprocess, "run", run)
    result = launcher.execute(plan, confirm_claim=plan["launch_claim_sha256"])
    assert result["stage"] == "scale4096"
    assert result["completion"] == {
        "completion_exit_code": "0",
        "terminal_kind": "clean_completion",
        "terminal_exit_code": "0",
    }
    assert phases == [("pre_launch", False), ("post_completion", False)]
    assert Path(result["namespace"], "launch_result.json").is_file()


def _scale4096_terminal_artifacts(tmp_path: Path):
    checkout = tmp_path / "checkpoint-checkout"
    wbt = checkout / launcher._B.WBT_RELATIVE
    root = wbt / "logs" / "rsl_rl" / launcher.EXPERIMENT_NAME
    root.mkdir(parents=True)
    namespace = tmp_path / launcher.EXPERIMENT_NAME / "scale-terminal-fixture"
    namespace.mkdir(parents=True)
    run_dir = root / (
        "2026-08-04_12-34-56_"
        + namespace.name
        + "-DIAGNOSTIC_UNAUTHORIZED"
    )
    run_dir.mkdir()
    claim_sha = "a" * 64
    checkpoint_path = run_dir / "model_5.pt"
    checkpoint = {
        "iter": 5,
        "infos": {"training_launch_claim_sha256": claim_sha},
        "model_state_dict": {"weight": _FakeTensor([1.0, 1.0])},
        "optimizer_state_dict": {"state": {0: {"momentum": _FakeTensor([1.0, 1.0])}}},
        "obs_norm_state_dict": {"running_mean": _FakeTensor([0.0, 0.0, 0.0])},
        "privileged_obs_norm_state_dict": {"running_var": _FakeTensor([1.0] * 4)},
    }
    checkpoint_path.write_bytes(b"trusted exact Pod checkpoint fixture\n")

    lines = ["[INFO] Task: fixture | experiment: fixture | log: %s" % run_dir]
    for update in range(5):
        lines.extend(
            (
                "HOPE_JOINT_SAFETY_UPDATE_JSON="
                + json.dumps(
                    {
                        "event": "hope_joint_safety_diagnostic_compact_update",
                        "schema_version": 1,
                        "status": (
                            "diagnostic_compact_optimizer_committed_and_ledger_acknowledged"
                        ),
                        "ppo_update": update,
                        "counter_totals": {"actual_hard_edge_events": 0},
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "HOPE_ACTUAL_JOINT_DIAGNOSTIC_UPDATE_JSON="
                + json.dumps(
                    {
                        "event": "action_ball_actual_joint_forbidden_diagnostic_update",
                        "schema_version": 2,
                        "ppo_update": update,
                        "enabled": True,
                        "total_hard_terminal_count": 0,
                        "physx_control_position_limits": {
                            "enabled": True,
                            "by_joint": [
                                {
                                    "joint": "joint_00",
                                    "sides": {
                                        "lower": {"nonfinite_readback_observed": False},
                                        "upper": {"nonfinite_readback_observed": False},
                                    },
                                }
                            ],
                        },
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "HOPE_REWARD_SAFETY_TRANSITION_UPDATE_JSON="
                + json.dumps(
                    {
                        "event": "hope_reward_safety_transition_update",
                        "schema_version": 2,
                        "ppo_update": update,
                        "coverage": "complete_update",
                        "terminal_transitions": [],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "HOPE_EXACT_BEHAVIOR_UPDATE_JSON="
                + json.dumps(
                    {
                        "event": "hope_exact_behavior_update",
                        "schema_version": 1,
                        "ppo_update": update,
                        "counters": {
                            "ready_nonfinite_value_count": 0,
                            "strike_window_entry_racket_target_distance_nonfinite_count": 0,
                            "virtual_contact_nonfinite_reject_count": 0,
                        },
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
        )
    log_path = namespace / "run.log"
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return checkout, namespace, claim_sha, checkpoint_path, checkpoint, log_path


def _audit_terminal_fixture(
    checkout, namespace, claim, checkpoint, monkeypatch, *, load_error=None
):
    _FakeTorch.checkpoint = checkpoint
    _FakeTorch.load_error = load_error
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch)
    return launcher._audit_scale4096_terminal(
        checkout=checkout,
        namespace=namespace,
        launch_claim_sha256=claim,
    )


def test_scale4096_terminal_checkpoint_and_safety_gate_accepts_valid_case(
    tmp_path, monkeypatch
):
    checkout, namespace, claim, checkpoint_path, _checkpoint, _log = (
        _scale4096_terminal_artifacts(tmp_path)
    )
    acceptance = _audit_terminal_fixture(
        checkout, namespace, claim, _checkpoint, monkeypatch
    )
    assert acceptance["checkpoint"]["path"] == str(checkpoint_path)
    assert acceptance["checkpoint"]["embedded_iteration"] == 5
    assert acceptance["checkpoint"]["load_mode"] == "torch_weights_only"
    assert acceptance["checkpoint"]["all_tensors_finite"] is True
    assert acceptance["safety_counters"] == {
        "observed_ppo_updates": 5,
        "actual_hard_edge_event_count": 0,
        "actual_hard_terminal_count": 0,
        "hard_termination_count": 0,
        "table_contact_count": 0,
        "nonfinite_count": 0,
    }


def test_scale4096_terminal_checkpoint_gate_rejects_missing_checkpoint(
    tmp_path, monkeypatch
):
    checkout, namespace, claim, checkpoint_path, _checkpoint, _log = (
        _scale4096_terminal_artifacts(tmp_path)
    )
    checkpoint_path.unlink()
    with pytest.raises(launcher.LaunchRefused, match="checkpoint.*missing"):
        _audit_terminal_fixture(
            checkout, namespace, claim, _checkpoint, monkeypatch
        )


def test_scale4096_terminal_checkpoint_gate_rejects_corrupt_checkpoint(
    tmp_path, monkeypatch
):
    checkout, namespace, claim, checkpoint_path, _checkpoint, _log = (
        _scale4096_terminal_artifacts(tmp_path)
    )
    checkpoint_path.write_bytes(b"not a PyTorch checkpoint\n")
    with pytest.raises(launcher.LaunchRefused, match="weights-only load"):
        _audit_terminal_fixture(
            checkout,
            namespace,
            claim,
            _checkpoint,
            monkeypatch,
            load_error=ValueError("corrupt checkpoint"),
        )


@pytest.mark.parametrize(
    "state_key",
    (
        "model_state_dict",
        "optimizer_state_dict",
        "obs_norm_state_dict",
        "privileged_obs_norm_state_dict",
    ),
)
def test_scale4096_terminal_checkpoint_gate_rejects_nonfinite_tensor(
    tmp_path, state_key, monkeypatch
):
    checkout, namespace, claim, checkpoint_path, checkpoint, _log = (
        _scale4096_terminal_artifacts(tmp_path)
    )
    checkpoint[state_key] = {"bad": _FakeTensor([float("nan")])}
    with pytest.raises(launcher.LaunchRefused, match="non-finite tensor"):
        _audit_terminal_fixture(
            checkout, namespace, claim, checkpoint, monkeypatch
        )


def test_scale4096_terminal_checkpoint_gate_rejects_wrong_iteration(
    tmp_path, monkeypatch
):
    checkout, namespace, claim, checkpoint_path, checkpoint, _log = (
        _scale4096_terminal_artifacts(tmp_path)
    )
    checkpoint["iter"] = 4
    with pytest.raises(launcher.LaunchRefused, match="iteration/launch-claim"):
        _audit_terminal_fixture(
            checkout, namespace, claim, checkpoint, monkeypatch
        )


def test_scale4096_terminal_gate_rejects_missing_safety_counters(
    tmp_path, monkeypatch
):
    checkout, namespace, claim, _checkpoint_path, _checkpoint, log_path = (
        _scale4096_terminal_artifacts(tmp_path)
    )
    rewritten = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        prefix = "HOPE_EXACT_BEHAVIOR_UPDATE_JSON="
        if line.startswith(prefix):
            row = json.loads(line[len(prefix) :])
            row["counters"].pop("virtual_contact_nonfinite_reject_count")
            line = prefix + json.dumps(row, sort_keys=True, separators=(",", ":"))
        rewritten.append(line)
    log_path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")
    with pytest.raises(launcher.LaunchRefused, match="nonfinite counters.*missing"):
        _audit_terminal_fixture(
            checkout, namespace, claim, _checkpoint, monkeypatch
        )


@pytest.mark.parametrize("counter_kind", ("actual_hard", "table", "nonfinite"))
def test_scale4096_terminal_gate_rejects_observed_safety_event(
    tmp_path, counter_kind, monkeypatch
):
    checkout, namespace, claim, _checkpoint_path, checkpoint, log_path = (
        _scale4096_terminal_artifacts(tmp_path)
    )
    rewritten = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        prefix, separator, payload = line.partition("=")
        if not separator or not prefix.startswith("HOPE_"):
            rewritten.append(line)
            continue
        row = json.loads(payload)
        if row.get("ppo_update") == 0:
            if counter_kind == "actual_hard" and prefix == "HOPE_JOINT_SAFETY_UPDATE_JSON":
                row["counter_totals"]["actual_hard_edge_events"] = 1
            elif counter_kind == "table" and prefix == "HOPE_REWARD_SAFETY_TRANSITION_UPDATE_JSON":
                row["terminal_transitions"] = [
                    {"termination_terms": ["robot_hit_table"]}
                ]
            elif counter_kind == "nonfinite" and prefix == "HOPE_EXACT_BEHAVIOR_UPDATE_JSON":
                row["counters"]["ready_nonfinite_value_count"] = 1
        rewritten.append(
            prefix + "=" + json.dumps(row, sort_keys=True, separators=(",", ":"))
        )
    log_path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")
    with pytest.raises(
        launcher.LaunchRefused, match="hard/table/nonfinite safety counters are nonzero"
    ):
        _audit_terminal_fixture(
            checkout, namespace, claim, checkpoint, monkeypatch
        )


@pytest.mark.parametrize(
    "mutation",
    (
        lambda result: result.__setitem__("completion", None),
        lambda result: result["completion"].__setitem__("terminal_exit_code", "9"),
        lambda result: result["output_contract"].__setitem__("ppo_update_count", 4),
        lambda result: result["output_contract"].__setitem__(
            "finite_model_save_interval", 100
        ),
    ),
)
def test_long4096_rejects_launch_accepted_without_exact_scale_terminal_receipt(
    tmp_path, monkeypatch, mutation
):
    _patch_plan_environment(monkeypatch)
    spec_path, spec, _ = _case(
        tmp_path, arm_id=launcher.ARM_IDS[0], stage="long4096"
    )
    predecessor_path = Path(spec["predecessor_result"]["path"])
    predecessor = json.loads(predecessor_path.read_text())
    predecessor.pop("content_sha256")
    mutation(predecessor)
    spec["predecessor_result"]["sha256"] = _write(
        predecessor_path, _sealed(predecessor)
    )
    _write(spec_path, spec)
    with pytest.raises(launcher.LaunchRefused, match="finite natural-exit receipt"):
        launcher.build_plan(spec_path)


def test_long4096_rejects_failure_branch_predecessor(tmp_path, monkeypatch):
    _patch_plan_environment(monkeypatch)
    spec_path, spec, _ = _case(
        tmp_path, arm_id=launcher.ARM_IDS[0], stage="long4096"
    )
    probe_path = tmp_path / (launcher.ARM_IDS[0] + ".probe512.json")
    spec["predecessor_result"] = {
        "path": str(probe_path),
        "sha256": hashlib.sha256(probe_path.read_bytes()).hexdigest(),
    }
    _write(spec_path, spec)
    with pytest.raises(launcher.LaunchRefused, match="scale4096 predecessor result"):
        launcher.build_plan(spec_path)


@pytest.mark.parametrize(
    "field",
    (
        "arm_materialization",
        "policy_recipe_materialization",
        "oracle32_receipt",
    ),
)
def test_long4096_rejects_scale_reward_policy_or_oracle_lineage_drift(
    tmp_path, monkeypatch, field
):
    _patch_plan_environment(monkeypatch)
    spec_path, spec, _ = _case(
        tmp_path, arm_id=launcher.ARM_IDS[0], stage="long4096"
    )
    predecessor_path = Path(spec["predecessor_result"]["path"])
    predecessor = json.loads(predecessor_path.read_text())
    predecessor.pop("content_sha256")
    predecessor[field]["content_sha256"] = "8" * 64
    spec["predecessor_result"]["sha256"] = _write(
        predecessor_path, _sealed(predecessor)
    )
    _write(spec_path, spec)
    with pytest.raises(launcher.LaunchRefused, match="predecessor arm/oracle lineage"):
        launcher.build_plan(spec_path)


def test_confirm_digest_mismatch_blocks_before_source_lock_or_namespace(tmp_path, monkeypatch):
    _patch_plan_environment(monkeypatch)
    spec_path, _, _ = _case(tmp_path, arm_id=launcher.ARM_IDS[0], stage="materialize")
    plan = launcher.build_plan(spec_path)
    monkeypatch.setattr(launcher._B, "_verify_clean_source", lambda *args: pytest.fail("source touched"))
    with pytest.raises(launcher.LaunchRefused, match="confirm-claim differs"):
        launcher.execute(plan, confirm_claim="f" * 64)
    assert not Path(plan["canonical_payload"]["spec"]["namespace"]).exists()


def test_claim_namespace_is_no_clobber(tmp_path, monkeypatch):
    _patch_plan_environment(monkeypatch)
    spec_path, _, _ = _case(tmp_path, arm_id=launcher.ARM_IDS[0], stage="materialize")
    plan = launcher.build_plan(spec_path)
    namespace = launcher._B._claim_namespace(plan)
    original = (namespace / "launch_claim.json").read_bytes()
    with pytest.raises(launcher.LaunchRefused):
        launcher._B._claim_namespace(plan)
    assert (namespace / "launch_claim.json").read_bytes() == original


def test_pre_exec_admission_race_refuses_before_execve(tmp_path, monkeypatch):
    _patch_plan_environment(monkeypatch)
    spec_path, _, _ = _case(tmp_path, arm_id=launcher.ARM_IDS[0], stage="materialize")
    plan = launcher.build_plan(spec_path)
    namespace = launcher._B._claim_namespace(plan)
    monkeypatch.setattr(launcher._B, "_validate_runtime_asset_claim", lambda value: value)
    monkeypatch.setattr(launcher, "_lock_gpu_admission", lambda fd: None)
    monkeypatch.setattr(launcher, "_unlock_gpu_admission", lambda fd: None)
    monkeypatch.setattr(
        launcher,
        "_verify_gpu_admission",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            launcher.LaunchRefused("pre_exec race occupied GPU")
        ),
    )
    monkeypatch.setattr(os, "execve", lambda *args: pytest.fail("execve reached"))
    lock_path = Path(plan["canonical_payload"]["spec"]["gpu"]["lock_path"])
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        with pytest.raises(launcher.LaunchRefused, match="pre_exec race"):
            launcher._internal_exec(
                namespace / "launch_claim.json", plan["launch_claim_sha256"], lock_fd
            )
    finally:
        os.close(lock_fd)
    assert not (namespace / "pre_exec_gpu_admission.json").exists()


def test_post_boot_admission_failure_routes_exact_cleanup_and_spends_namespace(
    tmp_path, monkeypatch
):
    _patch_plan_environment(monkeypatch)
    spec_path, _, _ = _case(tmp_path, arm_id=launcher.ARM_IDS[0], stage="long512")
    plan = launcher.build_plan(spec_path)
    monkeypatch.setattr(launcher._B, "_validate_runtime_asset_claim", lambda value: value)
    lock_file = tmp_path / "gpu.lock"
    monkeypatch.setattr(
        launcher, "_open_gpu_shared_lock", lambda path: os.open(lock_file, os.O_RDWR | os.O_CREAT, 0o600)
    )
    monkeypatch.setattr(launcher, "_lock_gpu_admission", lambda fd: None)
    monkeypatch.setattr(launcher, "_unlock_gpu_admission", lambda fd: None)
    phases = []

    def admission(spec, *, phase, current_namespace, require_current_compute=False, **kwargs):
        phases.append((phase, require_current_compute))
        if phase == "post_boot":
            raise launcher.LaunchRefused("post_boot unknown pid")
        return {"phase": phase}

    monkeypatch.setattr(launcher, "_verify_gpu_admission", admission)
    monkeypatch.setattr(
        launcher, "_reservation_document", lambda spec, digest: {"claim": digest}
    )
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda *args, **kwargs: type("Completed", (), {"returncode": 0})(),
    )
    cleanup_calls = []

    def cleanup(namespace, state, claim_sha, reason):
        cleanup_calls.append((namespace, state, claim_sha, reason))
        return {"cleanup": {"completed": True}, "path": namespace / "cleanup-failure.json"}

    monkeypatch.setattr(launcher, "_cleanup_post_boot_admission_failure", cleanup)
    with pytest.raises(launcher.LaunchRefused, match=r"cleanup completed.*cleanup-failure.json"):
        launcher.execute(plan, confirm_claim=plan["launch_claim_sha256"])
    namespace = Path(plan["canonical_payload"]["spec"]["namespace"])
    assert namespace.is_dir()
    assert (namespace / "launch_claim.json").is_file()
    assert phases == [("pre_launch", False), ("post_boot", True)]
    assert len(cleanup_calls) == 1
    assert cleanup_calls[0][0] == namespace
    assert cleanup_calls[0][2] == plan["launch_claim_sha256"]


def test_claim_revalidation_detects_code_owned_bundle_mutation(tmp_path, monkeypatch):
    _patch_plan_environment(monkeypatch)
    spec_path, _, _ = _case(tmp_path, arm_id=launcher.ARM_IDS[0], stage="materialize")
    plan = launcher.build_plan(spec_path)
    namespace = Path(plan["canonical_payload"]["spec"]["namespace"])
    namespace.mkdir()
    payload = copy.deepcopy(plan["canonical_payload"])
    payload["bundle"]["normalizers"]["actor"]["state"] = "donor"
    monkeypatch.setattr(launcher._B, "_validate_runtime_asset_claim", lambda value: value)
    with pytest.raises(launcher.LaunchRefused, match="drifted"):
        launcher._revalidate_claim_payload(payload)


@pytest.mark.parametrize("retired_key", ("target_recipe", "target_validity_mask", "resume_path", "checkpoint_path"))
def test_spec_rejects_retired_control_keys(tmp_path, retired_key):
    _, spec, _ = _case(tmp_path, arm_id=launcher.ARM_IDS[0], stage="materialize")
    spec[retired_key] = "forbidden"
    with pytest.raises(launcher.LaunchRefused):
        launcher._validate_spec(spec)


def test_template_colocation_and_python_symlink(tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    real_python = tmp_path / "real-python"
    real_python.write_text("#!/bin/sh\n")
    real_python.chmod(0o755)
    venv_python = tmp_path / "venv-python"
    venv_python.symlink_to(real_python)
    output = tmp_path / "template.json"
    root = tmp_path / launcher.EXPERIMENT_NAME
    root.mkdir()
    args = launcher._parser().parse_args([
        "template", "--output", str(output), "--checkout", str(checkout),
        "--commit-sha", "a" * 40, "--isaac-python", str(venv_python),
        "--arm-id", launcher.ARM_IDS[0], "--lineage-path", "a211.json",
        "--lineage-sha256", "b" * 64, "--stage", "materialize",
        "--gpu-index", "2", "--gpu-uuid", "GPU-12345678", "--owner", "Franco",
        "--namespace", str(root / "fresh"), "--allow-colocation",
    ])
    launcher._write_template(args)
    document = json.loads(output.read_text())
    assert document["source"]["isaac_python"] == str(venv_python)
    assert document[launcher.COLOCATION_SPEC_KEY] is True
    assert document["gpu"]["require_empty"] is False


def test_parser_exposes_explicit_execute_and_hidden_exec():
    parser = launcher._parser()
    subparsers = next(action for action in parser._actions if action.__class__.__name__ == "_SubParsersAction")
    assert set(subparsers.choices) == {"template", "plan", "execute", "_exec"}


def test_launcher_never_sets_or_repurposes_home():
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"HOME"' not in source
    assert "$HOME" not in source
