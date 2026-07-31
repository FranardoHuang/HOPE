#!/usr/bin/env python3
"""Materialize the first A3-vendor shared-ready identity, fail closed.

This launcher exists only to break the one-time bootstrap cycle in which the
new A3 plant needs a live ``training_contract.json`` before an action-specific
dynamic-ready bundle can be rematerialized.  It has exactly two stages:

* ``recipe``: one environment, no PPO update, and one no-clobber shared-ready
  policy recipe at ``<namespace>/vendor_shared_ready_policy_recipe.json``;
* ``smoke``: one environment and exactly two PPO updates, saving every update,
  so the run emits its live training contract and bounded smoke checkpoints.

The smoke spec must pin the policy-contract SHA emitted by the recipe stage.
Both stages use only one fixed, tracked stable-v2 ``bh_loop_c`` motion and its
exact N=1 manifest.  They intentionally consume no bundle at all, so no
schema-v2/dynamic-ready artifact can enter the cycle.  The selected task is
always ``HOPEPingPongActionBallA3VendorV1`` and all task-owned vendor PD, push
and control-step-delay settings remain untouched.

This is diagnostic identity materialization, not a training wave.  It cannot
resume, launch a long stage, mint formal evidence, promote a curriculum,
export a policy, judge a checkpoint, or authorize hardware.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence


_THIS_FILE = Path(__file__).resolve()
_SAFETY_FILE = _THIS_FILE.with_name("launch_n1_reward_screen_diagnostic.py")
_SAFETY_SPEC = importlib.util.spec_from_file_location(
    "_hope_vendor_identity_safety", _SAFETY_FILE
)
if _SAFETY_SPEC is None or _SAFETY_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"cannot load launch safety implementation: {_SAFETY_FILE}")
_S = importlib.util.module_from_spec(_SAFETY_SPEC)
_SAFETY_SPEC.loader.exec_module(_S)


SCHEMA_VERSION = 1
SPEC_KIND = "a3_vendor_identity_smoke_spec_v1"
CLAIM_KIND = "a3_vendor_identity_smoke_claim_v1"
RESULT_KIND = "a3_vendor_identity_smoke_launch_result_v1"
EXPERIMENT_NAME = "agibot_a3_action_ball_vendor_identity_smoke"
TASK_PROFILE_ID = "HOPEPingPongActionBallA3VendorV1"
ACTION_ID = "bh_loop_c"
SCOPE = "upper"
SEED = 0
ACTOR_OBS_CONTRACT = (
    "action_ball_table_pose_twist_heading_task_teacher_start_v2"
)
RECIPE_SENTINEL_POLICY_SHA256 = "0" * 64
RECIPE_FILENAME = "vendor_shared_ready_policy_recipe.json"
DIAGNOSTIC_SUFFIX = "DIAGNOSTIC_UNAUTHORIZED"
# Exact receipt emitted by the adopted ActionBall current-low reward subtree.
# The vendor leaf inherits that subtree byte-for-byte and changes only action
# delay and push event configuration, neither of which is part of the reward
# receipt.  Existing successful N1 diagnostics pin this same value; the live
# recipe stage independently recomputes it before scene construction.
EXPECTED_REWARD_RECIPE_SHA256 = (
    "c2f13419a22fd12d1ab93d936516f8e990dad1b5b51a03f4e93c4d02e4e26c11"
)

LAUNCHER_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "launch_a3_vendor_identity_smoke.py"
)
SAFETY_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "launch_n1_reward_screen_diagnostic.py"
)
TRAIN_SOURCE = "hope_training/whole_body_tracking/scripts/train.py"
TASK_SOURCE = (
    "hope_training/whole_body_tracking/cfg/task/"
    "HOPEPingPongActionBallA3VendorV1.yaml"
)
ROBOT_SOURCE = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/robots/agibot_a3.py"
)
TRAINING_CONTRACT_SOURCE = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/utils/training_contract.py"
)
ACTION_SOURCE = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp/hope_actions.py"
)
RUNNER_SOURCE = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/utils/my_on_policy_runner.py"
)
KIT_LAUNCHER_SOURCE = (
    "hope_training/whole_body_tracking/scripts/launch_kit_training_locked.sh"
)
WBT_RELATIVE = Path("hope_training/whole_body_tracking")

MOTION_PIN: Mapping[str, str] = {
    "path": (
        "assets/motions/fivebind_20260727/"
        "bh_loop_c_upper_stable_v2.npz"
    ),
    "sha256": "0fa46ad66d57edd006b0a70a7de0542d8d53945ee3ae9802fdbd937555a0c85b",
}
MANIFEST_PIN: Mapping[str, str] = {
    "path": (
        "configs/n1_contact_20260730_stable_v2/"
        "bh_loop_c.manifest.v3.775f74183e58.json"
    ),
    "sha256": "775f74183e58683df48f5f44084e89320736d1533a4d962f43f455664830d8e5",
}

_SPEC_KEYS = (
    "schema_version",
    "kind",
    "source",
    "motion",
    "manifest",
    "expected_effective_reward_recipe_sha256",
    "policy_contract_sha256",
    "seed",
    "stage",
    "num_envs",
    "max_iterations",
    "save_interval",
    "gpu",
    "namespace",
    "log_path",
)
_SOURCE_KEYS = ("checkout", "commit_sha", "isaac_python")
_PIN_KEYS = ("path", "sha256")

LaunchRefused = _S.LaunchRefused
canonical_sha256 = _S.canonical_sha256


def _validate_stage(
    stage: Any,
    num_envs: Any,
    max_iterations: Any,
    save_interval: Any,
    policy_contract_sha256: Any,
) -> dict[str, Any]:
    if stage not in ("recipe", "smoke") or type(stage) is not str:
        raise LaunchRefused("stage must be exactly 'recipe' or 'smoke'")
    envs = _S._plain_int(num_envs, name="num_envs", minimum=1)
    iterations = _S._plain_int(
        max_iterations, name="max_iterations", minimum=1
    )
    save = _S._plain_int(save_interval, name="save_interval", minimum=1)
    expected_budget = (1, 1, 1) if stage == "recipe" else (1, 2, 1)
    if (envs, iterations, save) != expected_budget:
        explanation = (
            "recipe is exactly 1 env / materialization-only / configured "
            "iteration budget 1 / save interval 1"
            if stage == "recipe"
            else "smoke is exactly 1 env / 2 PPO updates / save interval 1"
        )
        raise LaunchRefused(explanation)
    if stage == "recipe":
        if policy_contract_sha256 is not None:
            raise LaunchRefused(
                "recipe policy_contract_sha256 must be null; the stage "
                "materializes that identity"
            )
        policy_sha = None
    else:
        policy_sha = _S._sha256(
            policy_contract_sha256, name="policy_contract_sha256"
        )
        if policy_sha == RECIPE_SENTINEL_POLICY_SHA256:
            raise LaunchRefused(
                "smoke cannot use the recipe-stage policy SHA sentinel"
            )
    return {
        "stage": stage,
        "num_envs": envs,
        "max_iterations": iterations,
        "save_interval": save,
        "policy_contract_sha256": policy_sha,
    }


def _validate_spec_document(
    document: dict[str, Any], *, namespace_claimed: bool = False
) -> dict[str, Any]:
    row = _S._exact_dict(document, _SPEC_KEYS, name="identity-smoke spec")
    if row["schema_version"] != SCHEMA_VERSION or row["kind"] != SPEC_KIND:
        raise LaunchRefused(
            f"identity-smoke spec must be schema {SCHEMA_VERSION} / {SPEC_KIND!r}"
        )
    source = _S._exact_dict(row["source"], _SOURCE_KEYS, name="spec.source")
    checkout = _S._absolute_path(
        source["checkout"], name="spec.source.checkout", must_exist=True
    )
    commit_sha = source["commit_sha"]
    if type(commit_sha) is not str or _S.COMMIT_RE.fullmatch(commit_sha) is None:
        raise LaunchRefused("spec.source.commit_sha must be 40 lowercase hex")
    isaac_python = _S._absolute_path(
        source["isaac_python"], name="spec.source.isaac_python", must_exist=True
    )
    try:
        python_stat = isaac_python.stat()
    except OSError as exc:
        raise LaunchRefused(f"cannot stat Isaac Python: {exc}") from exc
    if not stat.S_ISREG(python_stat.st_mode) or not os.access(isaac_python, os.X_OK):
        raise LaunchRefused("spec.source.isaac_python must be executable")

    motion = _S._exact_dict(
        row["motion"], _PIN_KEYS, name="spec.motion"
    )
    manifest = _S._exact_dict(
        row["manifest"], _PIN_KEYS, name="spec.manifest"
    )
    if dict(motion) != dict(MOTION_PIN):
        raise LaunchRefused(
            "motion must be the fixed tracked bh_loop_c stable-v2 motion"
        )
    if dict(manifest) != dict(MANIFEST_PIN):
        raise LaunchRefused("manifest must be the fixed tracked bh_loop_c N=1 manifest")
    if row["seed"] != SEED or type(row["seed"]) is not int:
        raise LaunchRefused("identity-smoke seed must be exactly 0")
    stage = _validate_stage(
        row["stage"],
        row["num_envs"],
        row["max_iterations"],
        row["save_interval"],
        row["policy_contract_sha256"],
    )
    reward_sha = _S._sha256(
        row["expected_effective_reward_recipe_sha256"],
        name="expected_effective_reward_recipe_sha256",
    )
    if reward_sha != EXPECTED_REWARD_RECIPE_SHA256:
        raise LaunchRefused(
            "expected_effective_reward_recipe_sha256 must equal the fixed "
            "vendor-task inherited ActionBall reward receipt"
        )
    gpu = _S._validate_gpu(row["gpu"])
    namespace = _S._absolute_path(row["namespace"], name="namespace")
    if (
        namespace.name in ("", ".", "..")
        or _S.SAFE_COMPONENT_RE.fullmatch(namespace.name) is None
        or not namespace.name.startswith("a3vendor-identity-")
    ):
        raise LaunchRefused(
            "namespace basename must start with 'a3vendor-identity-' and be safe"
        )
    log_path = _S._absolute_path(row["log_path"], name="log_path")
    if log_path != namespace / "run.log":
        raise LaunchRefused("log_path must be exactly <namespace>/run.log")
    if os.path.lexists(namespace):
        if not namespace_claimed:
            raise LaunchRefused(
                f"run namespace already exists and is permanently spent: {namespace}"
            )
        try:
            namespace_stat = namespace.lstat()
        except OSError as exc:
            raise LaunchRefused(f"claimed namespace cannot be inspected: {exc}") from exc
        if (
            not stat.S_ISDIR(namespace_stat.st_mode)
            or namespace.resolve(strict=True) != namespace
        ):
            raise LaunchRefused("claimed namespace must remain a real directory")
    elif namespace_claimed:
        raise LaunchRefused("claimed namespace vanished before trainer exec")
    parent = namespace.parent
    if not parent.exists() or parent.resolve(strict=True) != parent:
        raise LaunchRefused(
            "namespace parent must be an existing real absolute directory"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": SPEC_KIND,
        "source": {
            "checkout": str(checkout),
            "commit_sha": commit_sha,
            "isaac_python": str(isaac_python),
        },
        "motion": dict(MOTION_PIN),
        "manifest": dict(MANIFEST_PIN),
        "expected_effective_reward_recipe_sha256": reward_sha,
        "policy_contract_sha256": stage["policy_contract_sha256"],
        "seed": SEED,
        "stage": stage["stage"],
        "num_envs": stage["num_envs"],
        "max_iterations": stage["max_iterations"],
        "save_interval": stage["save_interval"],
        "gpu": gpu,
        "namespace": str(namespace),
        "log_path": str(log_path),
    }


def _validate_runtime_sources(
    checkout: Path, commit_sha: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    sources = (
        (LAUNCHER_SOURCE, "A3 vendor identity-smoke launcher"),
        (SAFETY_SOURCE, "identity-smoke launch safety implementation"),
        (TRAIN_SOURCE, "training entrypoint"),
        (TASK_SOURCE, f"immutable task profile {TASK_PROFILE_ID}"),
        (ROBOT_SOURCE, "A3 vendor actuator source"),
        (TRAINING_CONTRACT_SOURCE, "training-contract implementation"),
        (ACTION_SOURCE, "control-step action-delay implementation"),
        (RUNNER_SOURCE, "runtime ABI/std receipt implementation"),
        (KIT_LAUNCHER_SOURCE, "Kit locked launcher"),
    )
    for relative, label in sources:
        pin = {"path": relative, "sha256": _S.sha256_file(checkout / relative)}
        normalized, _path = _S._verify_tracked_file(
            checkout, commit_sha, pin, name=label
        )
        result[label] = normalized
    if _THIS_FILE != checkout / LAUNCHER_SOURCE:
        raise LaunchRefused(
            "running identity-smoke launcher is not the selected checkout path"
        )
    return result


def _validate_scientific_inputs(
    checkout: Path,
    commit_sha: str,
    motion_pin: Mapping[str, Any],
    manifest_pin: Mapping[str, Any],
) -> dict[str, Any]:
    normalized_motion, _motion_path = _S._verify_tracked_file(
        checkout,
        commit_sha,
        dict(motion_pin),
        name="identity-smoke stable-v2 motion",
    )
    normalized_manifest, manifest = _S._load_tracked_json(
        checkout,
        commit_sha,
        dict(manifest_pin),
        name="identity-smoke N=1 manifest",
    )
    actions = manifest.get("actions") if type(manifest) is dict else None
    if (
        normalized_motion != dict(MOTION_PIN)
        or normalized_manifest["path"] != MANIFEST_PIN["path"]
        or normalized_manifest["sha256"] != MANIFEST_PIN["sha256"]
        or manifest.get("schema_version") != 3
        or manifest.get("action_order") != [ACTION_ID]
        or manifest.get("mobility_mode") != "no_move"
        or not isinstance(actions, list)
        or len(actions) != 1
        or type(actions[0]) is not dict
        or actions[0].get("action_id") != ACTION_ID
        or actions[0].get("motion_path") != MOTION_PIN["path"]
        or actions[0].get("motion_sha256") != MOTION_PIN["sha256"]
    ):
        raise LaunchRefused(
            "identity-smoke manifest is not exact N=1 bh_loop_c/no_move/stable-v2"
        )
    return {"motion": normalized_motion, "manifest": normalized_manifest}


def _rsl_output_contract(spec: Mapping[str, Any]) -> dict[str, Any]:
    namespace_name = Path(spec["namespace"]).name
    suffix = f"_{namespace_name}-{DIAGNOSTIC_SUFFIX}"
    root = (
        Path(spec["source"]["checkout"])
        / WBT_RELATIVE
        / "logs/rsl_rl"
        / EXPERIMENT_NAME
    )
    if spec["stage"] == "recipe":
        return {
            "namespace_recipe": str(Path(spec["namespace"]) / RECIPE_FILENAME),
            "ppo_update_count": 0,
            "rsl_experiment_root": str(root),
            "rsl_run_suffix": suffix,
            "training_contract": None,
            "checkpoints": [],
            "required_runtime_log_events": [],
        }
    return {
        "namespace_recipe": None,
        "ppo_update_count": 2,
        "rsl_experiment_root": str(root),
        "rsl_run_suffix": suffix,
        "training_contract": "params/training_contract.json",
        "checkpoints": ["model_0.pt", "model_1.pt"],
        "required_runtime_log_events": [
            "HOPE_CONTROL_STEP_ACTION_DELAY_RUNTIME_JSON=",
            "HOPE_RSL_RL_RUNTIME_ABI_JSON=",
            "HOPE_POLICY_STD_UPDATE_JSON=",
        ],
    }


def _check_rsl_namespace_available(spec: Mapping[str, Any]) -> None:
    output = _rsl_output_contract(spec)
    root = Path(output["rsl_experiment_root"])
    if not os.path.lexists(root):
        return
    if root.resolve(strict=True) != root or not root.is_dir():
        raise LaunchRefused("identity-smoke RSL experiment root is not real")
    spent = sorted(
        child.name
        for child in root.iterdir()
        if child.name.endswith(output["rsl_run_suffix"])
    )
    if spent:
        raise LaunchRefused(f"identity-smoke trainer run_name is spent: {spent[0]}")


def _build_training_argv(
    spec: Mapping[str, Any], scientific_inputs: Mapping[str, Any]
) -> list[str]:
    checkout = Path(spec["source"]["checkout"])
    wbt = checkout / WBT_RELATIVE
    motion = checkout / scientific_inputs["motion"]["path"]
    manifest = checkout / scientific_inputs["manifest"]["path"]
    json_list = lambda values: json.dumps(  # noqa: E731
        values, separators=(",", ":"), ensure_ascii=False
    )
    policy_sha = (
        RECIPE_SENTINEL_POLICY_SHA256
        if spec["stage"] == "recipe"
        else spec["policy_contract_sha256"]
    )
    argv = [
        spec["source"]["isaac_python"],
        str(wbt / "scripts/train.py"),
        f"task={TASK_PROFILE_ID}",
        "algo=ppo",
        "algo.policy.init_noise_std=0.02",
        "action_ball_shared_ready_bootstrap=true",
        "headless=true",
        "logger=tensorboard",
        "video=false",
        "device=cuda:0",
        f"seed={SEED}",
        f"num_envs={spec['num_envs']}",
        f"max_iterations={spec['max_iterations']}",
        f"algo.runner.save_interval={spec['save_interval']}",
        f"run_name={Path(spec['namespace']).name}",
        f"task.experiment_name={EXPERIMENT_NAME}",
        (
            "expected_effective_reward_recipe_sha256="
            f"{spec['expected_effective_reward_recipe_sha256']}"
        ),
        f"task.actor_obs_contract={ACTOR_OBS_CONTRACT}",
        "task.rewards.full_body_mimic=false",
        f"motion_file={json_list([str(motion)])}",
        f"task.racket.clip_names={json_list([ACTION_ID])}",
        "task.racket.target_mode=action_ball",
        f"task.racket.action_ball_manifest_path={manifest}",
        (
            "task.racket.action_ball_manifest_sha256="
            f"{scientific_inputs['manifest']['sha256']}"
        ),
        f"task.racket.action_ball_policy_contract_sha256={policy_sha}",
        "task.racket.action_ball_diagnostic_unauthorized=true",
        "+task.racket.reference_guard_mode=metrics_only",
        f"task.racket.action_ball_seed={SEED}",
        "task.racket.question_bank=",
        "task.racket.question_bank_allow_legacy=false",
        "task.racket.cq_anchor_bank=",
        "task.racket.exam_bank=",
    ]
    if spec["stage"] == "recipe":
        argv.append(
            "action_ball_policy_recipe_output_path="
            f"{Path(spec['namespace']) / RECIPE_FILENAME}"
        )
    forbidden = (
        "action_ball_dynamic_ready",
        "checkpoint_path=",
        "stable_ready_plant",
        "task.push",
        "push_robot",
        "randomize_pd_gains",
        "kp_gain_range",
        "kd_gain_range",
        "control_step_action_delay_",
    )
    if any(fragment in item for item in argv for fragment in forbidden):
        raise LaunchRefused(
            "identity-smoke argv overrides a forbidden resume/dynamic/task-owned axis"
        )
    return argv


def build_plan(spec_path: Path) -> dict[str, Any]:
    absolute = _S._absolute_path(str(spec_path), name="--spec", must_exist=True)
    _S._stable_regular_file(absolute, name="identity-smoke spec")
    raw = absolute.read_bytes()
    document = _S._strict_json_bytes(raw, name="identity-smoke spec")
    if raw != _S._canonical_bytes(document) + b"\n":
        raise LaunchRefused("identity-smoke spec must be canonical JSON plus newline")
    spec = _validate_spec_document(document)
    checkout = Path(spec["source"]["checkout"])
    commit_sha = spec["source"]["commit_sha"]
    source = _S._verify_clean_source(checkout, commit_sha)
    runtime_sources = _validate_runtime_sources(checkout, commit_sha)
    runtime_assets = _S._validate_runtime_asset_environment()
    scientific_inputs = _validate_scientific_inputs(
        checkout, commit_sha, spec["motion"], spec["manifest"]
    )
    _check_rsl_namespace_available(spec)
    argv = _build_training_argv(spec, scientific_inputs)
    output_contract = _rsl_output_contract(spec)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": CLAIM_KIND,
        "diagnostic_unauthorized": True,
        "identity_materialization_only": True,
        "formal_evidence_prohibited": True,
        "curriculum_promotion_prohibited": True,
        "resume_prohibited": True,
        "export_prohibited": True,
        "judge_prohibited": True,
        "hardware_authority_prohibited": True,
        "long_stage_prohibited": True,
        "spec_file_sha256": hashlib.sha256(raw).hexdigest(),
        "spec": spec,
        "fixed_identity": {
            "task_profile": TASK_PROFILE_ID,
            "action_id": ACTION_ID,
            "scope": SCOPE,
            "actor_obs_contract": ACTOR_OBS_CONTRACT,
            "bootstrap": "shared_ready_fresh",
            "dynamic_ready": False,
        },
        "source": source,
        "runtime_sources": runtime_sources,
        "runtime_assets": runtime_assets,
        "scientific_inputs": scientific_inputs,
        "output_contract": output_contract,
        "boot_marker": (
            "ACTION_BALL_POLICY_RECIPE_MATERIALIZED"
            if spec["stage"] == "recipe"
            else "Learning iteration"
        ),
        "training_argv": argv,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": CLAIM_KIND,
        "launch_claim_sha256": canonical_sha256(payload),
        "canonical_payload": payload,
    }


def _load_internal_claim(
    claim_path: Path, expected_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    absolute = _S._absolute_path(
        str(claim_path), name="internal claim", must_exist=True
    )
    _S._stable_regular_file(absolute, name="internal identity-smoke claim")
    raw = absolute.read_bytes()
    plan = _S._strict_json_bytes(raw, name="internal identity-smoke claim")
    if raw != _S._canonical_bytes(plan) + b"\n":
        raise LaunchRefused("internal identity-smoke claim is not canonical")
    outer = _S._exact_dict(
        plan,
        ("schema_version", "kind", "launch_claim_sha256", "canonical_payload"),
        name="internal identity-smoke claim",
    )
    if (
        outer["schema_version"] != SCHEMA_VERSION
        or outer["kind"] != CLAIM_KIND
        or outer["launch_claim_sha256"] != expected_sha256
        or canonical_sha256(outer["canonical_payload"]) != expected_sha256
    ):
        raise LaunchRefused("internal identity-smoke claim digest differs")
    payload = outer["canonical_payload"]
    if type(payload) is not dict or payload.get("kind") != CLAIM_KIND:
        raise LaunchRefused("internal identity-smoke payload kind differs")
    return plan, payload


def _internal_exec(claim_path: Path, expected_sha256: str, lock_fd: int) -> int:
    _plan, payload = _load_internal_claim(claim_path, expected_sha256)
    spec = _validate_spec_document(payload["spec"], namespace_claimed=True)
    checkout = Path(spec["source"]["checkout"])
    commit_sha = spec["source"]["commit_sha"]
    source = _S._verify_clean_source(checkout, commit_sha)
    if source != payload["source"]:
        raise LaunchRefused("source identity drifted after namespace claim")
    runtime = _validate_runtime_sources(checkout, commit_sha)
    if runtime != payload["runtime_sources"]:
        raise LaunchRefused("runtime source identity drifted after namespace claim")
    runtime_assets = _S._validate_runtime_asset_claim(payload.get("runtime_assets"))
    scientific_inputs = _validate_scientific_inputs(
        checkout, commit_sha, spec["motion"], spec["manifest"]
    )
    if scientific_inputs != payload["scientific_inputs"]:
        raise LaunchRefused("scientific inputs drifted after namespace claim")
    argv = _build_training_argv(spec, scientific_inputs)
    if argv != payload["training_argv"]:
        raise LaunchRefused("training argv differs from immutable claim")
    if _rsl_output_contract(spec) != payload["output_contract"]:
        raise LaunchRefused("output contract differs from immutable claim")

    lock_path = Path(spec["gpu"]["lock_path"])
    try:
        lock_stat = os.fstat(lock_fd)
        path_stat = lock_path.lstat()
    except OSError as exc:
        raise LaunchRefused(f"inherited GPU lock cannot be verified: {exc}") from exc
    if (
        not stat.S_ISREG(lock_stat.st_mode)
        or not stat.S_ISREG(path_stat.st_mode)
        or (lock_stat.st_dev, lock_stat.st_ino)
        != (path_stat.st_dev, path_stat.st_ino)
    ):
        raise LaunchRefused("inherited GPU lock differs from shared lock path")
    try:
        _S.fcntl.flock(lock_fd, _S.fcntl.LOCK_EX | _S.fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise LaunchRefused("inherited GPU lock is not owned by this launch") from exc
    second_gpu = _S._verify_gpu_empty(
        spec["gpu"]["index"], spec["gpu"]["uuid"]
    )
    _S._write_exclusive_json(
        Path(spec["namespace"]) / "pre_exec_gpu_admission.json",
        {
            "schema_version": 1,
            "kind": "a3_vendor_identity_smoke_pre_exec_gpu_admission_v1",
            "launch_claim_sha256": expected_sha256,
            "gpu": second_gpu,
        },
    )
    wbt = checkout / WBT_RELATIVE
    environment = {
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "HOME": os.environ.get("HOME", "/root"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(wbt / "source/whole_body_tracking"),
        "CUDA_VISIBLE_DEVICES": str(spec["gpu"]["index"]),
        "HYDRA_FULL_ERROR": "1",
        "WANDB_MODE": "offline",
        "HOPE_URDF_IMPORTER_NO_UI": runtime_assets["urdf_importer_no_ui"],
        "HOPE_AGIBOT_A3_USD_PATH": runtime_assets["a3_preconverted_usd"]["path"],
        "LD_LIBRARY_PATH": runtime_assets["private_glu"]["directory"],
    }
    os.chdir(wbt)
    os.execve(argv[0], argv, environment)
    raise AssertionError("os.execve returned")


def launch(plan: dict[str, Any], *, confirm_claim: str) -> dict[str, Any]:
    expected = _S._sha256(confirm_claim, name="--confirm-claim")
    if expected != plan["launch_claim_sha256"]:
        raise LaunchRefused(
            "--confirm-claim differs from the freshly recomputed plan"
        )
    payload = plan["canonical_payload"]
    spec = payload["spec"]
    checkout = Path(spec["source"]["checkout"])
    _S._verify_clean_source(checkout, spec["source"]["commit_sha"])
    _S._validate_runtime_asset_claim(payload.get("runtime_assets"))
    lock_fd = _S._open_gpu_lock(Path(spec["gpu"]["lock_path"]))
    namespace: Path | None = None
    try:
        first_gpu = _S._verify_gpu_empty(
            spec["gpu"]["index"], spec["gpu"]["uuid"]
        )
        namespace = _S._claim_namespace(plan)
        _S._write_exclusive_json(
            namespace / "pre_launch_gpu_admission.json",
            {
                "schema_version": 1,
                "kind": "a3_vendor_identity_smoke_pre_launch_gpu_admission_v1",
                "launch_claim_sha256": expected,
                "gpu": first_gpu,
            },
        )
        launcher = checkout / KIT_LAUNCHER_SOURCE
        state_path = Path(spec["log_path"] + ".launch")
        internal_command = [
            str(Path(sys.executable).resolve()),
            str(checkout / LAUNCHER_SOURCE),
            "_exec",
            "--claim",
            str(namespace / "launch_claim.json"),
            "--claim-sha256",
            expected,
            "--gpu-lock-fd",
            str(lock_fd),
        ]
        environment = {
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "HOME": os.environ.get("HOME", "/root"),
            "LANG": "C",
            "LC_ALL": "C",
            "KIT_BOOT_MARKER": payload["boot_marker"],
            "KIT_BOOT_TIMEOUT_S": "2700",
            "KIT_BOOT_STALE_TIMEOUT_S": "1800",
            "KIT_BOOT_POLL_S": "5",
            "KIT_BOOT_STATE_FILE": str(state_path),
        }
        result = subprocess.run(
            [str(launcher), spec["log_path"], *internal_command],
            cwd=checkout / WBT_RELATIVE,
            env=environment,
            pass_fds=(lock_fd,),
            check=False,
        )
        if result.returncode != 0:
            raise LaunchRefused(
                f"locked Kit launcher returned {result.returncode}; "
                f"namespace remains spent at {namespace}"
            )
        return {
            "schema_version": 1,
            "kind": RESULT_KIND,
            "launch_claim_sha256": expected,
            "stage": spec["stage"],
            "namespace": str(namespace),
            "log_path": spec["log_path"],
            "state_path": str(state_path),
            "gpu": spec["gpu"],
            "output_contract": payload["output_contract"],
            "diagnostic_unauthorized": True,
            "accepted": True,
        }
    finally:
        os.close(lock_fd)


def _template_document(args: argparse.Namespace) -> dict[str, Any]:
    namespace = Path(args.namespace)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": SPEC_KIND,
        "source": {
            "checkout": args.checkout,
            "commit_sha": args.commit_sha,
            "isaac_python": args.isaac_python,
        },
        "motion": dict(MOTION_PIN),
        "manifest": dict(MANIFEST_PIN),
        "expected_effective_reward_recipe_sha256": (
            EXPECTED_REWARD_RECIPE_SHA256
        ),
        "policy_contract_sha256": (
            None if args.stage == "recipe" else args.policy_contract_sha256
        ),
        "seed": SEED,
        "stage": args.stage,
        "num_envs": 1,
        "max_iterations": 1 if args.stage == "recipe" else 2,
        "save_interval": 1,
        "gpu": {
            "index": args.gpu_index,
            "uuid": args.gpu_uuid,
            "owner": args.owner,
            "lock_path": f"/tmp/hope_lean_queue_gpu{args.gpu_index}.lock",
            "require_empty": True,
        },
        "namespace": str(namespace),
        "log_path": str(namespace / "run.log"),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    template = sub.add_parser(
        "template", help="print one canonical concrete Pod spec"
    )
    template.add_argument("--stage", choices=("recipe", "smoke"), required=True)
    template.add_argument("--checkout", required=True)
    template.add_argument("--commit-sha", required=True)
    template.add_argument("--isaac-python", required=True)
    template.add_argument("--policy-contract-sha256")
    template.add_argument("--gpu-index", required=True, type=int)
    template.add_argument("--gpu-uuid", required=True)
    template.add_argument("--owner", required=True)
    template.add_argument("--namespace", required=True)
    for command in ("plan", "launch"):
        child = sub.add_parser(command)
        child.add_argument("--spec", required=True)
        if command == "launch":
            child.add_argument("--confirm-claim", required=True)
    internal = sub.add_parser("_exec", help=argparse.SUPPRESS)
    internal.add_argument("--claim", required=True)
    internal.add_argument("--claim-sha256", required=True)
    internal.add_argument("--gpu-lock-fd", required=True, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "template":
            if args.stage == "recipe" and args.policy_contract_sha256 is not None:
                raise LaunchRefused(
                    "recipe template does not accept --policy-contract-sha256"
                )
            if args.stage == "smoke" and args.policy_contract_sha256 is None:
                raise LaunchRefused(
                    "smoke template requires --policy-contract-sha256"
                )
            print(
                json.dumps(
                    _template_document(args),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
            )
            return 0
        if args.command == "_exec":
            return _internal_exec(
                Path(args.claim), args.claim_sha256, args.gpu_lock_fd
            )
        plan = build_plan(Path(args.spec))
        if args.command == "plan":
            print(
                json.dumps(
                    plan,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
            )
            return 0
        result = launch(plan, confirm_claim=args.confirm_claim)
        print(
            json.dumps(
                result,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
        )
        return 0
    except LaunchRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
