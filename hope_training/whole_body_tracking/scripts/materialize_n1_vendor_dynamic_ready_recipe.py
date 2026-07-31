#!/usr/bin/env python3
"""Materialize one vendor-bound N1 dynamic-ready policy recipe, fail closed.

This is a recipe-only adapter over the existing A3 vendor diagnostic and
identity-smoke safety implementations.  It fixes ``bh_loop_c``, seed 0, one
environment, and zero PPO updates.  Before Kit construction it revalidates the
code-owned required identity, the actual vendor runtime-authority receipt, and
one exact schema-v2 dynamic-ready/nominal-hold/bundle chain.  The resulting
policy-contract SHA is suitable only as the later vendor diagnostic smoke's
policy input.

The wrapper is diagnostic materialization infrastructure.  It does not launch
PPO, resume a checkpoint, export a policy, judge a checkpoint, promote a
curriculum, mint formal evidence, or authorize deployment or hardware.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping, Sequence


_THIS_FILE = Path(__file__).resolve()


def _load_sibling(name: str, filename: str):
    path = _THIS_FILE.with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"cannot load required sibling module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_V = _load_sibling(
    "_hope_vendor_dynamic_recipe_diagnostic",
    "launch_n1_vendor_baseline_diagnostic.py",
)
_I = _load_sibling(
    "_hope_vendor_dynamic_recipe_identity_safety",
    "launch_a3_vendor_identity_smoke.py",
)
_S = _I._S


SCHEMA_VERSION = 1
SPEC_KIND = "n1_vendor_dynamic_ready_recipe_spec_v1"
CLAIM_KIND = "n1_vendor_dynamic_ready_recipe_claim_v1"
RESULT_KIND = "n1_vendor_dynamic_ready_recipe_result_v1"
EXPERIMENT_NAME = "agibot_a3_action_ball_vendor_dynamic_ready_recipe"
TASK_PROFILE_ID = _V.TASK_PROFILE_ID
ACTION_ID = "bh_loop_c"
SCOPE = "upper"
SEED = 0
NUM_ENVS = 1
MAX_ITERATIONS = 0
SAVE_INTERVAL = 1
STAGE = "recipe"
RECIPE_FILENAME = "vendor_dynamic_ready_policy_recipe.json"
RECIPE_SENTINEL_POLICY_SHA256 = "0" * 64
OLD_SHARED_READY_POLICY_SHA256 = (
    "27bf405e5677fe2e7bab6fcc15c166901734048dd334b8b0abc3a8ffef3ce416"
)
VENDOR_RUNTIME_CONTRACT_SHA256 = (
    "38974f1bc5da8140aec24e07dd2d59d9b7cc90ed52acdd20f54564dd70368fba"
)
BUNDLE_PIN: Mapping[str, str] = _V.CANONICAL_BUNDLE_PIN

LAUNCHER_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "materialize_n1_vendor_dynamic_ready_recipe.py"
)
VENDOR_LAUNCHER_SOURCE = _V.LAUNCHER_SOURCE
IDENTITY_SAFETY_SOURCE = _I.LAUNCHER_SOURCE
TRAINING_CONTRACT_SOURCE = _I.TRAINING_CONTRACT_SOURCE
WBT_RELATIVE = _I.WBT_RELATIVE

_SPEC_KEYS = (
    "schema_version",
    "kind",
    "source",
    "action_id",
    "bundle",
    "vendor_runtime_training_contract_sha256",
    "stage",
    "seed",
    "num_envs",
    "max_iterations",
    "gpu",
    "namespace",
    "log_path",
)
_SOURCE_KEYS = ("checkout", "commit_sha", "isaac_python")
_PIN_KEYS = ("path", "sha256")

LaunchRefused = _S.LaunchRefused
canonical_sha256 = _S.canonical_sha256


def _configure_identity_launch_safety() -> None:
    """Point the proven lock/no-clobber launcher at this narrow wrapper."""

    _I.LAUNCHER_SOURCE = LAUNCHER_SOURCE
    _I.CLAIM_KIND = CLAIM_KIND
    _I.RESULT_KIND = RESULT_KIND


def _validate_spec_document(
    document: dict[str, Any], *, namespace_claimed: bool = False
) -> dict[str, Any]:
    row = _S._exact_dict(document, _SPEC_KEYS, name="vendor recipe spec")
    if row["schema_version"] != SCHEMA_VERSION or row["kind"] != SPEC_KIND:
        raise LaunchRefused(
            f"vendor recipe spec must be schema {SCHEMA_VERSION} / {SPEC_KIND!r}"
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

    if row["action_id"] != ACTION_ID:
        raise LaunchRefused("vendor dynamic-ready recipe action must be bh_loop_c")
    bundle = _S._exact_dict(row["bundle"], _PIN_KEYS, name="spec.bundle")
    if dict(bundle) != dict(BUNDLE_PIN):
        raise LaunchRefused("vendor dynamic-ready recipe requires the exact code pin")

    contract_sha = _S._sha256(
        row["vendor_runtime_training_contract_sha256"],
        name="vendor_runtime_training_contract_sha256",
    )
    if contract_sha == OLD_SHARED_READY_POLICY_SHA256:
        raise LaunchRefused(
            "old 27bf shared-ready policy is not a vendor runtime contract"
        )
    if contract_sha != VENDOR_RUNTIME_CONTRACT_SHA256:
        raise LaunchRefused(
            "vendor runtime contract differs from the fixed live contract"
        )
    if row["stage"] != STAGE:
        raise LaunchRefused("vendor dynamic-ready wrapper is recipe-only")
    if row["seed"] != SEED or type(row["seed"]) is not int:
        raise LaunchRefused("vendor dynamic-ready recipe seed must be exactly 0")
    if row["num_envs"] != NUM_ENVS or type(row["num_envs"]) is not int:
        raise LaunchRefused("vendor dynamic-ready recipe requires exactly one env")
    if (
        row["max_iterations"] != MAX_ITERATIONS
        or type(row["max_iterations"]) is not int
    ):
        raise LaunchRefused("vendor dynamic-ready recipe permits zero PPO iterations only")

    gpu = _S._validate_gpu(row["gpu"])
    namespace = _S._absolute_path(row["namespace"], name="namespace")
    if (
        namespace.name in ("", ".", "..")
        or _S.SAFE_COMPONENT_RE.fullmatch(namespace.name) is None
        or not namespace.name.startswith("a3vendor-dynamic-recipe-")
    ):
        raise LaunchRefused(
            "namespace basename must start with 'a3vendor-dynamic-recipe-'"
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
        "action_id": ACTION_ID,
        "bundle": dict(BUNDLE_PIN),
        "vendor_runtime_training_contract_sha256": contract_sha,
        "stage": STAGE,
        "seed": SEED,
        "num_envs": NUM_ENVS,
        "max_iterations": MAX_ITERATIONS,
        "gpu": gpu,
        "namespace": str(namespace),
        "log_path": str(log_path),
    }


def _validate_runtime_sources(
    checkout: Path, commit_sha: str
) -> dict[str, dict[str, Any]]:
    result = _V._validate_runtime_sources(checkout, commit_sha)
    for relative, label in (
        (LAUNCHER_SOURCE, "vendor dynamic-ready recipe wrapper"),
        (IDENTITY_SAFETY_SOURCE, "vendor identity lock/no-clobber safety"),
        (TRAINING_CONTRACT_SOURCE, "dynamic-ready training-contract loader"),
    ):
        pin = {"path": relative, "sha256": _S.sha256_file(checkout / relative)}
        normalized, _path = _S._verify_tracked_file(
            checkout, commit_sha, pin, name=label
        )
        result[label] = normalized
    if _THIS_FILE != checkout / LAUNCHER_SOURCE:
        raise LaunchRefused(
            "running vendor recipe wrapper is not the selected checkout path"
        )
    return result


def _load_training_contract_module(checkout: Path):
    module_path = checkout / TRAINING_CONTRACT_SOURCE
    spec = importlib.util.spec_from_file_location(
        "_hope_vendor_dynamic_recipe_training_contract", module_path
    )
    if spec is None or spec.loader is None:
        raise LaunchRefused("cannot load dynamic-ready training-contract validator")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise LaunchRefused(
            f"cannot import dynamic-ready training-contract validator: {exc}"
        ) from exc
    return module


def _validate_vendor_inputs(
    checkout: Path, commit_sha: str, spec: Mapping[str, Any]
) -> dict[str, Any]:
    bundle = _V._B._validate_bundle(
        checkout,
        commit_sha,
        spec["bundle"],
        expected_action=ACTION_ID,
        expected_scope=SCOPE,
        require_dynamic_ready=True,
    )
    identity = _V._validate_vendor_identity_manifest(checkout, commit_sha)
    authoritative_sha = identity["runtime_training_contract_sha256"]
    if authoritative_sha != spec["vendor_runtime_training_contract_sha256"]:
        raise LaunchRefused(
            "recipe spec runtime contract differs from tracked required identity"
        )
    actual_authority = _V._validate_actual_vendor_authority(
        checkout, commit_sha, bundle, authoritative_sha
    )
    runtime_binding = _V._validate_vendor_runtime_binding(
        checkout, commit_sha, bundle, authoritative_sha
    )
    authority_contract = actual_authority.get("runtime_training_contract")
    if (
        type(authority_contract) is not dict
        or authority_contract.get("sha256") != authoritative_sha
        or runtime_binding["runtime_training_contract_sha256"]
        != authoritative_sha
    ):
        raise LaunchRefused(
            "required identity, actual authority, and dynamic-ready contract differ"
        )

    dynamic_ready = bundle.get("dynamic_ready")
    if type(dynamic_ready) is not dict:
        raise LaunchRefused("vendor recipe requires one dynamic-ready bundle")
    training_contract = _load_training_contract_module(checkout)
    try:
        binding = training_contract.load_action_ball_dynamic_ready_runtime_binding(
            artifact_path=str(
                checkout / dynamic_ready["artifact"]["path"]
            ),
            artifact_sha256=dynamic_ready["artifact"]["sha256"],
            nominal_hold_receipt_path=str(
                checkout / dynamic_ready["nominal_hold_receipt"]["path"]
            ),
            nominal_hold_receipt_sha256=(
                dynamic_ready["nominal_hold_receipt"]["sha256"]
            ),
            action_order=[ACTION_ID],
            motion_paths=[str(checkout / bundle["motion"]["path"])],
        )
    except (OSError, TypeError, ValueError) as exc:
        raise LaunchRefused(
            f"dynamic-ready runtime binding refused exact bundle pins: {exc}"
        ) from exc
    if (
        binding.get("schema_version") != 2
        or binding.get("kind")
        != training_contract.ACTION_BALL_DYNAMIC_READY_RUNTIME_BINDING_KIND_V2
        or binding.get("action_order") != [ACTION_ID]
        or binding.get("motion_sha256_per_action")
        != [bundle["motion"]["sha256"]]
        or type(binding.get("binding_sha256")) is not str
    ):
        raise LaunchRefused("vendor recipe requires the exact schema-v2 binding")
    binding_sha = _S._sha256(
        binding["binding_sha256"], name="dynamic-ready binding SHA"
    )
    return {
        "bundle": bundle,
        "required_identity": identity,
        "actual_authority": actual_authority,
        "runtime_binding": runtime_binding,
        "dynamic_ready_binding": {
            "schema_version": binding["schema_version"],
            "kind": binding["kind"],
            "action_order": list(binding["action_order"]),
            "motion_sha256_per_action": list(
                binding["motion_sha256_per_action"]
            ),
            "binding_sha256": binding_sha,
        },
    }


def _training_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source": dict(spec["source"]),
        "action_id": ACTION_ID,
        "scope": SCOPE,
        "reward_profile": _V.REWARD_PROFILE,
        "seed": SEED,
        "num_envs": NUM_ENVS,
        "max_iterations": MAX_ITERATIONS,
        "save_interval": SAVE_INTERVAL,
        "namespace": spec["namespace"],
        "expected_effective_reward_recipe_sha256": (
            _I.EXPECTED_REWARD_RECIPE_SHA256
        ),
        "policy_contract_sha256": RECIPE_SENTINEL_POLICY_SHA256,
    }


def _build_training_argv(
    spec: Mapping[str, Any], vendor_inputs: Mapping[str, Any]
) -> list[str]:
    argv = _V._build_training_argv(
        _training_spec(spec), vendor_inputs["bundle"]
    )
    recipe_path = Path(spec["namespace"]) / RECIPE_FILENAME
    argv.append(f"action_ball_policy_recipe_output_path={recipe_path}")
    if f"max_iterations={MAX_ITERATIONS}" not in argv:
        raise LaunchRefused("vendor recipe argv lost the zero-iteration gate")
    if (
        f"num_envs={NUM_ENVS}" not in argv
        or f"seed={SEED}" not in argv
        or f"task={TASK_PROFILE_ID}" not in argv
        or "action_ball_dynamic_ready_bootstrap=true" not in argv
    ):
        raise LaunchRefused("vendor recipe argv lost its fixed identity")
    if any(OLD_SHARED_READY_POLICY_SHA256 in item for item in argv):
        raise LaunchRefused("old 27bf shared-ready policy entered recipe argv")
    if argv.count(_V.STABLE_READY_PLANT_OVERRIDE) != 1:
        raise LaunchRefused(
            "vendor recipe argv must carry exactly one stable-ready plant override"
        )
    forbidden = (
        "checkpoint_path=",
        "action_ball_shared_ready_bootstrap=true",
        "task.push",
        "push_robot",
        "randomize_pd_gains",
        "kp_gain_range",
        "kd_gain_range",
        "control_step_action_delay_",
    )
    if any(fragment in item for item in argv for fragment in forbidden):
        raise LaunchRefused(
            "vendor recipe argv overrides resume/shared-ready/task-owned physics"
        )
    return argv


def _output_contract(
    spec: Mapping[str, Any], vendor_inputs: Mapping[str, Any]
) -> dict[str, Any]:
    root = (
        Path(spec["source"]["checkout"])
        / WBT_RELATIVE
        / "logs/rsl_rl"
        / EXPERIMENT_NAME
    )
    return {
        "recipe": str(Path(spec["namespace"]) / RECIPE_FILENAME),
        "ppo_update_count": 0,
        "checkpoints": [],
        "training_contract": None,
        "policy_training_contract_sha256_source": (
            "recipe.policy_contract_sha256"
        ),
        "dynamic_ready_binding_sha256": vendor_inputs[
            "dynamic_ready_binding"
        ]["binding_sha256"],
        "rsl_experiment_root": str(root),
        "rsl_run_suffix": (
            f"_{Path(spec['namespace']).name}-DIAGNOSTIC_UNAUTHORIZED"
        ),
    }


def _check_rsl_namespace_available(output_contract: Mapping[str, Any]) -> None:
    root = Path(output_contract["rsl_experiment_root"])
    if not os.path.lexists(root):
        return
    if root.resolve(strict=True) != root or not root.is_dir():
        raise LaunchRefused("vendor recipe RSL experiment root is not real")
    spent = sorted(
        child.name
        for child in root.iterdir()
        if child.name.endswith(output_contract["rsl_run_suffix"])
    )
    if spent:
        raise LaunchRefused(f"vendor recipe trainer run_name is spent: {spent[0]}")


def _validate_materialized_recipe(
    recipe_path: Path, *, expected_binding_sha256: str
) -> dict[str, Any]:
    _S._stable_regular_file(recipe_path, name="materialized vendor policy recipe")
    raw = recipe_path.read_bytes()
    document = _S._strict_json_bytes(raw, name="materialized vendor policy recipe")
    if raw != _S._canonical_bytes(document) + b"\n":
        raise LaunchRefused("materialized vendor policy recipe is not canonical")
    row = _S._exact_dict(
        document,
        (
            "schema_version",
            "kind",
            "action_count",
            "action_order",
            "policy_contract_sha256",
            "action_ball_ppo_runner_recipe",
            "policy_bootstrap",
        ),
        name="materialized vendor policy recipe",
    )
    policy_sha = _S._sha256(
        row["policy_contract_sha256"], name="materialized policy contract SHA"
    )
    runner_recipe = row["action_ball_ppo_runner_recipe"]
    bootstrap = row["policy_bootstrap"]
    ready_source = bootstrap.get("ready_source") if type(bootstrap) is dict else None
    identity = ready_source.get("identity") if type(ready_source) is dict else None
    if (
        row["schema_version"] != 1
        or row["kind"]
        != "action_ball_shared_ready_policy_recipe_materialization_v1"
        or row["action_count"] != 1
        or row["action_order"] != [ACTION_ID]
        or policy_sha == OLD_SHARED_READY_POLICY_SHA256
        or type(runner_recipe) is not dict
        or runner_recipe.get("sha256") != policy_sha
        or type(runner_recipe.get("recipe")) is not dict
        or runner_recipe["recipe"].get("policy_initialization") != bootstrap
        or type(bootstrap) is not dict
        or bootstrap.get("schema_version") != 2
        or bootstrap.get("action_count") != 1
        or bootstrap.get("action_order") != [ACTION_ID]
        or type(identity) is not dict
        or identity.get("binding_sha256") != expected_binding_sha256
    ):
        raise LaunchRefused(
            "materialized policy recipe is not the exact vendor dynamic-ready contract"
        )
    return {
        "path": str(recipe_path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "policy_contract_sha256": policy_sha,
        "policy_training_contract_sha256": policy_sha,
        "dynamic_ready_binding_sha256": expected_binding_sha256,
        "diagnostic_unauthorized": True,
        "launch_authorized": False,
        "export_authorized": False,
        "judge_authorized": False,
        "hardware_authorized": False,
    }


def build_plan(spec_path: Path) -> dict[str, Any]:
    absolute = _S._absolute_path(str(spec_path), name="--spec", must_exist=True)
    _S._stable_regular_file(absolute, name="vendor recipe spec")
    raw = absolute.read_bytes()
    document = _S._strict_json_bytes(raw, name="vendor recipe spec")
    if raw != _S._canonical_bytes(document) + b"\n":
        raise LaunchRefused("vendor recipe spec must be canonical JSON plus newline")
    spec = _validate_spec_document(document)
    checkout = Path(spec["source"]["checkout"])
    commit_sha = spec["source"]["commit_sha"]
    source = _S._verify_clean_source(checkout, commit_sha)
    runtime_sources = _validate_runtime_sources(checkout, commit_sha)
    runtime_assets = _S._validate_runtime_asset_environment()
    vendor_inputs = _validate_vendor_inputs(checkout, commit_sha, spec)
    output_contract = _output_contract(spec, vendor_inputs)
    _check_rsl_namespace_available(output_contract)
    argv = _build_training_argv(spec, vendor_inputs)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": CLAIM_KIND,
        "diagnostic_unauthorized": True,
        "recipe_materialization_only": True,
        "ppo_updates_authorized": 0,
        "resume_prohibited": True,
        "launch_prohibited": True,
        "formal_evidence_prohibited": True,
        "curriculum_promotion_prohibited": True,
        "export_prohibited": True,
        "judge_prohibited": True,
        "deployment_prohibited": True,
        "hardware_authority_prohibited": True,
        "spec_file_sha256": hashlib.sha256(raw).hexdigest(),
        "spec": spec,
        "source": source,
        "runtime_sources": runtime_sources,
        "runtime_assets": runtime_assets,
        "vendor_inputs": vendor_inputs,
        "output_contract": output_contract,
        "boot_marker": "ACTION_BALL_POLICY_RECIPE_MATERIALIZED",
        "training_argv": argv,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": CLAIM_KIND,
        "launch_claim_sha256": canonical_sha256(payload),
        "canonical_payload": payload,
    }


def _internal_exec(claim_path: Path, expected_sha256: str, lock_fd: int) -> int:
    _plan, payload = _I._load_internal_claim(claim_path, expected_sha256)
    spec = _validate_spec_document(payload["spec"], namespace_claimed=True)
    checkout = Path(spec["source"]["checkout"])
    commit_sha = spec["source"]["commit_sha"]
    source = _S._verify_clean_source(checkout, commit_sha)
    if source != payload["source"]:
        raise LaunchRefused("source identity drifted after namespace claim")
    runtime_sources = _validate_runtime_sources(checkout, commit_sha)
    if runtime_sources != payload["runtime_sources"]:
        raise LaunchRefused("runtime source identity drifted after namespace claim")
    runtime_assets = _S._validate_runtime_asset_claim(payload["runtime_assets"])
    vendor_inputs = _validate_vendor_inputs(checkout, commit_sha, spec)
    if vendor_inputs != payload["vendor_inputs"]:
        raise LaunchRefused("vendor authority/bundle inputs drifted after claim")
    argv = _build_training_argv(spec, vendor_inputs)
    if argv != payload["training_argv"]:
        raise LaunchRefused("training argv differs from immutable claim")
    if _output_contract(spec, vendor_inputs) != payload["output_contract"]:
        raise LaunchRefused("recipe output contract differs from immutable claim")

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
            "kind": "n1_vendor_dynamic_ready_recipe_gpu_admission_v1",
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
    result = _I.launch(plan, confirm_claim=confirm_claim)
    payload = plan["canonical_payload"]
    materialized = _validate_materialized_recipe(
        Path(payload["output_contract"]["recipe"]),
        expected_binding_sha256=payload["output_contract"][
            "dynamic_ready_binding_sha256"
        ],
    )
    result["materialized_policy_recipe"] = materialized
    result["policy_training_contract_sha256"] = materialized[
        "policy_training_contract_sha256"
    ]
    result["launch_authorized"] = False
    result["export_authorized"] = False
    result["judge_authorized"] = False
    result["hardware_authorized"] = False
    return result


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
        "action_id": ACTION_ID,
        "bundle": dict(BUNDLE_PIN),
        "vendor_runtime_training_contract_sha256": (
            VENDOR_RUNTIME_CONTRACT_SHA256
        ),
        "stage": STAGE,
        "seed": SEED,
        "num_envs": NUM_ENVS,
        "max_iterations": MAX_ITERATIONS,
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
    template = sub.add_parser("template", help="print one canonical Pod spec")
    template.add_argument("--checkout", required=True)
    template.add_argument("--commit-sha", required=True)
    template.add_argument("--isaac-python", required=True)
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


_configure_identity_launch_safety()


if __name__ == "__main__":
    raise SystemExit(main())
