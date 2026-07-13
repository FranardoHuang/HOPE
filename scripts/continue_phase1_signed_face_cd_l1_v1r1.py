#!/usr/bin/env python3
"""One-shot C2-evidence consumer and D2-only continuation for signed-face L1.

The frozen v1 launcher successfully started and completed C2, but its outer
post-boot verifier compared the emitted float mount signs with integer
expectations and therefore did not write ``runtime_verified.json``.  This
consumer never launches or retries C2.  It first replays the immutable v1 C2
claim, log, hard contract and terminal checkpoint, then permits exactly one
new D2 claim in the original unclaimed namespace.

There is no activation, judge, L2, second-seed, retry, signal, or robot path.
The reviewed Kit wrapper remains the only component that may manage the exact
PGID it creates before the hard-contract marker.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
from typing import Any


HERE = Path(__file__).absolute()


def _early_no_symlink_components(path: Path, label: str) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise RuntimeError(f"{label} contains a symlink component: {current}")


_early_no_symlink_components(HERE, "v1r1 launcher path")
if HERE.parent.name != "scripts":
    raise RuntimeError("v1r1 launcher must be installed under control/v1r1/scripts/")
ROOT = HERE.parent.parent
V1_SCRIPT = ROOT / "scripts/run_phase1_signed_face_cd_l1.py"
_early_no_symlink_components(V1_SCRIPT, "v1 helper path")
_SPEC = importlib.util.spec_from_file_location("phase1_signed_face_cd_l1_v1", V1_SCRIPT)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - import failure is fatal
    raise RuntimeError(f"cannot load frozen v1 helper: {V1_SCRIPT}")
v1 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(v1)

ContractError = v1.ContractError
CONTINUATION_ID = "phase1-signed-face-c2-d2-l1-v1r1-d2-only-20260714"
CONTINUATION_MANIFEST_SHA256 = (
    "f31fcf7bf500dde26a347af15feacececda1b5e1fd870c74759aca7d60c5def8"
)
V1_MANIFEST_SHA256 = "785ad96dd53e1809ddcf86d1ecd80572b02e3c96ffd6d6599cab20a73b559895"
V1_LAUNCHER_SHA256 = "0fa250207246e8bf69b6475125882b45e817f9e777d13039614c82dad9a803ba"
ROOT_CONFIRMATION = "ROOT_APPROVES_SIM_ONLY_SIGNED_FACE_D2_ONLY_V1R1"
CELL_ID = "D2"


def require_safe_relative_path(value: Any, label: str) -> Path:
    if type(value) is not str or not value:
        raise ContractError(f"{label} must be a non-empty relative path string")
    relative = Path(value)
    if relative.is_absolute() or not relative.parts or any(
        part in ("", ".", "..") for part in relative.parts
    ):
        raise ContractError(f"{label} must be relative and contain no dot traversal")
    return relative


def safe_control_path(root: Path, value: Any, label: str) -> Path:
    relative = require_safe_relative_path(value, label)
    target = root / relative
    v1.no_symlink_existing_components(target, label)
    return target


def continuation_manifest_path(root: Path = ROOT) -> Path:
    return root / "configs/phase1_signed_face_cd_l1_v1r1_continuation_20260714.json"


def read_manifest(path: Path) -> dict[str, Any]:
    v1.require_regular(path, "v1r1 continuation manifest")
    if v1.sha256_file(path) != CONTINUATION_MANIFEST_SHA256:
        raise ContractError("v1r1 continuation manifest bytes changed")
    value = v1.read_json_object(path, "v1r1 continuation manifest")
    validate_manifest(value)
    return value


def validate_manifest(manifest: dict[str, Any]) -> None:
    expected_top = {
        "schema_version": 1,
        "continuation_id": CONTINUATION_ID,
        "status": "machine_preregistered_one_shot_continuation_root_launch_switch_required",
        "human_owner": "Franco",
        "executor": "Codex",
        "simulation_only": True,
        "real_robot_commands_forbidden": True,
        "automatic_retry_forbidden": True,
        "c2_relaunch_forbidden": True,
        "activation_authorized": False,
        "judge_authorized": False,
        "l2_authorized": False,
        "second_seed_authorized": False,
    }
    for key, expected in expected_top.items():
        v1.require_exact(manifest.get(key), expected, f"v1r1 manifest {key}")
    original = manifest.get("original_v1_control")
    control = manifest.get("continuation_control")
    c2 = manifest.get("preserved_c2")
    d2 = manifest.get("d2_only_continuation")
    outputs = manifest.get("outputs")
    boundary = manifest.get("decision_boundary")
    if not all(isinstance(value, dict) for value in (original, control, c2, d2, outputs, boundary)):
        raise ContractError("v1r1 control/evidence/output sections must be objects")
    v1.require_exact(original, {
        "root": "/workspace/codexschema/phase1_signed_face_cd_l1_20260714/control/v1",
        "manifest_relative_path": "phase1_signed_face_cd_l1_prereg_20260714.json",
        "manifest_sha256": V1_MANIFEST_SHA256,
        "launcher_relative_path": "run_phase1_signed_face_cd_l1.py",
        "launcher_sha256": V1_LAUNCHER_SHA256,
    }, "original v1 control")
    v1.require_exact(control, {
        "root": "/workspace/codexschema/phase1_signed_face_cd_l1_20260714/control/v1r1",
        "manifest_relative_path": "configs/phase1_signed_face_cd_l1_v1r1_continuation_20260714.json",
        "launcher_relative_path": "scripts/continue_phase1_signed_face_cd_l1_v1r1.py",
        "v1_helper_relative_path": "scripts/run_phase1_signed_face_cd_l1.py",
        "v1_helper_sha256": V1_LAUNCHER_SHA256,
        "v1_manifest_relative_path": "configs/phase1_signed_face_cd_l1_prereg_20260714.json",
        "v1_manifest_sha256": V1_MANIFEST_SHA256,
        "root_launch_confirmation": ROOT_CONFIRMATION,
    }, "v1r1 mini-tree control")
    for key in ("manifest_relative_path", "launcher_relative_path"):
        require_safe_relative_path(original[key], f"original v1 {key}")
    for key in (
        "manifest_relative_path", "launcher_relative_path",
        "v1_helper_relative_path", "v1_manifest_relative_path",
    ):
        require_safe_relative_path(control[key], f"v1r1 {key}")
    v1.require_exact(c2.get("cell_id"), "C2", "preserved cell")
    v1.require_exact(c2.get("run_name"), "phase1_signed_face_l1_c2d2_v1_C2_fresh_control_seed3", "C2 run")
    v1.require_exact(c2.get("pid_equals_pgid"), 1820092, "C2 pid=pgid")
    require_exact_float_mount_signs(c2.get("hard_contract_mount_normal_sign_per_clip"))
    for key in (
        "launch_contract_sha256", "launch_state_sha256", "training_launch_claim_sha256",
        "training_log_sha256", "hard_contract_sha256", "terminal_checkpoint_sha256",
    ):
        v1.require_sha(c2.get(key), f"preserved C2 {key}")
    for key in (
        "old_runtime_verified_must_be_absent", "old_launch_failure_must_be_absent",
        "old_terminal_result_must_be_absent",
    ):
        v1.require_exact(c2.get(key), True, f"preserved C2 {key}")
    v1.require_exact(c2.get("training_recipe_changed"), False, "C2 recipe boundary")
    v1.require_exact(d2, {
        "cell_id": "D2",
        "run_name": "phase1_signed_face_l1_c2d2_v1_D2_fresh_guidance_seed3",
        "arm_path": "/workspace/codexschema/phase1_signed_face_cd_l1_20260714/runs/l1/phase1_signed_face_l1_c2d2_v1_D2_fresh_guidance_seed3",
        "physical_gpu": 2,
        "face_guidance_weight": -0.4,
        "arm_and_exact_training_run_must_be_absent_before_claim": True,
        "one_new_atomic_claim": True,
        "same_v1_training_source_recipe_and_runtime": True,
    }, "D2-only continuation")
    v1.require_exact(outputs, {
        "continuation_evidence_root": "/workspace/codexschema/phase1_signed_face_cd_l1_20260714/continuations/v1r1",
        "c2_attestation_path": "/workspace/codexschema/phase1_signed_face_cd_l1_20260714/continuations/v1r1/c2_terminal_attestation.json",
        "d2_runtime_verified_basename": "runtime_verified_v1r1.json",
        "d2_launch_failure_basename": "launch_failure_v1r1.json",
        "d2_terminal_result_basename": "terminal_result_v1r1.json",
        "paired_result_path": "/workspace/codexschema/phase1_signed_face_cd_l1_20260714/runs/l1/paired_l1_result_v1r1.json",
    }, "v1r1 output paths")
    v1.require_exact(boundary, {
        "c2_relaunch": False, "d2_automatic_retry": False,
        "activation": False, "judge": False, "l2": False,
        "second_seed": False, "stop_or_promote": False,
        "real_robot_commands": False,
    }, "v1r1 decision boundary")


def require_exact_float_mount_signs(value: Any) -> list[float]:
    """Accept only the emitted contract's exact float representation.

    ``bool`` is an ``int`` subclass and ``1 == 1.0`` in Python.  The explicit
    ``type(...) is float`` checks are therefore part of the contract.
    """

    if type(value) is not list or len(value) != 2:
        raise ContractError("mount_normal_sign_per_clip must be a two-float list")
    if any(type(item) is not float or not math.isfinite(item) for item in value):
        raise ContractError("mount_normal_sign_per_clip must contain exact floats, never bool/int")
    if value != [1.0, -1.0]:
        raise ContractError("mount_normal_sign_per_clip must be exactly [1.0,-1.0]")
    return value


def verify_hard_contract(
    path: Path, original_manifest: dict[str, Any], cell_id: str,
) -> tuple[str, dict[str, Any]]:
    """Verify the v1 training contract with the corrected float wire type."""

    v1.require_regular(path, "adjacent hard contract")
    contract = v1.read_json_object(path, "adjacent hard contract")
    require_exact_float_mount_signs(contract.get("mount_normal_sign_per_clip"))
    expected = {
        "schema_version": 3,
        "actor_obs_contract": "deploy_parity_face179",
        "actor_obs_total_dim": 179,
        "face_command_pairing": "shared_plus_y",
        "strike_phase_per_clip": [0.471, 0.338],
        "motion_kinematics_exact": True,
        "motion_allow_legacy_link_origin_velocity": False,
        "motion_event_timing": {"mode": "disabled"},
        "racket_guidance_reward": {
            "position": {"weight": 0.0, "command_name": "racket_target", "d_max": 0.5},
            "signed_face": {
                "weight": v1.cells(original_manifest)[cell_id]["face_guidance_weight"],
                "command_name": "racket_target",
                "theta_max": math.pi,
            },
        },
    }
    for key, wanted in expected.items():
        v1.require_exact(contract.get(key), wanted, f"hard contract {key}")
    if len(contract.get("joint_names", [])) != 31 or len(contract.get("action_joint_ids", [])) != 31:
        raise ContractError("hard contract does not bind 31 joints/actions")
    friction = contract.get("joint_friction_coefficients")
    if (
        not isinstance(friction, list) or len(friction) != 31
        or any(type(item) not in (int, float) or float(item) != 0.0 for item in friction)
    ):
        raise ContractError("hard contract is not 31/31 zero-friction")
    clips = contract.get("motion_clips")
    expected_clips = [
        original_manifest["inputs"]["forehand_motion"]["sha256"],
        original_manifest["inputs"]["backhand_motion"]["sha256"],
    ]
    if (
        not isinstance(clips, list)
        or [item.get("sha256") for item in clips if isinstance(item, dict)] != expected_clips
    ):
        raise ContractError("hard contract motion order/SHA changed")
    bank = contract.get("question_bank")
    expected_bank = original_manifest["inputs"]["schema3_train_bank"]
    if not isinstance(bank, dict):
        raise ContractError("hard contract lacks train bank")
    for key in ("sha256", "physics_contract_sha256", "source_family_sha256", "schema_version", "split"):
        v1.require_exact(bank.get(key), expected_bank[key], f"hard contract train bank {key}")
    v1.require_exact(bank.get("exact"), True, "hard contract train bank exactness")
    return v1.sha256_file(path), contract


def original_control_paths(manifest: dict[str, Any]) -> tuple[Path, Path]:
    item = manifest["original_v1_control"]
    root = Path(item["root"])
    return (
        safe_control_path(root, item["manifest_relative_path"], "original v1 manifest"),
        safe_control_path(root, item["launcher_relative_path"], "original v1 launcher"),
    )


def checked_in_original_paths() -> tuple[Path, Path]:
    return ROOT / "configs/phase1_signed_face_cd_l1_prereg_20260714.json", V1_SCRIPT


def verify_checked_in_original() -> dict[str, Any]:
    manifest_path, launcher_path = checked_in_original_paths()
    v1.require_regular(manifest_path, "mini-tree v1 manifest")
    v1.require_regular(launcher_path, "mini-tree v1 helper")
    if v1.sha256_file(manifest_path) != V1_MANIFEST_SHA256:
        raise ContractError("checked-in v1 manifest bytes changed")
    if v1.sha256_file(launcher_path) != V1_LAUNCHER_SHA256:
        raise ContractError("checked-in v1 launcher bytes changed")
    original = v1.load_manifest(manifest_path)
    return {"manifest": original, "source": v1.verify_static_source(original)}


def verify_original_for_plan(manifest: dict[str, Any]) -> dict[str, Any]:
    """Validate the same four-file mini-tree locally and on the Pod."""

    del manifest
    return verify_checked_in_original()


def load_original_runtime(manifest: dict[str, Any]) -> tuple[dict[str, Any], Path, Path, dict[str, Any]]:
    manifest_path, launcher_path = original_control_paths(manifest)
    if v1.sha256_file(manifest_path) != V1_MANIFEST_SHA256:
        raise ContractError("runtime v1 manifest bytes changed")
    if v1.sha256_file(launcher_path) != V1_LAUNCHER_SHA256:
        raise ContractError("runtime v1 launcher bytes changed")
    original = v1.load_manifest(manifest_path)
    receipt = v1.verify_external_control_location(original, manifest_path, launcher_path)
    return original, manifest_path, launcher_path, receipt


def continuation_control_receipt(
    manifest: dict[str, Any], manifest_path: Path, launcher_path: Path,
) -> dict[str, Any]:
    item = manifest["continuation_control"]
    root = Path(item["root"])
    v1.require_directory(root, "v1r1 external control root")
    expected_manifest = safe_control_path(
        root, item["manifest_relative_path"], "v1r1 manifest"
    )
    expected_launcher = safe_control_path(
        root, item["launcher_relative_path"], "v1r1 launcher"
    )
    helper_path = safe_control_path(
        root, item["v1_helper_relative_path"], "v1r1 helper"
    )
    v1_manifest_path = safe_control_path(
        root, item["v1_manifest_relative_path"], "v1r1 v1 manifest"
    )
    if (
        ROOT != root or V1_SCRIPT != helper_path
        or manifest_path != expected_manifest or launcher_path != expected_launcher
    ):
        raise ContractError("runtime modes require exact v1r1 external control paths")
    result = {}
    for label, path in (
        ("manifest", manifest_path), ("launcher", launcher_path),
        ("v1_helper", helper_path), ("v1_manifest", v1_manifest_path),
    ):
        info = v1.require_regular(path, f"v1r1 external control {label}")
        if info.st_mode & 0o222:
            raise ContractError(f"v1r1 external control {label} must be read-only")
        result[label] = {
            "path": str(path), "sha256": v1.sha256_file(path),
            "device": info.st_dev, "inode": info.st_ino,
        }
    v1.require_exact(result["manifest"]["sha256"], CONTINUATION_MANIFEST_SHA256, "runtime v1r1 manifest")
    v1.require_exact(result["v1_helper"]["sha256"], V1_LAUNCHER_SHA256, "runtime v1 helper")
    v1.require_exact(result["v1_manifest"]["sha256"], V1_MANIFEST_SHA256, "runtime v1 manifest copy")
    return result


def c2_attestation_path(manifest: dict[str, Any], original: dict[str, Any]) -> Path:
    del original
    path = Path(manifest["outputs"]["c2_attestation_path"])
    if path.parent != Path(manifest["outputs"]["continuation_evidence_root"]):
        raise ContractError("C2 attestation escaped the independent continuation evidence root")
    return path


def _stable_checkpoint(path: Path, delay: float) -> os.stat_result:
    before = v1.require_regular(path, "terminal checkpoint")
    if before.st_size <= 0:
        raise ContractError("terminal checkpoint is empty")
    if delay:
        time.sleep(delay)
    after = v1.require_regular(path, "stable terminal checkpoint")
    keys = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink")
    if any(getattr(before, key) != getattr(after, key) for key in keys):
        raise ContractError("terminal checkpoint is still changing")
    return after


def audit_preserved_c2(
    manifest: dict[str, Any], *, stable_delay: float = 1.0,
    require_current_c2_gpu_empty: bool = True,
) -> dict[str, Any]:
    original, original_manifest_path, original_launcher_path, original_control = load_original_runtime(manifest)
    expected = manifest["preserved_c2"]
    paths = v1.expected_arm_paths(original, "C2")
    v1.require_exact(str(paths["arm"]), expected["arm_path"], "C2 arm path")
    v1.require_directory(paths["arm"], "preserved C2 arm")
    for key in ("runtime", "failure", "result"):
        if paths[key].exists():
            raise ContractError(f"preserved C2 {key} must remain absent; continuation assumption changed")
    if v1.sha256_file(paths["launch"]) != expected["launch_contract_sha256"]:
        raise ContractError("preserved C2 launch contract bytes changed")
    if v1.sha256_file(paths["state"]) != expected["launch_state_sha256"]:
        raise ContractError("preserved C2 launch state bytes changed")
    launch = v1.read_json_object(paths["launch"], "preserved C2 launch contract")
    claim, claim_sha = v1.reconstruct_claim(
        original, V1_MANIFEST_SHA256, V1_LAUNCHER_SHA256, "C2", launch
    )
    v1.require_exact(claim_sha, expected["training_launch_claim_sha256"], "C2 canonical claim SHA")
    # Rebuilding the command proves that the accepted trainer recipe is still the
    # original v1 C2 recipe.  The continuation never rewrites this launch record.
    v1.require_exact(launch.get("command"), v1.build_command(original, "C2", claim_sha), "C2 command")
    v1.require_exact(launch.get("external_control"), original_control, "C2 original control")
    v1.require_exact(launch.get("training_source"), v1.expected_source_identity(original), "C2 source")
    v1.require_exact(launch.get("optimization_recipe"), v1.optimization_recipe(original, "C2"), "C2 recipe")
    arm_info = v1.require_directory(paths["arm"], "preserved C2 atomic claim")
    v1.require_exact(v1.identity(arm_info), {
        "device": claim["claim_directory"]["st_dev"],
        "inode": claim["claim_directory"]["st_ino"],
    }, "C2 atomic claim directory")
    state = v1.parse_state(paths["state"])
    v1.require_exact(state.get("pid"), str(expected["pid_equals_pgid"]), "C2 state pid")
    v1.require_exact(state.get("pgid"), state.get("pid"), "C2 state pgid")
    if v1.process_starttime(expected["pid_equals_pgid"]) != -1:
        raise ContractError("preserved C2 pid is present/reused; natural-exit identity is ambiguous")

    preflight = v1.verify_runtime(
        original, "C2", require_empty_gpu=require_current_c2_gpu_empty
    )
    run_dir = Path(expected["training_run_dir"])
    v1.require_directory(run_dir, "preserved C2 training run")
    logs_root = preflight["wbt"] / "logs/rsl_rl/agibot_a3_hope_virtualball"
    if run_dir.parent != logs_root or not run_dir.name.endswith(f"_{expected['run_name']}"):
        raise ContractError("preserved C2 training run path changed")
    contract_path = run_dir / "params/training_contract.json"
    hard_sha, hard_contract = verify_hard_contract(contract_path, original, "C2")
    v1.require_exact(hard_sha, expected["hard_contract_sha256"], "C2 hard-contract SHA")
    log_info = v1.require_regular(paths["log"], "preserved C2 training log")
    v1.require_exact(v1.sha256_file(paths["log"]), expected["training_log_sha256"], "C2 training log SHA")
    log_text = paths["log"].read_text(encoding="utf-8", errors="replace")
    if v1.FAILURE_RE.search(log_text):
        raise ContractError("preserved C2 log contains a hard failure signature")
    if log_text.count(v1.THREAD_MARKER) != 1 or v1.HARD_CONTRACT_MARKER not in log_text:
        raise ContractError("preserved C2 log lacks exact Kit/hard-contract markers")
    checkpoint = run_dir / expected["terminal_checkpoint_basename"]
    checkpoint_info = _stable_checkpoint(checkpoint, stable_delay)
    checkpoint_sha = v1.sha256_file(checkpoint)
    v1.require_exact(checkpoint_sha, expected["terminal_checkpoint_sha256"], "C2 checkpoint SHA")
    checkpoint_audit = v1.checkpoint_audit(preflight["python"], checkpoint)
    for key, wanted in {
        "iter": 24,
        "training_contract_schema_version": 3,
        "training_contract_sha256": hard_sha,
        "training_contract_lineage_exact": 1,
        "training_launch_claim_sha256": claim_sha,
        "nonfinite_floating_elements": 0,
    }.items():
        v1.require_exact(checkpoint_audit.get(key), wanted, f"C2 checkpoint {key}")
    return {
        "artifact_kind": "phase1_signed_face_c2_v1r1_preserved_terminal_attestation",
        "schema_version": 1,
        "continuation_id": CONTINUATION_ID,
        "continuation_manifest_sha256": CONTINUATION_MANIFEST_SHA256,
        "original_v1_manifest_sha256": V1_MANIFEST_SHA256,
        "original_v1_launcher_sha256": V1_LAUNCHER_SHA256,
        "cell_id": "C2",
        "run_name": expected["run_name"],
        "training_source": v1.expected_source_identity(original),
        "optimization_recipe": v1.optimization_recipe(original, "C2"),
        "training_launch_claim": claim,
        "training_launch_claim_sha256": claim_sha,
        "launch_contract_sha256": expected["launch_contract_sha256"],
        "launch_state_sha256": expected["launch_state_sha256"],
        "training_run_dir": str(run_dir),
        "hard_contract_path": str(contract_path),
        "hard_contract_sha256": hard_sha,
        "hard_contract": hard_contract,
        "terminal_checkpoint_path": str(checkpoint),
        "terminal_checkpoint_sha256": checkpoint_sha,
        "terminal_checkpoint_file_identity": v1.identity(checkpoint_info),
        "checkpoint_audit": checkpoint_audit,
        "training_log_sha256": expected["training_log_sha256"],
        "training_log_file_identity": v1.identity(log_info),
        "old_runtime_verified_absent": True,
        "old_launch_failure_absent": True,
        "old_terminal_result_absent": True,
        "exact_trainer_pid_absent": True,
        "assigned_gpu_empty_terminal_barrier_observed": True,
        "outer_false_rejection_only": expected["outer_false_rejection"],
        "training_recipe_changed": False,
        "c2_relaunch_authorized": False,
        "activation": False,
        "judge": False,
        "l2": False,
        "second_seed": False,
        "real_robot_commands_executed": False,
    }


def write_c2_attestation(
    manifest: dict[str, Any], manifest_path: Path, launcher_path: Path,
) -> dict[str, Any]:
    continuation_control_receipt(manifest, manifest_path, launcher_path)
    original, _, _, _ = load_original_runtime(manifest)
    output = c2_attestation_path(manifest, original)
    if output.exists():
        raise ContractError("C2 v1r1 attestation already exists; no-clobber replay only")
    value = audit_preserved_c2(manifest, require_current_c2_gpu_empty=True)
    evidence_root = Path(manifest["outputs"]["continuation_evidence_root"])
    v1.no_symlink_existing_components(evidence_root, "v1r1 continuation evidence root")
    evidence_root.mkdir(parents=True, exist_ok=False)
    v1.require_directory(evidence_root, "v1r1 continuation evidence root")
    try:
        v1.write_json_exclusive(output, value)
    except BaseException:
        try:
            evidence_root.rmdir()
        except OSError:
            pass
        raise
    return {
        "status": "preserved_c2_terminal_attested_without_relaunch",
        "attestation_path": str(output),
        "attestation_sha256": v1.sha256_file(output),
        "c2_relaunch": False,
        "next_action": "launch-d2 may claim only the still-absent D2 arm",
    }


def load_c2_attestation(manifest: dict[str, Any]) -> tuple[dict[str, Any], Path, str]:
    original, _, _, _ = load_original_runtime(manifest)
    path = c2_attestation_path(manifest, original)
    recorded = v1.read_json_object(path, "C2 v1r1 terminal attestation")
    expected = audit_preserved_c2(
        manifest, stable_delay=0.0, require_current_c2_gpu_empty=False
    )
    v1.require_exact(recorded, expected, "C2 v1r1 terminal attestation replay")
    return recorded, path, v1.sha256_file(path)


def d2_paths(manifest: dict[str, Any], original: dict[str, Any]) -> dict[str, Path]:
    base = v1.expected_arm_paths(original, CELL_ID)
    return {
        **base,
        "runtime": base["arm"] / manifest["outputs"]["d2_runtime_verified_basename"],
        "failure": base["arm"] / manifest["outputs"]["d2_launch_failure_basename"],
        "result": base["arm"] / manifest["outputs"]["d2_terminal_result_basename"],
    }


def build_d2_claim(
    manifest: dict[str, Any], original: dict[str, Any], *,
    launcher_sha: str, arm_identity: dict[str, int], c2_attestation_sha: str,
) -> dict[str, Any]:
    v1.require_sha(launcher_sha, "v1r1 launcher SHA")
    v1.require_sha(c2_attestation_sha, "C2 attestation SHA")
    if set(arm_identity) != {"device", "inode"} or any(
        type(value) is not int or value <= 0 for value in arm_identity.values()
    ):
        raise ContractError("D2 atomic claim directory identity is invalid")
    arm = d2_paths(manifest, original)["arm"]
    return {
        "schema_version": 1,
        "continuation_id": CONTINUATION_ID,
        "continuation_manifest_sha256": CONTINUATION_MANIFEST_SHA256,
        "continuation_launcher_sha256": launcher_sha,
        "original_v1_manifest_sha256": V1_MANIFEST_SHA256,
        "original_v1_launcher_sha256": V1_LAUNCHER_SHA256,
        "c2_terminal_attestation_sha256": c2_attestation_sha,
        "training_source": v1.expected_source_identity(original),
        "stage": "l1",
        "cell_id": CELL_ID,
        "run_name": v1.cells(original)[CELL_ID]["run_name"],
        "optimization_recipe": v1.optimization_recipe(original, CELL_ID),
        "execution_lane": {
            "host": original["runtime"]["pod"],
            "physical_gpu": 2,
            "cuda_visible_devices": "2",
            "local_training_device": "cuda:0",
            "training_environment_sha256": v1.TRAINING_ENV_SHA256_BY_CELL[CELL_ID],
        },
        "expected_terminal_checkpoint_iteration": 24,
        "claim_directory": {
            "path": str(arm),
            "st_dev": arm_identity["device"],
            "st_ino": arm_identity["inode"],
        },
        "c2_relaunch_authorized": False,
        "automatic_retry_authorized": False,
    }


def reconstruct_d2_claim(
    manifest: dict[str, Any], original: dict[str, Any], launch: dict[str, Any],
    launcher_sha: str, c2_attestation_sha: str,
) -> tuple[dict[str, Any], str]:
    claim = launch.get("training_launch_claim")
    digest = v1.require_sha(launch.get("training_launch_claim_sha256"), "D2 launch claim")
    if not isinstance(claim, dict) or v1.canonical_sha256(claim) != digest:
        raise ContractError("D2 recorded claim bytes/digest changed")
    directory = claim.get("claim_directory")
    if not isinstance(directory, dict):
        raise ContractError("D2 claim lacks atomic directory identity")
    expected = build_d2_claim(
        manifest, original, launcher_sha=launcher_sha,
        arm_identity={"device": directory.get("st_dev"), "inode": directory.get("st_ino")},
        c2_attestation_sha=c2_attestation_sha,
    )
    v1.require_exact(claim, expected, "reconstructed D2 v1r1 claim")
    return claim, digest


def _d2_training_run_absent(original: dict[str, Any]) -> None:
    wbt = Path(original["source"]["training_checkout"]) / original["source"]["wbt_relative_path"]
    logs = wbt / "logs/rsl_rl/agibot_a3_hope_virtualball"
    run_name = v1.cells(original)[CELL_ID]["run_name"]
    if logs.is_dir() and list(logs.glob(f"*_{run_name}")):
        raise ContractError("D2 exact training run name already exists; no retry/reuse")


def launch_d2(
    manifest: dict[str, Any], manifest_path: Path, launcher_path: Path,
    root_confirmation: str | None,
) -> dict[str, Any]:
    if os.geteuid() != 0:
        raise ContractError("launch-d2 requires root on the simulator Pod")
    if root_confirmation != ROOT_CONFIRMATION:
        raise ContractError("launch-d2 requires the exact D2-only simulation confirmation")
    continuation_control = continuation_control_receipt(manifest, manifest_path, launcher_path)
    original, _, _, original_control = load_original_runtime(manifest)
    c2_attestation, c2_path, c2_sha = load_c2_attestation(manifest)
    paths = d2_paths(manifest, original)
    v1.require_exact(str(paths["arm"]), manifest["d2_only_continuation"]["arm_path"], "D2 arm path")
    if paths["arm"].exists():
        raise ContractError("D2 arm is already claimed; v1r1 has no retry path")
    _d2_training_run_absent(original)
    preflight = v1.verify_runtime(original, CELL_ID, require_empty_gpu=True)
    run_root = Path(original["runtime"]["run_root"])
    v1.no_symlink_existing_components(run_root, "run root")
    run_root.mkdir(parents=True, exist_ok=True)
    v1.require_directory(run_root, "run root")
    paths["arm"].mkdir(exist_ok=False)
    arm_identity = v1.identity(paths["arm"].stat())
    launcher_sha = v1.sha256_file(launcher_path)
    claim = build_d2_claim(
        manifest, original, launcher_sha=launcher_sha,
        arm_identity=arm_identity, c2_attestation_sha=c2_sha,
    )
    claim_sha = v1.canonical_sha256(claim)
    command = v1.build_command(original, CELL_ID, claim_sha)
    launch_contract = {
        "artifact_kind": "phase1_signed_face_d2_v1r1_launch_contract",
        "schema_version": 1,
        "continuation_id": CONTINUATION_ID,
        "continuation_manifest_sha256": CONTINUATION_MANIFEST_SHA256,
        "continuation_launcher_sha256": launcher_sha,
        "original_v1_manifest_sha256": V1_MANIFEST_SHA256,
        "original_v1_launcher_sha256": V1_LAUNCHER_SHA256,
        "cell_id": CELL_ID,
        "run_name": v1.cells(original)[CELL_ID]["run_name"],
        "optimization_recipe": v1.optimization_recipe(original, CELL_ID),
        "training_source": v1.expected_source_identity(original),
        "training_launch_claim": claim,
        "training_launch_claim_sha256": claim_sha,
        "command": command,
        "original_v1_control": original_control,
        "continuation_control": continuation_control,
        "c2_terminal_attestation_path": str(c2_path),
        "c2_terminal_attestation_sha256": c2_sha,
        "c2_optimization_recipe": c2_attestation["optimization_recipe"],
        "execution_lane": claim["execution_lane"],
        "gpu_snapshot_before": preflight["gpu_snapshot"],
        "training_environment_sha256": preflight["training_environment_sha256"],
        "explicit_pythonpath": preflight["environment"]["PYTHONPATH"],
        "kit_thread_caps": original["runtime"]["kit_thread_cap_contract"],
        "verified_inputs": preflight["verified_inputs"],
        "runtime_closure": preflight["runtime_closure"],
        "c2_relaunch": False,
        "automatic_retry": False,
        "activation": False,
        "judge": False,
        "l2": False,
        "second_seed": False,
        "real_robot_commands_forbidden": True,
    }
    v1.write_json_exclusive(paths["launch"], launch_contract)
    try:
        environment = preflight["environment"].copy()
        environment.update({
            "KIT_BOOT_MARKER": v1.HARD_CONTRACT_MARKER,
            "KIT_BOOT_TIMEOUT_S": str(original["runtime"]["kit_boot_timeout_seconds"]),
            "KIT_BOOT_POLL_S": str(original["runtime"]["poll_seconds"]),
            "KIT_BOOT_STATE_FILE": str(paths["state"]),
        })
        completed = subprocess.run(
            [str(preflight["locked"]), str(paths["log"]), *command],
            cwd=preflight["wbt"], env=environment, check=False,
        )
        if completed.returncode != 0:
            raise ContractError(f"reviewed D2 wrapper failed rc={completed.returncode}")
        state = v1.parse_state(paths["state"])
        if not state.get("pid", "").isdigit() or state.get("pgid") != state.get("pid"):
            raise ContractError("D2 wrapper did not record isolated pid==pgid")
        pid = int(state["pid"])
        starttime = v1.process_starttime(pid)
        if starttime <= 0:
            raise ContractError("D2 trainer exited before runtime verification")
        run_dir = v1.locate_training_run(preflight["wbt"], v1.cells(original)[CELL_ID]["run_name"])
        run_info = v1.require_directory(run_dir, "D2 training run")
        contract_path = run_dir / "params/training_contract.json"
        contract_info = v1.require_regular(contract_path, "D2 adjacent hard contract")
        hard_sha, _ = verify_hard_contract(contract_path, original, CELL_ID)
        if (run_dir / "model_24.pt").exists():
            raise ContractError("D2 terminal checkpoint appeared before runtime verification")
        log_text = paths["log"].read_text(encoding="utf-8", errors="replace")
        if v1.THREAD_MARKER not in log_text:
            raise ContractError("D2 log lacks verified 16/16 thread marker")
        runtime = {
            "artifact_kind": "phase1_signed_face_d2_v1r1_runtime_verified",
            "schema_version": 1,
            "continuation_id": CONTINUATION_ID,
            "continuation_manifest_sha256": CONTINUATION_MANIFEST_SHA256,
            "continuation_launcher_sha256": launcher_sha,
            "original_v1_manifest_sha256": V1_MANIFEST_SHA256,
            "original_v1_launcher_sha256": V1_LAUNCHER_SHA256,
            "cell_id": CELL_ID,
            "run_name": v1.cells(original)[CELL_ID]["run_name"],
            "optimization_recipe": v1.optimization_recipe(original, CELL_ID),
            "training_source": v1.expected_source_identity(original),
            "training_launch_claim": claim,
            "training_launch_claim_sha256": claim_sha,
            "command": command,
            "original_v1_control": original_control,
            "continuation_control": continuation_control,
            "c2_terminal_attestation_path": str(c2_path),
            "c2_terminal_attestation_sha256": c2_sha,
            "execution_lane": claim["execution_lane"],
            "training_environment_sha256": preflight["training_environment_sha256"],
            "explicit_pythonpath": preflight["environment"]["PYTHONPATH"],
            "pid": pid,
            "pgid": pid,
            "process_starttime_ticks": starttime,
            "training_run_dir": str(run_dir),
            "training_run_directory_identity": v1.identity(run_info),
            "hard_contract_path": str(contract_path),
            "hard_contract_file_identity": v1.identity(contract_info),
            "hard_contract_sha256": hard_sha,
            "checkpoint_absent_at_runtime_verification": True,
            "c2_relaunch_started": False,
            "activation_started": False,
            "judge_started": False,
            "l2_started": False,
            "second_seed_started": False,
            "real_robot_commands_executed": False,
        }
        v1.write_json_exclusive(paths["runtime"], runtime)
    except BaseException as exc:
        failure = {
            "artifact_kind": "phase1_signed_face_d2_v1r1_launch_failure",
            "schema_version": 1,
            "continuation_id": CONTINUATION_ID,
            "continuation_manifest_sha256": CONTINUATION_MANIFEST_SHA256,
            "continuation_launcher_sha256": launcher_sha,
            "cell_id": CELL_ID,
            "launch_contract_sha256": v1.sha256_file(paths["launch"]),
            "failure_type": type(exc).__name__,
            "failure_message": str(exc),
            "automatic_retry": False,
            "c2_relaunch": False,
            "manual_diagnosis_required": True,
        }
        if paths["state"].is_file():
            failure["launch_state_sha256"] = v1.sha256_file(paths["state"])
        if paths["log"].is_file():
            failure["training_log_sha256_at_failure"] = v1.sha256_file(paths["log"])
        if not paths["failure"].exists():
            v1.write_json_exclusive(paths["failure"], failure)
        raise
    return {
        "status": "d2_only_claimed_and_runtime_verified",
        "cell_id": CELL_ID,
        "pid": runtime["pid"],
        "training_launch_claim_sha256": claim_sha,
        "hard_contract_sha256": runtime["hard_contract_sha256"],
        "c2_relaunch": False,
        "next_action": "wait for D2 natural exit, then finalize-d2",
    }


def verify_d2_launch_and_runtime(
    manifest: dict[str, Any], manifest_path: Path, launcher_path: Path,
) -> dict[str, Any]:
    continuation_control = continuation_control_receipt(manifest, manifest_path, launcher_path)
    original, _, _, original_control = load_original_runtime(manifest)
    _, c2_path, c2_sha = load_c2_attestation(manifest)
    paths = d2_paths(manifest, original)
    if paths["failure"].exists():
        raise ContractError("D2 v1r1 failure is preserved; no automatic retry is allowed")
    launch = v1.read_json_object(paths["launch"], "D2 v1r1 launch contract")
    runtime = v1.read_json_object(paths["runtime"], "D2 v1r1 runtime verification")
    launcher_sha = v1.sha256_file(launcher_path)
    claim, claim_sha = reconstruct_d2_claim(
        manifest, original, launch, launcher_sha, c2_sha
    )
    command = v1.build_command(original, CELL_ID, claim_sha)
    common = {
        "continuation_id": CONTINUATION_ID,
        "continuation_manifest_sha256": CONTINUATION_MANIFEST_SHA256,
        "continuation_launcher_sha256": launcher_sha,
        "original_v1_manifest_sha256": V1_MANIFEST_SHA256,
        "original_v1_launcher_sha256": V1_LAUNCHER_SHA256,
        "cell_id": CELL_ID,
        "run_name": v1.cells(original)[CELL_ID]["run_name"],
        "optimization_recipe": v1.optimization_recipe(original, CELL_ID),
        "training_source": v1.expected_source_identity(original),
        "training_launch_claim": claim,
        "training_launch_claim_sha256": claim_sha,
        "command": command,
        "original_v1_control": original_control,
        "continuation_control": continuation_control,
        "c2_terminal_attestation_path": str(c2_path),
        "c2_terminal_attestation_sha256": c2_sha,
        "execution_lane": claim["execution_lane"],
        "training_environment_sha256": v1.TRAINING_ENV_SHA256_BY_CELL[CELL_ID],
    }
    for key, expected in common.items():
        v1.require_exact(launch.get(key), expected, f"D2 launch {key}")
        v1.require_exact(runtime.get(key), expected, f"D2 runtime {key}")
    v1.require_exact(launch.get("explicit_pythonpath"), v1.EXPECTED_PYTHONPATH, "D2 launch PYTHONPATH")
    v1.require_exact(runtime.get("explicit_pythonpath"), v1.EXPECTED_PYTHONPATH, "D2 runtime PYTHONPATH")
    snapshot = launch.get("gpu_snapshot_before")
    if not isinstance(snapshot, dict):
        raise ContractError("D2 launch lacks assigned-GPU snapshot")
    v1.require_exact(snapshot.get("gpu"), 2, "D2 assigned GPU")
    v1.require_exact(snapshot.get("compute_pids"), [], "D2 pre-claim compute PIDs")
    arm_info = v1.require_directory(paths["arm"], "D2 atomic claim directory")
    v1.require_exact(v1.identity(arm_info), {
        "device": claim["claim_directory"]["st_dev"],
        "inode": claim["claim_directory"]["st_ino"],
    }, "D2 atomic claim directory")
    state = v1.parse_state(paths["state"])
    if not state.get("pid", "").isdigit() or state.get("pgid") != state.get("pid"):
        raise ContractError("D2 launch state lost pid==pgid")
    v1.require_exact(runtime.get("pid"), int(state["pid"]), "D2 runtime pid")
    v1.require_exact(runtime.get("pgid"), int(state["pgid"]), "D2 runtime pgid")
    if type(runtime.get("process_starttime_ticks")) is not int or runtime["process_starttime_ticks"] <= 0:
        raise ContractError("D2 runtime starttime is invalid")
    run_dir = Path(str(runtime.get("training_run_dir", "")))
    expected_logs = (
        Path(original["source"]["training_checkout"])
        / original["source"]["wbt_relative_path"]
        / "logs/rsl_rl/agibot_a3_hope_virtualball"
    )
    v1.require_directory(run_dir, "D2 training run")
    if run_dir.parent != expected_logs or not run_dir.name.endswith(f"_{v1.cells(original)[CELL_ID]['run_name']}"):
        raise ContractError("D2 runtime binds a foreign training run")
    v1.require_exact(v1.identity(run_dir.stat()), runtime.get("training_run_directory_identity"), "D2 run identity")
    contract_path = Path(str(runtime.get("hard_contract_path", "")))
    if contract_path != run_dir / "params/training_contract.json":
        raise ContractError("D2 adjacent hard-contract path changed")
    contract_info = v1.require_regular(contract_path, "D2 adjacent hard contract")
    v1.require_exact(v1.identity(contract_info), runtime.get("hard_contract_file_identity"), "D2 contract identity")
    hard_sha, hard_contract = verify_hard_contract(contract_path, original, CELL_ID)
    v1.require_exact(runtime.get("hard_contract_sha256"), hard_sha, "D2 hard-contract SHA")
    for key in (
        "checkpoint_absent_at_runtime_verification", "c2_relaunch_started",
        "activation_started", "judge_started", "l2_started", "second_seed_started",
    ):
        v1.require_exact(runtime.get(key), True if key == "checkpoint_absent_at_runtime_verification" else False, f"D2 {key}")
    return {
        "original": original,
        "paths": paths,
        "launch": launch,
        "runtime": runtime,
        "claim": claim,
        "claim_sha": claim_sha,
        "command": command,
        "run_dir": run_dir,
        "contract_path": contract_path,
        "hard_sha": hard_sha,
        "hard_contract": hard_contract,
    }


def audit_d2_terminal(
    manifest: dict[str, Any], manifest_path: Path, launcher_path: Path,
    *, stable_delay: float = 1.0,
) -> dict[str, Any]:
    evidence = verify_d2_launch_and_runtime(manifest, manifest_path, launcher_path)
    original = evidence["original"]
    preflight = v1.verify_runtime(original, CELL_ID, require_empty_gpu=True)
    runtime = evidence["runtime"]
    if v1.process_starttime(runtime["pid"]) == runtime["process_starttime_ticks"]:
        raise ContractError("D2 exact trainer is still alive; finalize is read-only")
    log_path = evidence["paths"]["log"]
    log_info = v1.require_regular(log_path, "D2 training log")
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if v1.FAILURE_RE.search(text):
        raise ContractError("D2 log contains a hard failure signature")
    if text.count(v1.THREAD_MARKER) != 1 or v1.HARD_CONTRACT_MARKER not in text:
        raise ContractError("D2 log lacks exact Kit/hard-contract markers")
    checkpoint = evidence["run_dir"] / "model_24.pt"
    before = v1.require_regular(checkpoint, "D2 model_24.pt")
    runtime_info = v1.require_regular(evidence["paths"]["runtime"], "D2 v1r1 runtime verification")
    if before.st_ctime_ns <= runtime_info.st_ctime_ns:
        raise ContractError("D2 model_24.pt does not postdate runtime verification")
    checkpoint_info = _stable_checkpoint(checkpoint, stable_delay)
    checkpoint_sha = v1.sha256_file(checkpoint)
    checkpoint_audit = v1.checkpoint_audit(preflight["python"], checkpoint)
    for key, expected in {
        "iter": 24,
        "training_contract_schema_version": 3,
        "training_contract_sha256": evidence["hard_sha"],
        "training_contract_lineage_exact": 1,
        "training_launch_claim_sha256": evidence["claim_sha"],
        "nonfinite_floating_elements": 0,
    }.items():
        v1.require_exact(checkpoint_audit.get(key), expected, f"D2 checkpoint {key}")
    return {
        "artifact_kind": "phase1_signed_face_d2_v1r1_terminal_result",
        "schema_version": 1,
        "continuation_id": CONTINUATION_ID,
        "continuation_manifest_sha256": CONTINUATION_MANIFEST_SHA256,
        "continuation_launcher_sha256": v1.sha256_file(launcher_path),
        "original_v1_manifest_sha256": V1_MANIFEST_SHA256,
        "original_v1_launcher_sha256": V1_LAUNCHER_SHA256,
        "cell_id": CELL_ID,
        "run_name": v1.cells(original)[CELL_ID]["run_name"],
        "optimization_recipe": v1.optimization_recipe(original, CELL_ID),
        "training_source": v1.expected_source_identity(original),
        "training_launch_claim": evidence["claim"],
        "training_launch_claim_sha256": evidence["claim_sha"],
        "execution_lane": evidence["claim"]["execution_lane"],
        "training_run_dir": str(evidence["run_dir"]),
        "hard_contract_path": str(evidence["contract_path"]),
        "hard_contract_sha256": evidence["hard_sha"],
        "hard_contract": evidence["hard_contract"],
        "terminal_checkpoint_path": str(checkpoint),
        "terminal_checkpoint_sha256": checkpoint_sha,
        "terminal_checkpoint_file_identity": v1.identity(checkpoint_info),
        "checkpoint_audit": checkpoint_audit,
        "training_log_sha256": v1.sha256_file(log_path),
        "training_log_file_identity": v1.identity(log_info),
        "exact_trainer_natural_exit_observed": True,
        "assigned_gpu_empty_terminal_barrier_observed": True,
        "c2_relaunch": False,
        "automatic_retry": False,
        "activation": False,
        "judge": False,
        "l2": False,
        "second_seed": False,
        "real_robot_commands_executed": False,
    }


def finalize_d2(
    manifest: dict[str, Any], manifest_path: Path, launcher_path: Path,
) -> dict[str, Any]:
    original, _, _, _ = load_original_runtime(manifest)
    path = d2_paths(manifest, original)["result"]
    if path.exists():
        raise ContractError("D2 v1r1 terminal result already exists; no-clobber")
    result = audit_d2_terminal(manifest, manifest_path, launcher_path)
    v1.write_json_exclusive(path, result)
    return {
        "status": "d2_v1r1_terminal_checkpoint_finite_exactly_bound",
        "terminal_result_path": str(path),
        "terminal_result_sha256": v1.sha256_file(path),
        "c2_relaunch": False,
        "activation": False,
        "judge": False,
        "l2": False,
        "second_seed": False,
    }


def load_d2_result(
    manifest: dict[str, Any], manifest_path: Path, launcher_path: Path,
) -> tuple[dict[str, Any], Path, str]:
    original, _, _, _ = load_original_runtime(manifest)
    path = d2_paths(manifest, original)["result"]
    recorded = v1.read_json_object(path, "D2 v1r1 terminal result")
    expected = audit_d2_terminal(manifest, manifest_path, launcher_path, stable_delay=0.0)
    v1.require_exact(recorded, expected, "D2 v1r1 terminal result replay")
    return recorded, path, v1.sha256_file(path)


def normalized_recipe(recipe: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(recipe)
    value["signed_face_guidance_weight"] = "<causal-axis>"
    return value


def finalize_pair(
    manifest: dict[str, Any], manifest_path: Path, launcher_path: Path,
) -> dict[str, Any]:
    c2, c2_path, c2_sha = load_c2_attestation(manifest)
    d2, d2_path, d2_sha = load_d2_result(manifest, manifest_path, launcher_path)
    v1.require_exact(c2["optimization_recipe"]["signed_face_guidance_weight"], 0.0, "C2 weight")
    v1.require_exact(d2["optimization_recipe"]["signed_face_guidance_weight"], -0.4, "D2 weight")
    v1.require_exact(
        normalized_recipe(c2["optimization_recipe"]),
        normalized_recipe(d2["optimization_recipe"]),
        "mixed-control normalized trainer recipe",
    )
    contracts = {"C2": c2["hard_contract"], "D2": d2["hard_contract"]}
    v1.verify_pair_contracts(contracts)
    if c2["hard_contract_sha256"] == d2["hard_contract_sha256"]:
        raise ContractError("C2/D2 distinct signed weights produced the same hard-contract SHA")
    output = Path(manifest["outputs"]["paired_result_path"])
    if output.exists():
        raise ContractError("v1r1 paired result already exists; no-clobber")
    content = {
        "artifact_kind": "phase1_signed_face_c2_d2_l1_v1r1_mixed_outer_control_pair",
        "schema_version": 1,
        "continuation_id": CONTINUATION_ID,
        "continuation_manifest_sha256": CONTINUATION_MANIFEST_SHA256,
        "continuation_launcher_sha256": v1.sha256_file(launcher_path),
        "original_v1_manifest_sha256": V1_MANIFEST_SHA256,
        "original_v1_launcher_sha256": V1_LAUNCHER_SHA256,
        "ordered_cells": ["C2", "D2"],
        "outer_control_by_cell": {"C2": "v1", "D2": "v1r1"},
        "terminal_evidence_path_by_cell": {"C2": str(c2_path), "D2": str(d2_path)},
        "terminal_evidence_sha256_by_cell": {"C2": c2_sha, "D2": d2_sha},
        "training_launch_claim_sha256_by_cell": {
            "C2": c2["training_launch_claim_sha256"],
            "D2": d2["training_launch_claim_sha256"],
        },
        "terminal_checkpoint_sha256_by_cell": {
            "C2": c2["terminal_checkpoint_sha256"],
            "D2": d2["terminal_checkpoint_sha256"],
        },
        "hard_contract_sha256_by_cell": {
            "C2": c2["hard_contract_sha256"],
            "D2": d2["hard_contract_sha256"],
        },
        "mixed_outer_control_acknowledged": True,
        "normalized_trainer_recipe_differs_only_signed_face_weight": True,
        "normalized_hard_contract_differs_only_signed_face_weight": True,
        "both_model_24_finite_iter24_lineage1": True,
        "checkpoint_binds_adjacent_hard_contract_and_own_outer_claim": True,
        "c2_relaunch": False,
        "automatic_retry": False,
        "activation": False,
        "judge": False,
        "l2": False,
        "second_seed": False,
        "stop_or_promote": False,
        "same_immutable_signed_paper_still_required": True,
        "real_robot_commands_executed": False,
    }
    v1.write_json_exclusive(output, content)
    return {
        "status": "paired_l1_mixed_control_provenance_complete_decision_still_blocked",
        "paired_result_path": str(output),
        "paired_result_sha256": v1.sha256_file(output),
        "activation": False,
        "judge": False,
        "l2": False,
        "second_seed": False,
    }


def build_plan(manifest: dict[str, Any], manifest_path: Path, launcher_path: Path) -> dict[str, Any]:
    checked = verify_original_for_plan(manifest)
    original = checked["manifest"]
    if v1.cells(original)[CELL_ID]["run_name"] != manifest["d2_only_continuation"]["run_name"]:
        raise ContractError("D2 continuation run name changed")
    return {
        "artifact_kind": "phase1_signed_face_c2_d2_l1_v1r1_plan_only",
        "schema_version": 1,
        "continuation_id": CONTINUATION_ID,
        "continuation_manifest_sha256": v1.sha256_file(manifest_path),
        "continuation_launcher_sha256": v1.sha256_file(launcher_path),
        "original_v1_manifest_sha256": V1_MANIFEST_SHA256,
        "original_v1_launcher_sha256": V1_LAUNCHER_SHA256,
        "training_source": checked["source"],
        "preserved_c2_action": "verify_exact_terminal_evidence_only",
        "only_launchable_cell": CELL_ID,
        "d2_command_template": v1.build_command(original, CELL_ID, v1.CLAIM_PLACEHOLDER),
        "c2_launch_or_retry_mode_present": False,
        "writes_or_launches_performed": False,
        "decision_boundary": manifest["decision_boundary"],
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--manifest", type=Path, default=continuation_manifest_path())
    value.add_argument(
        "--mode",
        choices=(
            "plan", "static-validate", "validate-runtime", "attest-c2",
            "launch-d2", "finalize-d2", "finalize-pair",
        ),
        default="plan",
    )
    value.add_argument("--root-confirm")
    value.add_argument("--plan-output", type=Path)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    manifest_path = args.manifest.absolute()
    launcher_path = HERE
    try:
        manifest = read_manifest(manifest_path)
        if args.mode == "static-validate":
            result = {
                "status": "v1r1_static_valid_no_writes",
                "original_v1": verify_original_for_plan(manifest),
                "c2_relaunch_or_retry_mode_present": False,
            }
        elif args.mode == "plan":
            result = build_plan(manifest, manifest_path, launcher_path)
            if args.plan_output is not None:
                v1.write_json_exclusive(args.plan_output.resolve(), result)
        elif args.mode == "validate-runtime":
            continuation_control_receipt(manifest, manifest_path, launcher_path)
            original, _, _, _ = load_original_runtime(manifest)
            c2 = v1.verify_runtime(original, "C2", require_empty_gpu=True)
            d2 = v1.verify_runtime(original, CELL_ID, require_empty_gpu=True)
            result = {
                "status": "v1r1_runtime_validated_no_launch",
                "c2_gpu_snapshot": c2["gpu_snapshot"],
                "d2_gpu_snapshot": d2["gpu_snapshot"],
                "c2_relaunch_or_retry_mode_present": False,
            }
        elif args.mode == "attest-c2":
            result = write_c2_attestation(manifest, manifest_path, launcher_path)
        elif args.mode == "launch-d2":
            result = launch_d2(manifest, manifest_path, launcher_path, args.root_confirm)
        elif args.mode == "finalize-d2":
            result = finalize_d2(manifest, manifest_path, launcher_path)
        else:
            result = finalize_pair(manifest, manifest_path, launcher_path)
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    except (ContractError, OSError, subprocess.CalledProcessError) as exc:
        print(f"[signed-face-c2-d2-l1-v1r1] FATAL: {exc}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
