#!/usr/bin/env python3
"""One-time, fail-closed continuation from the live v1 A0 arm to unclaimed A1.

The v1 trainer is scientifically valid, but its outer verifier falsely expected
``physics_contract_sha256`` inside the compact schema-3 hard-contract bank
record.  The full bank SHA already binds that metadata.  This continuation
reproduces that exact false rejection, independently validates the bank metadata,
attests the already-live A0 evidence without restarting it, and may then create
exactly one new A1 claim.  It contains no robot command and never discovers or
signals a pre-existing process.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any


SHA_RE = re.compile(r"^[0-9a-f]{64}$")
MANIFEST_ID = (
    "phase1-non-striking-arm-imitation-a0-a1-single-seed-20260714-"
    "v1r1-continuation"
)
EXPECTED_PRIOR = {
    "manifest_relative_path": "configs/phase1_non_striking_arm_imitation_a01_prereg_20260714.json",
    "manifest_runtime_path": (
        "/workspace/codexschema/phase1_non_striking_arm_20260714/control/v1/"
        "phase1_non_striking_arm_imitation_a01_prereg_20260714.json"
    ),
    "manifest_sha256": "b2462527b6573ce6accaf8e626fe264c3da10e8994dba133d8f0aeaeed870506",
    "launcher_relative_path": "scripts/run_phase1_non_striking_arm_imitation_a01.py",
    "launcher_runtime_path": (
        "/workspace/codexschema/phase1_non_striking_arm_20260714/control/v1/"
        "run_phase1_non_striking_arm_imitation_a01.py"
    ),
    "launcher_sha256": "716279ec68ea1b1e22cc32e634e38cd9e81d4fc969b059d21ec7a1f8e081489f",
    "manifest_id": "phase1-non-striking-arm-imitation-a0-a1-single-seed-20260714-v1",
    "training_commit": "353a11419ae8589ed4a374ed97169cd7a50d50a3",
    "training_tree": "184fcb296c09988a7d4b2f5b08168f1584b44b9d",
    "exact_failure_message": "hard contract train-bank binding changed",
    "exact_failure_line": (
        "[non-striking-arm-a01] FATAL: hard contract train-bank binding changed"
    ),
    "failure_reproduction_required": True,
}
EXPECTED_A0 = {
    "cell_id": "A0",
    "run_name": "phase1_non_striking_arm_A0_full_imitation_seed17",
    "arm_dir": (
        "/workspace/codexschema/phase1_non_striking_arm_20260714/runs/diagnostic/"
        "phase1_non_striking_arm_A0_full_imitation_seed17"
    ),
    "training_run_dir": (
        "/workspace/codexschema/nohope_non_striking_arm_353a114/hope_training/"
        "whole_body_tracking/logs/rsl_rl/agibot_a3_hope_virtualball/"
        "2026-07-13_19-48-44_phase1_non_striking_arm_A0_full_imitation_seed17"
    ),
    "pid": 1811464,
    "pgid": 1811464,
    "started_utc": "2026-07-13T19:48:35Z",
    "ready_utc": "2026-07-13T19:49:15Z",
    "launch_contract_sha256": "4c059aa610479a0aea86e437903daaf350f63c1a38f844fa23c517032d418153",
    "launch_state_sha256": "045518bc488bdf5f80cc96a56ed6efa018785283eeb8e7c8f3bff2c27805a342",
    "hard_contract_sha256": "14ef410be5bdcc341901b3678d5331a59af89382e07939ad2049210bf68c29f1",
    "runtime_verified_must_be_absent": True,
    "mask_log_marker_count": 0,
}
EXPECTED_A1 = {
    "cell_id": "A1",
    "run_name": "phase1_non_striking_arm_A1_left_arm_free_seed17",
    "arm_dir": (
        "/workspace/codexschema/phase1_non_striking_arm_20260714/runs/diagnostic/"
        "phase1_non_striking_arm_A1_left_arm_free_seed17"
    ),
    "claim_must_be_absent_before_attestation": True,
    "training_run_glob_must_be_absent_before_attestation": True,
    "root_launch_confirmation": "ROOT_APPROVES_SIM_ONLY_A1_V1R1_CONTINUATION",
    "launch_contract_basename": "launch_contract.json",
    "launch_state_basename": "run.log.launch",
    "training_log_basename": "run.log",
    "runtime_verified_basename": "runtime_verified.json",
}
EXPECTED_BANK = {
    "path": (
        "/workspace/codexschema/phase1_signed_face_rescue_20260713/assets/"
        "schema3_bank_rebind_v2/s1_v4rg_runtime_order_schema3_train_882fea4_rebound.npz"
    ),
    "sha256": "3a9d8851c1c0b13ef82f58228ea1cf83213157c70d72daa514f1bed3a3885b71",
    "schema_version": 3,
    "split": "train",
    "source_family_sha256": "9603a1788eb17ce03598cdde4efff946039613cf61fcc686f90a385706dba9db",
    "physics_contract_sha256": "09dfe8999c54e36b258fe54b5ec3da5d9816ff3be3675963b919371d7f4afb95",
    "source_family_contract_physics_must_match": True,
}
EXPECTED_CONTROL = {
    "runtime_root": "/workspace/codexschema/phase1_non_striking_arm_20260714/control/v1r1",
    "recovery_attestation_basename": "a0_v1r1_recovery_attestation.json",
    "paired_final_result_basename": "a0_a1_v1r1_checkpoint_result.json",
    "manifest_basename": "phase1_non_striking_arm_imitation_a01_v1r1_continuation_20260714.json",
    "launcher_basename": "run_phase1_non_striking_arm_imitation_a01_v1r1.py",
}
EXPECTED_INVARIANTS = {
    "a0_must_be_live_and_exact_before_attestation": True,
    "a0_must_remain_live_through_a1_runtime_verification": True,
    "a0_launch_is_never_reissued": True,
    "a1_is_the_only_new_claim": True,
    "recovery_attestation_written_before_a1_claim": True,
    "recovery_attestation_is_no_clobber": True,
    "a1_launch_contract_binds_prior_and_continuation_control": True,
    "bank_file_sha_and_metadata_physics_are_both_required": True,
    "old_false_rejection_must_reproduce_exactly": True,
    "corrected_verifier_accepts_only_actual_schema3_question_bank_fields": True,
    "checkpoint_contract_and_lineage_checks_unchanged": True,
    "judge_started": False,
    "real_robot_commands_executed": False,
}


class ContinuationError(RuntimeError):
    """The unique v1 -> v1r1 continuation contract was violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContinuationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_pairs
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContinuationError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContinuationError(f"{label} must be one JSON object")
    return value


def require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA_RE.fullmatch(value):
        raise ContinuationError(f"{label} must be one lowercase SHA-256")
    return value


def require_exact(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ContinuationError(f"{label} changed")


def require_no_symlink_components(path: Path, label: str, *, must_exist: bool = True) -> Path:
    if not path.is_absolute():
        raise ContinuationError(f"{label} path must be absolute")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise ContinuationError(f"{label} contains a symlink component: {current}")
        if not current.exists():
            if must_exist:
                raise ContinuationError(f"{label} is missing: {current}")
            break
    return path


def require_regular(path: Path, label: str) -> Path:
    require_no_symlink_components(path, label)
    if not path.is_file() or path.stat().st_nlink != 1:
        raise ContinuationError(f"{label} must be one single-link regular file")
    return path


def write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    payload = (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def load_manifest(path: Path) -> dict[str, Any]:
    data = read_json(path, "v1r1 continuation manifest")
    required = {
        "schema_version", "manifest_id", "status", "purpose", "simulation_only",
        "real_robot_commands_forbidden", "automatic_retry_forbidden",
        "a0_restart_forbidden", "prior_control", "a0_existing_evidence",
        "a1_continuation", "schema3_bank_metadata", "continuation_control",
        "continuation_invariants",
    }
    if set(data) != required:
        raise ContinuationError("continuation manifest top-level keys changed")
    if data["schema_version"] != 1 or data["manifest_id"] != MANIFEST_ID:
        raise ContinuationError("unexpected continuation schema or identity")
    if data["status"] != "machine_preregistered_one_time_a1_continuation_only":
        raise ContinuationError("continuation status changed")
    if not all(data.get(key) is True for key in (
        "simulation_only", "real_robot_commands_forbidden",
        "automatic_retry_forbidden", "a0_restart_forbidden",
    )):
        raise ContinuationError("continuation safety booleans changed")
    require_exact(data["prior_control"], EXPECTED_PRIOR, "prior control")
    require_exact(data["a0_existing_evidence"], EXPECTED_A0, "A0 frozen evidence")
    require_exact(data["a1_continuation"], EXPECTED_A1, "A1 continuation")
    require_exact(data["schema3_bank_metadata"], EXPECTED_BANK, "schema-3 bank metadata")
    require_exact(data["continuation_control"], EXPECTED_CONTROL, "continuation control")
    require_exact(data["continuation_invariants"], EXPECTED_INVARIANTS, "continuation invariants")
    return data


def _control_paths(
    manifest: dict[str, Any], *, runtime_paths: bool, repo_root: Path
) -> tuple[Path, Path]:
    prior = manifest["prior_control"]
    if runtime_paths:
        return Path(prior["manifest_runtime_path"]), Path(prior["launcher_runtime_path"])
    return (
        repo_root / prior["manifest_relative_path"],
        repo_root / prior["launcher_relative_path"],
    )


def load_prior_control(
    manifest: dict[str, Any], *, runtime_paths: bool, repo_root: Path
) -> tuple[Any, dict[str, Any], Path, Path]:
    manifest_path, launcher_path = _control_paths(
        manifest, runtime_paths=runtime_paths, repo_root=repo_root
    )
    require_regular(manifest_path, "prior manifest")
    require_regular(launcher_path, "prior launcher")
    if sha256_file(manifest_path) != manifest["prior_control"]["manifest_sha256"]:
        raise ContinuationError("prior manifest SHA changed")
    if sha256_file(launcher_path) != manifest["prior_control"]["launcher_sha256"]:
        raise ContinuationError("prior launcher SHA changed")
    spec = importlib.util.spec_from_file_location(
        "non_striking_arm_v1_frozen_control", launcher_path
    )
    if spec is None or spec.loader is None:
        raise ContinuationError("cannot load the frozen prior launcher")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        prior_manifest = module.load_manifest(manifest_path)
    except Exception as exc:
        raise ContinuationError(f"frozen prior manifest rejected itself: {exc}") from exc
    prior = manifest["prior_control"]
    if prior_manifest.get("manifest_id") != prior["manifest_id"]:
        raise ContinuationError("prior manifest identity changed")
    source = prior_manifest.get("source", {})
    if (
        source.get("expected_training_commit") != prior["training_commit"]
        or source.get("expected_training_tree") != prior["training_tree"]
    ):
        raise ContinuationError("prior training source identity changed")
    return module, prior_manifest, manifest_path, launcher_path


def verify_schema3_bank_metadata(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    require_regular(path, "schema-3 train bank")
    if sha256_file(path) != expected["sha256"]:
        raise ContinuationError("schema-3 train-bank file SHA changed")
    try:
        import numpy as np

        with np.load(path, allow_pickle=False) as archive:
            if "meta_json" not in archive.files:
                raise ContinuationError("schema-3 train bank has no meta_json")
            raw = bytes(np.asarray(archive["meta_json"], dtype=np.uint8).tolist())
        meta = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_pairs)
    except ContinuationError:
        raise
    except Exception as exc:
        raise ContinuationError(f"cannot parse schema-3 train-bank metadata: {exc}") from exc
    if not isinstance(meta, dict):
        raise ContinuationError("schema-3 train-bank metadata must be an object")
    for key in ("schema_version", "split", "source_family_sha256", "physics_contract_sha256"):
        if meta.get(key) != expected[key]:
            raise ContinuationError(f"schema-3 train-bank metadata {key} changed")
    family = meta.get("source_family_contract")
    if not isinstance(family, dict):
        raise ContinuationError("schema-3 train bank lacks source_family_contract")
    if family.get("physics_contract_sha256") != expected["physics_contract_sha256"]:
        raise ContinuationError("source-family physics-contract SHA changed")
    if canonical_sha256(family) != expected["source_family_sha256"]:
        raise ContinuationError("source-family contract no longer hashes to its declared SHA")
    return {
        "path": str(path),
        "sha256": expected["sha256"],
        "schema_version": expected["schema_version"],
        "split": expected["split"],
        "source_family_sha256": expected["source_family_sha256"],
        "physics_contract_sha256": expected["physics_contract_sha256"],
        "source_family_contract_sha256_recomputed": canonical_sha256(family),
    }


def verify_hard_contract_v1r1(
    path: Path, prior_manifest: dict[str, Any], cell_id: str,
    bank_metadata: dict[str, Any], prior_module: Any,
) -> tuple[str, dict[str, Any]]:
    """Verify the actual compact schema-3 contract and the bank metadata separately."""

    contract = read_json(path, f"{cell_id} emitted hard contract")
    shared = prior_manifest["shared_training_contract"]
    expected_scalars = {
        "schema_version": 3,
        "actor_obs_contract": shared["actor_observation_contract"],
        "actor_obs_total_dim": shared["actor_observation_dim"],
        "face_command_pairing": shared["face_command_pairing"],
        "mount_normal_sign_per_clip": shared["mount_normal_sign_per_clip"],
        "strike_phase_per_clip": shared["strike_phase_per_clip"],
        "motion_kinematics_exact": True,
        "motion_allow_legacy_link_origin_velocity": False,
    }
    for key, expected in expected_scalars.items():
        if contract.get(key) != expected:
            raise ContinuationError(f"{cell_id} emitted hard contract {key} changed")
    body_names = prior_module.cell_map(prior_manifest)[cell_id]["body_names"]
    if contract.get("motion_imitation_body_names") != body_names:
        raise ContinuationError(f"{cell_id} hard contract body mask changed")
    if contract.get("motion_event_timing") != {"mode": "disabled"}:
        raise ContinuationError(f"{cell_id} unexpectedly enabled event timing")
    if len(contract.get("joint_names", [])) != 31 or len(contract.get("action_joint_ids", [])) != 31:
        raise ContinuationError(f"{cell_id} hard contract does not bind 31 joints/actions")
    friction = contract.get("joint_friction_coefficients")
    if (
        not isinstance(friction, list) or len(friction) != 31
        or any(type(value) not in (int, float) or not math.isfinite(value) or value != 0 for value in friction)
    ):
        raise ContinuationError(f"{cell_id} hard contract is not 31/31 zero-friction")
    clips = contract.get("motion_clips")
    expected_clip_shas = [
        prior_manifest["inputs"]["forehand_motion"]["sha256"],
        prior_manifest["inputs"]["backhand_motion"]["sha256"],
    ]
    if (
        not isinstance(clips, list) or len(clips) != 2
        or not all(isinstance(item, dict) for item in clips)
        or [item.get("sha256") for item in clips] != expected_clip_shas
    ):
        raise ContinuationError(f"{cell_id} hard contract motion order/SHA changed")
    bank = contract.get("question_bank")
    actual_bank_keys = {"sha256", "schema_version", "split", "source_family_sha256", "exact"}
    if not isinstance(bank, dict) or set(bank) != actual_bank_keys:
        raise ContinuationError(
            f"{cell_id} hard contract question_bank shape changed: "
            f"{sorted(bank) if isinstance(bank, dict) else type(bank).__name__}"
        )
    expected_bank = prior_manifest["inputs"]["schema3_train_bank"]
    if (
        bank["sha256"] != expected_bank["sha256"]
        or bank["source_family_sha256"] != expected_bank["source_family_sha256"]
        or bank["schema_version"] != 3 or bank["split"] != "train" or bank["exact"] is not True
    ):
        raise ContinuationError(f"{cell_id} compact train-bank hard contract changed")
    if (
        bank_metadata["sha256"] != bank["sha256"]
        or bank_metadata["source_family_sha256"] != bank["source_family_sha256"]
        or bank_metadata["physics_contract_sha256"] != expected_bank["physics_contract_sha256"]
    ):
        raise ContinuationError(f"{cell_id} hard contract and independently parsed bank metadata differ")
    return sha256_file(path), contract


def reproduce_exact_v1_false_rejection(
    prior_module: Any, contract_path: Path, prior_manifest: dict[str, Any],
    expected_message: str,
) -> dict[str, Any]:
    try:
        prior_module.verify_hard_contract(contract_path, prior_manifest, "A0")
    except prior_module.ContractError as exc:
        if str(exc) != expected_message:
            raise ContinuationError(
                f"v1 verifier failed differently: {exc}"
            ) from exc
    else:
        raise ContinuationError("v1 false rejection no longer reproduces")
    return {
        "old_verifier_rejected": True,
        "exact_message": expected_message,
        "exact_stderr_line": f"[non-striking-arm-a01] FATAL: {expected_message}",
        "classification": "outer_verifier_false_rejection_only",
    }


def process_identity(pid: int) -> dict[str, Any] | None:
    stat_path = Path(f"/proc/{pid}/stat")
    cmdline_path = Path(f"/proc/{pid}/cmdline")
    try:
        raw = stat_path.read_text(encoding="utf-8")
        close = raw.rfind(")")
        fields = raw[close + 2 :].split()
        state = fields[0]
        pgid = int(fields[2])
        argv = [
            item.decode("utf-8", errors="surrogateescape")
            for item in cmdline_path.read_bytes().split(b"\0") if item
        ]
    except (OSError, ValueError, IndexError):
        return None
    if state in {"Z", "X"}:
        return None
    return {"pid": pid, "pgid": pgid, "state": state, "argv": argv}


def require_a1_absent(prior_manifest: dict[str, Any], prior_module: Any) -> dict[str, Any]:
    a1 = EXPECTED_A1
    arm_dir = Path(a1["arm_dir"])
    require_no_symlink_components(arm_dir.parent, "A1 claim parent")
    if arm_dir.exists() or arm_dir.is_symlink():
        raise ContinuationError("A1 claim already exists; automatic retry/continuation is forbidden")
    checkout = Path(prior_manifest["source"]["training_checkout"])
    logs_root = (
        checkout / prior_manifest["source"]["wbt_relative_path"] / "logs" / "rsl_rl"
        / "agibot_a3_hope_virtualball"
    )
    require_no_symlink_components(logs_root, "training logs root")
    matches = list(logs_root.glob(f"*_{a1['run_name']}"))
    if matches:
        raise ContinuationError(f"A1 training run already exists without a claim: {matches}")
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            command = (proc / "cmdline").read_bytes()
        except OSError:
            continue
        if a1["run_name"].encode() in command:
            raise ContinuationError(f"A1 run_name already appears in live PID {proc.name}")
    return {
        "arm_dir": str(arm_dir),
        "arm_claim_absent": True,
        "training_run_match_count": 0,
        "live_run_name_match_count": 0,
    }


def _verify_old_launch_contract(
    value: dict[str, Any], prior_manifest: dict[str, Any], prior_module: Any,
) -> None:
    cell = prior_module.cell_map(prior_manifest)["A0"]
    expected_subset = {
        "artifact_kind": "phase1_non_striking_arm_a01_launch_contract",
        "schema_version": 1,
        "manifest_id": prior_manifest["manifest_id"],
        "manifest_sha256": EXPECTED_PRIOR["manifest_sha256"],
        "launcher_sha256": EXPECTED_PRIOR["launcher_sha256"],
        "training_commit": EXPECTED_PRIOR["training_commit"],
        "training_tree": EXPECTED_PRIOR["training_tree"],
        "cell_id": "A0",
        "run_name": cell["run_name"],
        "seed": 17,
        "fresh_initialization": True,
        "expected_body_names": cell["body_names"],
        "command": prior_module.build_command(prior_manifest, "A0"),
        "training_environment_sha256": prior_manifest["runtime"]["training_environment_sha256"],
        "automatic_judge_launch": False,
        "real_robot_commands_forbidden": True,
    }
    for key, expected in expected_subset.items():
        if value.get(key) != expected:
            raise ContinuationError(f"A0 frozen launch contract {key} changed")
    before = value.get("gpu_snapshot_before")
    if not isinstance(before, dict) or before.get("gpu") != 0 or before.get("compute_pids") != []:
        raise ContinuationError("A0 launch contract no longer proves an initially empty GPU0")
    verified = value.get("verified_inputs")
    if not isinstance(verified, dict):
        raise ContinuationError("A0 launch contract lacks verified inputs")
    for name, (path, digest) in prior_module.input_paths(prior_manifest).items():
        if verified.get(name) != {"path": str(path), "sha256": digest}:
            raise ContinuationError(f"A0 launch contract verified input {name} changed")


def verify_a0_live_evidence(
    manifest: dict[str, Any], prior_module: Any, prior_manifest: dict[str, Any],
    preflight: dict[str, Any], *, require_a1_absence: bool,
    allowed_gpu_pids: set[int] | None = None,
    require_a1_launch_memory_floor: bool = True,
) -> dict[str, Any]:
    spec = manifest["a0_existing_evidence"]
    arm_dir = require_no_symlink_components(Path(spec["arm_dir"]), "A0 arm directory")
    if not arm_dir.is_dir():
        raise ContinuationError("A0 arm directory is not a real directory")
    launch_path = require_regular(arm_dir / "launch_contract.json", "A0 launch contract")
    state_path = require_regular(arm_dir / "run.log.launch", "A0 launch state")
    log_path = require_regular(arm_dir / "run.log", "A0 training log")
    runtime_verified = arm_dir / "runtime_verified.json"
    if runtime_verified.exists() or runtime_verified.is_symlink():
        raise ContinuationError("A0 unexpectedly has v1 runtime_verified evidence")
    if sha256_file(launch_path) != spec["launch_contract_sha256"]:
        raise ContinuationError("A0 launch-contract SHA changed")
    if sha256_file(state_path) != spec["launch_state_sha256"]:
        raise ContinuationError("A0 launch-state SHA changed")
    launch_contract = read_json(launch_path, "A0 launch contract")
    _verify_old_launch_contract(launch_contract, prior_manifest, prior_module)
    state = prior_module.parse_launch_state(state_path)
    expected_state = {
        "pid": str(spec["pid"]), "pgid": str(spec["pgid"]),
        "started_utc": spec["started_utc"], "ready_utc": spec["ready_utc"],
        "marker": prior_manifest["runtime"]["kit_boot_marker"],
    }
    for key, expected in expected_state.items():
        if state.get(key) != expected:
            raise ContinuationError(f"A0 launch state {key} changed")
    identity = process_identity(spec["pid"])
    expected_command = prior_module.build_command(prior_manifest, "A0")
    if identity is None or identity["pgid"] != spec["pgid"]:
        raise ContinuationError("A0 exact PID/PGID is not live")
    if identity["argv"] != expected_command:
        raise ContinuationError("A0 live argv changed")
    gpu = prior_module.gpu_snapshot(prior_manifest["runtime"]["gpu"])
    allowed = {spec["pid"]} if allowed_gpu_pids is None else allowed_gpu_pids
    if set(gpu.get("compute_pids", [])) != allowed or set(gpu.get("trainer_pids", [])) != allowed:
        raise ContinuationError(
            f"GPU0 ownership changed: compute={gpu.get('compute_pids')} trainers={gpu.get('trainer_pids')}"
        )
    if (
        require_a1_launch_memory_floor
        and gpu.get("free_memory_mib", 0)
        < prior_manifest["runtime"]["minimum_free_gpu_memory_mib_before_each_launch"]
    ):
        raise ContinuationError("GPU0 free memory is below the frozen A1 launch floor")
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if prior_module.FAILURE_RE.search(text):
        raise ContinuationError("A0 live log contains a hard failure signature")
    if prior_manifest["runtime"]["kit_boot_marker"] not in text:
        raise ContinuationError("A0 live log lost its ready marker")
    markers = prior_module.verify_mask_log(log_path, "A0")
    if len(markers) != spec["mask_log_marker_count"]:
        raise ContinuationError("A0 live log body-mask evidence changed")
    training_run = require_no_symlink_components(
        Path(spec["training_run_dir"]), "A0 training run directory"
    )
    if not training_run.is_dir():
        raise ContinuationError("A0 training run directory changed")
    hard_path = require_regular(
        training_run / "params" / "training_contract.json", "A0 hard contract"
    )
    if sha256_file(hard_path) != spec["hard_contract_sha256"]:
        raise ContinuationError("A0 hard-contract SHA changed")
    bank_metadata = verify_schema3_bank_metadata(
        Path(manifest["schema3_bank_metadata"]["path"]),
        manifest["schema3_bank_metadata"],
    )
    hard_sha, _ = verify_hard_contract_v1r1(
        hard_path, prior_manifest, "A0", bank_metadata, prior_module
    )
    failure = reproduce_exact_v1_false_rejection(
        prior_module, hard_path, prior_manifest,
        manifest["prior_control"]["exact_failure_message"],
    )
    absence = require_a1_absent(prior_manifest, prior_module) if require_a1_absence else None
    return {
        "cell_id": "A0",
        "run_name": spec["run_name"],
        "pid": spec["pid"],
        "pgid": spec["pgid"],
        "process_state": identity["state"],
        "exact_argv_verified": True,
        "arm_dir": str(arm_dir),
        "launch_contract_path": str(launch_path),
        "launch_contract_sha256": spec["launch_contract_sha256"],
        "launch_state_path": str(state_path),
        "launch_state_sha256": spec["launch_state_sha256"],
        "training_log_path": str(log_path),
        "training_log_live_not_hashed": True,
        "runtime_verified_absent": True,
        "training_run_dir": str(training_run),
        "hard_contract_path": str(hard_path),
        "hard_contract_sha256": hard_sha,
        "mask_log_marker_count": len(markers),
        "gpu_snapshot": gpu,
        "bank_metadata": bank_metadata,
        "old_failure_reproduction": failure,
        "a1_absence": absence,
        "training_source": prior_module.verify_static_source(prior_manifest),
        "preflight_training_module_path": preflight["training_module_path"],
    }


def build_recovery_attestation(
    manifest: dict[str, Any], manifest_path: Path, launcher_path: Path,
    prior_manifest_path: Path, prior_launcher_path: Path, a0_evidence: dict[str, Any],
) -> dict[str, Any]:
    content = {
        "manifest_id": manifest["manifest_id"],
        "continuation_manifest_path": str(manifest_path),
        "continuation_manifest_sha256": sha256_file(manifest_path),
        "continuation_launcher_path": str(launcher_path),
        "continuation_launcher_sha256": sha256_file(launcher_path),
        "prior_control": {
            "manifest_path": str(prior_manifest_path),
            "manifest_sha256": manifest["prior_control"]["manifest_sha256"],
            "launcher_path": str(prior_launcher_path),
            "launcher_sha256": manifest["prior_control"]["launcher_sha256"],
            "training_commit": manifest["prior_control"]["training_commit"],
            "training_tree": manifest["prior_control"]["training_tree"],
        },
        "a0_existing_evidence": a0_evidence,
        "a0_restart_forbidden": True,
        "a1_was_unclaimed": True,
        "only_a1_may_be_claimed_after_this_attestation": True,
        "automatic_retry_forbidden": True,
        "judge_started": False,
        "real_robot_commands_executed": False,
    }
    return {
        "artifact_kind": "phase1_non_striking_arm_a01_v1r1_recovery_attestation",
        "schema_version": 1,
        "content": content,
        "content_sha256": canonical_sha256(content),
    }


def _runtime_control_paths(manifest: dict[str, Any]) -> tuple[Path, Path, Path, Path]:
    root = Path(manifest["continuation_control"]["runtime_root"])
    return (
        root,
        root / manifest["continuation_control"]["manifest_basename"],
        root / manifest["continuation_control"]["launcher_basename"],
        root / manifest["continuation_control"]["recovery_attestation_basename"],
    )


def runtime_preflight(
    manifest: dict[str, Any], manifest_path: Path, launcher_path: Path,
    *, require_a1_absence: bool,
) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any], Path, Path]:
    root, expected_manifest, expected_launcher, _ = _runtime_control_paths(manifest)
    require_no_symlink_components(root, "v1r1 control root")
    if manifest_path != expected_manifest or launcher_path != expected_launcher:
        raise ContinuationError("v1r1 runtime must use the exact external control paths")
    require_regular(manifest_path, "v1r1 manifest")
    require_regular(launcher_path, "v1r1 launcher")
    repo_root = launcher_path.parents[1]
    prior_module, prior_manifest, prior_manifest_path, prior_launcher_path = load_prior_control(
        manifest, runtime_paths=True, repo_root=repo_root
    )
    try:
        preflight = prior_module.verify_runtime(prior_manifest, require_initial_empty=False)
    except Exception as exc:
        raise ContinuationError(f"frozen v1 runtime preflight failed: {exc}") from exc
    a0 = verify_a0_live_evidence(
        manifest, prior_module, prior_manifest, preflight,
        require_a1_absence=require_a1_absence,
    )
    return prior_module, prior_manifest, preflight, a0, prior_manifest_path, prior_launcher_path


def expected_a1_launch_contract(
    manifest: dict[str, Any], manifest_path: Path, launcher_path: Path,
    prior_manifest: dict[str, Any], prior_module: Any, preflight: dict[str, Any],
    attestation_path: Path, attestation_sha: str, gpu_before: dict[str, Any],
) -> dict[str, Any]:
    return {
        "artifact_kind": "phase1_non_striking_arm_a01_v1r1_a1_launch_contract",
        "schema_version": 1,
        "manifest_id": manifest["manifest_id"],
        "continuation_manifest_sha256": sha256_file(manifest_path),
        "continuation_launcher_sha256": sha256_file(launcher_path),
        "prior_manifest_sha256": manifest["prior_control"]["manifest_sha256"],
        "prior_launcher_sha256": manifest["prior_control"]["launcher_sha256"],
        "prior_a0_launch_contract_sha256": manifest["a0_existing_evidence"]["launch_contract_sha256"],
        "prior_a0_launch_state_sha256": manifest["a0_existing_evidence"]["launch_state_sha256"],
        "prior_a0_hard_contract_sha256": manifest["a0_existing_evidence"]["hard_contract_sha256"],
        "recovery_attestation_path": str(attestation_path),
        "recovery_attestation_sha256": attestation_sha,
        "training_commit": manifest["prior_control"]["training_commit"],
        "training_tree": manifest["prior_control"]["training_tree"],
        "cell_id": "A1",
        "run_name": manifest["a1_continuation"]["run_name"],
        "seed": prior_manifest["shared_training_contract"]["training_seed"],
        "fresh_initialization": True,
        "expected_body_names": prior_module.cell_map(prior_manifest)["A1"]["body_names"],
        "command": prior_module.build_command(prior_manifest, "A1"),
        "gpu_snapshot_before": gpu_before,
        "verified_inputs": preflight["verified_inputs"],
        "ignored_asset": preflight["ignored_asset"],
        "training_environment_sha256": prior_manifest["runtime"]["training_environment_sha256"],
        "training_module_path": preflight["training_module_path"],
        "a0_restart_performed": False,
        "automatic_retry": False,
        "automatic_judge_launch": False,
        "real_robot_commands_forbidden": True,
    }


def validate_runtime(
    manifest: dict[str, Any], manifest_path: Path, launcher_path: Path
) -> dict[str, Any]:
    _, _, _, a0, prior_manifest_path, prior_launcher_path = runtime_preflight(
        manifest, manifest_path, launcher_path, require_a1_absence=True
    )
    return {
        "status": "v1r1_runtime_validated_no_write_no_launch",
        "prior_manifest_path": str(prior_manifest_path),
        "prior_launcher_path": str(prior_launcher_path),
        "a0": a0,
        "a1_claim_absent": True,
    }


def launch_a1(
    manifest: dict[str, Any], manifest_path: Path, launcher_path: Path,
    root_confirm: str | None,
) -> dict[str, Any]:
    if os.geteuid() != 0:
        raise ContinuationError("v1r1 A1 launch requires root on the simulator Pod")
    if root_confirm != manifest["a1_continuation"]["root_launch_confirmation"]:
        raise ContinuationError("v1r1 A1 launch requires the exact confirmation token")
    (
        prior_module, prior_manifest, preflight, a0, prior_manifest_path,
        prior_launcher_path,
    ) = runtime_preflight(
        manifest, manifest_path, launcher_path, require_a1_absence=True
    )
    root, _, _, attestation_path = _runtime_control_paths(manifest)
    final_path = root / manifest["continuation_control"]["paired_final_result_basename"]
    if attestation_path.exists() or attestation_path.is_symlink():
        raise ContinuationError("recovery attestation already exists; automatic continuation retry is forbidden")
    if final_path.exists() or final_path.is_symlink():
        raise ContinuationError("paired final result already exists")
    attestation = build_recovery_attestation(
        manifest, manifest_path, launcher_path, prior_manifest_path,
        prior_launcher_path, a0,
    )
    write_json_exclusive(attestation_path, attestation)
    attestation_sha = sha256_file(attestation_path)
    require_regular(attestation_path, "v1r1 recovery attestation")
    # Close the only material race before the A1 no-clobber claim.
    verify_a0_live_evidence(
        manifest, prior_module, prior_manifest, preflight,
        require_a1_absence=True,
    )
    a1 = manifest["a1_continuation"]
    arm_dir = Path(a1["arm_dir"])
    arm_dir.mkdir(exist_ok=False)
    log_path = arm_dir / a1["training_log_basename"]
    state_path = arm_dir / a1["launch_state_basename"]
    launch_path = arm_dir / a1["launch_contract_basename"]
    gpu_before = prior_module.gpu_snapshot(prior_manifest["runtime"]["gpu"])
    if set(gpu_before["compute_pids"]) != {EXPECTED_A0["pid"]}:
        raise ContinuationError("GPU ownership changed after recovery attestation")
    launch_contract = expected_a1_launch_contract(
        manifest, manifest_path, launcher_path, prior_manifest, prior_module,
        preflight, attestation_path, attestation_sha, gpu_before,
    )
    write_json_exclusive(launch_path, launch_contract)
    environment = preflight["environment"].copy()
    environment.update({
        "KIT_BOOT_MARKER": prior_manifest["runtime"]["kit_boot_marker"],
        "KIT_BOOT_TIMEOUT_S": str(prior_manifest["runtime"]["kit_boot_timeout_seconds"]),
        "KIT_BOOT_POLL_S": str(prior_manifest["runtime"]["poll_seconds"]),
        "KIT_BOOT_STATE_FILE": str(state_path),
    })
    command = prior_module.build_command(prior_manifest, "A1")
    subprocess.run(
        [str(preflight["locked"]), str(log_path), *command],
        cwd=preflight["wbt"], env=environment, check=True,
    )
    state = prior_module.parse_launch_state(state_path)
    if not state.get("pid", "").isdigit() or state.get("pid") != state.get("pgid"):
        raise ContinuationError("A1 locked launcher did not record isolated pid==pgid")
    a1_pid = int(state["pid"])
    identity = process_identity(a1_pid)
    if identity is None or identity["pgid"] != a1_pid or identity["argv"] != command:
        raise ContinuationError("A1 exact process identity/argv changed after boot")
    run_dir = prior_module.locate_training_run(preflight["wbt"], a1["run_name"])
    hard_path = require_regular(
        run_dir / "params" / "training_contract.json", "A1 emitted hard contract"
    )
    bank_meta = verify_schema3_bank_metadata(
        Path(manifest["schema3_bank_metadata"]["path"]),
        manifest["schema3_bank_metadata"],
    )
    hard_sha, _ = verify_hard_contract_v1r1(
        hard_path, prior_manifest, "A1", bank_meta, prior_module
    )
    markers = prior_module.verify_mask_log(log_path, "A1")
    if len(markers) != 4:
        raise ContinuationError("A1 log does not prove all four post-override body masks")
    a0_after = verify_a0_live_evidence(
        manifest, prior_module, prior_manifest, preflight,
        require_a1_absence=False,
        allowed_gpu_pids={EXPECTED_A0["pid"], a1_pid},
        require_a1_launch_memory_floor=False,
    )
    verified = {
        "artifact_kind": "phase1_non_striking_arm_a01_v1r1_a1_runtime_verified",
        "schema_version": 1,
        "manifest_id": manifest["manifest_id"],
        "continuation_manifest_sha256": sha256_file(manifest_path),
        "continuation_launcher_sha256": sha256_file(launcher_path),
        "prior_manifest_sha256": manifest["prior_control"]["manifest_sha256"],
        "prior_launcher_sha256": manifest["prior_control"]["launcher_sha256"],
        "recovery_attestation_sha256": attestation_sha,
        "launch_contract_sha256": sha256_file(launch_path),
        "cell_id": "A1",
        "run_name": a1["run_name"],
        "pid": a1_pid,
        "pgid": a1_pid,
        "training_run_dir": str(run_dir),
        "hard_contract_path": str(hard_path),
        "hard_contract_sha256": hard_sha,
        "mask_log_markers": markers,
        "a0_pid": EXPECTED_A0["pid"],
        "a0_remained_live_and_exact": a0_after["exact_argv_verified"],
        "a0_restart_performed": False,
        "automatic_retry": False,
        "boot_marker_observed": prior_manifest["runtime"]["kit_boot_marker"],
        "judge_started": False,
        "real_robot_commands_executed": False,
    }
    verified_path = arm_dir / a1["runtime_verified_basename"]
    write_json_exclusive(verified_path, verified)
    return {
        "status": "a1_continuation_launched_a0_untouched",
        "a0_pid": EXPECTED_A0["pid"],
        "a1_pid": a1_pid,
        "a1_hard_contract_sha256": hard_sha,
        "recovery_attestation_sha256": attestation_sha,
        "a1_runtime_verified_sha256": sha256_file(verified_path),
    }


def _require_process_absent(pid: int, label: str) -> None:
    if process_identity(pid) is not None:
        raise ContinuationError(f"{label} is still live; finalizer is read-only")


def _checkpoint_rows(
    prior_module: Any, preflight: dict[str, Any], run_dir: Path,
    hard_sha: str, milestones: list[int], cell_id: str,
) -> list[dict[str, Any]]:
    rows = []
    for iteration in milestones:
        checkpoint = require_regular(run_dir / f"model_{iteration}.pt", f"{cell_id} checkpoint")
        before = (checkpoint.stat().st_size, checkpoint.stat().st_mtime_ns)
        time.sleep(1)
        if before != (checkpoint.stat().st_size, checkpoint.stat().st_mtime_ns):
            raise ContinuationError(f"{cell_id} model_{iteration}.pt is unstable")
        audit = prior_module.checkpoint_audit(preflight["python"], checkpoint)
        expected = {
            "iter": iteration,
            "training_contract_schema_version": 3,
            "training_contract_sha256": hard_sha,
            "training_contract_lineage_exact": 1,
            "nonfinite_floating_elements": 0,
        }
        for key, value in expected.items():
            if audit.get(key) != value:
                raise ContinuationError(f"{cell_id} model_{iteration}.pt {key} changed")
        rows.append({
            "iteration": iteration,
            "path": str(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "sha256": sha256_file(checkpoint),
            "audit": audit,
        })
    return rows


def finalize(
    manifest: dict[str, Any], manifest_path: Path, launcher_path: Path
) -> dict[str, Any]:
    root, expected_manifest, expected_launcher, attestation_path = _runtime_control_paths(manifest)
    if manifest_path != expected_manifest or launcher_path != expected_launcher:
        raise ContinuationError("v1r1 finalizer must use exact external control paths")
    attestation = read_json(require_regular(attestation_path, "recovery attestation"), "recovery attestation")
    if attestation.get("artifact_kind") != "phase1_non_striking_arm_a01_v1r1_recovery_attestation":
        raise ContinuationError("recovery attestation kind changed")
    if not isinstance(attestation.get("content"), dict):
        raise ContinuationError("recovery attestation content is not an object")
    if canonical_sha256(attestation["content"]) != attestation.get("content_sha256"):
        raise ContinuationError("recovery attestation content SHA changed")
    content = attestation["content"]
    if (
        content.get("continuation_manifest_sha256") != sha256_file(manifest_path)
        or content.get("continuation_launcher_sha256") != sha256_file(launcher_path)
        or content.get("prior_control", {}).get("manifest_sha256") != EXPECTED_PRIOR["manifest_sha256"]
        or content.get("prior_control", {}).get("launcher_sha256") != EXPECTED_PRIOR["launcher_sha256"]
        or content.get("a0_existing_evidence", {}).get("hard_contract_sha256") != EXPECTED_A0["hard_contract_sha256"]
    ):
        raise ContinuationError("recovery attestation control/A0 binding changed")
    repo_root = launcher_path.parents[1]
    prior_module, prior_manifest, _, _ = load_prior_control(
        manifest, runtime_paths=True, repo_root=repo_root
    )
    try:
        preflight = prior_module.verify_runtime(prior_manifest, require_initial_empty=False)
    except Exception as exc:
        raise ContinuationError(f"finalizer source/runtime preflight failed: {exc}") from exc
    a1 = manifest["a1_continuation"]
    a1_arm = require_no_symlink_components(Path(a1["arm_dir"]), "A1 arm directory")
    launch_path = require_regular(a1_arm / a1["launch_contract_basename"], "A1 launch contract")
    state_path = require_regular(a1_arm / a1["launch_state_basename"], "A1 launch state")
    log_path = require_regular(a1_arm / a1["training_log_basename"], "A1 training log")
    verified_path = require_regular(a1_arm / a1["runtime_verified_basename"], "A1 runtime evidence")
    launch_contract = read_json(launch_path, "A1 launch contract")
    verified = read_json(verified_path, "A1 runtime evidence")
    attestation_sha = sha256_file(attestation_path)
    gpu_before = launch_contract.get("gpu_snapshot_before")
    if not isinstance(gpu_before, dict):
        raise ContinuationError("A1 launch contract lacks its pre-launch GPU snapshot")
    expected_launch = expected_a1_launch_contract(
        manifest, manifest_path, launcher_path, prior_manifest, prior_module,
        preflight, attestation_path, attestation_sha, gpu_before,
    )
    if launch_contract != expected_launch:
        raise ContinuationError("A1 launch contract differs from the exact v1r1 contract")
    for value, label in ((launch_contract, "A1 launch contract"), (verified, "A1 runtime evidence")):
        if (
            value.get("continuation_manifest_sha256") != sha256_file(manifest_path)
            or value.get("continuation_launcher_sha256") != sha256_file(launcher_path)
            or value.get("prior_manifest_sha256") != EXPECTED_PRIOR["manifest_sha256"]
            or value.get("prior_launcher_sha256") != EXPECTED_PRIOR["launcher_sha256"]
            or value.get("recovery_attestation_sha256") != attestation_sha
            or value.get("cell_id") != "A1" or value.get("run_name") != a1["run_name"]
        ):
            raise ContinuationError(f"{label} continuation chain changed")
    if verified.get("launch_contract_sha256") != sha256_file(launch_path):
        raise ContinuationError("A1 runtime evidence binds a different launch contract")
    state = prior_module.parse_launch_state(state_path)
    if not state.get("pid", "").isdigit() or state.get("pid") != state.get("pgid"):
        raise ContinuationError("A1 final launch state lost pid==pgid")
    if int(state["pid"]) != verified.get("pid") or verified.get("pgid") != verified.get("pid"):
        raise ContinuationError("A1 runtime PID/PGID binding changed")
    _require_process_absent(EXPECTED_A0["pid"], "A0")
    _require_process_absent(int(state["pid"]), "A1")
    for path, cell_id in ((Path(EXPECTED_A0["arm_dir"]) / "run.log", "A0"), (log_path, "A1")):
        text = require_regular(path, f"{cell_id} terminal log").read_text(
            encoding="utf-8", errors="replace"
        )
        if prior_module.FAILURE_RE.search(text):
            raise ContinuationError(f"{cell_id} terminal log contains a hard failure")
        prior_module.verify_mask_log(path, cell_id)
    bank_meta = verify_schema3_bank_metadata(Path(EXPECTED_BANK["path"]), EXPECTED_BANK)
    a0_arm = require_no_symlink_components(Path(EXPECTED_A0["arm_dir"]), "A0 arm directory")
    if sha256_file(require_regular(a0_arm / "launch_contract.json", "A0 launch contract")) != EXPECTED_A0["launch_contract_sha256"]:
        raise ContinuationError("A0 launch-contract SHA changed before finalization")
    if sha256_file(require_regular(a0_arm / "run.log.launch", "A0 launch state")) != EXPECTED_A0["launch_state_sha256"]:
        raise ContinuationError("A0 launch-state SHA changed before finalization")
    hard_contracts = {}
    hard_shas = {}
    run_dirs = {
        "A0": Path(EXPECTED_A0["training_run_dir"]),
        "A1": Path(verified["training_run_dir"]),
    }
    for cell_id in ("A0", "A1"):
        hard_path = require_regular(
            run_dirs[cell_id] / "params" / "training_contract.json",
            f"{cell_id} terminal hard contract",
        )
        hard_sha, hard_contract = verify_hard_contract_v1r1(
            hard_path, prior_manifest, cell_id, bank_meta, prior_module
        )
        if cell_id == "A0" and hard_sha != EXPECTED_A0["hard_contract_sha256"]:
            raise ContinuationError("A0 terminal hard contract changed after recovery")
        if cell_id == "A0":
            failure = reproduce_exact_v1_false_rejection(
                prior_module, hard_path, prior_manifest,
                manifest["prior_control"]["exact_failure_message"],
            )
            if failure != content["a0_existing_evidence"].get("old_failure_reproduction"):
                raise ContinuationError("A0 exact v1 false-rejection evidence changed")
        if cell_id == "A1" and hard_sha != verified.get("hard_contract_sha256"):
            raise ContinuationError("A1 terminal hard contract changed after runtime verification")
        hard_shas[cell_id] = hard_sha
        hard_contracts[cell_id] = hard_contract
    prior_module.verify_pair_contracts_differ_only_by_imitation_body_names(hard_contracts)
    if len(set(hard_shas.values())) != 2:
        raise ContinuationError("A0/A1 hard contracts must have distinct body-mask SHAs")
    milestones = prior_manifest["shared_training_contract"]["relative_checkpoint_milestones"]
    cells = {
        cell_id: {
            "run_name": prior_module.cell_map(prior_manifest)[cell_id]["run_name"],
            "training_run_dir": str(run_dirs[cell_id]),
            "hard_contract_sha256": hard_shas[cell_id],
            "checkpoints": _checkpoint_rows(
                prior_module, preflight, run_dirs[cell_id], hard_shas[cell_id],
                milestones, cell_id,
            ),
        }
        for cell_id in ("A0", "A1")
    }
    result = {
        "artifact_kind": "phase1_non_striking_arm_a01_v1r1_paired_checkpoint_result",
        "schema_version": 1,
        "manifest_id": manifest["manifest_id"],
        "continuation_manifest_sha256": sha256_file(manifest_path),
        "continuation_launcher_sha256": sha256_file(launcher_path),
        "prior_manifest_sha256": EXPECTED_PRIOR["manifest_sha256"],
        "prior_launcher_sha256": EXPECTED_PRIOR["launcher_sha256"],
        "recovery_attestation_sha256": attestation_sha,
        "a0_restarted": False,
        "only_hard_contract_difference": "motion_imitation_body_names",
        "cells": cells,
        "same_immutable_signed_paper_judged": False,
        "stop_or_promote_authorized": False,
        "second_seed_authorized": False,
        "hardware_authorized": False,
    }
    output = root / manifest["continuation_control"]["paired_final_result_basename"]
    write_json_exclusive(output, result)
    return {"status": "paired_checkpoints_finite_bound_judging_still_blocked", "path": str(output), "sha256": sha256_file(output)}


def build_plan(
    manifest: dict[str, Any], manifest_path: Path, launcher_path: Path
) -> dict[str, Any]:
    repo_root = launcher_path.parents[1]
    prior_module, prior_manifest, prior_manifest_path, prior_launcher_path = load_prior_control(
        manifest, runtime_paths=False, repo_root=repo_root
    )
    command = prior_module.build_command(prior_manifest, "A1")
    if any(EXPECTED_A0["run_name"] in item for item in command):
        raise ContinuationError("continuation plan unexpectedly contains an A0 launch")
    return {
        "artifact_kind": "phase1_non_striking_arm_a01_v1r1_plan_only",
        "schema_version": 1,
        "manifest_id": manifest["manifest_id"],
        "continuation_manifest_sha256": sha256_file(manifest_path),
        "continuation_launcher_sha256": sha256_file(launcher_path),
        "prior_manifest_path": str(prior_manifest_path),
        "prior_manifest_sha256": sha256_file(prior_manifest_path),
        "prior_launcher_path": str(prior_launcher_path),
        "prior_launcher_sha256": sha256_file(prior_launcher_path),
        "only_new_cell": "A1",
        "a0_restart_forbidden": True,
        "a1_command": command,
        "writes_or_launches_performed": False,
        "simulation_only": True,
        "real_robot_commands_forbidden": True,
    }


def parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--manifest", type=Path,
        default=root / "configs/phase1_non_striking_arm_imitation_a01_v1r1_continuation_20260714.json",
    )
    value.add_argument(
        "--mode", choices=("plan", "validate-runtime", "launch-a1", "finalize"),
        default="plan",
    )
    value.add_argument("--root-confirm")
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    manifest_path = args.manifest.resolve()
    launcher_path = Path(__file__).resolve()
    try:
        manifest = load_manifest(manifest_path)
        if args.mode == "plan":
            result = build_plan(manifest, manifest_path, launcher_path)
        elif args.mode == "validate-runtime":
            result = validate_runtime(manifest, manifest_path, launcher_path)
        elif args.mode == "launch-a1":
            result = launch_a1(
                manifest, manifest_path, launcher_path, args.root_confirm
            )
        else:
            result = finalize(manifest, manifest_path, launcher_path)
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    except (ContinuationError, OSError, subprocess.SubprocessError) as exc:
        print(f"[non-striking-arm-a01-v1r1] FATAL: {exc}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
