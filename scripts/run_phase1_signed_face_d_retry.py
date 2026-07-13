#!/usr/bin/env python3
"""Strict one-cell retry/finalizer for the failed Phase-1 v6 D boot.

This program owns only the new v6r1 D run name.  It never mutates the failed
v6 D claim, has no direct process-signal API, and cannot launch L2, a judge, or
a second seed.  The frozen locked Kit wrapper may perform its existing
TERM-then-KILL cleanup only against the isolated arm PGID on a pre-marker boot
timeout.  The original exact v6 launcher remains the semantic authority: v6r1
asks it to rebuild D's command, proves the failed claim used that exact command,
and changes exactly one argument (``run_name``).
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from types import ModuleType
from typing import Any


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
RUN_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
CURRENT_MANIFEST_ID = "phase1-signed-face-rescue-d-single-cell-retry-20260713-v6r1"
FOREIGN_MANIFEST_ID = "phase1-signed-face-rescue-single-seed-funnel-20260713-v6"
FOREIGN_CONFIG_SHA256 = "97779cee50819ae6ff34d62f6f3c2aed6b13c360b1bf7f0d075aec1f07feebf2"
FOREIGN_LAUNCHER_SHA256 = "9463f228b26e0a2af548dc749b42428cc3dd1a6379c9d11448e854cfa9d85052"
LOCKED_LAUNCHER_SHA256 = "b250ec6d1cb3700bd45b7ede79e3d124125a0ae586a12dee16510b7cf647fa14"
TRAINING_COMMIT = "50c49e58a9413ec6ac1c3ed2565d9a78acdb5e64"
EXPECTED_HARD_CONTRACT_SHA256 = (
    "dfc583d49362b86dcf3b90e92d0a847f64c14b6b5d71b5d70045c68baa3888a5"
)
OLD_RUN_NAME = "phase1_signed_face_l1_v6_D_fresh_guidance_seed3"
NEW_RUN_NAME = "phase1_signed_face_l1_v6r1_D_fresh_guidance_seed3"
CONTROL_ROOT = Path(
    "/workspace/codexschema/phase1_signed_face_rescue_20260713/control/v6r1"
)
EXPECTED_ORIGINAL_CELLS = {
    "A": (
        "phase1_signed_face_l1_v6_A_hot_control_seed3",
        "hot_parent",
        False,
        13824,
    ),
    "B": (
        "phase1_signed_face_l1_v6_B_hot_guidance_seed3",
        "hot_parent",
        False,
        13824,
    ),
    "C": (
        "phase1_signed_face_l1_v6_C_fresh_control_seed3",
        "fresh",
        True,
        24,
    ),
}
EXPECTED_ORIGINAL_CHECKPOINTS = {
    "A": (
        "/workspace/codexschema/nohope_signed_face_rescue_epoch1_50c49e5/hope_training/"
        "whole_body_tracking/logs/rsl_rl/agibot_a3_hope_virtualball/"
        "2026-07-13_14-18-05_phase1_signed_face_l1_v6_A_hot_control_seed3/model_13824.pt",
        "a1fbb766b2f642e39a4d9d8ce2f89134d6e0c6c1f4202ff7cf94a44f95068d2e",
        7074119,
    ),
    "B": (
        "/workspace/codexschema/nohope_signed_face_rescue_epoch1_50c49e5/hope_training/"
        "whole_body_tracking/logs/rsl_rl/agibot_a3_hope_virtualball/"
        "2026-07-13_14-18-28_phase1_signed_face_l1_v6_B_hot_guidance_seed3/model_13824.pt",
        "c73f59dcfc34a0425f20611117ce3fc1ca97c61681fe3a29b3d2cb4b17620f3d",
        7074119,
    ),
    "C": (
        "/workspace/codexschema/nohope_signed_face_rescue_epoch1_50c49e5/hope_training/"
        "whole_body_tracking/logs/rsl_rl/agibot_a3_hope_virtualball/"
        "2026-07-13_14-19-00_phase1_signed_face_l1_v6_C_fresh_control_seed3/model_24.pt",
        "5ce4de67965a4d9eeba259689ef228cb8fab0cb90cf969fc3a5d859d2ec66b11",
        7073617,
    ),
}


class ContractError(RuntimeError):
    """A frozen retry/finalization invariant was violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ContractError(f"{label} must be one lowercase SHA-256")
    return value


def require_exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        raise ContractError(
            f"{label} keys changed: missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}"
        )
    return value


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{label} must contain one JSON object")
    return value


def write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    """Durably publish one no-clobber JSON artifact."""

    if not path.parent.is_dir():
        raise ContractError(f"output parent must already exist: {path.parent}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o444)
    complete = False
    try:
        raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        complete = True
    finally:
        os.close(fd)
        if not complete:
            try:
                path.unlink()
            except OSError:
                pass
    dir_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _absolute_path(value: Any, label: str) -> Path:
    if not isinstance(value, str):
        raise ContractError(f"{label} must be an absolute path")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ContractError(f"{label} must be an absolute traversal-free path")
    return path


def load_manifest(path: Path) -> dict[str, Any]:
    data = read_json(path, "v6r1 manifest")
    require_exact_keys(
        data,
        {
            "schema_version",
            "manifest_id",
            "purpose",
            "status",
            "simulation_only",
            "real_robot_commands_forbidden",
            "direct_signals_by_retry_tool_forbidden",
            "locked_launcher_exact_pgid_boot_timeout_cleanup_allowed",
            "broad_signals_forbidden",
            "automatic_second_retry_forbidden",
            "automatic_judge_launch",
            "l2_training_launch_authorized",
            "second_seed_authorized",
            "foreign_v6",
            "retry_authority",
            "original_terminal_cells",
            "original_l1_checkpoint_audit",
            "runtime",
            "mixed_finalizer",
        },
        "manifest",
    )
    if data["schema_version"] != 1 or data["manifest_id"] != CURRENT_MANIFEST_ID:
        raise ContractError("unexpected v6r1 manifest identity")
    exact_bools = {
        "simulation_only": True,
        "real_robot_commands_forbidden": True,
        "direct_signals_by_retry_tool_forbidden": True,
        "locked_launcher_exact_pgid_boot_timeout_cleanup_allowed": True,
        "broad_signals_forbidden": True,
        "automatic_second_retry_forbidden": True,
        "automatic_judge_launch": False,
        "l2_training_launch_authorized": False,
        "second_seed_authorized": False,
    }
    for key, expected in exact_bools.items():
        if data.get(key) is not expected:
            raise ContractError(f"manifest {key} changed")

    foreign = require_exact_keys(
        data["foreign_v6"],
        {
            "manifest_id",
            "config_path",
            "config_sha256",
            "launcher_path",
            "launcher_sha256",
            "training_commit",
            "training_checkout",
            "expected_hard_contract_sha256",
            "source_recovery_bundle",
            "source_git_evidence",
        },
        "foreign_v6",
    )
    expected_foreign = {
        "manifest_id": FOREIGN_MANIFEST_ID,
        "config_path": (
            "/workspace/codexschema/phase1_signed_face_rescue_20260713/control/v6/"
            "phase1_signed_face_rescue_funnel_prereg_v6_20260713.json"
        ),
        "config_sha256": FOREIGN_CONFIG_SHA256,
        "launcher_path": (
            "/workspace/codexschema/phase1_signed_face_rescue_20260713/control/v6/"
            "run_phase1_signed_face_rescue_funnel.py"
        ),
        "launcher_sha256": FOREIGN_LAUNCHER_SHA256,
        "training_commit": TRAINING_COMMIT,
        "training_checkout": (
            "/workspace/codexschema/nohope_signed_face_rescue_epoch1_50c49e5"
        ),
        "expected_hard_contract_sha256": EXPECTED_HARD_CONTRACT_SHA256,
    }
    for key, expected in expected_foreign.items():
        if foreign.get(key) != expected:
            raise ContractError(f"foreign_v6 {key} changed")
    bundle = require_exact_keys(
        foreign["source_recovery_bundle"],
        {"path", "sha256", "advertised_commit", "required_prerequisite_commit"},
        "source recovery bundle",
    )
    if bundle != {
        "path": "/workspace/codexschema/phase1_signed_face_rescue_20260713/source_50c49e5.bundle",
        "sha256": "2a794e2c0f9c4adefd5194d94c404bbdf137cf5368f9c2c2aedf2bc50cc0a39e",
        "advertised_commit": TRAINING_COMMIT,
        "required_prerequisite_commit": "882fea4285f0cf9a97ba79d79ae8af31d26ea1ed",
    }:
        raise ContractError("source recovery bundle identity changed")
    git_evidence = require_exact_keys(
        foreign["source_git_evidence"], {"path", "sha256"}, "source git evidence"
    )
    if git_evidence != {
        "path": (
            "/workspace/codexschema/phase1_signed_face_rescue_20260713/"
            "source_50c49e5_git_evidence.txt"
        ),
        "sha256": "12dc839fc76217cd714cfd8ef8f61c42c7e8231cce2b218f34fd42da4a008c99",
    }:
        raise ContractError("source git evidence identity changed")
    for label in ("config_path", "launcher_path", "training_checkout"):
        _absolute_path(foreign[label], f"foreign {label}")

    retry = require_exact_keys(
        data["retry_authority"],
        {
            "classification",
            "cell_id",
            "causal_role",
            "initialization",
            "training_seed",
            "face_guidance_weight",
            "expected_lineage_exact",
            "old_run_name",
            "new_run_name",
            "num_envs",
            "max_iterations",
            "expected_terminal_checkpoint_iteration",
            "old_claim_must_remain_immutable",
            "only_command_change_allowed",
            "old_failed_training_run_dir",
            "old_outer_evidence",
        },
        "retry authority",
    )
    expected_retry = {
        "classification": "versioned_single_cell_retry_after_pre_runtime_boot_timeout",
        "cell_id": "D",
        "causal_role": "fresh_guidance",
        "initialization": "fresh",
        "training_seed": 3,
        "face_guidance_weight": -0.4,
        "expected_lineage_exact": True,
        "old_run_name": OLD_RUN_NAME,
        "new_run_name": NEW_RUN_NAME,
        "num_envs": 512,
        "max_iterations": 25,
        "expected_terminal_checkpoint_iteration": 24,
        "old_claim_must_remain_immutable": True,
        "only_command_change_allowed": "run_name",
        "old_failed_training_run_dir": (
            "/workspace/codexschema/nohope_signed_face_rescue_epoch1_50c49e5/"
            "hope_training/whole_body_tracking/logs/rsl_rl/agibot_a3_hope_virtualball/"
            "2026-07-13_14-19-36_phase1_signed_face_l1_v6_D_fresh_guidance_seed3"
        ),
    }
    for key, expected in expected_retry.items():
        if retry.get(key) != expected:
            raise ContractError(f"retry authority {key} changed")
    if not RUN_NAME_RE.fullmatch(retry["new_run_name"]):
        raise ContractError("retry run name is unsafe")
    if retry["new_run_name"] == retry["old_run_name"]:
        raise ContractError("retry must use a new run name")
    old = require_exact_keys(
        retry["old_outer_evidence"],
        {
            "launch_contract",
            "launch_state",
            "training_log",
            "timeout_diagnostic",
            "original_pid",
            "runtime_verified_must_be_absent",
            "all_model_checkpoints_must_be_absent",
        },
        "old D outer evidence",
    )
    expected_old_artifacts = {
        "launch_contract": "f6dd2fd222ea3f67a65c7c5b2e1ea74ea5f5725a5546b601c4691b8786de0b63",
        "launch_state": "4e1ab699aa3a6763a74c9680beba7b55cc3fb23675edb58f5282acd04497f350",
        "training_log": "baa02f52b645b29ec33a49c1117446591f8e3bfaf334d31cc7b7cb0decb93610",
        "timeout_diagnostic": "ae7de7a37329eddfaa264adb99f9a38c0b7ead33579d5b1dac0d63f4a74b5a0c",
    }
    for name, expected_sha in expected_old_artifacts.items():
        spec = require_exact_keys(old[name], {"path", "sha256"}, f"old D {name}")
        _absolute_path(spec["path"], f"old D {name}")
        if spec["sha256"] != expected_sha:
            raise ContractError(f"old D {name} SHA changed")
    if (
        old["original_pid"] != 1759428
        or old["runtime_verified_must_be_absent"] is not True
        or old["all_model_checkpoints_must_be_absent"] is not True
    ):
        raise ContractError("old D failure classification changed")

    originals = require_exact_keys(
        data["original_terminal_cells"], {"A", "B", "C"}, "original terminal cells"
    )
    for cell_id, (run_name, initialization, lineage, terminal) in EXPECTED_ORIGINAL_CELLS.items():
        spec = originals[cell_id]
        expected_keys = {
            "run_name",
            "initialization",
            "expected_lineage_exact",
            "expected_terminal_checkpoint_iteration",
            "checkpoint_path",
            "checkpoint_sha256",
            "checkpoint_bytes",
            "launch_contract_sha256",
            "runtime_verified_sha256",
            "launch_state_sha256",
            "training_log_sha256",
        }
        if cell_id == "B":
            expected_keys.add("post_terminal_exact_pgid_action")
        require_exact_keys(spec, expected_keys, f"original cell {cell_id}")
        if (
            spec["run_name"] != run_name
            or spec["initialization"] != initialization
            or spec["expected_lineage_exact"] is not lineage
            or spec["expected_terminal_checkpoint_iteration"] != terminal
        ):
            raise ContractError(f"original cell {cell_id} identity changed")
        checkpoint_path, checkpoint_sha, checkpoint_bytes = EXPECTED_ORIGINAL_CHECKPOINTS[cell_id]
        if (
            spec["checkpoint_path"] != checkpoint_path
            or spec["checkpoint_sha256"] != checkpoint_sha
            or spec["checkpoint_bytes"] != checkpoint_bytes
        ):
            raise ContractError(f"original cell {cell_id} terminal checkpoint identity changed")
        require_sha(spec["checkpoint_sha256"], f"original cell {cell_id} checkpoint")
        for key in (
            "launch_contract_sha256",
            "runtime_verified_sha256",
            "launch_state_sha256",
            "training_log_sha256",
        ):
            require_sha(spec[key], f"original cell {cell_id} {key}")
    b_action = require_exact_keys(
        originals["B"]["post_terminal_exact_pgid_action"],
        {"path", "sha256", "pgid", "classification"},
        "B post-terminal action",
    )
    if b_action != {
        "path": "/workspace/codexschema/phase1_signed_face_rescue_20260713/b_kill_action.txt",
        "sha256": "cf619541487f6fb87182df0dd33b73e09c79ab9b1a30037a40901208bcddcafe",
        "pgid": 1758211,
        "classification": "post_terminal_hung_process_exact_pgid_cleanup",
    }:
        raise ContractError("B post-terminal exact-PGID evidence changed")

    checkpoint_audit = require_exact_keys(
        data["original_l1_checkpoint_audit"],
        {"path", "sha256", "required_cells", "d_run_dirs_must_be_empty"},
        "original L1 checkpoint audit",
    )
    if checkpoint_audit != {
        "path": "/workspace/codexschema/phase1_signed_face_rescue_20260713/l1_checkpoint_audit.jsonl",
        "sha256": "620767581cb47dda23843822129b09c66507b0cdc887e283d619d4b51fb0d354",
        "required_cells": ["A", "B", "C", "D"],
        "d_run_dirs_must_be_empty": True,
    }:
        raise ContractError("original L1 checkpoint audit identity changed")

    runtime = require_exact_keys(
        data["runtime"],
        {
            "pod",
            "gpu",
            "artifact_root",
            "control_root",
            "config_path",
            "launcher_path",
            "launch_contract_path",
            "training_log_path",
            "launch_state_path",
            "runtime_verified_path",
            "mixed_activation_path",
            "kit_lock_path",
            "kit_boot_marker",
            "locked_launcher_sha256",
            "kit_boot_timeout_seconds",
            "post_contract_ready_timeout_seconds",
            "post_contract_timeout_requires_manual_exact_state_pgid_audit",
            "poll_seconds",
            "minimum_free_gpu_memory_mib",
            "gpu_must_be_empty_before_launch",
            "kit_lock_must_be_free_before_launch",
            "control_outputs_are_no_clobber",
        },
        "runtime",
    )
    expected_runtime = {
        "pod": "pod1",
        "gpu": 0,
        "artifact_root": "/workspace/codexschema/phase1_signed_face_rescue_20260713",
        "control_root": str(CONTROL_ROOT),
        "config_path": str(CONTROL_ROOT / "phase1_signed_face_d_retry_prereg_v6r1_20260713.json"),
        "launcher_path": str(CONTROL_ROOT / "run_phase1_signed_face_d_retry.py"),
        "launch_contract_path": str(CONTROL_ROOT / "d_retry_launch_contract.json"),
        "training_log_path": str(CONTROL_ROOT / "run.log"),
        "launch_state_path": str(CONTROL_ROOT / "run.log.launch"),
        "runtime_verified_path": str(CONTROL_ROOT / "runtime_verified.json"),
        "mixed_activation_path": str(CONTROL_ROOT / "l1_mixed_activation.json"),
        "kit_lock_path": "/workspace/.kit_boot.lock",
        "kit_boot_marker": "[train.py] hard training contract:",
        "locked_launcher_sha256": LOCKED_LAUNCHER_SHA256,
        "kit_boot_timeout_seconds": 900,
        "post_contract_ready_timeout_seconds": 900,
        "post_contract_timeout_requires_manual_exact_state_pgid_audit": True,
        "poll_seconds": 5,
        "minimum_free_gpu_memory_mib": 4500,
        "gpu_must_be_empty_before_launch": True,
        "kit_lock_must_be_free_before_launch": True,
        "control_outputs_are_no_clobber": True,
    }
    if runtime != expected_runtime:
        raise ContractError("v6r1 runtime/write-path contract changed")
    artifact_root = Path(runtime["artifact_root"])
    for key in (
        "control_root",
        "config_path",
        "launcher_path",
        "launch_contract_path",
        "training_log_path",
        "launch_state_path",
        "runtime_verified_path",
        "mixed_activation_path",
    ):
        candidate = _absolute_path(runtime[key], f"runtime {key}")
        try:
            candidate.relative_to(artifact_root)
        except ValueError as exc:
            raise ContractError(f"runtime {key} escapes artifact root") from exc

    finalizer = require_exact_keys(
        data["mixed_finalizer"],
        {
            "required_sources",
            "required_lineage_exact",
            "all_four_hard_contract_sha256_must_equal",
            "all_checkpoints_must_be_finite",
            "retry_d_must_exit_naturally_after_iteration_24",
            "automatic_judge_launch",
            "l2_training_launch_authorized",
            "second_seed_authorized",
            "stop_or_promote_authorized",
        },
        "mixed finalizer",
    )
    if finalizer != {
        "required_sources": {
            "A": "original_v6",
            "B": "original_v6",
            "C": "original_v6",
            "D": "v6r1_single_cell_retry",
        },
        "required_lineage_exact": {"A": False, "B": False, "C": True, "D": True},
        "all_four_hard_contract_sha256_must_equal": EXPECTED_HARD_CONTRACT_SHA256,
        "all_checkpoints_must_be_finite": True,
        "retry_d_must_exit_naturally_after_iteration_24": True,
        "automatic_judge_launch": False,
        "l2_training_launch_authorized": False,
        "second_seed_authorized": False,
        "stop_or_promote_authorized": False,
    }:
        raise ContractError("mixed finalizer contract changed")
    return data


def load_foreign_runtime(manifest: dict[str, Any]) -> tuple[ModuleType, dict[str, Any]]:
    spec = manifest["foreign_v6"]
    config_path = Path(spec["config_path"])
    launcher_path = Path(spec["launcher_path"])
    for label, path, expected in (
        ("foreign v6 config", config_path, spec["config_sha256"]),
        ("foreign v6 launcher", launcher_path, spec["launcher_sha256"]),
    ):
        if not path.is_file() or sha256_file(path) != expected:
            raise ContractError(f"{label} is missing or has the wrong SHA")
    module_spec = importlib.util.spec_from_file_location(
        "phase1_signed_face_frozen_v6_runtime", launcher_path
    )
    if module_spec is None or module_spec.loader is None:
        raise ContractError("cannot construct the exact foreign v6 module loader")
    module = importlib.util.module_from_spec(module_spec)
    try:
        module_spec.loader.exec_module(module)
        foreign_manifest = module.load_manifest(config_path)
    except Exception as exc:
        raise ContractError(f"exact foreign v6 control rejected itself: {exc}") from exc
    if (
        foreign_manifest.get("manifest_id") != FOREIGN_MANIFEST_ID
        or foreign_manifest.get("source", {}).get("expected_training_commit") != TRAINING_COMMIT
    ):
        raise ContractError("foreign v6 manifest/source identity changed")
    return module, foreign_manifest


def foreign_runtime_preflight(
    manifest: dict[str, Any], foreign: ModuleType, foreign_manifest: dict[str, Any]
) -> dict[str, Any]:
    spec = manifest["foreign_v6"]
    try:
        result = foreign.runtime_preflight(
            foreign_manifest,
            Path(spec["config_path"]),
            Path(spec["launcher_path"]),
            config_sha=spec["config_sha256"],
            launcher_sha=spec["launcher_sha256"],
            stage_name="l1",
            activation_path=None,
            activation_sha=None,
        )
    except Exception as exc:
        raise ContractError(f"foreign v6 source/runtime/bank/report preflight failed: {exc}") from exc
    if result.get("runtime_closure_sha256") != canonical_sha256(
        foreign_manifest["runtime"]["runtime_closure"]
    ):
        raise ContractError("foreign v6 runtime closure SHA changed")
    return result


def verify_production_locations(
    manifest: dict[str, Any], config_path: Path, launcher_path: Path
) -> None:
    runtime = manifest["runtime"]
    if config_path.resolve() != Path(runtime["config_path"]).resolve():
        raise ContractError("production v6r1 config path changed")
    if launcher_path.resolve() != Path(runtime["launcher_path"]).resolve():
        raise ContractError("production v6r1 launcher path changed")
    checkout = Path(manifest["foreign_v6"]["training_checkout"]).resolve()
    for label, path in (("config", config_path.resolve()), ("launcher", launcher_path.resolve())):
        try:
            path.relative_to(checkout)
        except ValueError:
            pass
        else:
            raise ContractError(f"v6r1 {label} may not live inside the training checkout")


def verify_recovery_evidence(manifest: dict[str, Any]) -> dict[str, Any]:
    foreign = manifest["foreign_v6"]
    bundle = foreign["source_recovery_bundle"]
    git_evidence = foreign["source_git_evidence"]
    for label, item in (("source recovery bundle", bundle), ("source git evidence", git_evidence)):
        path = Path(item["path"])
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise ContractError(f"{label} is missing or has the wrong SHA")
    text = Path(git_evidence["path"]).read_text(encoding="utf-8", errors="replace")
    if f"HEAD {TRAINING_COMMIT}" not in text or "PORCELAIN_BEGIN\nPORCELAIN_END" not in text:
        raise ContractError("source git evidence does not prove clean exact epoch-1 checkout")
    return {
        "bundle_path": bundle["path"],
        "bundle_sha256": bundle["sha256"],
        "source_git_evidence_path": git_evidence["path"],
        "source_git_evidence_sha256": git_evidence["sha256"],
    }


def verify_original_checkpoint_audit(manifest: dict[str, Any]) -> dict[str, Any]:
    spec = manifest["original_l1_checkpoint_audit"]
    path = Path(spec["path"])
    if not path.is_file() or sha256_file(path) != spec["sha256"]:
        raise ContractError("original L1 checkpoint audit is missing or has the wrong SHA")
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("row is not an object")
            rows.append(value)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ContractError(f"cannot parse original L1 checkpoint audit: {exc}") from exc
    by_cell = {row.get("cell"): row for row in rows}
    if len(rows) != 4 or set(by_cell) != {"A", "B", "C", "D"}:
        raise ContractError("original L1 checkpoint audit must contain A/B/C/D exactly once")
    for cell_id in ("A", "B", "C"):
        cell = manifest["original_terminal_cells"][cell_id]
        row = by_cell[cell_id]
        audit = row.get("audit")
        if not isinstance(audit, dict):
            raise ContractError(f"original L1 checkpoint audit {cell_id} lacks audit payload")
        expected = {
            "checkpoint_exists": True,
            "checkpoint_path": cell["checkpoint_path"],
            "checkpoint_sha": cell["checkpoint_sha256"],
            "checkpoint_bytes": cell["checkpoint_bytes"],
            "contract_exists": True,
            "contract_sha": EXPECTED_HARD_CONTRACT_SHA256,
            "expected_lineage": int(cell["expected_lineage_exact"]),
        }
        for key, value in expected.items():
            if row.get(key) != value:
                raise ContractError(f"original L1 checkpoint audit {cell_id} {key} changed")
        expected_audit = {
            "iter": cell["expected_terminal_checkpoint_iteration"],
            "lineage": int(cell["expected_lineage_exact"]),
            "schema": 3,
            "contract_sha": EXPECTED_HARD_CONTRACT_SHA256,
            "nonfinite": 0,
            "floating_tensor_count": 74,
            "floating_elements": 1762715,
        }
        for key, value in expected_audit.items():
            if audit.get(key) != value:
                raise ContractError(
                    f"original L1 checkpoint audit {cell_id} audit.{key} changed"
                )
    d_row = by_cell["D"]
    if d_row != {"cell": "D", "expected_lineage": 1, "run_dirs": []}:
        raise ContractError("original L1 checkpoint audit no longer proves D had no run dir")
    return {"path": str(path), "sha256": spec["sha256"], "cells": ["A", "B", "C", "D"]}


def parse_launch_state(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def process_entry_exists(pid: int) -> bool:
    return Path(f"/proc/{pid}/stat").is_file()


def process_argv(pid: int) -> list[str]:
    path = Path(f"/proc/{pid}/cmdline")
    if not path.is_file():
        raise ContractError("training process cmdline disappeared")
    return [
        item.decode("utf-8", errors="surrogateescape")
        for item in path.read_bytes().split(b"\0")
        if item
    ]


def gpu_snapshot(gpu: int) -> dict[str, Any]:
    try:
        free_lines = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                str(gpu),
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip().splitlines()
        pid_lines = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                str(gpu),
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        ).splitlines()
    except subprocess.CalledProcessError as exc:
        raise ContractError(f"cannot inspect GPU {gpu}: {exc.output}") from exc
    if len(free_lines) != 1 or not free_lines[0].strip().isdigit():
        raise ContractError(f"cannot parse GPU {gpu} free memory")
    pids: list[int] = []
    for raw in pid_lines:
        if raw.strip().isdigit() and int(raw.strip()) not in pids:
            pids.append(int(raw.strip()))
    trainers = []
    for pid in pids:
        cmdline = Path(f"/proc/{pid}/cmdline")
        if cmdline.is_file() and b"scripts/train.py" in cmdline.read_bytes():
            trainers.append(pid)
    return {
        "gpu": gpu,
        "free_memory_mib": int(free_lines[0].strip()),
        "compute_pids": pids,
        "trainer_pids": trainers,
    }


def verify_kit_lock_free(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ContractError("Kit boot lock must be one existing regular non-symlink file")
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ContractError("Kit boot lock is currently held") from exc
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
    finally:
        os.close(fd)
    return {"path": str(path), "free_at_preflight": True}


def build_retry_command(original: list[str], old_run_name: str, new_run_name: str) -> list[str]:
    old_token = f"run_name={old_run_name}"
    new_token = f"run_name={new_run_name}"
    matches = [index for index, value in enumerate(original) if value == old_token]
    if len(matches) != 1:
        raise ContractError("original D command must contain exactly one old run_name argument")
    index = matches[0]
    result = list(original)
    result[index] = new_token
    differences = [i for i, (left, right) in enumerate(zip(original, result)) if left != right]
    if len(result) != len(original) or differences != [index]:
        raise ContractError("retry command changed more than run_name")
    if original[index] != old_token or result[index] != new_token:
        raise ContractError("retry command run_name delta is not the frozen old-to-new pair")
    forbidden = ("ros2 ", "run_deploy", "real_robot", "joint_command", "/dev/")
    if any(any(token in part.lower() for token in forbidden) for part in result):
        raise ContractError("retry command contains a robot/runtime token")
    if any("actor_leg_ref_mask" in part for part in result):
        raise ContractError("retry command may not override actor_leg_ref_mask")
    return result


def verify_failed_d_claim(
    manifest: dict[str, Any],
    foreign: ModuleType,
    foreign_manifest: dict[str, Any],
    preflight: dict[str, Any],
) -> dict[str, Any]:
    retry = manifest["retry_authority"]
    outer = retry["old_outer_evidence"]
    artifacts: dict[str, Path] = {}
    for name in ("launch_contract", "launch_state", "training_log", "timeout_diagnostic"):
        path = Path(outer[name]["path"])
        if not path.is_file() or sha256_file(path) != outer[name]["sha256"]:
            raise ContractError(f"old D {name} evidence is missing or changed")
        artifacts[name] = path
    run_dir = artifacts["launch_contract"].parent
    runtime_verified = run_dir / "runtime_verified.json"
    if runtime_verified.exists():
        raise ContractError("old D runtime_verified unexpectedly exists; retry authority is invalid")
    failed_training_run = Path(retry["old_failed_training_run_dir"])
    if not failed_training_run.is_dir():
        raise ContractError("old D failed training run directory is missing")
    checkpoints = sorted(failed_training_run.glob("model_*.pt"))
    if checkpoints:
        raise ContractError(f"old D now has checkpoints; retry authority is invalid: {checkpoints}")
    state = parse_launch_state(artifacts["launch_state"])
    pid = outer["original_pid"]
    if state.get("pid") != str(pid) or state.get("pgid") != str(pid):
        raise ContractError("old D launch state no longer binds its exact PID=PGID")
    if process_entry_exists(pid):
        raise ContractError("old D PID still exists; a versioned retry is unsafe")
    text = artifacts["training_log"].read_text(encoding="utf-8", errors="replace")
    if manifest["runtime"]["kit_boot_marker"] in text or "Learning iteration" in text:
        raise ContractError("old D unexpectedly reached runtime contract or learning")
    if str(failed_training_run) not in text:
        raise ContractError("old D log does not bind the frozen failed training run directory")

    launch = read_json(artifacts["launch_contract"], "old D launch contract")
    expected_identity = {
        "manifest_id": FOREIGN_MANIFEST_ID,
        "manifest_file_sha256": FOREIGN_CONFIG_SHA256,
        "launcher_file_sha256": FOREIGN_LAUNCHER_SHA256,
        "training_commit": TRAINING_COMMIT,
        "stage": "l1",
        "cell_id": "D",
        "causal_role": "fresh_guidance",
        "initialization": "fresh",
        "expected_lineage_exact": True,
        "run_name": OLD_RUN_NAME,
    }
    for key, expected in expected_identity.items():
        if launch.get(key) != expected:
            raise ContractError(f"old D launch contract {key} changed")
    try:
        original_command = foreign.build_command(
            foreign_manifest, "l1", "D", wbt=preflight["wbt"]
        )
    except Exception as exc:
        raise ContractError(f"foreign v6 cannot rebuild original D command: {exc}") from exc
    if launch.get("command") != original_command:
        raise ContractError("old D launch command differs from exact foreign v6 reconstruction")
    retry_command = build_retry_command(original_command, OLD_RUN_NAME, NEW_RUN_NAME)
    diagnostic = artifacts["timeout_diagnostic"].read_text(
        encoding="utf-8", errors="replace"
    )
    if "D_GROUP_BEGIN\nD_GROUP_END" not in diagnostic or outer["training_log"]["sha256"] not in diagnostic:
        raise ContractError("old D timeout diagnostic does not prove dead group and frozen log")
    return {
        "old_launch_contract_path": str(artifacts["launch_contract"]),
        "old_launch_contract_sha256": outer["launch_contract"]["sha256"],
        "old_launch_state_path": str(artifacts["launch_state"]),
        "old_launch_state_sha256": outer["launch_state"]["sha256"],
        "old_training_log_path": str(artifacts["training_log"]),
        "old_training_log_sha256": outer["training_log"]["sha256"],
        "old_timeout_diagnostic_path": str(artifacts["timeout_diagnostic"]),
        "old_timeout_diagnostic_sha256": outer["timeout_diagnostic"]["sha256"],
        "old_pid": pid,
        "old_runtime_verified_absent": True,
        "old_checkpoint_count": 0,
        "original_command": original_command,
        "retry_command": retry_command,
        "command_delta": {"field": "run_name", "old": OLD_RUN_NAME, "new": NEW_RUN_NAME},
    }


def verify_retry_outputs_absent(manifest: dict[str, Any], *, allow_runtime: bool) -> None:
    runtime = manifest["runtime"]
    names = (
        "launch_contract_path",
        "training_log_path",
        "launch_state_path",
        "runtime_verified_path",
        "mixed_activation_path",
    )
    existing = [name for name in names if Path(runtime[name]).exists()]
    if allow_runtime:
        if Path(runtime["mixed_activation_path"]).exists():
            raise ContractError("mixed activation already exists; no-clobber finalizer refuses")
        return
    if existing:
        raise ContractError(
            "v6r1 control claim/output already exists; automatic second retry is forbidden: "
            f"{existing}"
        )


def launch_readiness(
    manifest: dict[str, Any], config_path: Path, launcher_path: Path
) -> tuple[ModuleType, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    verify_production_locations(manifest, config_path, launcher_path)
    foreign, foreign_manifest = load_foreign_runtime(manifest)
    preflight = foreign_runtime_preflight(manifest, foreign, foreign_manifest)
    locked = Path(preflight["locked"])
    if sha256_file(locked) != LOCKED_LAUNCHER_SHA256:
        raise ContractError("locked Kit launcher SHA differs from the frozen exact-PGID wrapper")
    recovery = verify_recovery_evidence(manifest)
    checkpoint_audit = verify_original_checkpoint_audit(manifest)
    failed = verify_failed_d_claim(manifest, foreign, foreign_manifest, preflight)
    verify_retry_outputs_absent(manifest, allow_runtime=False)
    snapshot = gpu_snapshot(manifest["runtime"]["gpu"])
    if snapshot["compute_pids"] or snapshot["trainer_pids"]:
        raise ContractError(f"Pod1 GPU0 is not empty: {snapshot}")
    if snapshot["free_memory_mib"] < manifest["runtime"]["minimum_free_gpu_memory_mib"]:
        raise ContractError("Pod1 GPU0 free memory is below the v6r1 launch floor")
    lock = verify_kit_lock_free(Path(manifest["runtime"]["kit_lock_path"]))
    return foreign, foreign_manifest, preflight, failed, {
        "recovery": recovery,
        "original_checkpoint_audit": checkpoint_audit,
        "gpu": snapshot,
        "kit_lock": lock,
    }


def verify_live_retry_process(pid: int, command: list[str], gpu: int) -> None:
    if not process_entry_exists(pid):
        raise ContractError("v6r1 D exited before the first learning iteration")
    if process_argv(pid) != command[3:]:
        raise ContractError("v6r1 D live argv differs from the frozen retry command")
    snapshot = gpu_snapshot(gpu)
    if snapshot["compute_pids"] != [pid] or snapshot["trainer_pids"] != [pid]:
        raise ContractError(f"v6r1 D is not the sole exact GPU0 trainer: {snapshot}")


def wait_ready_no_signal(
    log_path: Path,
    state_path: Path,
    command: list[str],
    *,
    gpu: int,
    timeout: int,
    poll: int,
    failure_re: re.Pattern[str],
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
        if failure_re.search(text):
            raise ContractError("v6r1 D log contains a hard failure before ready")
        state = parse_launch_state(state_path) if state_path.is_file() else {}
        pid_raw = state.get("pid", "")
        if not pid_raw.isdigit() or state.get("pgid") != pid_raw:
            raise ContractError("v6r1 D launch state lost PID=PGID")
        verify_live_retry_process(int(pid_raw), command, gpu)
        if "Learning iteration" in text:
            return
        time.sleep(poll)
    raise ContractError(
        "v6r1 D did not reach its first learning iteration before the post-contract "
        "timeout; the arm may still be live, so inspect only pid=pgid from the frozen "
        "launch state and do not start another retry"
    )


def run_locked_launcher(
    locked: Path,
    log_path: Path,
    state_path: Path,
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    boot_timeout: int,
) -> dict[str, Any]:
    """Run the frozen wrapper and classify its exact-PGID timeout contract."""

    if sha256_file(locked) != LOCKED_LAUNCHER_SHA256:
        raise ContractError("locked Kit launcher SHA changed before execution")
    try:
        subprocess.run(
            [str(locked), str(log_path), *command],
            cwd=cwd,
            env=environment,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        if exc.returncode == 124:
            state = parse_launch_state(state_path) if state_path.is_file() else {}
            pid = state.get("pid", "")
            if (
                not pid.isdigit()
                or state.get("pgid") != pid
                or state.get("boot_timeout_s") != str(boot_timeout)
            ):
                raise ContractError(
                    "locked launcher returned boot-timeout rc=124 without complete "
                    "pid=pgid cleanup evidence; preserve the claim and inspect manually"
                ) from exc
            raise ContractError(
                "frozen locked launcher recorded its allowed exact-PGID TERM-then-KILL "
                f"boot-timeout cleanup for pgid={pid}; preserve the no-clobber claim and "
                "do not start an automatic second retry"
            ) from exc
        raise ContractError(
            "v6r1 locked launch failed; the no-clobber claim is preserved and no automatic "
            f"second retry is allowed (rc={exc.returncode})"
        ) from exc
    return {
        "locked_launcher_sha256": LOCKED_LAUNCHER_SHA256,
        "ready_marker_return_code": 0,
        "exact_pgid_boot_timeout_cleanup_executed": False,
    }


def launch_retry(
    manifest: dict[str, Any],
    config_path: Path,
    launcher_path: Path,
    *,
    config_sha: str,
    launcher_sha: str,
) -> dict[str, Any]:
    foreign, foreign_manifest, preflight, failed, readiness = launch_readiness(
        manifest, config_path, launcher_path
    )
    runtime = manifest["runtime"]
    command = failed["retry_command"]
    claim = {
        "artifact_kind": "phase1_signed_face_rescue_d_versioned_retry_launch_contract",
        "schema_version": 1,
        "manifest_id": manifest["manifest_id"],
        "manifest_file_sha256": config_sha,
        "launcher_file_sha256": launcher_sha,
        "foreign_manifest_id": FOREIGN_MANIFEST_ID,
        "foreign_manifest_file_sha256": FOREIGN_CONFIG_SHA256,
        "foreign_launcher_file_sha256": FOREIGN_LAUNCHER_SHA256,
        "training_commit": TRAINING_COMMIT,
        "stage": "l1",
        "cell_id": "D",
        "run_name": NEW_RUN_NAME,
        "causal_role": "fresh_guidance",
        "initialization": "fresh",
        "expected_lineage_exact": True,
        "expected_terminal_checkpoint_iteration": 24,
        "old_failure_evidence": {
            key: value for key, value in failed.items() if key not in {"original_command", "retry_command"}
        },
        "original_command_sha256": canonical_sha256(failed["original_command"]),
        "command": command,
        "command_delta": failed["command_delta"],
        "runtime_closure_sha256": preflight["runtime_closure_sha256"],
        "training_environment_sha256": preflight["training_environment_sha256"],
        "training_module_path": preflight["training_module_path"],
        "verified_inputs": preflight["verified_inputs"],
        "recovery_evidence": readiness["recovery"],
        "original_checkpoint_audit": readiness["original_checkpoint_audit"],
        "gpu_snapshot_before": readiness["gpu"],
        "kit_lock_free_before": readiness["kit_lock"],
        "automatic_judge_launch": False,
        "l2_training_launch_authorized": False,
        "second_seed_authorized": False,
        "direct_signals_sent_by_retry_tool": False,
        "locked_launcher_sha256": LOCKED_LAUNCHER_SHA256,
        "locked_launcher_exact_pgid_boot_timeout_cleanup_allowed": True,
        "broad_signals_forbidden": True,
        "post_contract_timeout_requires_manual_exact_state_pgid_audit": True,
        "real_robot_commands_forbidden": True,
    }
    claim_path = Path(runtime["launch_contract_path"])
    write_json_exclusive(claim_path, claim)
    environment = preflight["training_environment"].copy()
    environment.update(
        {
            "KIT_BOOT_MARKER": runtime["kit_boot_marker"],
            "KIT_BOOT_TIMEOUT_S": str(runtime["kit_boot_timeout_seconds"]),
            "KIT_BOOT_POLL_S": str(runtime["poll_seconds"]),
            "KIT_BOOT_STATE_FILE": runtime["launch_state_path"],
        }
    )
    locked_result = run_locked_launcher(
        Path(preflight["locked"]),
        Path(runtime["training_log_path"]),
        Path(runtime["launch_state_path"]),
        command,
        cwd=Path(preflight["wbt"]),
        environment=environment,
        boot_timeout=runtime["kit_boot_timeout_seconds"],
    )
    log_path = Path(runtime["training_log_path"])
    state_path = Path(runtime["launch_state_path"])
    try:
        contract_path = foreign.emitted_contract_path(log_path, NEW_RUN_NAME)
        contract_sha, _ = foreign.verify_emitted_contract(
            contract_path, foreign_manifest, hot=False
        )
    except Exception as exc:
        raise ContractError(f"v6r1 D emitted hard contract is invalid: {exc}") from exc
    if contract_sha != EXPECTED_HARD_CONTRACT_SHA256:
        raise ContractError("v6r1 D hard contract differs from original A/B/C")
    wait_ready_no_signal(
        log_path,
        state_path,
        command,
        gpu=runtime["gpu"],
        timeout=runtime["post_contract_ready_timeout_seconds"],
        poll=runtime["poll_seconds"],
        failure_re=foreign.FAILURE_RE,
    )
    try:
        foreign.verify_guidance_log(log_path, foreign.cell_by_id(foreign_manifest, "D"))
    except Exception as exc:
        raise ContractError(f"v6r1 D applied-guidance proof failed: {exc}") from exc
    state = parse_launch_state(state_path)
    pid = int(state["pid"])
    verified = {
        "artifact_kind": "phase1_signed_face_rescue_d_retry_runtime_verified",
        "schema_version": 1,
        "manifest_file_sha256": config_sha,
        "launcher_file_sha256": launcher_sha,
        "foreign_manifest_file_sha256": FOREIGN_CONFIG_SHA256,
        "foreign_launcher_file_sha256": FOREIGN_LAUNCHER_SHA256,
        "launch_contract_sha256": sha256_file(claim_path),
        "cell_id": "D",
        "run_name": NEW_RUN_NAME,
        "pid": pid,
        "pgid": pid,
        "training_run_dir": str(contract_path.parent.parent),
        "emitted_hard_contract_path": str(contract_path),
        "emitted_hard_contract_sha256": contract_sha,
        "training_environment_sha256": preflight["training_environment_sha256"],
        "runtime_closure_sha256": preflight["runtime_closure_sha256"],
        "training_module_path": preflight["training_module_path"],
        "expected_lineage_exact": True,
        "actor_leg_ref_mask_provenance_epoch": 1,
        "actor_leg_ref_mask": False,
        "command_delta_only_run_name": True,
        "direct_signals_sent_by_retry_tool": False,
        "locked_launcher_sha256": locked_result["locked_launcher_sha256"],
        "locked_launcher_exact_pgid_boot_timeout_cleanup_allowed": True,
        "locked_launcher_exact_pgid_boot_timeout_cleanup_executed": locked_result[
            "exact_pgid_boot_timeout_cleanup_executed"
        ],
        "broad_signals_forbidden": True,
        "automatic_judge_launch": False,
    }
    write_json_exclusive(Path(runtime["runtime_verified_path"]), verified)
    return {"pid": pid, "pgid": pid, "contract_sha256": contract_sha}


def _stable_file(path: Path, delay: float) -> None:
    before = path.stat()
    if before.st_size <= 0:
        raise ContractError(f"empty terminal checkpoint: {path}")
    if delay > 0:
        time.sleep(delay)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ContractError(f"terminal checkpoint is still changing: {path}")


def _check_audit(
    audit: dict[str, Any], *, terminal: int, lineage: bool, contract_sha: str, label: str
) -> None:
    expected = {
        "iter": terminal,
        "training_contract_schema_version": 3,
        "training_contract_sha256": contract_sha,
        "training_contract_lineage_exact": int(lineage),
        "training_contract_provenance_location": "infos",
        "nonfinite_floating_elements": 0,
    }
    for key, value in expected.items():
        if audit.get(key) != value:
            raise ContractError(f"{label} checkpoint {key} mismatch")
    if audit.get("floating_tensor_count", 0) <= 0:
        raise ContractError(f"{label} checkpoint has no floating tensors")


def audit_original_terminal_cell(
    manifest: dict[str, Any],
    foreign: ModuleType,
    foreign_manifest: dict[str, Any],
    preflight: dict[str, Any],
    cell_id: str,
    *,
    stability_delay: float = 2.0,
) -> dict[str, Any]:
    spec = manifest["original_terminal_cells"][cell_id]
    run_name = spec["run_name"]
    run_dir = (
        Path(manifest["runtime"]["artifact_root"]) / "runs" / "l1" / run_name
    )
    paths = {
        "launch_contract": run_dir / "launch_contract.json",
        "runtime_verified": run_dir / "runtime_verified.json",
        "launch_state": run_dir / "run.log.launch",
        "training_log": run_dir / "run.log",
    }
    for name, path in paths.items():
        expected = spec[f"{name}_sha256"]
        if not path.is_file() or sha256_file(path) != expected:
            raise ContractError(f"original L1 {cell_id} {name} evidence changed")
    launch = read_json(paths["launch_contract"], f"original L1 {cell_id} launch contract")
    runtime_verified = read_json(
        paths["runtime_verified"], f"original L1 {cell_id} runtime verified"
    )
    expected_identity = {
        "manifest_file_sha256": FOREIGN_CONFIG_SHA256,
        "launcher_file_sha256": FOREIGN_LAUNCHER_SHA256,
        "training_commit": TRAINING_COMMIT,
        "stage": "l1",
        "cell_id": cell_id,
        "run_name": run_name,
    }
    for key, expected in expected_identity.items():
        if launch.get(key) != expected:
            raise ContractError(f"original L1 {cell_id} launch {key} changed")
    for key in ("manifest_file_sha256", "launcher_file_sha256", "cell_id", "run_name"):
        if runtime_verified.get(key) != expected_identity[key]:
            raise ContractError(f"original L1 {cell_id} runtime {key} changed")
    try:
        command = foreign.build_command(
            foreign_manifest, "l1", cell_id, wbt=preflight["wbt"]
        )
    except Exception as exc:
        raise ContractError(f"cannot rebuild original L1 {cell_id} command: {exc}") from exc
    if launch.get("command") != command:
        raise ContractError(f"original L1 {cell_id} command drifted")
    state = parse_launch_state(paths["launch_state"])
    if not state.get("pid", "").isdigit() or state.get("pid") != state.get("pgid"):
        raise ContractError(f"original L1 {cell_id} lost PID=PGID")
    if process_entry_exists(int(state["pid"])):
        raise ContractError(f"original L1 {cell_id} process still exists")
    text = paths["training_log"].read_text(encoding="utf-8", errors="replace")
    if foreign.FAILURE_RE.search(text):
        raise ContractError(f"original L1 {cell_id} log contains a hard failure")
    try:
        foreign.verify_guidance_log(
            paths["training_log"], foreign.cell_by_id(foreign_manifest, cell_id)
        )
        contract_path = Path(runtime_verified["emitted_hard_contract_path"])
        contract_sha, _ = foreign.verify_emitted_contract(
            contract_path,
            foreign_manifest,
            hot=spec["initialization"] == "hot_parent",
        )
    except Exception as exc:
        raise ContractError(f"original L1 {cell_id} contract/log proof failed: {exc}") from exc
    if contract_sha != EXPECTED_HARD_CONTRACT_SHA256:
        raise ContractError(f"original L1 {cell_id} hard contract changed")
    terminal = spec["expected_terminal_checkpoint_iteration"]
    checkpoint = Path(spec["checkpoint_path"])
    if checkpoint.parent != Path(runtime_verified["training_run_dir"]):
        raise ContractError(f"original L1 {cell_id} checkpoint path differs from runtime evidence")
    if not checkpoint.is_file():
        raise ContractError(f"original L1 {cell_id} terminal checkpoint is missing")
    if checkpoint.stat().st_size != spec["checkpoint_bytes"]:
        raise ContractError(f"original L1 {cell_id} terminal checkpoint byte size changed")
    if sha256_file(checkpoint) != spec["checkpoint_sha256"]:
        raise ContractError(f"original L1 {cell_id} terminal checkpoint SHA changed")
    _stable_file(checkpoint, stability_delay)
    try:
        audit = foreign.checkpoint_audit(preflight["python"], checkpoint)
    except Exception as exc:
        raise ContractError(f"original L1 {cell_id} checkpoint audit failed: {exc}") from exc
    _check_audit(
        audit,
        terminal=terminal,
        lineage=spec["expected_lineage_exact"],
        contract_sha=contract_sha,
        label=f"original L1 {cell_id}",
    )
    if cell_id == "B":
        action = spec["post_terminal_exact_pgid_action"]
        action_path = Path(action["path"])
        if not action_path.is_file() or sha256_file(action_path) != action["sha256"]:
            raise ContractError("B exact-PGID post-terminal action evidence changed")
        action_text = action_path.read_text(encoding="utf-8", errors="replace")
        if (
            f"KILL_SENT exact_pgid={action['pgid']}" not in action_text
            or "GROUP_AFTER_BEGIN\nGROUP_AFTER_END" not in action_text
        ):
            raise ContractError("B post-terminal action evidence is semantically incomplete")
    return {
        "source": "original_v6",
        "run_name": run_name,
        "initialization": spec["initialization"],
        "expected_lineage_exact": spec["expected_lineage_exact"],
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": spec["checkpoint_sha256"],
        "checkpoint_audit": audit,
        "training_contract_path": str(contract_path),
        "training_contract_sha256": contract_sha,
        "launch_contract_sha256": spec["launch_contract_sha256"],
        "runtime_verified_sha256": spec["runtime_verified_sha256"],
        "launch_state_sha256": spec["launch_state_sha256"],
        "training_log_sha256": spec["training_log_sha256"],
    }


def audit_retry_terminal_d(
    manifest: dict[str, Any],
    foreign: ModuleType,
    foreign_manifest: dict[str, Any],
    preflight: dict[str, Any],
    *,
    config_sha: str,
    launcher_sha: str,
    stability_delay: float = 2.0,
) -> dict[str, Any]:
    runtime = manifest["runtime"]
    paths = {
        "launch_contract": Path(runtime["launch_contract_path"]),
        "runtime_verified": Path(runtime["runtime_verified_path"]),
        "launch_state": Path(runtime["launch_state_path"]),
        "training_log": Path(runtime["training_log_path"]),
    }
    if not all(path.is_file() for path in paths.values()):
        raise ContractError("v6r1 D runtime evidence is incomplete")
    launch = read_json(paths["launch_contract"], "v6r1 D launch contract")
    verified = read_json(paths["runtime_verified"], "v6r1 D runtime verified")
    expected_common = {
        "manifest_file_sha256": config_sha,
        "launcher_file_sha256": launcher_sha,
        "foreign_manifest_file_sha256": FOREIGN_CONFIG_SHA256,
        "foreign_launcher_file_sha256": FOREIGN_LAUNCHER_SHA256,
        "cell_id": "D",
        "run_name": NEW_RUN_NAME,
    }
    for key, expected in expected_common.items():
        if launch.get(key) != expected or verified.get(key) != expected:
            raise ContractError(f"v6r1 D evidence {key} mismatch")
    expected_launch_signal_policy = {
        "direct_signals_sent_by_retry_tool": False,
        "locked_launcher_sha256": LOCKED_LAUNCHER_SHA256,
        "locked_launcher_exact_pgid_boot_timeout_cleanup_allowed": True,
        "broad_signals_forbidden": True,
        "post_contract_timeout_requires_manual_exact_state_pgid_audit": True,
    }
    for key, expected in expected_launch_signal_policy.items():
        if launch.get(key) != expected:
            raise ContractError(f"v6r1 D launch signal policy {key} mismatch")
    if (
        verified.get("direct_signals_sent_by_retry_tool") is not False
        or verified.get("locked_launcher_sha256") != LOCKED_LAUNCHER_SHA256
        or verified.get("locked_launcher_exact_pgid_boot_timeout_cleanup_allowed")
        is not True
        or verified.get("locked_launcher_exact_pgid_boot_timeout_cleanup_executed")
        is not False
        or verified.get("broad_signals_forbidden") is not True
    ):
        raise ContractError("v6r1 D runtime signal provenance changed")
    try:
        original = foreign.build_command(
            foreign_manifest, "l1", "D", wbt=preflight["wbt"]
        )
    except Exception as exc:
        raise ContractError(f"cannot rebuild original D command during finalization: {exc}") from exc
    command = build_retry_command(original, OLD_RUN_NAME, NEW_RUN_NAME)
    if launch.get("command") != command:
        raise ContractError("v6r1 D launch command changed after launch")
    if verified.get("launch_contract_sha256") != sha256_file(paths["launch_contract"]):
        raise ContractError("v6r1 D runtime did not bind its launch contract")
    state = parse_launch_state(paths["launch_state"])
    if not state.get("pid", "").isdigit() or state.get("pid") != state.get("pgid"):
        raise ContractError("v6r1 D final state lost PID=PGID")
    if process_entry_exists(int(state["pid"])):
        raise ContractError("v6r1 D is still running; mixed finalizer is read-only")
    text = paths["training_log"].read_text(encoding="utf-8", errors="replace")
    if foreign.FAILURE_RE.search(text):
        raise ContractError("v6r1 D log contains a hard failure")
    if "Learning iteration 24/25" not in text:
        raise ContractError("v6r1 D did not log its natural terminal iteration 24/25")
    try:
        foreign.verify_guidance_log(
            paths["training_log"], foreign.cell_by_id(foreign_manifest, "D")
        )
        contract_path = Path(verified["emitted_hard_contract_path"])
        contract_sha, _ = foreign.verify_emitted_contract(
            contract_path, foreign_manifest, hot=False
        )
    except Exception as exc:
        raise ContractError(f"v6r1 D contract/log proof failed: {exc}") from exc
    if contract_sha != EXPECTED_HARD_CONTRACT_SHA256:
        raise ContractError("v6r1 D hard contract differs from original A/B/C")
    checkpoint = Path(verified["training_run_dir"]) / "model_24.pt"
    if not checkpoint.is_file():
        raise ContractError("v6r1 D terminal model_24.pt is missing")
    _stable_file(checkpoint, stability_delay)
    try:
        audit = foreign.checkpoint_audit(preflight["python"], checkpoint)
    except Exception as exc:
        raise ContractError(f"v6r1 D checkpoint audit failed: {exc}") from exc
    _check_audit(
        audit,
        terminal=24,
        lineage=True,
        contract_sha=contract_sha,
        label="v6r1 D",
    )
    return {
        "source": "v6r1_single_cell_retry",
        "run_name": NEW_RUN_NAME,
        "initialization": "fresh",
        "expected_lineage_exact": True,
        "natural_terminal_iteration": 24,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_audit": audit,
        "training_contract_path": str(contract_path),
        "training_contract_sha256": contract_sha,
        "launch_contract_sha256": sha256_file(paths["launch_contract"]),
        "runtime_verified_sha256": sha256_file(paths["runtime_verified"]),
        "launch_state_sha256": sha256_file(paths["launch_state"]),
        "training_log_sha256": sha256_file(paths["training_log"]),
        "direct_signals_sent_by_retry_tool": False,
        "locked_launcher_exact_pgid_boot_timeout_cleanup_allowed": True,
        "locked_launcher_exact_pgid_boot_timeout_cleanup_executed": False,
    }


def build_mixed_activation_content(
    manifest: dict[str, Any],
    *,
    config_sha: str,
    launcher_sha: str,
    cells: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if set(cells) != {"A", "B", "C", "D"}:
        raise ContractError("mixed activation must contain A/B/C/D exactly")
    expected_sources = manifest["mixed_finalizer"]["required_sources"]
    expected_lineage = manifest["mixed_finalizer"]["required_lineage_exact"]
    for cell_id in ("A", "B", "C", "D"):
        if cells[cell_id].get("source") != expected_sources[cell_id]:
            raise ContractError(f"mixed activation {cell_id} source mismatch")
        if cells[cell_id].get("expected_lineage_exact") is not expected_lineage[cell_id]:
            raise ContractError(f"mixed activation {cell_id} lineage mismatch")
        if cells[cell_id].get("training_contract_sha256") != EXPECTED_HARD_CONTRACT_SHA256:
            raise ContractError(f"mixed activation {cell_id} hard contract mismatch")
        require_sha(cells[cell_id].get("checkpoint_sha256"), f"cell {cell_id} checkpoint")
    if cells["D"].get("locked_launcher_exact_pgid_boot_timeout_cleanup_executed") is not False:
        raise ContractError("mixed activation D must prove the boot-timeout cleanup did not run")
    retry = manifest["retry_authority"]["old_outer_evidence"]
    return {
        "manifest_id": manifest["manifest_id"],
        "manifest_file_sha256": config_sha,
        "launcher_file_sha256": launcher_sha,
        "foreign_manifest_id": FOREIGN_MANIFEST_ID,
        "foreign_manifest_file_sha256": FOREIGN_CONFIG_SHA256,
        "foreign_launcher_file_sha256": FOREIGN_LAUNCHER_SHA256,
        "training_commit": TRAINING_COMMIT,
        "status": "l1_mixed_v6_abc_plus_v6r1_d_terminal_l2_blocked",
        "training_seed": 3,
        "emitted_hard_contract_sha256": EXPECTED_HARD_CONTRACT_SHA256,
        "cells": cells,
        "retry_lineage": {
            "old_cell": "D",
            "old_run_name": OLD_RUN_NAME,
            "new_run_name": NEW_RUN_NAME,
            "classification": manifest["retry_authority"]["classification"],
            "only_command_change": "run_name",
            "old_launch_contract_sha256": retry["launch_contract"]["sha256"],
            "old_launch_state_sha256": retry["launch_state"]["sha256"],
            "old_training_log_sha256": retry["training_log"]["sha256"],
            "old_timeout_diagnostic_sha256": retry["timeout_diagnostic"]["sha256"],
            "new_launch_contract_sha256": cells["D"]["launch_contract_sha256"],
            "new_runtime_verified_sha256": cells["D"]["runtime_verified_sha256"],
            "new_checkpoint_sha256": cells["D"]["checkpoint_sha256"],
        },
        "automatic_judge_launch": False,
        "l2_training_launch_authorized": False,
        "second_seed_authorized": False,
        "stop_or_promote_authorized": False,
        "signal_policy": {
            "direct_signals_sent_by_retry_tool": False,
            "locked_launcher_sha256": LOCKED_LAUNCHER_SHA256,
            "locked_launcher_exact_pgid_boot_timeout_cleanup_allowed": True,
            "broad_signals_forbidden": True,
            "retry_d_boot_timeout_cleanup_executed": cells["D"].get(
                "locked_launcher_exact_pgid_boot_timeout_cleanup_executed"
            ),
        },
        "real_robot_commands_forbidden": True,
    }


def finalize_mixed_l1(
    manifest: dict[str, Any],
    config_path: Path,
    launcher_path: Path,
    *,
    config_sha: str,
    launcher_sha: str,
) -> dict[str, Any]:
    verify_production_locations(manifest, config_path, launcher_path)
    foreign, foreign_manifest = load_foreign_runtime(manifest)
    preflight = foreign_runtime_preflight(manifest, foreign, foreign_manifest)
    verify_recovery_evidence(manifest)
    verify_original_checkpoint_audit(manifest)
    verify_failed_d_claim(manifest, foreign, foreign_manifest, preflight)
    verify_retry_outputs_absent(manifest, allow_runtime=True)
    cells = {
        cell_id: audit_original_terminal_cell(
            manifest, foreign, foreign_manifest, preflight, cell_id
        )
        for cell_id in ("A", "B", "C")
    }
    cells["D"] = audit_retry_terminal_d(
        manifest,
        foreign,
        foreign_manifest,
        preflight,
        config_sha=config_sha,
        launcher_sha=launcher_sha,
    )
    content = build_mixed_activation_content(
        manifest, config_sha=config_sha, launcher_sha=launcher_sha, cells=cells
    )
    artifact = {
        "artifact_kind": "phase1_signed_face_rescue_l1_mixed_activation",
        "schema_version": 1,
        "content": content,
        "content_sha256": canonical_sha256(content),
    }
    output = Path(manifest["runtime"]["mixed_activation_path"])
    write_json_exclusive(output, artifact)
    return {"path": str(output), "sha256": sha256_file(output), "content_sha256": artifact["content_sha256"]}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-launcher-sha256", required=True)
    parser.add_argument(
        "action", choices=("static-validate", "validate", "plan", "launch", "finalize-mixed-l1")
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.resolve()
    launcher_path = Path(__file__).resolve()
    config_sha = require_sha(args.expected_config_sha256, "expected config SHA")
    launcher_sha = require_sha(args.expected_launcher_sha256, "expected launcher SHA")
    if not config_path.is_file() or sha256_file(config_path) != config_sha:
        raise ContractError("v6r1 manifest file SHA mismatch")
    if sha256_file(launcher_path) != launcher_sha:
        raise ContractError("v6r1 launcher file SHA mismatch")
    manifest = load_manifest(config_path)
    if args.action == "static-validate":
        print(
            json.dumps(
                {
                    "status": "static_valid",
                    "manifest_id": manifest["manifest_id"],
                    "config_sha256": config_sha,
                    "launcher_sha256": launcher_sha,
                    "only_cell": "D",
                    "only_command_change": "run_name",
                },
                sort_keys=True,
            )
        )
        return 0
    if args.action in {"validate", "plan"}:
        foreign, foreign_manifest, preflight, failed, readiness = launch_readiness(
            manifest, config_path, launcher_path
        )
        if args.action == "validate":
            print(
                json.dumps(
                    {
                        "status": "runtime_validated_no_writes",
                        "source_commit": TRAINING_COMMIT,
                        "gpu": readiness["gpu"],
                        "kit_lock": readiness["kit_lock"],
                        "old_failure": {
                            key: value
                            for key, value in failed.items()
                            if key not in {"original_command", "retry_command"}
                        },
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.action == "plan":
            print(
                json.dumps(
                    {
                        "status": "plan_only_no_writes",
                        "cell_id": "D",
                        "old_run_name": OLD_RUN_NAME,
                        "new_run_name": NEW_RUN_NAME,
                        "original_command": failed["original_command"],
                        "retry_command": failed["retry_command"],
                        "command_delta": failed["command_delta"],
                        "automatic_judge_launch": False,
                        "l2_training_launch_authorized": False,
                        "second_seed_authorized": False,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
    if args.action == "launch":
        result = launch_retry(
            manifest,
            config_path,
            launcher_path,
            config_sha=config_sha,
            launcher_sha=launcher_sha,
        )
        print(json.dumps({"status": "v6r1_d_launched_verified", **result}, sort_keys=True))
        return 0
    result = finalize_mixed_l1(
        manifest,
        config_path,
        launcher_path,
        config_sha=config_sha,
        launcher_sha=launcher_sha,
    )
    print(json.dumps({"status": "mixed_l1_activation_written", **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(2)
