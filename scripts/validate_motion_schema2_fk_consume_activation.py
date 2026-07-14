#!/usr/bin/env python3
"""Dependency-light source gate for the B/C schema-2/FK one-shot runner.

This validator binds the historical no-write inspection receipt, v2 activation, and exact runner.
It has no consume or subprocess-launch mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ASSET_IDS = ("franco_backhand_loop_b", "franco_backhand_loop_c")
INSPECTION_COMMIT = "748b6d5fe24bfe58915c34d8dfe09f254f8e4957"
ACTIVATION_CONTRACT_COMMIT = "047b0d8ac5b7df48400b33b91092bd1749651012"
SOURCE_CHECKOUT = "/workspace/codexschema/nohope_schema2_fk_inspect_748b6d5"
SUCCESS_PYTHON = "/workspace/hope_mjeval_venv/bin/python"
DONOR_PATH = (
    "/workspace/codexschema/gate3_face179_b5762fa/isolated_assets/"
    "formal_sz_seed3_model2000_11f3a288/exported_2fa3534/policy.onnx"
)
DONOR_SHA = "0c428ddf9968b047acbe7bbd5a39069a8e661ab0421038ea3b635284deb7b155"
RECEIPT_PATH = (
    "configs/motion_backhand_loop_bc_schema2_fk_runtime_inspection_receipt_20260714.json"
)
VALIDATOR_PATH = "scripts/validate_motion_schema2_fk_consume_activation.py"
RUNNER_PATH = "scripts/run_motion_schema2_fk_consume_once.py"
TOOL_PATH = "scripts/materialize_motion_schema2_fk.py"
PLAN_PATHS = {
    "franco_backhand_loop_b": "configs/motion_backhand_loop_b_schema2_fk_prereg_20260714.json",
    "franco_backhand_loop_c": "configs/motion_backhand_loop_c_schema2_fk_prereg_20260714.json",
}
PLAN_SHAS = {
    "franco_backhand_loop_b": "3d71cc02c6ae68d0ecedf280e8341d763ad39ec0aac1757367c9719e761d33ae",
    "franco_backhand_loop_c": "662b8c4c0851d2f6d9d5c23313dc0c27334528a2b5fb2b62ad90bc3447257e31",
}
OUTPUT_ROOTS = {
    "franco_backhand_loop_b": (
        "/workspace/codexschema/motion_video_intake_20260711/schema2_fk_primary_v1/"
        "franco_backhand_loop_b_98e7b883b29d"
    ),
    "franco_backhand_loop_c": (
        "/workspace/codexschema/motion_video_intake_20260711/schema2_fk_primary_v1/"
        "franco_backhand_loop_c_aa0c86fd3509"
    ),
}
OUTPUT_FILENAMES = {
    "franco_backhand_loop_b": "franco_backhand_loop_b.98e7b883b29d.schema2_fk.npz",
    "franco_backhand_loop_c": "franco_backhand_loop_c.aa0c86fd3509.schema2_fk.npz",
}
CONTROL_ROOT = (
    "/workspace/codexschema/motion_video_intake_20260711/schema2_fk_primary_v1/"
    ".bc_schema2_fk_consume_control_v2"
)


class ActivationContractError(ValueError):
    """The receipt, activation, runner binding, or authorization changed."""


def _reject_duplicate_json_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ActivationContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json_bytes(data: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ActivationContractError(f"non-finite JSON constant in {label}: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ActivationContractError(f"invalid {label} JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ActivationContractError(f"{label} must be a JSON object")
    return value


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ActivationContractError(f"{label} must be a lowercase SHA-256")
    return value


def exact_keys(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ActivationContractError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        raise ActivationContractError(
            f"{label} keys changed: missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}"
        )
    return value


def _reject_symlink_components(path: Path, label: str) -> None:
    probe = path
    while probe != probe.parent:
        if os.path.lexists(probe) and probe.is_symlink():
            raise ActivationContractError(f"{label} contains symlink component: {probe}")
        probe = probe.parent


def require_regular_repo_binding(
    value: Any, label: str, *, repo_root: Path = REPO_ROOT, expected_path: str | None = None
) -> Path:
    binding = exact_keys(value, {"path", "bytes", "sha256"}, label)
    raw = binding["path"]
    if expected_path is not None and raw != expected_path:
        raise ActivationContractError(f"{label}.path must equal {expected_path}")
    if not isinstance(raw, str) or not raw or Path(raw).is_absolute() or ".." in Path(raw).parts:
        raise ActivationContractError(f"{label}.path must be safe and repo-relative")
    path = repo_root / raw
    _reject_symlink_components(path, label)
    try:
        info = path.stat()
    except OSError as exc:
        raise ActivationContractError(f"cannot stat {label}: {exc}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise ActivationContractError(f"{label} must be a regular file")
    if (
        isinstance(binding["bytes"], bool)
        or not isinstance(binding["bytes"], int)
        or info.st_size != binding["bytes"]
    ):
        raise ActivationContractError(f"{label} byte binding changed")
    expected_sha = require_sha(binding["sha256"], f"{label}.sha256")
    if sha256_file(path) != expected_sha:
        raise ActivationContractError(f"{label} SHA binding changed")
    return path


def require_historical_repo_binding(
    value: Any,
    label: str,
    *,
    repo_root: Path,
    commit: str,
    expected_path: str,
) -> None:
    """Verify immutable source provenance from the recorded commit, not the current file."""

    binding = exact_keys(value, {"path", "bytes", "sha256"}, label)
    if binding["path"] != expected_path:
        raise ActivationContractError(f"{label}.path must equal {expected_path}")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ActivationContractError(f"{label} historical commit is malformed")
    environment = os.environ.copy()
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    tree = subprocess.run(
        [
            "git", "--no-optional-locks", "-C", str(repo_root), "ls-tree",
            commit, "--", expected_path,
        ],
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    fields = tree.stdout.decode("utf-8", errors="strict").strip().split(None, 3)
    if tree.returncode != 0 or len(fields) != 4 or fields[0] not in {"100644", "100755"}:
        raise ActivationContractError(f"cannot prove regular historical {label}")
    blob = subprocess.run(
        [
            "git", "--no-optional-locks", "-C", str(repo_root), "show",
            f"{commit}:{expected_path}",
        ],
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if blob.returncode != 0:
        raise ActivationContractError(f"cannot read historical {label}")
    payload = blob.stdout
    if (
        type(binding["bytes"]) is not int
        or len(payload) != binding["bytes"]
        or hashlib.sha256(payload).hexdigest() != binding["sha256"]
    ):
        raise ActivationContractError(f"historical {label} content binding changed")


def read_bound_json(path: Path, expected_sha: str, label: str) -> tuple[dict[str, Any], int, str]:
    require_sha(expected_sha, f"expected {label} SHA")
    _reject_symlink_components(path, label)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ActivationContractError(f"cannot read {label}: {exc}") from exc
    actual_sha = sha256_bytes(data)
    if actual_sha != expected_sha:
        raise ActivationContractError(f"{label} SHA {actual_sha} != {expected_sha}")
    return strict_json_bytes(data, label), len(data), actual_sha


def _absolute_binding(value: Any, label: str) -> Mapping[str, Any]:
    binding = exact_keys(value, {"path", "bytes", "sha256"}, label)
    if not isinstance(binding["path"], str) or not Path(binding["path"]).is_absolute():
        raise ActivationContractError(f"{label}.path must be absolute")
    if (
        isinstance(binding["bytes"], bool)
        or not isinstance(binding["bytes"], int)
        or binding["bytes"] <= 0
    ):
        raise ActivationContractError(f"{label}.bytes must be positive")
    require_sha(binding["sha256"], f"{label}.sha256")
    return binding


def validate_receipt(receipt: Mapping[str, Any], *, repo_root: Path = REPO_ROOT) -> None:
    exact_keys(
        receipt,
        {
            "schema_version", "receipt_id", "status", "scope", "timing", "pod",
            "inspection_checkout", "runtime", "donor_onnx", "vendor_mjcf", "assets",
            "honesty_boundary",
        },
        "inspection receipt",
    )
    if (
        receipt["schema_version"] != 1
        or receipt["receipt_id"]
        != "motion-backhand-loop-bc-schema2-fk-runtime-inspection-20260714-v1"
        or receipt["status"] != "complete_exact_read_only_inspection_b_and_c_passed"
    ):
        raise ActivationContractError("inspection receipt identity/status changed")
    if receipt["timing"] != {
        "exact_command_start_and_end_not_captured": True,
        "inspection_completed_before_observation_utc": "2026-07-13T20:19:37Z",
        "receipt_observation_utc": "2026-07-13T20:19:37Z",
    }:
        raise ActivationContractError("receipt timing honesty changed")
    if receipt["pod"] != {"name": "pod1", "ssh_endpoint": "root@162.43.172.171:18333"}:
        raise ActivationContractError("receipt Pod identity changed")
    checkout = exact_keys(
        receipt["inspection_checkout"],
        {"path", "commit", "detached", "clean_before", "clean_after", "tracked_files"},
        "receipt checkout",
    )
    if (
        checkout["path"] != SOURCE_CHECKOUT
        or checkout["commit"] != INSPECTION_COMMIT
        or checkout["detached"] is not True
        or checkout["clean_before"] is not True
        or checkout["clean_after"] is not True
    ):
        raise ActivationContractError("receipt checkout is not exact detached/clean")
    tracked = exact_keys(
        checkout["tracked_files"],
        {
            "consumer", "b_preregistration", "c_preregistration", "shared_runtime",
            "donor_metadata_expectation",
        },
        "receipt tracked files",
    )
    expected = {
        "consumer": (TOOL_PATH, "33cf23eecff514a0e89bfe245db5b63470c4cd1dc9a433d0b920dfd84b9caebd"),
        "b_preregistration": (PLAN_PATHS[ASSET_IDS[0]], PLAN_SHAS[ASSET_IDS[0]]),
        "c_preregistration": (PLAN_PATHS[ASSET_IDS[1]], PLAN_SHAS[ASSET_IDS[1]]),
        "shared_runtime": (
            "configs/motion_backhand_loop_bc_schema2_fk_runtime_v1.json",
            "3d32b146e72029960ebf9cb2777f484804dafc87097e9cd3d0513dc277eed6e8",
        ),
        "donor_metadata_expectation": (
            "configs/a3_schema2_fk_donor_metadata_v1.json",
            "0e55b78806e3a5ae26f3ace6df067ec5cd145ed985a778f7962e4bfe24116f1c",
        ),
    }
    for name, (path, expected_sha) in expected.items():
        require_regular_repo_binding(tracked[name], name, repo_root=repo_root, expected_path=path)
        if tracked[name]["sha256"] != expected_sha:
            raise ActivationContractError(f"receipt {name} identity changed")
    runtime = receipt["runtime"]
    if runtime["failed_default_python_attempt"] != {
        "executable": "/usr/bin/python3",
        "resolved_executable": "/usr/bin/python3.12",
        "executable_bytes": 8021824,
        "executable_sha256": "1d3cf64f97cadc79fdc6fe2496a21b7b456cb94211978cfef5a65f616af74fd5",
        "python_version": "3.12.3",
        "packages": {"numpy": "2.1.2", "onnxruntime": None, "mujoco": None},
        "asset_id": ASSET_IDS[0],
        "returncode": 2,
        "failure": "onnxruntime is required for runtime inspection",
        "writes_observed": False,
        "accepted_as_inspection": False,
    }:
        raise ActivationContractError("default-Python fail-closed evidence changed")
    if runtime["successful_python"] != {
        "executable": SUCCESS_PYTHON,
        "resolved_executable": "/usr/bin/python3.12",
        "executable_bytes": 8021824,
        "executable_sha256": "1d3cf64f97cadc79fdc6fe2496a21b7b456cb94211978cfef5a65f616af74fd5",
        "python_version": "3.12.3",
        "packages": {"numpy": "2.5.0", "onnxruntime": "1.27.0", "mujoco": "3.10.0"},
    }:
        raise ActivationContractError("successful runtime receipt changed")
    donor = exact_keys(
        receipt["donor_onnx"],
        {"path", "bytes", "sha256", "required_metadata_subset_exact"},
        "receipt donor",
    )
    if donor != {
        "path": DONOR_PATH,
        "bytes": 1325146,
        "sha256": DONOR_SHA,
        "required_metadata_subset_exact": True,
    }:
        raise ActivationContractError("receipt donor changed")
    vendor = receipt["vendor_mjcf"]
    if (
        vendor.get("canonical_sha256")
        != "2ab1cd31bffaaef979b4d9f35699bf1e6bec3a127be96c9266af131eee3feb97"
        or vendor.get("closure")
        != {
            "file_count": 75,
            "xml_file_count": 1,
            "include_reference_count": 0,
            "unique_external_file_count": 74,
            "mesh_reference_count": 74,
            "total_bytes": 14127373,
            "manifest_sha256": "e0381752eab46013c08559b331abb261beaa88a207a3c2f1155ab00857b962de",
        }
        or vendor.get("model_loaded") is not True
        or vendor.get("joint_and_body_names_exact") is not True
        or vendor.get("dynamics_steps") != 0
    ):
        raise ActivationContractError("vendor MJCF receipt changed")
    assets = exact_keys(receipt["assets"], set(ASSET_IDS), "receipt assets")
    for asset, frames in zip(ASSET_IDS, (91, 98), strict=True):
        row = assets[asset]
        _absolute_binding(row["source_motion"], f"{asset} source motion")
        _absolute_binding(row["source_materialization_report"], f"{asset} source report")
        if (
            row["output_root"] != OUTPUT_ROOTS[asset]
            or row["output_root_absent_before"] is not True
            or row["output_root_absent_after"] is not True
            or row["input_frames"] != frames
            or row["returncode"] != 0
            or row["no_write"] is not True
            or row["stdout"]
            != (
                f"[schema2-fk] PASS inspect asset={asset} frames={frames} "
                "donor_exact=true no_write=true"
            )
        ):
            raise ActivationContractError(f"{asset} no-write receipt changed")
    if receipt["honesty_boundary"] != {
        "restricted_pickle_loaded": True,
        "donor_metadata_reextracted": True,
        "vendor_model_loaded": True,
        "forward_kinematics_trajectory_evaluated": False,
        "schema2_npz_written": False,
        "simulation_run": False,
        "training_run": False,
        "hardware_run": False,
        "motion_certificate_count": 0,
        "only_unlocked_next_step": (
            "reviewed_one_attempt_no_clobber_schema2_fk_consume_activation_per_asset"
        ),
    }:
        raise ActivationContractError("receipt honesty boundary changed")


def expected_child_command(asset: str) -> list[str]:
    peer = ASSET_IDS[1] if asset == ASSET_IDS[0] else ASSET_IDS[0]
    return [
        SUCCESS_PYTHON,
        f"{SOURCE_CHECKOUT}/{TOOL_PATH}",
        "--prereg",
        f"{SOURCE_CHECKOUT}/{PLAN_PATHS[asset]}",
        "--expected-prereg-sha256",
        PLAN_SHAS[asset],
        "--peer-prereg",
        f"{SOURCE_CHECKOUT}/{PLAN_PATHS[peer]}",
        "--expected-peer-prereg-sha256",
        PLAN_SHAS[peer],
        "--hope_frame",
        "off",
        "--donor",
        DONOR_PATH,
        "consume",
    ]


def validate_activation(
    activation: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
    historical_binding_commit: str | None = None,
) -> None:
    exact_keys(
        activation,
        {
            "schema_version", "activation_id", "status", "scope", "inspection_receipt",
            "source_gate_validator", "runner", "source_checkout", "runtime", "donor_onnx",
            "assets", "control", "execution_contract", "commands", "authorization",
        },
        "v2 activation",
    )
    if (
        activation["schema_version"] != 2
        or activation["activation_id"]
        != "motion-backhand-loop-bc-schema2-fk-consume-20260714-v2"
        or activation["status"] != "review_required_runner_not_executed"
    ):
        raise ActivationContractError("v2 activation identity/status changed")
    if activation["scope"] != (
        "Source-only v2 authorization for one irreversible claimed schema-2/FK child attempt per "
        "inspected B/C asset after human review; the runner has not executed on Pod."
    ):
        raise ActivationContractError("v2 activation scope changed")
    receipt_path = require_regular_repo_binding(
        activation["inspection_receipt"], "receipt binding", repo_root=repo_root,
        expected_path=RECEIPT_PATH,
    )
    if strict_json_bytes(receipt_path.read_bytes(), "bound receipt") != receipt:
        raise ActivationContractError("activation-bound receipt differs")
    if historical_binding_commit is None:
        require_regular_repo_binding(
            activation["source_gate_validator"],
            "source validator",
            repo_root=repo_root,
            expected_path=VALIDATOR_PATH,
        )
        require_regular_repo_binding(
            activation["runner"],
            "one-shot runner",
            repo_root=repo_root,
            expected_path=RUNNER_PATH,
        )
    else:
        if historical_binding_commit != ACTIVATION_CONTRACT_COMMIT:
            raise ActivationContractError(
                "portable historical activation commit is not the frozen contract commit"
            )
        require_historical_repo_binding(
            activation["source_gate_validator"],
            "source validator",
            repo_root=repo_root,
            commit=historical_binding_commit,
            expected_path=VALIDATOR_PATH,
        )
        require_historical_repo_binding(
            activation["runner"],
            "one-shot runner",
            repo_root=repo_root,
            commit=historical_binding_commit,
            expected_path=RUNNER_PATH,
        )
    if activation["source_checkout"] != {
        "path": SOURCE_CHECKOUT,
        "commit": INSPECTION_COMMIT,
        "must_be_detached": True,
        "must_be_clean_before_and_after": True,
        "may_not_be_archive_or_live_a0": True,
    }:
        raise ActivationContractError("source checkout contract changed")
    expected_runtime = {
        "executable": SUCCESS_PYTHON,
        "resolved_executable": "/usr/bin/python3.12",
        "executable_bytes": 8021824,
        "executable_sha256": "1d3cf64f97cadc79fdc6fe2496a21b7b456cb94211978cfef5a65f616af74fd5",
        "python_version": "3.12.3",
        "prefix": "/workspace/hope_mjeval_venv",
        "base_prefix": "/usr",
        "packages": {"numpy": "2.5.0", "onnxruntime": "1.27.0", "mujoco": "3.10.0"},
        "module_origins": {
            "numpy": "/workspace/hope_mjeval_venv/lib/python3.12/site-packages/numpy/__init__.py",
            "onnxruntime": (
                "/workspace/hope_mjeval_venv/lib/python3.12/site-packages/"
                "onnxruntime/__init__.py"
            ),
            "mujoco": "/workspace/hope_mjeval_venv/lib/python3.12/site-packages/mujoco/__init__.py",
        },
        "environment_overrides": {
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        },
    }
    if activation["runtime"] != expected_runtime:
        raise ActivationContractError("runtime executable/package/module origins changed")
    for key in (
        "executable", "resolved_executable", "executable_bytes", "executable_sha256",
        "python_version", "packages",
    ):
        if activation["runtime"][key] != receipt["runtime"]["successful_python"][key]:
            raise ActivationContractError("activation runtime differs from inspected runtime")
    if activation["donor_onnx"] != receipt["donor_onnx"]:
        raise ActivationContractError("activation donor differs from receipt")
    assets = exact_keys(activation["assets"], set(ASSET_IDS), "activation assets")
    for asset in ASSET_IDS:
        row = exact_keys(
            assets[asset],
            {
                "preregistration", "source_motion", "source_materialization_report", "output_root",
                "output_motion_filename", "report_filename", "claim_path", "failure_ledger_path",
                "success_ledger_path", "output_root_must_be_absent", "attempts_authorized",
                "attempts_started", "schema2_materialized",
            },
            f"activation {asset}",
        )
        require_regular_repo_binding(
            row["preregistration"], f"{asset} preregistration", repo_root=repo_root,
            expected_path=PLAN_PATHS[asset],
        )
        receipt_row = receipt["assets"][asset]
        if row != {
            "preregistration": row["preregistration"],
            "source_motion": receipt_row["source_motion"],
            "source_materialization_report": receipt_row["source_materialization_report"],
            "output_root": OUTPUT_ROOTS[asset],
            "output_motion_filename": OUTPUT_FILENAMES[asset],
            "report_filename": "schema2_fk_report.json",
            "claim_path": f"{CONTROL_ROOT}/{asset}.claim.json",
            "failure_ledger_path": f"{CONTROL_ROOT}/{asset}.failure.json",
            "success_ledger_path": f"{CONTROL_ROOT}/{asset}.success.json",
            "output_root_must_be_absent": True,
            "attempts_authorized": 1,
            "attempts_started": 0,
            "schema2_materialized": False,
        }:
            raise ActivationContractError(f"{asset} one-shot contract changed")
    if activation["control"] != {
        "root": CONTROL_ROOT,
        "shared_flock_path": f"{CONTROL_ROOT}/bc.shared.lock",
        "asset_parallelism": 1,
        "claim_publish": "atomic_hardlink_no_replace_before_child",
        "claim_is_permanent_after_any_child_outcome": True,
        "failure_ledger_is_permanent": True,
        "success_ledger_is_completion_last": True,
    }:
        raise ActivationContractError("shared control/flock contract changed")
    if activation["execution_contract"] != {
        "mode": "CPU-only exact schema-2/FK materialization",
        "asset_parallelism": 1,
        "hope_frame": "off",
        "shared_flock_required": True,
        "runtime_revalidation_before_claim": True,
        "rehash_checkout_runner_activation_receipt_inputs_and_mjcf_before_claim": True,
        "claim_required_before_child_start": True,
        "claim_is_irreversible_even_if_child_or_validation_fails": True,
        "child_execution": "synchronous_setsid_capture_stdout_stderr_returncode",
        "no_clobber": True,
        "failure_ledger_permanent": True,
        "failure_cleanup_grants_retry": False,
        "success_ledger_published_last": True,
        "formal_result_requires_claim_runner_activation_receipt_npz_and_report": True,
        "direct_materializer_consume_forbidden": True,
        "direct_output_without_success_ledger_accepted": False,
        "automatic_retry": False,
        "failure_action": "stop_that_asset_preserve_evidence_require_new_human_review",
        "operator_review_required_before_first_attempt": True,
        "activation_does_not_override_plan_failure_policy": True,
    }:
        raise ActivationContractError("one-shot execution contract changed")
    commands = exact_keys(activation["commands"], set(ASSET_IDS), "activation commands")
    for asset in ASSET_IDS:
        command = exact_keys(
            commands[asset], {"child_argv", "expected_inspect_stdout"}, f"{asset} command"
        )
        frames = receipt["assets"][asset]["input_frames"]
        if command != {
            "child_argv": expected_child_command(asset),
            "expected_inspect_stdout": (
                f"[schema2-fk] PASS inspect asset={asset} frames={frames} "
                "donor_exact=true no_write=true"
            ),
        }:
            raise ActivationContractError(f"{asset} child command changed")
    if activation["authorization"] != {
        "runner_preflight_authorized": True,
        "schema2_fk_one_shot_runner_authorized_after_review": True,
        "direct_materializer_consume_authorized": False,
        "schema2_materialized": False,
        "l0_authorized": False,
        "vendor_l1_authorized": False,
        "table_net_authorized": False,
        "dynamics_authorized": False,
        "simulator_authorized": False,
        "training_authorized": False,
        "formal_motion_authorized": False,
        "hardware_authorized": False,
    }:
        raise ActivationContractError("activation over-authorizes a downstream gate")


def load_validated_contract(
    activation_path: Path,
    expected_activation_sha: str,
    *,
    repo_root: Path = REPO_ROOT,
    historical_binding_commit: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    activation, activation_bytes, activation_sha = read_bound_json(
        activation_path, expected_activation_sha, "consume activation"
    )
    receipt_binding = exact_keys(
        activation.get("inspection_receipt"), {"path", "bytes", "sha256"},
        "inspection receipt binding",
    )
    receipt_path = repo_root / str(receipt_binding["path"])
    receipt, receipt_bytes, _receipt_sha = read_bound_json(
        receipt_path, str(receipt_binding["sha256"]), "inspection receipt"
    )
    if receipt_bytes != receipt_binding["bytes"]:
        raise ActivationContractError("inspection receipt byte binding changed")
    validate_receipt(receipt, repo_root=repo_root)
    validate_activation(
        activation,
        receipt,
        repo_root=repo_root,
        historical_binding_commit=historical_binding_commit,
    )
    return activation, receipt, {
        "path": str(activation_path), "bytes": activation_bytes, "sha256": activation_sha
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--expected-activation-sha256", required=True)
    parser.add_argument("command", choices=("static",))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        activation, _receipt, _meta = load_validated_contract(
            args.activation.resolve(),
            args.expected_activation_sha256,
            historical_binding_commit=ACTIVATION_CONTRACT_COMMIT,
        )
        print(
            "[schema2-fk-activation] PASS static schema=v2 runner_exact=true "
            f"assets={len(activation['assets'])} attempts_started=0 pod_run=false"
        )
        return 0
    except (ActivationContractError, OSError, TypeError, ValueError) as exc:
        print(f"[schema2-fk-activation] FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
