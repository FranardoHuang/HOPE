#!/usr/bin/env python3
"""Validate the B/C schema-2/FK inspection receipt and consume activation.

This is a dependency-light source gate.  It validates tracked bytes and the historical receipt;
it does not read private motion/ONNX payloads, create an attempt, invoke the materializer, or write
an output.  The separately reviewed operation command remains the only runtime consume entrypoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import sys
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ASSET_IDS = ("franco_backhand_loop_b", "franco_backhand_loop_c")
INSPECTION_COMMIT = "748b6d5fe24bfe58915c34d8dfe09f254f8e4957"
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


class ActivationContractError(ValueError):
    """The historical receipt or future activation is incomplete or changed."""


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


def require_regular_repo_binding(
    value: Any, label: str, *, repo_root: Path = REPO_ROOT, expected_path: str | None = None
) -> Path:
    binding = exact_keys(value, {"path", "bytes", "sha256"}, label)
    raw = binding["path"]
    if expected_path is not None and raw != expected_path:
        raise ActivationContractError(f"{label}.path must equal {expected_path}")
    if not isinstance(raw, str) or not raw or Path(raw).is_absolute() or ".." in Path(raw).parts:
        raise ActivationContractError(f"{label}.path must be a safe repo-relative path")
    path = repo_root / raw
    try:
        info = path.lstat()
    except OSError as exc:
        raise ActivationContractError(f"cannot stat {label}: {exc}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ActivationContractError(f"{label} must be a regular non-symlink file")
    size = binding["bytes"]
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0 or info.st_size != size:
        raise ActivationContractError(f"{label} byte binding changed")
    expected_sha = require_sha(binding["sha256"], f"{label}.sha256")
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise ActivationContractError(f"{label} SHA {actual_sha} != {expected_sha}")
    return path


def require_absolute_binding_shape(value: Any, label: str) -> Mapping[str, Any]:
    binding = exact_keys(value, {"path", "bytes", "sha256"}, label)
    if not isinstance(binding["path"], str) or not Path(binding["path"]).is_absolute():
        raise ActivationContractError(f"{label}.path must be absolute")
    if (
        isinstance(binding["bytes"], bool)
        or not isinstance(binding["bytes"], int)
        or binding["bytes"] <= 0
    ):
        raise ActivationContractError(f"{label}.bytes must be a positive integer")
    require_sha(binding["sha256"], f"{label}.sha256")
    return binding


def read_bound_json(path: Path, expected_sha: str, label: str) -> tuple[dict[str, Any], int, str]:
    require_sha(expected_sha, f"expected {label} SHA")
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ActivationContractError(f"cannot read {label} {path}: {exc}") from exc
    actual_sha = sha256_bytes(data)
    if actual_sha != expected_sha:
        raise ActivationContractError(f"{label} SHA {actual_sha} != {expected_sha}")
    return strict_json_bytes(data, label), len(data), actual_sha


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
    if receipt["schema_version"] != 1:
        raise ActivationContractError("receipt schema_version changed")
    if receipt["receipt_id"] != "motion-backhand-loop-bc-schema2-fk-runtime-inspection-20260714-v1":
        raise ActivationContractError("receipt identity changed")
    if receipt["status"] != "complete_exact_read_only_inspection_b_and_c_passed":
        raise ActivationContractError("receipt status is not an exact two-asset pass")
    if receipt["scope"] != (
        "Historical receipt for two exact CPU-only runtime inspections. It authorizes no write, "
        "simulation, training, L0/L1 audit, formal motion, deployment, or hardware action."
    ):
        raise ActivationContractError("receipt scope changed or overclaims")
    timing = exact_keys(
        receipt["timing"],
        {
            "exact_command_start_and_end_not_captured",
            "inspection_completed_before_observation_utc",
            "receipt_observation_utc",
        },
        "receipt timing",
    )
    if timing != {
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
        "inspection checkout",
    )
    if (
        checkout["path"] != SOURCE_CHECKOUT
        or checkout["commit"] != INSPECTION_COMMIT
        or checkout["detached"] is not True
        or checkout["clean_before"] is not True
        or checkout["clean_after"] is not True
    ):
        raise ActivationContractError("inspection checkout was not exact detached and clean")
    tracked = exact_keys(
        checkout["tracked_files"],
        {
            "consumer", "b_preregistration", "c_preregistration", "shared_runtime",
            "donor_metadata_expectation",
        },
        "inspection tracked files",
    )
    expected_tracked = {
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
    for name, (path, expected_sha) in expected_tracked.items():
        require_regular_repo_binding(tracked[name], name, repo_root=repo_root, expected_path=path)
        if tracked[name]["sha256"] != expected_sha:
            raise ActivationContractError(f"{name} expected inspection SHA changed")

    runtime = exact_keys(
        receipt["runtime"], {"failed_default_python_attempt", "successful_python"}, "runtime"
    )
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
    expected_success_runtime = {
        "executable": SUCCESS_PYTHON,
        "resolved_executable": "/usr/bin/python3.12",
        "executable_bytes": 8021824,
        "executable_sha256": "1d3cf64f97cadc79fdc6fe2496a21b7b456cb94211978cfef5a65f616af74fd5",
        "python_version": "3.12.3",
        "packages": {"numpy": "2.5.0", "onnxruntime": "1.27.0", "mujoco": "3.10.0"},
    }
    if runtime["successful_python"] != expected_success_runtime:
        raise ActivationContractError("successful runtime identity changed")

    donor = exact_keys(
        receipt["donor_onnx"],
        {"path", "bytes", "sha256", "required_metadata_subset_exact"},
        "donor ONNX",
    )
    if donor != {
        "path": DONOR_PATH,
        "bytes": 1325146,
        "sha256": DONOR_SHA,
        "required_metadata_subset_exact": True,
    }:
        raise ActivationContractError("donor runtime evidence changed")
    vendor = exact_keys(
        receipt["vendor_mjcf"],
        {
            "canonical_path", "canonical_bytes", "canonical_sha256", "closure", "model_loaded",
            "joint_and_body_names_exact", "dynamics_steps",
        },
        "vendor MJCF evidence",
    )
    if vendor != {
        "canonical_path": (
            "agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/"
            "a3_pingpong/a3_pingpong.xml"
        ),
        "canonical_bytes": 49107,
        "canonical_sha256": "2ab1cd31bffaaef979b4d9f35699bf1e6bec3a127be96c9266af131eee3feb97",
        "closure": {
            "file_count": 75,
            "xml_file_count": 1,
            "include_reference_count": 0,
            "unique_external_file_count": 74,
            "mesh_reference_count": 74,
            "total_bytes": 14127373,
            "manifest_sha256": "e0381752eab46013c08559b331abb261beaa88a207a3c2f1155ab00857b962de",
        },
        "model_loaded": True,
        "joint_and_body_names_exact": True,
        "dynamics_steps": 0,
    }:
        raise ActivationContractError("vendor MJCF receipt changed")

    assets = exact_keys(receipt["assets"], set(ASSET_IDS), "receipt assets")
    expected_frames = {ASSET_IDS[0]: 91, ASSET_IDS[1]: 98}
    expected_sources = {
        ASSET_IDS[0]: (
            "/workspace/codexschema/motion_video_intake_20260711/"
            "gmr_spatial_retarget_primary_v1/franco_backhand_loop_b_98e7b883b29d/"
            "franco_backhand_loop_b.98e7b883b29d.se2.gmr.pkl",
            27927,
            "278279125528c827e0a980389b040d54d16140620c59c67c878286be9d1c8ad6",
            "/workspace/codexschema/motion_video_intake_20260711/"
            "gmr_spatial_retarget_primary_v1/franco_backhand_loop_b_98e7b883b29d/"
            "materialization_report.json",
            4051,
            "a238c077524586b2f1181cd24cb84ee29aa985ab274cfb43292f3159c0daadf3",
        ),
        ASSET_IDS[1]: (
            "/workspace/codexschema/motion_video_intake_20260711/"
            "gmr_spatial_retarget_primary_v1/franco_backhand_loop_c_aa0c86fd3509/"
            "franco_backhand_loop_c.aa0c86fd3509.se2.gmr.pkl",
            30055,
            "0dd981a6d29c0c5321c905d1591a59fbb79763de6e43d92d4d76aefdc29ff48b",
            "/workspace/codexschema/motion_video_intake_20260711/"
            "gmr_spatial_retarget_primary_v1/franco_backhand_loop_c_aa0c86fd3509/"
            "materialization_report.json",
            4068,
            "b3b93d2cdb0a288f04aed764e5fdca92182cee625715a953600439088f59ff67",
        ),
    }
    for asset in ASSET_IDS:
        row = exact_keys(
            assets[asset],
            {
                "source_motion", "source_materialization_report", "output_root",
                "output_root_absent_before", "output_root_absent_after", "input_frames",
                "returncode", "stdout", "no_write",
            },
            f"receipt {asset}",
        )
        motion = require_absolute_binding_shape(row["source_motion"], f"{asset} source motion")
        report = require_absolute_binding_shape(
            row["source_materialization_report"], f"{asset} source report"
        )
        (
            motion_path,
            motion_bytes,
            motion_sha,
            report_path,
            report_bytes,
            report_sha,
        ) = expected_sources[asset]
        if (
            motion["path"] != motion_path
            or motion["bytes"] != motion_bytes
            or motion["sha256"] != motion_sha
        ):
            raise ActivationContractError(f"{asset} source motion receipt changed")
        if (
            report["path"] != report_path
            or report["bytes"] != report_bytes
            or report["sha256"] != report_sha
        ):
            raise ActivationContractError(f"{asset} source report receipt changed")
        frames = expected_frames[asset]
        expected_stdout = (
            f"[schema2-fk] PASS inspect asset={asset} frames={frames} "
            "donor_exact=true no_write=true"
        )
        if (
            row["output_root"] != OUTPUT_ROOTS[asset]
            or row["output_root_absent_before"] is not True
            or row["output_root_absent_after"] is not True
            or row["input_frames"] != frames
            or row["returncode"] != 0
            or row["stdout"] != expected_stdout
            or row["no_write"] is not True
        ):
            raise ActivationContractError(f"{asset} no-write PASS receipt changed")

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
        raise ActivationContractError("inspection honesty boundary changed")


def expected_command(asset: str) -> list[str]:
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
    activation: Mapping[str, Any], receipt: Mapping[str, Any], *, repo_root: Path = REPO_ROOT
) -> None:
    exact_keys(
        activation,
        {
            "schema_version", "activation_id", "status", "scope", "inspection_receipt",
            "validator", "source_checkout", "runtime", "donor_onnx", "assets",
            "execution_contract", "commands", "authorization",
        },
        "consume activation",
    )
    if activation["schema_version"] != 1:
        raise ActivationContractError("activation schema_version changed")
    if activation["activation_id"] != "motion-backhand-loop-bc-schema2-fk-consume-20260714-v1":
        raise ActivationContractError("activation identity changed")
    if activation["status"] != "review_required_not_consumed":
        raise ActivationContractError("activation must remain unconsumed and review-gated")
    if activation["scope"] != (
        "Source-only authorization for at most one no-clobber schema-2/FK consume attempt per "
        "inspected B/C asset after human review; no consume has run under this activation."
    ):
        raise ActivationContractError("activation scope changed or overclaims")
    receipt_path = require_regular_repo_binding(
        activation["inspection_receipt"],
        "inspection receipt binding",
        repo_root=repo_root,
        expected_path=RECEIPT_PATH,
    )
    if strict_json_bytes(receipt_path.read_bytes(), "bound inspection receipt") != receipt:
        raise ActivationContractError("activation-bound receipt differs from validated receipt")
    require_regular_repo_binding(
        activation["validator"],
        "activation validator binding",
        repo_root=repo_root,
        expected_path=VALIDATOR_PATH,
    )
    if activation["source_checkout"] != {
        "path": SOURCE_CHECKOUT,
        "commit": INSPECTION_COMMIT,
        "must_be_detached": True,
        "must_be_clean_before_and_after": True,
        "may_not_be_archive_or_live_a0": True,
    }:
        raise ActivationContractError("activation source checkout changed")
    success_runtime = receipt["runtime"]["successful_python"]
    if activation["runtime"] != success_runtime:
        raise ActivationContractError("activation runtime differs from passing inspection")
    if activation["donor_onnx"] != receipt["donor_onnx"]:
        raise ActivationContractError("activation donor differs from passing inspection")

    assets = exact_keys(activation["assets"], set(ASSET_IDS), "activation assets")
    for asset in ASSET_IDS:
        row = exact_keys(
            assets[asset],
            {
                "preregistration", "output_root", "output_motion_filename", "report_filename",
                "output_root_must_be_absent", "attempts_authorized", "attempts_started",
                "schema2_materialized",
            },
            f"activation {asset}",
        )
        require_regular_repo_binding(
            row["preregistration"],
            f"{asset} preregistration",
            repo_root=repo_root,
            expected_path=PLAN_PATHS[asset],
        )
        if row["preregistration"]["sha256"] != PLAN_SHAS[asset]:
            raise ActivationContractError(f"{asset} preregistration SHA changed")
        if row != {
            "preregistration": row["preregistration"],
            "output_root": OUTPUT_ROOTS[asset],
            "output_motion_filename": OUTPUT_FILENAMES[asset],
            "report_filename": "schema2_fk_report.json",
            "output_root_must_be_absent": True,
            "attempts_authorized": 1,
            "attempts_started": 0,
            "schema2_materialized": False,
        }:
            raise ActivationContractError(f"{asset} one-attempt activation changed")

    if activation["execution_contract"] != {
        "mode": "CPU-only exact schema-2/FK materialization",
        "asset_parallelism": 1,
        "hope_frame": "off",
        "rehash_all_bound_inputs_before_each_attempt": True,
        "no_clobber": True,
        "report_published_last": True,
        "automatic_retry": False,
        "failure_action": "stop_that_asset_preserve_evidence_require_new_human_review",
        "operator_review_required_before_first_attempt": True,
        "activation_does_not_override_plan_failure_policy": True,
    }:
        raise ActivationContractError("activation execution contract changed")
    commands = exact_keys(activation["commands"], set(ASSET_IDS), "activation commands")
    for asset in ASSET_IDS:
        command = exact_keys(commands[asset], {"cwd", "environment", "argv"}, f"{asset} command")
        if command != {
            "cwd": SOURCE_CHECKOUT,
            "environment": {"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"},
            "argv": expected_command(asset),
        }:
            raise ActivationContractError(f"{asset} exact consume command changed")
    if activation["authorization"] != {
        "schema2_fk_consume_once_per_asset": True,
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--expected-receipt-sha256", required=True)
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--expected-activation-sha256", required=True)
    parser.add_argument("command", choices=("static",))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        receipt, _receipt_bytes, _receipt_sha = read_bound_json(
            args.receipt.resolve(), args.expected_receipt_sha256, "inspection receipt"
        )
        activation, _activation_bytes, _activation_sha = read_bound_json(
            args.activation.resolve(), args.expected_activation_sha256, "consume activation"
        )
        validate_receipt(receipt)
        validate_activation(activation, receipt)
        print(
            "[schema2-fk-activation] PASS static receipt_exact=true assets=2 "
            "attempts_started=0 consume_not_run=true"
        )
        return 0
    except (ActivationContractError, OSError, TypeError, ValueError) as exc:
        print(f"[schema2-fk-activation] FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
