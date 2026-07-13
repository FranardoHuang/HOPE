#!/usr/bin/env python3
"""Static v6r2 correction for the superseded v6r1 validator.

This consumer validates tracked source bytes only.  It does not inspect remote
runtime state, reconstruct a training command, manage a process, launch a
trainer, or finalize an experiment.  Repeated fourth-cell Kit/scene-start boot
timeouts require root-cause work and a new v6r3-or-later preregistration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


CURRENT_MANIFEST_ID = (
    "phase1-signed-face-rescue-d-validator-correction-20260714-v6r2"
)
EXPECTED_MANIFEST_CONTENT_SHA256 = (
    "a9d2338621d5af74d5a91ffd16220cb238700eb698df0c72156da8a90a96e082"
)
V6R1_MANIFEST_ID = "phase1-signed-face-rescue-d-single-cell-retry-20260713-v6r1"
V6R1_CONFIG_SHA256 = "e0a677ec1b8adf328a5d73e74206b54f60e84bd22fb37f02b428511c0109ae90"
V6R1_CONSUMER_SHA256 = "1f03823eefad888fb1a9484af5349afec349dfa43be144aa10af33eaacee3a10"
V8_FAILURE_SHA256 = "0e5bb13b072c2c1e029a93e9d2d037adbfc1d74da9e1c9450acea333173f98a9"
V8_FINAL_STATE_SHA256 = (
    "80939e6df8184590df121f5edbf9f2de9188116438b6b328abd63152cdd2c90e"
)


class ContractError(RuntimeError):
    """A frozen v6r2 source invariant was violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{label} must contain one JSON object")
    return value


def require_regular_file(path: Path, label: str, expected_sha: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"{label} must be one existing regular non-symlink file")
    if sha256_file(path) != expected_sha:
        raise ContractError(f"{label} SHA changed")


def load_manifest(path: Path) -> dict[str, Any]:
    data = read_json(path, "v6r2 manifest")
    if canonical_sha256(data) != EXPECTED_MANIFEST_CONTENT_SHA256:
        raise ContractError("v6r2 manifest semantic content changed")
    if data.get("manifest_id") != CURRENT_MANIFEST_ID:
        raise ContractError("unexpected v6r2 manifest identity")

    correction = data.get("superseded_validator_correction", {})
    if (
        correction.get("superseded_manifest_id") != V6R1_MANIFEST_ID
        or correction.get("superseded_config", {}).get("sha256")
        != V6R1_CONFIG_SHA256
        or correction.get("superseded_consumer", {}).get("sha256")
        != V6R1_CONSUMER_SHA256
        or correction.get("historical_bug_correction", {}).get(
            "correct_required_state"
        )
        != "absent"
        or correction.get("historical_bug_correction", {})
        .get("immutable_checkpoint_audit", {})
        .get("d_row_run_dirs")
        != []
    ):
        raise ContractError("superseded v6r1 correction changed")

    barrier = data.get("foreign_v8_terminal_barrier", {})
    if (
        barrier.get("serial_launcher_cell_ordinal_zero_based") != 3
        or barrier.get("preceding_cell_c_terminal_checkpoint") != "model_24.pt"
        or barrier.get("failure_evidence", {}).get("sha256")
        != V8_FAILURE_SHA256
        or barrier.get("final_launch_state_evidence", {}).get("sha256")
        != V8_FINAL_STATE_SHA256
        or barrier.get("failure_scope")
        != "pre_contract_boot_not_learning_or_recipe_result"
        or barrier.get("automatic_retry_forbidden") is not True
        or barrier.get("boot_root_cause_and_new_preregistration_required")
        is not True
    ):
        raise ContractError("foreign-v8 terminal barrier changed")

    for key, expected in {
        "simulation_only": True,
        "real_robot_commands_forbidden": True,
        "direct_signals_forbidden": True,
        "broad_signals_forbidden": True,
        "automatic_retry_forbidden": True,
        "runtime_preflight_authorized": False,
        "launch_authorized": False,
        "finalize_authorized": False,
        "automatic_judge_launch": False,
        "l2_training_launch_authorized": False,
        "second_seed_authorized": False,
    }.items():
        if data.get(key) is not expected:
            raise ContractError("v6r2 authorization boundary changed")
    if data.get("next_step", {}).get("v6r2_must_not_be_reused_as_launcher") is not True:
        raise ContractError("v6r2 future-use boundary changed")
    return data


def require_sha(value: str, label: str) -> str:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ContractError(f"{label} must be one lowercase SHA-256")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-validator-sha256", required=True)
    parser.add_argument(
        "action", choices=("static-validate", "validate", "plan", "launch", "finalize")
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.resolve()
    validator_path = Path(__file__).resolve()
    config_sha = require_sha(args.expected_config_sha256, "expected config SHA")
    validator_sha = require_sha(
        args.expected_validator_sha256, "expected validator SHA"
    )
    require_regular_file(config_path, "v6r2 manifest", config_sha)
    require_regular_file(validator_path, "v6r2 validator", validator_sha)
    load_manifest(config_path)
    if args.action == "static-validate":
        print(
            json.dumps(
                {
                    "status": "static_valid",
                    "manifest_id": CURRENT_MANIFEST_ID,
                    "config_sha256": config_sha,
                    "validator_sha256": validator_sha,
                    "validator_revision": "v6r2",
                    "runtime_surface": "none",
                    "next_step": "new_v6r3_or_later_after_boot_root_cause",
                },
                sort_keys=True,
            )
        )
        return 0
    raise ContractError(
        "v6r2 is source-only and NOT LAUNCHED; runtime validation, planning, launch, "
        "and finalization are forbidden. Diagnose the repeated fourth-cell Kit/scene-"
        "start boot timeout and create a new v6r3-or-later preregistration."
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(2)
