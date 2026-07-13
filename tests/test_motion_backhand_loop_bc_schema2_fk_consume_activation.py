from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/validate_motion_schema2_fk_consume_activation.py"
RECEIPT = ROOT / "configs/motion_backhand_loop_bc_schema2_fk_runtime_inspection_receipt_20260714.json"
ACTIVATION = ROOT / "configs/motion_backhand_loop_bc_schema2_fk_consume_activation_20260714.json"
SPEC = importlib.util.spec_from_file_location("schema2_fk_activation", SCRIPT)
assert SPEC and SPEC.loader
V = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _validated() -> tuple[dict, dict]:
    receipt = _json(RECEIPT)
    activation = _json(ACTIVATION)
    V.validate_receipt(receipt)
    V.validate_activation(activation, receipt)
    return receipt, activation


def test_tracked_receipt_and_activation_validate_exactly():
    receipt, activation = _validated()
    assert receipt["inspection_checkout"]["commit"] == V.INSPECTION_COMMIT
    assert receipt["runtime"]["successful_python"]["packages"] == {
        "numpy": "2.5.0",
        "onnxruntime": "1.27.0",
        "mujoco": "3.10.0",
    }
    assert activation["authorization"]["schema2_fk_consume_once_per_asset"] is True
    assert activation["authorization"]["l0_authorized"] is False


def test_cli_static_passes_and_reports_no_consume():
    run = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--receipt",
            str(RECEIPT),
            "--expected-receipt-sha256",
            V.sha256_file(RECEIPT),
            "--activation",
            str(ACTIVATION),
            "--expected-activation-sha256",
            V.sha256_file(ACTIVATION),
            "static",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    assert "receipt_exact=true" in run.stdout
    assert "attempts_started=0" in run.stdout
    assert "consume_not_run=true" in run.stdout


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["inspection_checkout"].__setitem__("clean_after", False),
        lambda value: value["runtime"]["failed_default_python_attempt"].__setitem__(
            "accepted_as_inspection", True
        ),
        lambda value: value["runtime"]["failed_default_python_attempt"]["packages"].__setitem__(
            "onnxruntime", "1.27.0"
        ),
        lambda value: value["runtime"]["successful_python"]["packages"].__setitem__(
            "mujoco", "3.9.0"
        ),
        lambda value: value["donor_onnx"].__setitem__("sha256", "0" * 64),
        lambda value: value["vendor_mjcf"].__setitem__("dynamics_steps", 1),
        lambda value: value["assets"]["franco_backhand_loop_b"].__setitem__(
            "output_root_absent_after", False
        ),
        lambda value: value["assets"]["franco_backhand_loop_c"].__setitem__("input_frames", 97),
        lambda value: value["assets"]["franco_backhand_loop_c"].__setitem__("no_write", False),
        lambda value: value["assets"]["franco_backhand_loop_c"]["source_motion"].__setitem__(
            "path", "/tmp/same-bytes-wrong-lineage.pkl"
        ),
        lambda value: value["honesty_boundary"].__setitem__(
            "forward_kinematics_trajectory_evaluated", True
        ),
    ],
)
def test_receipt_mutations_fail_closed(mutate):
    receipt = _json(RECEIPT)
    mutate(receipt)
    with pytest.raises(V.ActivationContractError):
        V.validate_receipt(receipt)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.__setitem__("status", "consumed"),
        lambda value: value["source_checkout"].__setitem__("commit", "0" * 40),
        lambda value: value["assets"]["franco_backhand_loop_b"].__setitem__(
            "attempts_authorized", 2
        ),
        lambda value: value["assets"]["franco_backhand_loop_b"].__setitem__(
            "attempts_started", 1
        ),
        lambda value: value["assets"]["franco_backhand_loop_c"].__setitem__(
            "output_root", "/tmp/changed"
        ),
        lambda value: value["execution_contract"].__setitem__("asset_parallelism", 2),
        lambda value: value["execution_contract"].__setitem__("automatic_retry", True),
        lambda value: value["commands"]["franco_backhand_loop_b"]["argv"].__setitem__(
            -1, "inspect"
        ),
        lambda value: value["commands"]["franco_backhand_loop_c"]["environment"].__setitem__(
            "CUDA_VISIBLE_DEVICES", "0"
        ),
        lambda value: value["authorization"].__setitem__("l0_authorized", True),
        lambda value: value["authorization"].__setitem__("training_authorized", True),
        lambda value: value["authorization"].__setitem__("hardware_authorized", True),
    ],
)
def test_activation_mutations_fail_closed(mutate):
    receipt, activation = _validated()
    changed = copy.deepcopy(activation)
    mutate(changed)
    with pytest.raises(V.ActivationContractError):
        V.validate_activation(changed, receipt)


def test_commands_are_one_asset_at_a_time_and_bind_peer():
    _receipt, activation = _validated()
    for asset, peer in (
        ("franco_backhand_loop_b", "franco_backhand_loop_c"),
        ("franco_backhand_loop_c", "franco_backhand_loop_b"),
    ):
        argv = activation["commands"][asset]["argv"]
        assert argv == V.expected_command(asset)
        assert argv[-1] == "consume"
        assert f"{V.SOURCE_CHECKOUT}/{V.PLAN_PATHS[asset]}" in argv
        assert f"{V.SOURCE_CHECKOUT}/{V.PLAN_PATHS[peer]}" in argv
        assert V.DONOR_PATH in argv


def test_duplicate_json_and_wrong_outer_sha_fail_closed(tmp_path: Path):
    with pytest.raises(V.ActivationContractError, match="duplicate JSON key"):
        V.strict_json_bytes(b'{"schema_version":1,"schema_version":1}', "duplicate")
    copied = tmp_path / "receipt.json"
    copied.write_bytes(RECEIPT.read_bytes())
    with pytest.raises(V.ActivationContractError, match="receipt SHA"):
        V.read_bound_json(copied, "0" * 64, "inspection receipt")


def test_cli_has_no_consume_subcommand():
    run = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--receipt",
            str(RECEIPT),
            "--expected-receipt-sha256",
            V.sha256_file(RECEIPT),
            "--activation",
            str(ACTIVATION),
            "--expected-activation-sha256",
            V.sha256_file(ACTIVATION),
            "consume",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert run.returncode != 0
    assert "invalid choice" in run.stderr
