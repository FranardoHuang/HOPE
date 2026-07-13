from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT / "hope_training/whole_body_tracking/scripts/a3_joint_order_contract.py"
)
SPEC = importlib.util.spec_from_file_location("a3_joint_order_contract_test", MODULE_PATH)
CONTRACT = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = CONTRACT
SPEC.loader.exec_module(CONTRACT)


def _metadata(contract, **overrides):
    payload = {
        "joint_names": ",".join(contract.target_names),
        "articulation_joint_names": ",".join(contract.target_names),
        "action_joint_ids": ",".join(str(index) for index in range(31)),
    }
    payload.update(overrides)
    return payload


def test_checked_in_contract_is_distinct_bijective_and_reorders_columns():
    contract = CONTRACT.load_contract(repo_root=ROOT)
    assert contract.contract_id == "a3-gmr-dof-pos-to-runtime-articulation-v1"
    assert len(contract.source_names) == len(contract.target_names) == 31
    assert contract.source_names != contract.target_names
    assert sorted(contract.target_from_source_indices) == list(range(31))
    source = np.arange(2 * 31, dtype=np.float32).reshape(2, 31)
    target = CONTRACT.reorder_source_to_target(source, contract)
    for target_index, source_index in enumerate(contract.target_from_source_indices):
        np.testing.assert_array_equal(target[:, target_index], source[:, source_index])
    restored = target[:, list(contract.source_from_target_indices)]
    np.testing.assert_array_equal(restored, source)


def test_checked_in_cli_is_source_gate_only():
    run = subprocess.run(
        [sys.executable, str(MODULE_PATH)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    result = json.loads(run.stdout)
    assert result == {
        "bijection_valid": True,
        "contract_id": "a3-gmr-dof-pos-to-runtime-articulation-v1",
        "contract_sha256": CONTRACT._sha256(
            ROOT / "configs/a3_joint_order_bijection_v1.json"
        ),
        "joint_count": 31,
        "runtime_metadata_checked": False,
        "runtime_metadata_bytes": None,
        "runtime_metadata_sha256": None,
        "schema2_materialization_authorized": False,
        "source_equals_target": False,
    }


@pytest.mark.parametrize(
    "values,match",
    [
        (["a_joint"] * 31, "duplicate"),
        ([f"j{i}_joint" for i in range(30)], "length 30"),
        ([f"j{i}_joint" for i in range(30)] + [" bad_joint"], "malformed"),
    ],
)
def test_name_validator_rejects_duplicate_wrong_length_and_malformed(values, match):
    with pytest.raises(CONTRACT.JointOrderContractError, match=match):
        CONTRACT.normalize_names(values, expected_count=31, label="test")


def test_bijection_rejects_missing_and_extra_name():
    source = [f"j{i}_joint" for i in range(31)]
    target = source[1:] + ["extra_joint"]
    with pytest.raises(CONTRACT.JointOrderContractError, match="missing_from_target.*j0_joint.*extra_in_target.*extra_joint"):
        CONTRACT.validate_bijection(source, target)


def test_bijection_rejects_accidental_same_order():
    names = [f"j{i}_joint" for i in range(31)]
    with pytest.raises(CONTRACT.JointOrderContractError, match="unexpectedly became identical"):
        CONTRACT.validate_bijection(names, names)


def test_runtime_metadata_requires_complete_exact_target_and_identity_ids():
    contract = CONTRACT.load_contract(repo_root=ROOT)
    CONTRACT.validate_runtime_metadata(_metadata(contract), contract)

    partial = _metadata(contract)
    del partial["articulation_joint_names"]
    with pytest.raises(CONTRACT.JointOrderContractError, match="partial"):
        CONTRACT.validate_runtime_metadata(partial, contract)

    wrong = list(contract.target_names)
    wrong[0], wrong[1] = wrong[1], wrong[0]
    with pytest.raises(CONTRACT.JointOrderContractError, match="does not equal runtime target"):
        CONTRACT.validate_runtime_metadata(
            _metadata(contract, joint_names=",".join(wrong)), contract
        )

    duplicate = list(contract.target_names)
    duplicate[1] = duplicate[0]
    with pytest.raises(CONTRACT.JointOrderContractError, match="duplicate"):
        CONTRACT.validate_runtime_metadata(
            _metadata(contract, articulation_joint_names=",".join(duplicate)), contract
        )

    with pytest.raises(CONTRACT.JointOrderContractError, match="length"):
        CONTRACT.validate_runtime_metadata(
            _metadata(contract, joint_names=",".join(contract.target_names[:-1])), contract
        )

    ids = list(range(31))
    ids[0], ids[1] = ids[1], ids[0]
    with pytest.raises(CONTRACT.JointOrderContractError, match="not identity"):
        CONTRACT.validate_runtime_metadata(
            _metadata(contract, action_joint_ids=",".join(map(str, ids))), contract
        )


def test_reorder_rejects_wrong_length_and_nonfinite():
    contract = CONTRACT.load_contract(repo_root=ROOT)
    with pytest.raises(CONTRACT.JointOrderContractError, match="last dimension"):
        CONTRACT.reorder_source_to_target(np.zeros((2, 30)), contract)
    values = np.zeros((2, 31))
    values[0, 0] = np.nan
    with pytest.raises(CONTRACT.JointOrderContractError, match="finite"):
        CONTRACT.reorder_source_to_target(values, contract)


def _copy_contract_repo(tmp_path: Path) -> Path:
    for relative in (
        "configs/a3_joint_order_bijection_v1.json",
        "configs/a3_gmr_dof_pos_joint_order.txt",
        "configs/a3_runtime_articulation_joint_order.txt",
        "hope_training/config/joint_order_agibot_a3.yaml",
        "hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/robots/agibot_a3.py",
        "hope_training/whole_body_tracking/scripts/audit_motion_npz.py",
    ):
        source = ROOT / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return tmp_path


def test_contract_rejects_declared_permutation_drift(tmp_path):
    repo = _copy_contract_repo(tmp_path)
    path = repo / "configs/a3_joint_order_bijection_v1.json"
    raw = json.loads(path.read_text())
    raw["target_from_source_indices"][0], raw["target_from_source_indices"][1] = (
        raw["target_from_source_indices"][1],
        raw["target_from_source_indices"][0],
    )
    path.write_text(json.dumps(raw))
    with pytest.raises(CONTRACT.JointOrderContractError, match="does not match the named tables"):
        CONTRACT.load_contract(repo_root=repo)


def test_contract_rejects_order_file_byte_drift_before_use(tmp_path):
    repo = _copy_contract_repo(tmp_path)
    path = repo / "configs/a3_runtime_articulation_joint_order.txt"
    path.write_text(path.read_text() + "# unbound drift\n")
    with pytest.raises(CONTRACT.JointOrderContractError, match="file SHA mismatch"):
        CONTRACT.load_contract(repo_root=repo)


def test_json_loader_rejects_duplicate_keys():
    with pytest.raises(CONTRACT.JointOrderContractError, match="duplicate JSON key"):
        CONTRACT._load_json('{"joint_names":"a","joint_names":"b"}', "metadata")
