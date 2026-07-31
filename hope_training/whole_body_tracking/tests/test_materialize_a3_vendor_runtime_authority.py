"""Host-only tests for the tracked Agibot A3 vendor runtime authority."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


SOURCE_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "materialize_a3_vendor_runtime_authority.py"
)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write(root: Path, relative: str, payload: bytes) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _load_fixture_module(root: Path):
    path = (
        root
        / "hope_training"
        / "whole_body_tracking"
        / "scripts"
        / "materialize_a3_vendor_runtime_authority.py"
    )
    spec = importlib.util.spec_from_file_location(
        f"fixture_vendor_authority_{id(root)}", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "head_yaw_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "head_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]

LEGACY_LOGICAL_JOINT_NAMES = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "head_yaw_joint",
    "head_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]


def _joint_values(name: str) -> tuple[float, float, float, float]:
    if name == "waist_yaw_joint":
        return 80.0, 3.0, 220.0, 0.066472
    if name == "waist_roll_joint":
        return 50.0, 2.0, 46.0, 0.014623
    if name == "waist_pitch_joint":
        return 50.0, 2.0, 115.0, 0.088220
    if name.startswith("head_"):
        return 40.0, 2.0, 6.0, 0.0008100893338
    if name.endswith(("_hip_pitch_joint", "_hip_yaw_joint")):
        return 80.0, 3.0, 220.0, 0.066472
    if name.endswith("_hip_roll_joint"):
        return 120.0, 4.0, 220.0, 0.066472
    if name.endswith("_knee_joint"):
        return 250.0, 8.0, 320.0, 0.120340
    if name.endswith("_ankle_pitch_joint"):
        return 50.0, 2.0, 118.19999694824219, 0.064449
    if name.endswith("_ankle_roll_joint"):
        return 50.0, 2.0, 54.75, 0.020129
    if name.endswith(("_shoulder_pitch_joint", "_shoulder_roll_joint")):
        return 40.0, 3.0, 60.0, 0.012085
    return 30.0, 2.0, 24.0, 0.004968


def _contract(
    stable_motion_sha: str, *, joint_names: list[str] = JOINT_NAMES
) -> dict:
    rows = [_joint_values(name) for name in joint_names]
    effort = [row[2] for row in rows]
    stiffness = [row[0] for row in rows]
    return {
        "schema_version": 3,
        "target_mode": "action_ball",
        "joint_names": joint_names,
        "articulation_joint_names": joint_names,
        "action_joint_ids": list(range(31)),
        "joint_stiffness": stiffness,
        "joint_damping": [row[1] for row in rows],
        "joint_effort_limits": effort,
        "joint_velocity_limits": [12.0] * 31,
        "joint_armature": [row[3] for row in rows],
        "default_joint_pos": [0.0] * 31,
        "action_scale": [0.25 * e / k for e, k in zip(effort, stiffness)],
        "qdes_joint_pos_limits": [[-1.0, 1.0] for _ in range(31)],
        "physics_step_dt_s": 0.005,
        "policy_step_dt_s": 0.02,
        "control_decimation": 4,
        "control_step_action_delay": {
            "schema_version": 1,
            "enabled": True,
            "semantic_unit": "policy_control_step",
            "sample_timing": "once_per_episode_reset",
            "distribution": "discrete_uniform_inclusive",
            "min_steps": 0,
            "max_steps": 2,
            "shared_across_all_31_joints": True,
            "history_fill": "safe_default_or_action_specific_hold",
        },
        "push_robot_event": {
            "schema_version": 2,
            "enabled": True,
            "semantics": "symmetric_6d_velocity_delta",
            "func": "push_by_setting_velocity",
            "mode": "interval",
            "interval_range_s": [5.0, 15.0],
            "velocity_range": {
                "x": [-0.25, 0.25],
                "y": [-0.25, 0.25],
                "z": [-0.1, 0.1],
                "roll": [-0.26, 0.26],
                "pitch": [-0.26, 0.26],
                "yaw": [-0.39, 0.39],
            },
        },
        "action_ball_training": {
            "preflight": {"action_order": ["bh_loop_c"]},
            "policy_bootstrap": {
                "schema_version": 1,
                "kind": "action_ball_shared_ready_actor_bootstrap_v1",
                "action_order": ["bh_loop_c"],
            },
            "motion_admission": {
                "motion_file_sha256": [stable_motion_sha]
            },
        },
    }


def _fixture(tmp_path: Path) -> dict:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "fixture@example.invalid")
    _git(root, "config", "user.name", "fixture")
    producer_relative = (
        "hope_training/whole_body_tracking/scripts/"
        "materialize_a3_vendor_runtime_authority.py"
    )
    _write(root, producer_relative, SOURCE_SCRIPT.read_bytes())
    module = _load_fixture_module(root)
    for role, relative in module.SOURCE_PATHS.items():
        payload = (
            b"stable-motion-fixture"
            if role == "stable_motion"
            else f"# {role} fixture\n".encode()
        )
        _write(root, relative, payload)
    stable_sha = _sha((root / module.STABLE_MOTION_REPO_PATH).read_bytes())
    contract = _contract(stable_sha)
    contract_bytes = module._canonical_bytes(contract) + b"\n"
    _write(root, module.RUNTIME_CONTRACT_REPO_PATH, contract_bytes)
    receipt = root / module.RECEIPT_REPO_PATH
    receipt.parent.mkdir(parents=True, exist_ok=True)

    tracked_a = [producer_relative, *module.SOURCE_PATHS.values()]
    _git(root, "add", "--", *tracked_a)
    _git(root, "commit", "-qm", "scientific source A")
    commit_a = _git(root, "rev-parse", "HEAD")
    digest_by_role = {
        role: _sha((root / relative).read_bytes())
        for role, relative in module.SOURCE_PATHS.items()
    }
    kwargs = {
        "repo_root": root,
        "source_commit": commit_a,
        "expected_vendor_task_sha256": digest_by_role["vendor_task_profile"],
        "expected_action_ball_task_sha256": digest_by_role[
            "action_ball_task_profile"
        ],
        "expected_hitter_task_sha256": digest_by_role["hitter_task_profile"],
        "expected_env_base_sha256": digest_by_role["environment_base_profile"],
        "expected_sim_base_sha256": digest_by_role["simulation_base_profile"],
        "expected_randomization_base_sha256": digest_by_role[
            "randomization_base_profile"
        ],
        "expected_robot_source_sha256": digest_by_role["robot_actuator_source"],
        "expected_env_cfg_source_sha256": digest_by_role[
            "environment_config_source"
        ],
        "expected_train_source_sha256": digest_by_role["training_entrypoint"],
        "expected_training_contract_source_sha256": digest_by_role[
            "training_contract_source"
        ],
        "expected_hope_actions_source_sha256": digest_by_role["action_source"],
        "expected_runner_source_sha256": digest_by_role["runner_source"],
        "expected_stable_motion_sha256": digest_by_role["stable_motion"],
        "runtime_training_contract": root / module.RUNTIME_CONTRACT_REPO_PATH,
        "expected_runtime_training_contract_sha256": _sha(contract_bytes),
        "output": receipt,
    }
    result = module.materialize_vendor_runtime_authority(**kwargs)
    return {
        "root": root,
        "module": module,
        "commit_a": commit_a,
        "kwargs": kwargs,
        "result": result,
        "receipt": receipt,
    }


def _commit_receipt(fixture: dict) -> str:
    root = fixture["root"]
    module = fixture["module"]
    _git(
        root,
        "add",
        "--",
        module.RUNTIME_CONTRACT_REPO_PATH,
        module.RECEIPT_REPO_PATH,
    )
    _git(root, "commit", "-qm", "tracked authority B")
    return _git(root, "rev-parse", "HEAD")


def _candidate(authority: dict) -> dict:
    plant = deepcopy(authority["runtime_plant_identity"])
    return {
        "schema_version": 2,
        "kind": "agibot_a3_action_dynamic_ready_candidate_v2",
        "action_id": "bh_loop_c",
        "robot": {
            "family": "AgiBot A3",
            "joint_names": deepcopy(plant["joint_names"]),
        },
        "sources": {
            "runtime_training_contract": deepcopy(
                authority["runtime_training_contract"]
            ),
            "stable_motion": deepcopy(authority["sources"]["stable_motion"]),
        },
        "runtime_plant": plant,
    }


def test_commit_a_materialize_commit_b_validate_passes(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)
    module = fx["module"]
    with pytest.raises(
        module.VendorRuntimeAuthorityError, match="launch-commit blob"
    ):
        module.load_and_validate_vendor_runtime_authority(
            fx["receipt"], repo_root=fx["root"]
        )
    commit_b = _commit_receipt(fx)
    authority = module.load_and_validate_vendor_runtime_authority(
        fx["receipt"],
        repo_root=fx["root"],
        expected_runtime_training_contract_sha256=fx["kwargs"][
            "expected_runtime_training_contract_sha256"
        ],
        launch_commit=commit_b,
    )
    assert authority["authorization"] == {
        "training_authorized": False,
        "deployment_authorized": False,
        "hardware_authorized": False,
    }
    assert (
        module.validate_candidate_runtime_plant_against_vendor_authority(
            _candidate(authority), authority
        )
        == authority["runtime_plant_identity"]
    )


def test_receipt_is_canonical_no_clobber_and_worktree_mutation_fails(
    tmp_path: Path,
) -> None:
    fx = _fixture(tmp_path)
    module = fx["module"]
    with pytest.raises(module.VendorRuntimeAuthorityError, match="no-clobber"):
        module.materialize_vendor_runtime_authority(**fx["kwargs"])
    _commit_receipt(fx)
    original = fx["receipt"].read_bytes()
    fx["receipt"].chmod(0o644)
    fx["receipt"].write_bytes(b" " + original)
    with pytest.raises(module.VendorRuntimeAuthorityError):
        module.load_and_validate_vendor_runtime_authority(
            fx["receipt"], repo_root=fx["root"]
        )


@pytest.mark.parametrize(
    "field",
    ["joint_stiffness", "joint_armature", "action_scale_rad"],
)
def test_candidate_plant_tamper_and_old_schema_are_refused(
    tmp_path: Path, field: str
) -> None:
    fx = _fixture(tmp_path)
    module = fx["module"]
    _commit_receipt(fx)
    authority = module.load_and_validate_vendor_runtime_authority(
        fx["receipt"], repo_root=fx["root"]
    )
    candidate = _candidate(authority)
    candidate["runtime_plant"][field][0] += 1.0
    with pytest.raises(module.VendorRuntimeAuthorityError):
        module.validate_candidate_runtime_plant_against_vendor_authority(
            candidate, authority
        )
    candidate = _candidate(authority)
    candidate["schema_version"] = 1
    candidate["kind"] = "agibot_a3_action_dynamic_ready_candidate_v1"
    with pytest.raises(module.VendorRuntimeAuthorityError, match="schema-v2"):
        module.validate_candidate_runtime_plant_against_vendor_authority(
            candidate, authority
        )


def test_candidate_contract_retag_and_transitive_source_drift_are_refused(
    tmp_path: Path,
) -> None:
    fx = _fixture(tmp_path)
    module = fx["module"]
    _commit_receipt(fx)
    authority = module.load_and_validate_vendor_runtime_authority(
        fx["receipt"], repo_root=fx["root"]
    )
    candidate = _candidate(authority)
    candidate["sources"]["runtime_training_contract"]["sha256"] = "f" * 64
    with pytest.raises(module.VendorRuntimeAuthorityError, match="contract pin"):
        module.validate_candidate_runtime_plant_against_vendor_authority(
            candidate, authority
        )

    inherited = fx["root"] / module.ACTION_BALL_TASK_REPO_PATH
    inherited.write_bytes(inherited.read_bytes() + b"# drift\n")
    _git(fx["root"], "add", "--", module.ACTION_BALL_TASK_REPO_PATH)
    _git(fx["root"], "commit", "-qm", "drift inherited task")
    with pytest.raises(module.VendorRuntimeAuthorityError, match="drifted"):
        module.load_and_validate_vendor_runtime_authority(
            fx["receipt"], repo_root=fx["root"]
        )


def test_full_vendor_joint_table_and_push_func_are_fail_loud(tmp_path: Path) -> None:
    root = tmp_path / "semantic"
    root.mkdir()
    module = _load_fixture_module_for_semantics(root)
    stable_sha = "a" * 64
    contract = _contract(stable_sha)
    for field in (
        "joint_stiffness",
        "joint_damping",
        "joint_effort_limits",
        "joint_armature",
        "action_scale",
    ):
        changed = deepcopy(contract)
        changed[field][22] += 1.0
        with pytest.raises(module.VendorRuntimeAuthorityError):
            module._verified_vendor_runtime(
                changed, stable_motion_sha256=stable_sha
            )
    changed = deepcopy(contract)
    changed["push_robot_event"]["func"] = "lookalike"
    with pytest.raises(module.VendorRuntimeAuthorityError):
        module._verified_vendor_runtime(changed, stable_motion_sha256=stable_sha)
    renamed = deepcopy(contract)
    hip_index = JOINT_NAMES.index("left_hip_pitch_joint")
    renamed["joint_names"][hip_index] = "middle_hip_pitch_joint"
    renamed["articulation_joint_names"][hip_index] = "middle_hip_pitch_joint"
    with pytest.raises(module.VendorRuntimeAuthorityError, match="31-joint"):
        module._verified_vendor_runtime(renamed, stable_motion_sha256=stable_sha)


def test_live_usd_articulation_order_passes_and_legacy_logical_order_is_refused(
    tmp_path: Path,
) -> None:
    root = tmp_path / "live-order"
    root.mkdir()
    module = _load_fixture_module_for_semantics(root)
    stable_sha = "a" * 64
    contract = _contract(stable_sha)

    assert list(module.RUNTIME_JOINT_NAMES) == JOINT_NAMES
    plant = module._canonical_runtime_plant_identity(contract)
    verified = module._verified_vendor_runtime(
        contract, stable_motion_sha256=stable_sha
    )
    assert plant["joint_names"] == JOINT_NAMES
    assert list(verified["vendor_joint_values"]) == JOINT_NAMES
    assert plant["action_joint_ids"] == list(range(31))

    legacy = _contract(
        stable_sha, joint_names=LEGACY_LOGICAL_JOINT_NAMES
    )
    with pytest.raises(module.VendorRuntimeAuthorityError, match="action order"):
        module._canonical_runtime_plant_identity(legacy)
    with pytest.raises(
        module.VendorRuntimeAuthorityError, match="articulation order"
    ):
        module._verified_vendor_runtime(
            legacy, stable_motion_sha256=stable_sha
        )


def _load_fixture_module_for_semantics(root: Path):
    relative = (
        "hope_training/whole_body_tracking/scripts/"
        "materialize_a3_vendor_runtime_authority.py"
    )
    _write(root, relative, SOURCE_SCRIPT.read_bytes())
    return _load_fixture_module(root)
