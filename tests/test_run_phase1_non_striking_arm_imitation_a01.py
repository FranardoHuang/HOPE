from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/phase1_non_striking_arm_imitation_a01_prereg_20260714.json"
RUNNER = ROOT / "scripts/run_phase1_non_striking_arm_imitation_a01.py"


def load_module():
    spec = importlib.util.spec_from_file_location("non_striking_arm_a01", RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


M = load_module()


def manifest():
    return M.load_manifest(MANIFEST)


def write_manifest(tmp_path: Path, value: dict) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_checked_manifest_is_exact_two_cell_single_seed_direct_mask():
    data = manifest()
    cells = M.cell_map(data)
    assert tuple(cells) == ("A0", "A1")
    assert data["source"]["expected_training_commit"] == (
        "353a11419ae8589ed4a374ed97169cd7a50d50a3"
    )
    assert data["shared_training_contract"]["training_seed"] == 17
    assert data["shared_training_contract"]["initialization"] == "fresh"
    assert data["shared_training_contract"]["relative_checkpoint_milestones"] == [200, 500, 1000]
    assert data["launch_design"]["separate_25_iteration_training_smoke"] is False
    assert data["a2_fixed_budget_reallocation"]["status"] == "blocked_not_materialized"
    assert cells["A0"]["free_non_striking_arm_mimic"] is False
    assert cells["A1"]["free_non_striking_arm_mimic"] is True
    for term in M.BODY_TERMS:
        before = cells["A0"]["body_names"][term]
        after = cells["A1"]["body_names"][term]
        assert [name for name in before if name not in M.LEFT_ARM] == after
        assert [name for name in before if name == "torso_Link" or name.startswith("right_")] == after


def test_commands_differ_only_by_run_name_and_one_mask_flag():
    data = manifest()
    a0 = M.build_command(data, "A0")
    a1 = M.build_command(data, "A1")
    assert M.normalized_paired_command(data, "A0") == M.normalized_paired_command(data, "A1")
    assert a0.count("++task.rewards.free_non_striking_arm_mimic=false") == 1
    assert a1.count("++task.rewards.free_non_striking_arm_mimic=true") == 1
    for command in (a0, a1):
        assert command.count("seed=17") == 1
        assert command.count("num_envs=4096") == 1
        assert command.count("max_iterations=1001") == 1
        assert command.count("algo.runner.save_interval=100") == 1
        assert "checkpoint_path=null" in command
        assert "task.actions.qdes_clamp=true" in command
        assert "task.plant.zero_joint_friction=true" in command
        assert "++task.motion.event_timing_mode=disabled" in command
        assert not any("ros2" in item or "run_deploy" in item for item in command)


@pytest.mark.parametrize(
    "mutator,match",
    [
        (lambda d: d["shared_training_contract"].__setitem__("training_seed", 18), "training_seed"),
        (lambda d: d["shared_training_contract"].__setitem__("max_iterations", 1000), "max_iterations"),
        (lambda d: d["runtime"].__setitem__("gpu", 2), "runtime gpu"),
        (lambda d: d["evaluation"].__setitem__("automatic_judge_launch", True), "judge"),
        (lambda d: d["a2_fixed_budget_reallocation"].__setitem__("status", "ready"), "A2"),
        (lambda d: d["cells"][1]["body_names"]["motion_body_pos"].remove("right_elbow_Link"), "body_names"),
        (lambda d: d["cells"][0].__setitem__("free_non_striking_arm_mimic", True), "A0 mask"),
    ],
)
def test_manifest_rejects_scientific_or_safety_drift(tmp_path, mutator, match):
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    mutator(data)
    with pytest.raises(M.ContractError, match=match):
        M.load_manifest(write_manifest(tmp_path, data))


def test_default_plan_is_read_only_and_binds_exact_source(tmp_path):
    data = manifest()
    plan = M.build_plan(data, MANIFEST, RUNNER)
    assert plan["writes_or_launches_performed"] is False
    assert plan["source"]["commit"] == data["source"]["expected_training_commit"]
    assert plan["source"]["tree"] == data["source"]["expected_training_tree"]
    assert plan["mechanism_milestones"] == [200, 500, 1000]
    assert "--mode launch" in plan["launch_invocation"]
    assert data["runtime"]["root_launch_confirmation"] in plan["launch_invocation"]
    assert list(tmp_path.iterdir()) == []


def test_no_clobber_plan_or_result_writer(tmp_path):
    path = tmp_path / "ledger.json"
    M.write_json_exclusive(path, {"first": True})
    with pytest.raises(FileExistsError):
        M.write_json_exclusive(path, {"first": False})
    assert json.loads(path.read_text(encoding="utf-8")) == {"first": True}


def test_launch_needs_root_and_exact_explicit_confirmation(monkeypatch):
    data = manifest()
    monkeypatch.setattr(M.os, "geteuid", lambda: 501)
    with pytest.raises(M.ContractError, match="requires root"):
        M.launch(data, MANIFEST, RUNNER, data["runtime"]["root_launch_confirmation"])

    monkeypatch.setattr(M.os, "geteuid", lambda: 0)
    with pytest.raises(M.ContractError, match="confirmation token"):
        M.launch(data, MANIFEST, RUNNER, "wrong")


def test_mask_log_proof_requires_zero_markers_for_a0_and_all_four_for_a1(tmp_path):
    a0 = tmp_path / "a0.log"
    a0.write_text("Learning iteration 0\n", encoding="utf-8")
    assert M.verify_mask_log(a0, "A0") == []

    a1 = tmp_path / "a1.log"
    a1.write_text(
        "\n".join(
            f"rewards.{term}.body_names=[] (left non-striking arm imitation removed)"
            for term in M.BODY_TERMS
        ),
        encoding="utf-8",
    )
    assert len(M.verify_mask_log(a1, "A1")) == 4

    a0.write_text(a1.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(M.ContractError, match="A0"):
        M.verify_mask_log(a0, "A0")

    a1.write_text("motion_body_pos (left non-striking arm imitation removed)\n", encoding="utf-8")
    with pytest.raises(M.ContractError, match="all four"):
        M.verify_mask_log(a1, "A1")


def test_source_contains_no_broad_process_signal_or_robot_command():
    text = RUNNER.read_text(encoding="utf-8")
    assert "pkill" not in text
    assert "killall" not in text
    assert "os.killpg" not in text
    assert 'subprocess.run(["ros2"' not in text
    assert 'subprocess.Popen(["ros2"' not in text
    assert "scripts/run_deploy" not in text
    assert '"joint_command"=' not in text


def minimal_hard_contract(data: dict, cell_id: str) -> dict:
    shared = data["shared_training_contract"]
    return {
        "schema_version": 3,
        "actor_obs_contract": shared["actor_observation_contract"],
        "actor_obs_total_dim": shared["actor_observation_dim"],
        "face_command_pairing": shared["face_command_pairing"],
        "mount_normal_sign_per_clip": shared["mount_normal_sign_per_clip"],
        "strike_phase_per_clip": shared["strike_phase_per_clip"],
        "motion_kinematics_exact": True,
        "motion_allow_legacy_link_origin_velocity": False,
        "motion_event_timing": {"mode": "disabled"},
        "motion_imitation_body_names": copy.deepcopy(M.cell_map(data)[cell_id]["body_names"]),
        "joint_names": [f"j{i}" for i in range(31)],
        "action_joint_ids": list(range(31)),
        "joint_friction_coefficients": [0.0] * 31,
        "motion_clips": [
            {"sha256": data["inputs"]["forehand_motion"]["sha256"]},
            {"sha256": data["inputs"]["backhand_motion"]["sha256"]},
        ],
        "question_bank": {
            "sha256": data["inputs"]["schema3_train_bank"]["sha256"],
            "physics_contract_sha256": data["inputs"]["schema3_train_bank"]["physics_contract_sha256"],
            "source_family_sha256": data["inputs"]["schema3_train_bank"]["source_family_sha256"],
            "schema_version": 3,
            "split": "train",
            "exact": True,
        },
    }


def write_contract(tmp_path: Path, name: str, value: dict) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def test_each_cell_hard_contract_binds_its_post_override_body_names(tmp_path):
    data = manifest()
    contracts = {}
    shas = {}
    for cell_id in M.CELL_IDS:
        path = write_contract(tmp_path, f"{cell_id}.json", minimal_hard_contract(data, cell_id))
        digest, contract = M.verify_hard_contract(path, data, cell_id)
        shas[cell_id] = digest
        contracts[cell_id] = contract
        assert contract["motion_imitation_body_names"] == M.cell_map(data)[cell_id]["body_names"]
    assert shas["A0"] != shas["A1"]
    M.verify_pair_contracts_differ_only_by_imitation_body_names(contracts)


def test_hard_contract_rejects_missing_swapped_or_forged_body_mask(tmp_path):
    data = manifest()
    missing = minimal_hard_contract(data, "A0")
    missing.pop("motion_imitation_body_names")
    with pytest.raises(M.ContractError, match="motion_imitation_body_names"):
        M.verify_hard_contract(write_contract(tmp_path, "missing.json", missing), data, "A0")

    swapped = minimal_hard_contract(data, "A0")
    swapped["motion_imitation_body_names"] = M.cell_map(data)["A1"]["body_names"]
    with pytest.raises(M.ContractError, match="A0 hard contract"):
        M.verify_hard_contract(write_contract(tmp_path, "swapped.json", swapped), data, "A0")

    forged = minimal_hard_contract(data, "A1")
    forged["motion_imitation_body_names"]["motion_body_pos"].append("left_elbow_Link")
    with pytest.raises(M.ContractError, match="A1 hard contract"):
        M.verify_hard_contract(write_contract(tmp_path, "forged.json", forged), data, "A1")


def test_pair_contract_comparison_rejects_any_second_difference():
    data = manifest()
    contracts = {cell: minimal_hard_contract(data, cell) for cell in M.CELL_IDS}
    M.verify_pair_contracts_differ_only_by_imitation_body_names(contracts)
    contracts["A1"]["episode_length_s"] = 11.0
    with pytest.raises(M.ContractError, match="outside motion_imitation_body_names"):
        M.verify_pair_contracts_differ_only_by_imitation_body_names(contracts)


def test_checkpoint_audit_code_reads_embedded_hard_contract_identity():
    assert "training_contract_schema_version" in M.CHECKPOINT_AUDIT
    assert "training_contract_sha256" in M.CHECKPOINT_AUDIT
    assert "training_contract_lineage_exact" in M.CHECKPOINT_AUDIT
