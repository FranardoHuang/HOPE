"""Dependency-light source/static/attack tests for A2/B2 hot L1."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_phase1_signed_face_a2b2_l1.py"
MANIFEST = ROOT / "configs/phase1_signed_face_a2b2_l1_prereg_20260714.json"
SPEC = importlib.util.spec_from_file_location("signed_face_a2b2_l1", SCRIPT)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def hard_contract(m, cell_id):
    return {
        "schema_version": 3,
        "actor_obs_contract": "deploy_parity_face179",
        "actor_obs_total_dim": 179,
        "face_command_pairing": "shared_plus_y",
        "mount_normal_sign_per_clip": [1.0, -1.0],
        "strike_phase_per_clip": [0.471, 0.338],
        "motion_kinematics_exact": True,
        "motion_allow_legacy_link_origin_velocity": False,
        "motion_event_timing": {"mode": "disabled"},
        "racket_guidance_reward": {
            "position": {"weight": 0.0, "command_name": "racket_target", "d_max": 0.5},
            "signed_face": {
                "weight": mod.cells(m)[cell_id]["face_guidance_weight"],
                "command_name": "racket_target",
                "theta_max": 3.141592653589793,
            },
        },
        "joint_names": [f"joint_{i}" for i in range(31)],
        "action_joint_ids": list(range(31)),
        "joint_friction_coefficients": [0.0] * 31,
        "motion_clips": [
            {"sha256": m["inputs"]["forehand_motion"]["sha256"]},
            {"sha256": m["inputs"]["backhand_motion"]["sha256"]},
        ],
        "question_bank": copy.deepcopy(
            m["hot_start_contract_transition"]["allowed_changed_common_fields"]
            ["question_bank"]["current"]
        ),
    }


def bank_metadata(m):
    bank = m["inputs"]["schema3_train_bank"]
    return {
        "sha256": bank["sha256"],
        "source_family_sha256": bank["source_family_sha256"],
        "physics_contract_sha256": bank["physics_contract_sha256"],
    }


def parent_and_current(m, cell_id="A2"):
    current = hard_contract(m, cell_id)
    current.update(copy.deepcopy(
        m["hot_start_contract_transition"]["expected_current_only_shared_values"]
    ))
    current["racket_guidance_reward"] = copy.deepcopy(
        m["hot_start_contract_transition"]["expected_racket_guidance_reward_by_cell"]
        [cell_id]
    )
    parent = copy.deepcopy(current)
    for key in m["hot_start_contract_transition"]["allowed_current_only_top_level_keys"]:
        parent.pop(key)
    parent["question_bank"] = copy.deepcopy(
        m["hot_start_contract_transition"]["allowed_changed_common_fields"]
        ["question_bank"]["parent"]
    )
    return parent, current


def test_checked_in_manifest_static_source_and_plan_are_exact():
    m = mod.load_manifest(MANIFEST)
    assert mod.verify_static_source(m) == {
        "commit": mod.TRAINING_COMMIT,
        "tree": mod.TRAINING_TREE,
    }
    plan = mod.build_plan(m, MANIFEST, SCRIPT)
    assert plan["ordered_cells"] == ["A2", "B2"]
    assert plan["execution_lanes"]["A2"] == {
        "host": "pod1", "physical_gpu": 0, "local_training_device": "cuda:0"
    }
    assert plan["execution_lanes"]["B2"] == {
        "host": "pod2", "physical_gpu": 0, "local_training_device": "cuda:0"
    }
    assert plan["cross_host_independent_one_shot"] is True
    assert all(value is False for value in plan["decision_boundary"].values())


@pytest.mark.parametrize("cell_id,weight", [("A2", "0.0"), ("B2", "-0.4")])
def test_command_binds_parent_budget_zero_friction_and_only_cell_weight(cell_id, weight):
    m = manifest()
    command = mod.build_command(m, cell_id, "a" * 64)
    parent = m["inputs"]["hot_parent_checkpoint"]["path"]
    assert command.count(f"checkpoint_path={parent}") == 1
    assert command.count("checkpoint_allow_missing_contract=false") == 1
    assert command.count("checkpoint_allow_contract_mismatch=true") == 1
    assert command.count("checkpoint_tolerant=false") == 1
    assert command.count(mod.ZERO_FRICTION_ARG) == 1
    assert command.count("num_envs=512") == 1
    assert command.count("max_iterations=25") == 1
    assert command.count(f"++task.rewards.racket_face_guidance_weight={weight}") == 1
    assert mod.normalized_command(m, "A2") == mod.normalized_command(m, "B2")


def test_environment_and_claim_bind_distinct_cross_pod_scientific_cells(tmp_path):
    m = manifest()
    wbt = Path(mod.SOURCE_CHECKOUT) / m["source"]["wbt_relative_path"]
    for cell_id in mod.CELL_IDS:
        assert mod.canonical_sha256(mod.exact_environment_payload(m, wbt, cell_id)) == (
            mod.TRAINING_ENV_SHA256_BY_CELL[cell_id]
        )
    m["runtime"]["run_root"] = str(tmp_path)
    for cell_id, host in (("A2", "pod1"), ("B2", "pod2")):
        arm = tmp_path / mod.cells(m)[cell_id]["run_name"]
        arm.mkdir()
        st = arm.stat()
        claim = mod.build_claim(
            m, manifest_sha="1" * 64, launcher_sha="2" * 64,
            cell_id=cell_id, arm_dir=arm,
            arm_identity={"device": st.st_dev, "inode": st.st_ino},
        )
        assert claim["execution_lane"]["host"] == host
        assert claim["execution_lane"]["physical_gpu"] == 0
        assert claim["parent_checkpoint"]["sha256"] == (
            m["inputs"]["hot_parent_checkpoint"]["sha256"]
        )
        assert claim["optimization_recipe"]["zero_joint_friction"] is True
        assert claim["expected_child_training_contract_lineage_exact"] is False


@pytest.mark.parametrize(
    "mutator",
    [
        lambda m: m["shared_training_contract"].__setitem__("zero_joint_friction", False),
        lambda m: m["shared_training_contract"]["base_recipe"].remove(mod.ZERO_FRICTION_ARG),
        lambda m: m["shared_training_contract"]["base_recipe"].append(
            "++task.plant.zero_joint_friction=true"
        ),
        lambda m: m["shared_training_contract"].__setitem__("max_iterations", 26),
        lambda m: m["cells"][0].__setitem__("host", "pod2"),
        lambda m: m["cells"][1].__setitem__("gpu", 1),
        lambda m: m["cells"][1].__setitem__("expected_lineage_exact", True),
        lambda m: m["evaluation"].__setitem__("judge", True),
    ],
)
def test_manifest_drift_fails_closed(mutator):
    m = manifest()
    mutator(m)
    with pytest.raises(mod.ContractError):
        mod.validate_manifest(m)


@pytest.mark.parametrize("bad", [0.1, True, float("nan")])
def test_hard_contract_requires_31_exact_zero_friction(tmp_path, bad):
    m = manifest()
    value = hard_contract(m, "A2")
    value["joint_friction_coefficients"][7] = bad
    path = tmp_path / "hard.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(mod.ContractError, match="31/31 zero-friction"):
        mod.verify_hard_contract(path, m, "A2", bank_metadata=bank_metadata(m))


def test_core_hot_contract_diff_and_pair_axis_are_exact():
    m = manifest()
    parent, current = parent_and_current(m)
    mod.verify_hot_transition(parent, current, m, "A2")
    broken = copy.deepcopy(current)
    broken["actor_obs_total_dim"] = 180
    with pytest.raises(mod.ContractError, match="actor_obs_total_dim"):
        mod.verify_hot_transition(parent, broken, m, "A2")
    broken = copy.deepcopy(current)
    broken["racket_target_delay_steps"] = 3
    with pytest.raises(mod.ContractError, match="current-only"):
        mod.verify_hot_transition(parent, broken, m, "A2")
    pair = {cell: hard_contract(m, cell) for cell in mod.CELL_IDS}
    mod.verify_pair_contracts(pair)
    pair["B2"]["actor_obs_total_dim"] = 180
    with pytest.raises(mod.ContractError, match="outside signed-face"):
        mod.verify_pair_contracts(pair)


def test_fresh_absence_wrong_host_and_preserved_failure_fail_closed(tmp_path):
    m = manifest()
    m["runtime"]["run_root"] = str(tmp_path)
    assert mod.select_one_shot_cell(m, "A2", "pod1") == "A2"
    with pytest.raises(mod.ContractError, match="cell host"):
        mod.select_one_shot_cell(m, "A2", "pod2")
    paths = mod.expected_arm_paths(m, "A2")
    paths["arm"].mkdir()
    with pytest.raises(mod.ContractError, match="already claimed"):
        mod.select_one_shot_cell(m, "A2", "pod1")
    paths["failure"].write_text("{}\n", encoding="utf-8")
    with pytest.raises(mod.ContractError, match="no automatic retry"):
        mod.select_one_shot_cell(m, "A2", "pod1")


def test_host_identity_uses_actual_gpu_uuid_not_host_cli():
    m = manifest()
    pod1 = {"gpu": 0, "uuid": m["runtime"]["gpu0_uuid_by_host"]["pod1"]}
    mod.verify_gpu_host_identity(m, "pod1", pod1)
    with pytest.raises(mod.ContractError, match="hardware GPU0 UUID"):
        mod.verify_gpu_host_identity(m, "pod2", pod1)


def test_cross_pod_finalize_pair_is_runtime_reachable_and_fail_closed(
    monkeypatch, tmp_path
):
    m = manifest()
    m["runtime"]["pair_input_root"] = str(tmp_path / "inputs")
    m["runtime"]["pair_result_path"] = str(tmp_path / "pair.json")
    paths = mod.pair_input_paths(m)
    for cell_id in mod.CELL_IDS:
        paths[cell_id]["terminal"].parent.mkdir(parents=True)
        paths[cell_id]["terminal"].write_text("{}\n", encoding="utf-8")
        paths[cell_id]["hard"].write_text("{}\n", encoding="utf-8")
    pair = {cell: hard_contract(m, cell) for cell in mod.CELL_IDS}
    terminals = {
        cell: {"terminal_checkpoint_sha256": ("a" if cell == "A2" else "b") * 64}
        for cell in mod.CELL_IDS
    }
    monkeypatch.setattr(mod.os, "geteuid", lambda: 0)
    monkeypatch.setattr(mod, "verify_external_control_location", lambda *a: {"exact": True})
    monkeypatch.setattr(
        mod, "gpu_snapshot",
        lambda _gpu: {
            "gpu": 0,
            "uuid": m["runtime"]["gpu0_uuid_by_host"]["pod1"],
            "compute_pids": [], "trainer_pids": [], "free_memory_mib": 32000,
        },
    )
    monkeypatch.setattr(
        mod, "verify_copied_terminal_result",
        lambda _m, _mp, _lp, cell, _tp, _hp: (terminals[cell], pair[cell]),
    )
    result = mod.finalize_pair(m, MANIFEST, SCRIPT, "pod1")
    assert result["status"] == "paired_l1_provenance_complete_no_decision"
    written = json.loads((tmp_path / "pair.json").read_text())
    assert written["only_hard_contract_difference"].endswith("signed_face.weight")
    assert written["judge"] is False and written["l2"] is False
    with pytest.raises(mod.ContractError, match="pair finalize host"):
        mod.finalize_pair(m, MANIFEST, SCRIPT, "pod2")


def test_runtime_control_is_external_read_only_and_cli_has_no_judge_retry(tmp_path):
    m = manifest()
    m["runtime"]["external_control_root"] = str(tmp_path)
    config = tmp_path / "phase1_signed_face_a2b2_l1_prereg_20260714.json"
    launcher = tmp_path / "run_phase1_signed_face_a2b2_l1.py"
    config.write_text("{}\n", encoding="utf-8")
    launcher.write_text("# launcher\n", encoding="utf-8")
    config.chmod(0o444)
    launcher.chmod(0o555)
    mod.verify_external_control_location(m, config, launcher)
    config.chmod(0o644)
    with pytest.raises(mod.ContractError, match="read-only"):
        mod.verify_external_control_location(m, config, launcher)
    modes = set(mod.parser()._option_string_actions["--mode"].choices)
    assert modes == {
        "plan", "static-validate", "validate-runtime", "launch-one",
        "finalize-cell", "finalize-pair",
    }
    assert not ({"judge", "retry", "l2", "activate"} & modes)
    source = SCRIPT.read_text(encoding="utf-8")
    assert "os.kill(" not in source
    assert "subprocess.Popen" not in source
    assert "pkill" not in source
    assert "killall" not in source


def test_external_control_default_manifest_is_sibling(tmp_path):
    launcher = tmp_path / "run_phase1_signed_face_a2b2_l1.py"
    launcher.write_text("# launcher\n", encoding="utf-8")
    sibling = tmp_path / "phase1_signed_face_a2b2_l1_prereg_20260714.json"
    sibling.write_text("{}\n", encoding="utf-8")
    assert mod.default_manifest_path(launcher) == sibling
