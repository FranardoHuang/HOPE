from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/phase1_signed_face_rescue_funnel_prereg_20260713.json"
LAUNCHER = ROOT / "scripts/run_phase1_signed_face_rescue_funnel.py"


def load_module():
    spec = importlib.util.spec_from_file_location("signed_face_funnel", LAUNCHER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


M = load_module()


def manifest():
    return M.load_manifest(CONFIG)


def write_manifest(tmp_path: Path, value: dict) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_checked_manifest_is_exact_four_cell_single_seed_funnel():
    data = manifest()
    assert data["manifest_id"].endswith("-v4")
    assert data["runtime"]["external_control_root"].endswith("/control/v4")
    assert data["runtime"]["training_environment_sha256"] == (
        "ddaa0effe2ed5318cc8ce34efbbf5b4ee042572052ab57232291079f41bed743"
    )
    assert data["source"]["critical_files"]["setup_train_env.sh"] == (
        "88c1d7307ec90483712f7f3b0d8535179b88bb132a8a5a06111bff6872034214"
    )
    assert [cell["cell_id"] for cell in data["cells"]] == ["A", "B", "C", "D"]
    assert data["shared_training_contract"]["training_seed"] == 3
    assert [(cell["initialization"], cell["face_guidance_weight"]) for cell in data["cells"]] == [
        ("hot_parent", 0.0), ("hot_parent", -0.4), ("fresh", 0.0), ("fresh", -0.4)
    ]
    assert data["runtime"]["maximum_trainers_on_gpu"] == 4
    assert data["runtime"]["initial_gpu_must_have_zero_compute_processes"] is True
    assert data["seed_replication_before_l2_decision_forbidden"] is True


@pytest.mark.parametrize(
    "mutator,match",
    [
        (lambda d: d["shared_training_contract"].__setitem__("training_seed", 4), "training_seed"),
        (lambda d: d["cells"][1].__setitem__("face_guidance_weight", -0.95), "causal grid"),
        (lambda d: d["cells"][0].__setitem__("expected_lineage_exact", True), "causal grid"),
        (lambda d: d["runtime"].__setitem__("maximum_trainers_on_gpu", 3), "exactly four"),
        (lambda d: d["stages"]["l2"].__setitem__("max_iterations", 1000), "budget"),
        (lambda d: d["stages"]["l2"].__setitem__("relative_milestones", [1000]), "milestones"),
        (lambda d: d["stages"]["l2"].__setitem__("launch_authorized", True), "remain blocked"),
        (lambda d: d["evaluation_contract"].__setitem__("automatic_judge_launch", True), "judge"),
    ],
)
def test_manifest_rejects_scientific_contract_drift(tmp_path, mutator, match):
    data = json.loads(CONFIG.read_text(encoding="utf-8"))
    mutator(data)
    with pytest.raises(M.ContractError, match=match):
        M.load_manifest(write_manifest(tmp_path, data))


def test_manifest_rejects_duplicate_or_nonseed3_run_names(tmp_path):
    data = json.loads(CONFIG.read_text(encoding="utf-8"))
    data["stages"]["l2"]["run_names"]["D"] = data["stages"]["l2"]["run_names"]["C"]
    with pytest.raises(M.ContractError, match="duplicate run name"):
        M.load_manifest(write_manifest(tmp_path, data))

    data = json.loads(CONFIG.read_text(encoding="utf-8"))
    data["stages"]["l2"]["run_names"]["D"] = "phase1_signed_face_l2_D_fresh_guidance_seed4"
    with pytest.raises(M.ContractError, match="bind cell and seed"):
        M.load_manifest(write_manifest(tmp_path, data))


def test_commands_bind_hot_transfer_and_fresh_lineage_without_duplicate_seed():
    data = manifest()
    expected_wbt = (
        Path(data["source"]["training_checkout"]).resolve()
        / data["source"]["wbt_relative_path"]
    )
    commands = {
        cell: M.build_command(data, "l2", cell, wbt=expected_wbt)
        for cell in ("A", "B", "C", "D")
    }
    for command in commands.values():
        assert command.count("seed=3") == 1
        assert "num_envs=4096" in command
        assert "max_iterations=1001" in command
        assert "algo.runner.save_interval=100" in command
        assert "task.plant.zero_joint_friction=true" in command
        assert "++task.motion.event_timing_mode=disabled" in command
        assert "++task.rewards.racket_face_guidance_theta_max=3.141592653589793" in command
        assert not any("seed=1" in item or "seed=2" in item or "seed=4" in item for item in command)

    for cell in ("A", "B"):
        command = commands[cell]
        assert any(item.startswith("checkpoint_path=") and item.endswith("model_13800.pt") for item in command)
        assert "checkpoint_allow_missing_contract=false" in command
        assert "checkpoint_allow_contract_mismatch=true" in command
        assert "checkpoint_tolerant=false" in command
    for cell in ("C", "D"):
        command = commands[cell]
        assert "checkpoint_path=null" in command
        assert "checkpoint_allow_contract_mismatch=false" in command
        assert not any(item.endswith("model_13800.pt") for item in command)

    assert "++task.rewards.racket_face_guidance_weight=0.0" in commands["A"]
    assert "++task.rewards.racket_face_guidance_weight=-0.4" in commands["B"]
    assert "++task.rewards.racket_face_guidance_weight=0.0" in commands["C"]
    assert "++task.rewards.racket_face_guidance_weight=-0.4" in commands["D"]


def test_matched_pairs_differ_only_by_run_name_and_face_guidance():
    data = manifest()
    wbt = Path(data["source"]["training_checkout"]) / data["source"]["wbt_relative_path"]

    def normalize(command):
        return sorted(
            "run_name=<paired>" if item.startswith("run_name=")
            else "++task.rewards.racket_face_guidance_weight=<paired>"
            if item.startswith("++task.rewards.racket_face_guidance_weight=")
            else item
            for item in command
        )

    assert normalize(M.build_command(data, "l2", "A", wbt=wbt)) == normalize(
        M.build_command(data, "l2", "B", wbt=wbt)
    )
    assert normalize(M.build_command(data, "l2", "C", wbt=wbt)) == normalize(
        M.build_command(data, "l2", "D", wbt=wbt)
    )


def minimal_current_contract(data: dict) -> dict:
    contract = {
        "schema_version": 3,
        "actor_obs_contract": "deploy_parity_face179",
        "actor_obs_total_dim": 179,
        "face_command_pairing": "shared_plus_y",
        "mount_normal_sign_per_clip": [1.0, -1.0],
        "strike_phase_per_clip": [0.471, 0.338],
        "motion_kinematics_exact": True,
        "motion_allow_legacy_link_origin_velocity": False,
        "joint_names": [f"j{i}" for i in range(31)],
        "action_joint_ids": list(range(31)),
        "joint_friction_coefficients": [0.0] * 31,
        "motion_clips": [
            {"sha256": data["inputs"]["forehand_motion"]["sha256"]},
            {"sha256": data["inputs"]["backhand_motion"]["sha256"]},
        ],
        "question_bank": {
            "sha256": data["inputs"]["schema3_train_bank"]["sha256"],
            "schema_version": 3,
            "split": "train",
            "exact": True,
        },
        "motion_event_timing": {"mode": "disabled"},
    }
    for key in M.EXPECTED_CURRENT_ONLY_KEYS:
        contract.setdefault(key, {"mode": "disabled"} if key == "motion_event_timing" else 0)
    return contract


def test_hot_contract_transition_allows_only_frozen_current_extensions(tmp_path):
    data = manifest()
    current = minimal_current_contract(data)
    parent = {key: value for key, value in current.items() if key not in M.EXPECTED_CURRENT_ONLY_KEYS}
    parent_path = tmp_path / "parent.json"
    current_path = tmp_path / "current.json"
    parent_path.write_text(json.dumps(parent), encoding="utf-8")
    current_path.write_text(json.dumps(current), encoding="utf-8")
    data["inputs"]["hot_parent_checkpoint"]["adjacent_training_contract_path"] = str(parent_path)
    digest, _ = M.verify_emitted_contract(current_path, data, hot=True)
    assert digest == M.sha256_file(current_path)

    bad = copy.deepcopy(current)
    bad["unregistered_runtime_fact"] = 1
    current_path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(M.ContractError, match="unregistered contract extension"):
        M.verify_emitted_contract(current_path, data, hot=True)

    bad = copy.deepcopy(current)
    bad["actor_obs_total_dim"] = 180
    current_path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(M.ContractError, match="actor_obs_total_dim"):
        M.verify_emitted_contract(current_path, data, hot=True)


def test_fresh_contract_rejects_nonzero_friction_or_unsigned_pairing(tmp_path):
    data = manifest()
    path = tmp_path / "contract.json"
    contract = minimal_current_contract(data)
    contract["joint_friction_coefficients"][10] = 0.1
    path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(M.ContractError, match="zero-friction"):
        M.verify_emitted_contract(path, data, hot=False)

    contract = minimal_current_contract(data)
    contract["face_command_pairing"] = "legacy_signed_vs_A"
    path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(M.ContractError, match="face_command_pairing"):
        M.verify_emitted_contract(path, data, hot=False)


def test_l2_activation_binds_all_four_and_is_no_clobber(tmp_path):
    data = manifest()
    config_sha = M.sha256_file(CONFIG)
    launcher_sha = M.sha256_file(LAUNCHER)
    common_sha = "a" * 64
    cells = {}
    for cell_id, (initialization, guidance, lineage, _role) in M.EXPECTED_CELLS.items():
        terminal = data["stages"]["l1"]["expected_terminal_checkpoint_iteration"][cell_id]
        cells[cell_id] = {
            "run_name": data["stages"]["l1"]["run_names"][cell_id],
            "initialization": initialization,
            "face_guidance_weight": guidance,
            "expected_lineage_exact": lineage,
            "checkpoint_path": f"/tmp/model_{terminal}.pt",
            "checkpoint_sha256": "b" * 64,
            "checkpoint_audit": {
                "iter": terminal,
                "training_contract_schema_version": 3,
                "training_contract_sha256": common_sha,
                "training_contract_lineage_exact": int(lineage),
                "training_contract_provenance_location": "infos",
                "floating_tensor_count": 1,
                "nonfinite_floating_elements": 0,
            },
            "training_contract_sha256": common_sha,
            "launch_state_sha256": "c" * 64,
            "training_log_sha256": "d" * 64,
        }
    content = {
        "manifest_id": data["manifest_id"],
        "manifest_file_sha256": config_sha,
        "launcher_file_sha256": launcher_sha,
        "training_commit": data["source"]["expected_training_commit"],
        "status": "l1_all_four_terminal_l2_blocked_pending_signed_directional_paper",
        "emitted_hard_contract_sha256": common_sha,
        "cells": cells,
    }
    artifact = {
        "artifact_kind": "phase1_signed_face_rescue_l1_activation",
        "schema_version": 1,
        "content": content,
        "content_sha256": M.canonical_sha256(content),
    }
    path = tmp_path / "activation.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    assert M.activation_payload(path, data, config_sha, launcher_sha)["content"] == content

    artifact["content"]["cells"].pop("D")
    artifact["content_sha256"] = M.canonical_sha256(artifact["content"])
    path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(M.ContractError, match="A/B/C/D"):
        M.activation_payload(path, data, config_sha, launcher_sha)


def test_validate_path_can_be_static_and_performs_no_runtime_access(monkeypatch, capsys):
    monkeypatch.setattr(
        M, "runtime_preflight",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("static validation touched runtime")),
    )
    args = [
        "--config", str(CONFIG),
        "--expected-config-sha256", M.sha256_file(CONFIG),
        "--expected-launcher-sha256", M.sha256_file(LAUNCHER),
        "static-validate",
    ]
    assert M.main(args) == 0
    assert "static_valid" in capsys.readouterr().out


def test_missing_or_unreadable_training_checkout_is_concise_contract_error(monkeypatch):
    data = manifest()
    monkeypatch.setattr(
        M,
        "git_output",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            subprocess.CalledProcessError(128, ["git"], output="not a repository")
        ),
    )
    with pytest.raises(M.ContractError, match="missing or unreadable by Git"):
        M.verify_training_source(data)


def test_checkpoint_audit_contract_targets_nested_runner_layout():
    source = M.CHECKPOINT_AUDIT_CODE
    assert "stack = [obj]" in source
    assert "stack.extend(value.values())" in source
    assert "infos = obj.get('infos')" in source
    assert "infos.get('training_contract_sha256')" in source
    assert "obj.get('training_contract_sha256')" not in source


def test_launcher_binds_source_first_environment_and_forbids_local_override():
    source = LAUNCHER.read_text(encoding="utf-8")
    assert 'local_override = wbt / "setup_train_env.local.sh"' in source
    assert '"PYTHONPATH": pythonpath' in source
    assert '"HOPE_WBT_PYTHONPATH": pythonpath' in source
    assert "verify_training_module_resolution" in source
    assert "importlib.util.find_spec('whole_body_tracking')" in source
    assert "import pathlib, whole_body_tracking" not in source
    assert 'environment = preflight["training_environment"].copy()' in source


def test_l2_is_fail_closed_until_separate_signed_directional_paper_activation():
    data = manifest()
    with pytest.raises(M.ContractError, match="L2 is blocked"):
        M.runtime_preflight(
            data,
            CONFIG,
            LAUNCHER,
            config_sha=M.sha256_file(CONFIG),
            launcher_sha=M.sha256_file(LAUNCHER),
            stage_name="l2",
            activation_path=None,
            activation_sha=None,
        )


def test_launcher_contains_no_broad_kill_pull_switch_or_robot_command():
    source = LAUNCHER.read_text(encoding="utf-8")
    for forbidden in ("pkill", "killall", "pgrep -f", "git pull", "git switch", "git checkout"):
        assert forbidden not in source
    assert "os.killpg" not in source
    assert "automatic_judge_launch" in source
    assert "real_robot_commands_forbidden" in source


def test_no_clobber_writer_refuses_existing_file(tmp_path):
    path = tmp_path / "state.json"
    M.write_json_exclusive(path, {"first": True})
    with pytest.raises(FileExistsError):
        M.write_json_exclusive(path, {"second": True})
    assert json.loads(path.read_text()) == {"first": True}


def test_partial_stage_claim_is_preserved_and_blocks_automatic_retry(tmp_path):
    data = manifest()
    data["runtime"]["artifact_root"] = str(tmp_path)
    run_name = data["stages"]["l1"]["run_names"]["A"]
    partial = tmp_path / "runs" / "l1" / run_name
    partial.mkdir(parents=True)
    (partial / "launch_contract.json").write_text("{}", encoding="utf-8")
    with pytest.raises(M.ContractError, match="partial no-clobber claim"):
        M.verify_existing_stage_cell(
            data,
            {"wbt": Path(data["source"]["training_checkout"]) / data["source"]["wbt_relative_path"]},
            config_sha=M.sha256_file(CONFIG),
            launcher_sha=M.sha256_file(LAUNCHER),
            stage_name="l1",
            cell_id="A",
        )
