"""Dependency-light contract tests for the C2/D2-only signed-face L1 gate."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_phase1_signed_face_cd_l1.py"
MANIFEST = ROOT / "configs/phase1_signed_face_cd_l1_prereg_20260714.json"
SPEC = importlib.util.spec_from_file_location("signed_face_cd_l1", SCRIPT)
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
        "mount_normal_sign_per_clip": [1, -1],
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
        "joint_names": [f"joint_{index}" for index in range(31)],
        "action_joint_ids": list(range(31)),
        "joint_friction_coefficients": [0.0] * 31,
        "motion_clips": [
            {"sha256": m["inputs"]["forehand_motion"]["sha256"]},
            {"sha256": m["inputs"]["backhand_motion"]["sha256"]},
        ],
        "question_bank": {
            "sha256": m["inputs"]["schema3_train_bank"]["sha256"],
            "physics_contract_sha256": m["inputs"]["schema3_train_bank"]["physics_contract_sha256"],
            "source_family_sha256": m["inputs"]["schema3_train_bank"]["source_family_sha256"],
            "schema_version": 3,
            "split": "train",
            "exact": True,
        },
    }


def test_checked_in_manifest_and_static_source_are_exact():
    m = mod.load_manifest(MANIFEST)
    source = mod.verify_static_source(m)
    assert source["commit"] == mod.TRAINING_COMMIT
    assert source["tree"] == mod.TRAINING_TREE
    assert "scripts/train.py" in source["critical_files"]


def test_plan_is_read_only_l1_only_and_pair_is_structurally_matched():
    m = mod.load_manifest(MANIFEST)
    plan = mod.build_plan(m, MANIFEST, SCRIPT)
    assert plan["writes_or_launches_performed"] is False
    assert plan["ordered_cells"] == ["C2", "D2"]
    assert all(value is False for value in plan["decision_boundary"].values())
    assert plan["execution_lanes"]["C2"]["physical_gpu"] == 1
    assert plan["execution_lanes"]["D2"]["physical_gpu"] == 2
    assert plan["kit_boot_serialized_training_may_overlap"] is True
    assert mod.normalized_command(m, "C2") == mod.normalized_command(m, "D2")


@pytest.mark.parametrize("cell_id,weight", [("C2", "0.0"), ("D2", "-0.4")])
def test_command_binds_claim_thread_caps_and_only_requested_weight(cell_id, weight):
    m = manifest()
    claim = "a" * 64
    command = mod.build_command(m, cell_id, claim)
    assert command.count(mod.CARB_THREAD_ARG) == 1
    assert command.count(mod.TBB_THREAD_ARG) == 1
    assert command.count(f"{mod.CLAIM_ARG_PREFIX}{claim}") == 1
    assert command.count("++task.rewards.racket_guidance_weight=0.0") == 1
    assert command.count(f"++task.rewards.racket_face_guidance_weight={weight}") == 1
    assert command.count("max_iterations=25") == 1
    assert command.count("num_envs=512") == 1
    assert command.count("checkpoint_path=null") == 1


def test_atomic_claim_is_non_self_referential_and_binds_source_and_recipe(tmp_path):
    m = manifest()
    m["runtime"]["run_root"] = str(tmp_path)
    cell = mod.cells(m)["D2"]
    arm = tmp_path / cell["run_name"]
    arm.mkdir()
    st = arm.stat()
    claim = mod.build_claim(
        m,
        manifest_sha="1" * 64,
        launcher_sha="2" * 64,
        cell_id="D2",
        arm_dir=arm,
        arm_identity={"device": st.st_dev, "inode": st.st_ino},
    )
    assert claim["training_source"]["commit"] == mod.TRAINING_COMMIT
    assert claim["optimization_recipe"]["signed_face_guidance_weight"] == -0.4
    assert claim["optimization_recipe"]["positional_guidance_weight"] == 0.0
    assert claim["execution_lane"] == {
        "host": "pod1",
        "physical_gpu": 2,
        "cuda_visible_devices": "2",
        "local_training_device": "cuda:0",
        "training_environment_sha256": mod.TRAINING_ENV_SHA256_BY_CELL["D2"],
    }
    assert "training_launch_claim_sha256" not in claim
    assert "launch_contract" not in claim
    assert len(mod.canonical_sha256(claim)) == 64


@pytest.mark.parametrize(
    "mutator",
    [
        lambda m: m.__setitem__("automatic_retry_forbidden", 1),
        lambda m: m.__setitem__("judge_authorized", True),
        lambda m: m["runtime"].__setitem__("maximum_live_trainers_per_assigned_gpu", 3),
        lambda m: m["runtime"]["kit_thread_cap_contract"].__setitem__("carb_tasking_thread_count", 15),
        lambda m: m["cells"][0].__setitem__("gpu", 0),
        lambda m: m["execution_schedule"].__setitem__("previous_cell_terminal_not_required_for_next_claim", False),
        lambda m: m["shared_training_contract"].__setitem__("max_iterations", 26),
        lambda m: m["shared_training_contract"]["base_recipe"].append("unexpected=true"),
        lambda m: m["cells"][1].__setitem__("face_guidance_weight", float("nan")),
        lambda m: m["cells"][1].__setitem__("face_guidance_weight", -0.2),
        lambda m: m["evaluation"].__setitem__("activation", True),
        lambda m: m["historical_read_only_evidence"].__setitem__("prior_claims_or_checkpoints_adopted", True),
        lambda m: m["source"].__setitem__("expected_training_commit", "0" * 40),
        lambda m: m["source"]["critical_files"].__setitem__("scripts/train.py", "0" * 64),
    ],
)
def test_manifest_drift_fails_closed(mutator):
    m = manifest()
    mutator(m)
    with pytest.raises(mod.ContractError):
        mod.validate_manifest(m)


def test_emitted_hard_contract_requires_guidance_recipe(tmp_path):
    m = manifest()
    for cell_id in mod.CELL_IDS:
        path = tmp_path / f"{cell_id}.json"
        path.write_text(json.dumps(hard_contract(m, cell_id)), encoding="utf-8")
        _, observed = mod.verify_hard_contract(path, m, cell_id)
        assert observed["racket_guidance_reward"]["signed_face"]["weight"] == (
            0.0 if cell_id == "C2" else -0.4
        )
    broken = hard_contract(m, "D2")
    del broken["racket_guidance_reward"]
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(mod.ContractError, match="racket_guidance_reward"):
        mod.verify_hard_contract(path, m, "D2")


def test_pair_contracts_differ_only_one_nested_weight():
    m = manifest()
    pair = {cell_id: hard_contract(m, cell_id) for cell_id in mod.CELL_IDS}
    mod.verify_pair_contracts(pair)
    pair["D2"]["actor_obs_total_dim"] = 180
    with pytest.raises(mod.ContractError, match="outside signed-face"):
        mod.verify_pair_contracts(pair)


def test_checkpoint_audit_rejects_missing_claim_and_nonfinite(monkeypatch, tmp_path):
    checkpoint = tmp_path / "model_24.pt"
    checkpoint.write_bytes(b"not-used-by-mocked-audit")
    base = {
        "iter": 24,
        "training_contract_schema_version": 3,
        "training_contract_sha256": "a" * 64,
        "training_contract_lineage_exact": 1,
        "training_launch_claim_sha256": "b" * 64,
        "floating_tensor_count": 1,
        "floating_elements": 2,
        "nonfinite_floating_elements": 0,
    }
    monkeypatch.setattr(mod.subprocess, "check_output", lambda *a, **k: json.dumps(base))
    assert mod.checkpoint_audit(Path("python"), checkpoint)["iter"] == 24
    for key, value in (("training_launch_claim_sha256", None), ("nonfinite_floating_elements", 1)):
        broken = {**base, key: value}
        monkeypatch.setattr(mod.subprocess, "check_output", lambda *a, _v=broken, **k: json.dumps(_v))
        with pytest.raises(mod.ContractError):
            mod.checkpoint_audit(Path("python"), checkpoint)


def test_exclusive_writer_never_clobbers(tmp_path):
    path = tmp_path / "result.json"
    mod.write_json_exclusive(path, {"first": True})
    with pytest.raises(FileExistsError):
        mod.write_json_exclusive(path, {"first": False})
    assert json.loads(path.read_text()) == {"first": True}


def test_runtime_control_must_be_exact_external_read_only_snapshot(tmp_path):
    m = manifest()
    m["runtime"]["external_control_root"] = str(tmp_path)
    config = tmp_path / "phase1_signed_face_cd_l1_prereg_20260714.json"
    launcher = tmp_path / "run_phase1_signed_face_cd_l1.py"
    config.write_text("{}\n", encoding="utf-8")
    launcher.write_text("# gate\n", encoding="utf-8")
    config.chmod(0o444)
    launcher.chmod(0o555)
    receipts = mod.verify_external_control_location(m, config, launcher)
    assert receipts["manifest"]["sha256"] == mod.sha256_file(config)
    config.chmod(0o644)
    with pytest.raises(mod.ContractError, match="read-only"):
        mod.verify_external_control_location(m, config, launcher)


def test_preserved_failure_stops_schedule_without_retry(tmp_path):
    m = manifest()
    m["runtime"]["run_root"] = str(tmp_path)
    c2 = mod.expected_arm_paths(m, "C2")
    c2["arm"].mkdir()
    c2["failure"].write_text("{}\n", encoding="utf-8")
    with pytest.raises(mod.ContractError, match="no automatic retry"):
        mod.choose_next_cell(m, MANIFEST, SCRIPT)


def test_d2_unlocks_after_c2_runtime_verified_not_terminal(monkeypatch, tmp_path):
    m = manifest()
    m["runtime"]["run_root"] = str(tmp_path)
    c2 = mod.expected_arm_paths(m, "C2")
    c2["arm"].mkdir()
    c2["runtime"].write_text("{}\n", encoding="utf-8")
    assert not c2["result"].exists()
    seen = []
    monkeypatch.setattr(
        mod,
        "verify_launch_and_runtime",
        lambda *_args, **_kwargs: seen.append("C2") or {
            "runtime": {"pid": 123, "process_starttime_ticks": 456}
        },
    )
    monkeypatch.setattr(mod, "process_starttime", lambda _pid: 456)
    assert mod.choose_next_cell(m, MANIFEST, SCRIPT) == "D2"
    assert seen == ["C2"]


def test_d2_stays_blocked_if_c2_dies_unfinalized(monkeypatch, tmp_path):
    m = manifest()
    m["runtime"]["run_root"] = str(tmp_path)
    c2 = mod.expected_arm_paths(m, "C2")
    c2["arm"].mkdir()
    c2["runtime"].write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        mod,
        "verify_launch_and_runtime",
        lambda *_args, **_kwargs: {
            "runtime": {"pid": 123, "process_starttime_ticks": 456}
        },
    )
    monkeypatch.setattr(mod, "process_starttime", lambda _pid: -1)
    with pytest.raises(mod.ContractError, match="exited after runtime verification"):
        mod.choose_next_cell(m, MANIFEST, SCRIPT)


def test_launcher_has_no_signal_or_hidden_execution_modes():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "os.kill(" not in source
    assert "subprocess.Popen" not in source
    assert "pkill" not in source
    assert "killall" not in source
    actions = mod.parser()._option_string_actions["--mode"].choices
    assert set(actions) == {
        "plan", "static-validate", "validate-runtime",
        "launch-next", "finalize-cell", "finalize-pair",
    }
    assert not ({"activate", "judge", "l2", "retry"} & set(actions))
