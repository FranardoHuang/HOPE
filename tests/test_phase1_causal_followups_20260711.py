from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/phase1_causal_followups_20260711.json"
LAUNCHER = ROOT / "scripts/launch_phase1_causal_followups_20260711.py"
WORKER = (
    ROOT
    / "hope_training/whole_body_tracking/scripts/phase1_checkpoint_curve_worker.py"
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launcher = load_module(LAUNCHER, "phase1_causal_followup_launcher")
curve_worker = load_module(WORKER, "phase1_curve_worker_for_followup_test")


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest():
    return launcher.load_manifest(CONFIG)


def arms_by_name(data):
    return {arm["run_name"]: arm for arm in data["arms"]}


def test_manifest_preregisters_four_exact_causal_slots_and_release_gate():
    data = manifest()
    arms = arms_by_name(data)
    assert set(arms) == {
        "phase1_M3_S1_only_guidance0_seed1",
        "phase1_M3_S1_only_guidance0_seed2",
        "phase1_M2_S1_guidance_m095_seed1",
        "phase1_M2_S1_guidance_m095_seed2",
    }
    assert (arms["phase1_M3_S1_only_guidance0_seed1"]["pod"],
            arms["phase1_M3_S1_only_guidance0_seed1"]["gpu"]) == ("pod1", 1)
    assert (arms["phase1_M2_S1_guidance_m095_seed1"]["pod"],
            arms["phase1_M2_S1_guidance_m095_seed1"]["gpu"]) == ("pod2", 0)
    assert (arms["phase1_M2_S1_guidance_m095_seed2"]["pod"],
            arms["phase1_M2_S1_guidance_m095_seed2"]["gpu"]) == ("pod2", 1)
    queued = arms["phase1_M3_S1_only_guidance0_seed2"]
    assert (queued["pod"], queued["gpu"], queued["status"]) == (
        "pod1", 0, "queue_only_waiting_for_exact_predecessor_terminal"
    )
    assert queued["release_gate"]["predecessor_recorded_pgid"] == 1310472
    assert queued["release_gate"]["predecessor_run_name"] == "phase1_M3_old_pairing"


def test_causal_triangles_change_only_the_intended_axes():
    data = manifest()
    triangles = data["causal_triangles"]
    for key in ("M3_swing_seed1", "M3_swing_seed2"):
        triangle = triangles[key]
        assert triangle["old_helper"]["face_command_pairing"] == "legacy_signed_vs_A"
        assert triangle["s1_plus_guidance"]["face_command_pairing"] == "shared_plus_y"
        assert triangle["s1_only"]["face_command_pairing"] == "shared_plus_y"
        assert triangle["old_helper"]["racket_guidance_weight"] == -0.95
        assert triangle["s1_plus_guidance"]["racket_guidance_weight"] == -0.95
        assert triangle["s1_only"]["racket_guidance_weight"] == 0.0
    for key in ("M2_v4rg_legacy_seed1", "M2_v4rg_legacy_seed2"):
        triangle = triangles[key]
        assert triangle["s1_only"]["face_command_pairing"] == "shared_plus_y"
        assert triangle["s1_plus_guidance"]["face_command_pairing"] == "shared_plus_y"
        assert triangle["s1_only"]["racket_guidance_weight"] == 0.0
        assert triangle["s1_plus_guidance"]["racket_guidance_weight"] == -0.95


def test_hard_contract_hashes_reuse_existing_same_pairing_contracts():
    data = manifest()
    m3 = json.loads(
        (ROOT / "configs/phase1_M3_S1_terminal_audit_20260711.json").read_text()
    )
    m2 = json.loads(
        (ROOT / "configs/phase1_M2_S1_terminal_audit_20260711.json").read_text()
    )
    assert data["families"]["M3_swing"]["shared_plus_y_hard_contract_sha256"] == (
        m3["training_contract"]["sha256"]
    )
    assert data["families"]["M2_v4rg_legacy"]["shared_plus_y_hard_contract_sha256"] == (
        m2["training_contract"]["sha256"]
    )
    train = (
        ROOT / "hope_training/whole_body_tracking/scripts/train.py"
    ).read_text(encoding="utf-8")
    hard_builder = train.split("def _build_training_hard_contract", 1)[1].split(
        "def _contract_diff", 1
    )[0]
    assert '"face_command_pairing"' in hard_builder
    assert "racket_guidance_weight" not in hard_builder


def test_training_commands_preserve_recipe_inputs_and_only_change_guidance_seed():
    data = manifest()
    arms = arms_by_name(data)
    m3_seed1 = launcher.build_training_command(
        data, arms["phase1_M3_S1_only_guidance0_seed1"]
    )
    m3_seed2 = launcher.build_training_command(
        data, arms["phase1_M3_S1_only_guidance0_seed2"]
    )
    assert m3_seed1[:5] == [
        "env", "CUDA_VISIBLE_DEVICES=1", "PYTHONUNBUFFERED=1",
        "/workspace/hope_isaac_venv/bin/python", "scripts/train.py",
    ]
    assert "seed=1" in m3_seed1 and "seed=2" in m3_seed2
    assert "++task.racket.face_command_pairing=shared_plus_y" in m3_seed1
    assert "++task.rewards.racket_guidance_weight=0.0" in m3_seed1
    assert "max_iterations=4000" in m3_seed1
    assert "checkpoint_allow_missing_contract=true" in m3_seed1
    assert "checkpoint_allow_contract_mismatch=false" in m3_seed1
    assert "task.plant.zero_joint_friction=false" in m3_seed1
    assert "++task.motion.allow_legacy_link_origin_velocity=true" in m3_seed1

    m2_seed1 = launcher.build_training_command(
        data, arms["phase1_M2_S1_guidance_m095_seed1"]
    )
    m2_seed2 = launcher.build_training_command(
        data, arms["phase1_M2_S1_guidance_m095_seed2"]
    )
    assert "CUDA_VISIBLE_DEVICES=0" in m2_seed1
    assert "CUDA_VISIBLE_DEVICES=1" in m2_seed2
    assert "++task.rewards.racket_guidance_weight=-0.95" in m2_seed1
    assert "seed=1" in m2_seed1 and "seed=2" in m2_seed2
    assert len(data["base_recipe"]) == 69


def test_frozen_launcher_assets_and_recipe_are_reused_verbatim():
    data = manifest()
    source = (
        ROOT / "hope_training/whole_body_tracking/scripts/launch_phase1_20260711.sh"
    ).read_text(encoding="utf-8")
    for family in data["families"].values():
        for key in (
            "parent_checkpoint_sha256", "forehand_motion_sha256",
            "backhand_motion_sha256", "question_bank_sha256",
        ):
            assert family[key] in source
    for override in data["base_recipe"]:
        assert override in source
    locked = ROOT / data["runtime"]["locked_launcher_relative_path"]
    assert file_sha(locked) == data["runtime"]["locked_launcher_sha256"]


def test_q10_is_screen_only_and_q50_is_inactive_triggered_paper():
    data = manifest()
    q10 = data["checkpoint_evaluation"]["q10_screen"]
    assert q10["milestones"] == [17000, 18000, 19000, 20000, 20998]
    assert q10["screen_only"] is True
    assert q10["stop_or_promote_allowed"] is False
    assert q10["extra_args"] == [
        "--schedule-k", "20", "--exam-extra", "--allow-inexact-contract"
    ]
    q50 = data["checkpoint_evaluation"]["q50_decision_paper"]
    assert q50["status"] == "inactive_template"
    assert q50["auto_activate"] is False
    assert q50["schedule_k"] == 100 and q50["attempts_per_side"] == 50
    assert q50["activation_requires_all"]
    assert q50["activation_requires_any_numeric_trigger"]


def test_materialized_q10_manifest_is_worker_valid_and_does_not_launder_parent(tmp_path):
    data = manifest()
    arm = arms_by_name(data)["phase1_M3_S1_only_guidance0_seed1"]
    artifact_root = tmp_path / "artifacts"
    data["runtime"]["artifact_root"] = str(artifact_root)
    family = data["families"][arm["family"]]
    for path_key, sha_key, content in (
        ("parent_checkpoint", "parent_checkpoint_sha256", b"parent"),
        ("forehand_motion", "forehand_motion_sha256", b"fh"),
        ("backhand_motion", "backhand_motion_sha256", b"bh"),
        ("question_bank", "question_bank_sha256", b"bank"),
    ):
        path = artifact_root / family[path_key]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        family[sha_key] = file_sha(path)
    training_run = tmp_path / "training" / arm["run_name"]
    training_run.mkdir(parents=True)
    external_run = tmp_path / "external" / arm["run_name"]
    external_run.mkdir(parents=True)
    q10_path, q50_path, command = launcher.materialize_cadence(
        data, arm, {"worker": "/eval/worker.py", "judge": "/eval/judge.sh"},
        training_run, external_run,
    )
    loaded = curve_worker.load_manifest(q10_path)
    assert [Path(job["checkpoint"]).name for job in loaded["jobs"]] == [
        "model_17000.pt", "model_18000.pt", "model_19000.pt",
        "model_20000.pt", "model_20998.pt",
    ]
    assert not (training_run / "model_16999.pt").exists()
    assert loaded["parent_reference"]["judged_under_new_run_contract"] is False
    assert "--wait-for-checkpoints" in command
    q50 = json.loads(q50_path.read_text())
    assert q50["status"] == "inactive_requires_trigger_evidence"
    assert q50["jobs"] == []


def test_gpu_gate_refuses_a_four_process_gpu_before_any_launch(monkeypatch):
    def fake_output(command, text=True, **_kwargs):
        if "--query-gpu=memory.free" in command:
            return "8000\n"
        return "101\n102\n103\n104\n"

    monkeypatch.setattr(launcher.subprocess, "check_output", fake_output)
    with pytest.raises(launcher.ContractError, match="no fourth-process slot"):
        launcher.gpu_capacity(0, 3, 5500)


def test_gpu_gate_deduplicates_nvidia_smi_rows_and_allows_exactly_three(monkeypatch):
    def fake_output(command, text=True, **_kwargs):
        if "--query-gpu=memory.free" in command:
            return "8000\n"
        return "1346430\n1346430\n1349699\n1349699\n1354525\n1354525\n"

    monkeypatch.setattr(launcher.subprocess, "check_output", fake_output)
    capacity = launcher.gpu_capacity(0, 3, 5500)
    assert capacity["compute_pids"] == [1346430, 1349699, 1354525]


def test_gpu_gate_still_refuses_four_unique_pids_when_rows_are_duplicated(monkeypatch):
    def fake_output(command, text=True, **_kwargs):
        if "--query-gpu=memory.free" in command:
            return "8000\n"
        return "101\n101\n102\n102\n103\n103\n104\n104\n"

    monkeypatch.setattr(launcher.subprocess, "check_output", fake_output)
    with pytest.raises(launcher.ContractError, match="no fourth-process slot"):
        launcher.gpu_capacity(0, 3, 5500)


def test_exact_cleanup_refuses_nonisolated_state_without_signalling(tmp_path, monkeypatch):
    state = tmp_path / "run.log.launch"
    state.write_text("pid=123\npgid=456\ncommand=run_name=arm\n")
    calls = []
    monkeypatch.setattr(launcher.os, "killpg", lambda *args: calls.append(args))
    with pytest.raises(launcher.ContractError, match="pid==pgid"):
        launcher.exact_stop_new_trainer(state, "arm")
    assert calls == []


def test_release_gate_is_read_only_and_never_signals_live_predecessor(tmp_path, monkeypatch):
    state = tmp_path / "run.log.launch"
    log = tmp_path / "run.log"
    checkpoint = tmp_path / "model_20998.pt"
    state.write_text("pid=1310472\npgid=1310472\ncommand=run_name=phase1_M3_old_pairing \n")
    log.write_text("Learning iteration 20998/20999\n")
    checkpoint.write_bytes(b"stable")
    arm = {
        "run_name": "queued",
        "release_gate": {
            "predecessor_launch_state": str(state),
            "predecessor_log": str(log),
            "required_terminal_checkpoint": str(checkpoint),
            "predecessor_recorded_pgid": 1310472,
            "predecessor_run_name": "phase1_M3_old_pairing",
        },
    }
    monkeypatch.setattr(launcher, "process_alive", lambda _pid: True)
    calls = []
    monkeypatch.setattr(launcher.os, "killpg", lambda *args: calls.append(args))
    with pytest.raises(launcher.ContractError, match="remains queue-only"):
        launcher.check_release_gate(arm)
    assert calls == []


def test_validate_path_performs_no_writes(monkeypatch, capsys):
    monkeypatch.setattr(launcher, "preflight", lambda *_args, **_kwargs: {"read_only": True})
    monkeypatch.setattr(
        launcher, "atomic_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("validate wrote state")),
    )
    monkeypatch.setattr(
        launcher, "launch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("validate launched")),
    )
    assert launcher.main([
        "--config", str(CONFIG),
        "--expected-config-sha256", file_sha(CONFIG),
        "--expected-launcher-sha256", file_sha(LAUNCHER),
        "--pod", "pod1", "--arm", "phase1_M3_S1_only_guidance0_seed1",
        "validate",
    ]) == 0
    assert "validated_no_writes" in capsys.readouterr().out


def test_atomic_run_claim_refuses_double_launch_or_preexisting_directory(tmp_path):
    run = tmp_path / "runs" / "arm"
    run.parent.mkdir()
    launcher.claim_run_directory(run)
    with pytest.raises(launcher.ContractError, match="claimed concurrently or already exists"):
        launcher.claim_run_directory(run)


def test_launcher_has_no_broad_kill_or_worktree_mutation_and_auto_starts_cadence():
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "pkill" not in source
    assert "git pull" not in source
    assert "git switch" not in source
    assert "git checkout" not in source
    assert "os.killpg(pid" in source
    assert "start_q10_worker(worker_command" in source
    assert "production manifest and launcher must live outside the training checkout" in source
