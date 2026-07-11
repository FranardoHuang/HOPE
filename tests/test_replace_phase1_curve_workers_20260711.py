from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/phase1_curve_worker_hardening_20260711.json"
TOOL = ROOT / "scripts/replace_phase1_curve_workers_20260711.py"
CURRENT_WORKER = (
    ROOT / "hope_training/whole_body_tracking/scripts/phase1_checkpoint_curve_worker.py"
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


replacement = load_module(TOOL, "phase1_worker_hardening_replacement")


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config():
    return replacement.load_config(CONFIG)


def test_config_binds_four_workers_in_two_pod_atomic_scopes():
    data = config()
    assert data["pods"]["pod1"]["arms"] == [
        "phase1_M3_S1_only_guidance0_seed1",
        "phase1_M3_S1_only_guidance0_seed2",
    ]
    assert data["pods"]["pod2"]["arms"] == [
        "phase1_M2_S1_guidance_m095_seed1",
        "phase1_M2_S1_guidance_m095_seed2",
    ]
    assert len(data["arms"]) == 4
    assert all(arm["first_job_id"].endswith("_17000_clean_q10") for arm in data["arms"])


def test_standalone_worker_sha_is_the_current_screen_policy_hardened_source():
    data = config()
    assert file_sha(CURRENT_WORKER) == (
        data["runtime"]["standalone_hardened_worker_sha256"]
    )
    source = CURRENT_WORKER.read_text(encoding="utf-8")
    for marker in (
        "screen_policy", "manifest_sha256", "job_spec_sha256", "job_contract_sha256"
    ):
        assert marker in source
    assert data["runtime"]["legacy_worker_sha256"] != file_sha(CURRENT_WORKER)


def test_new_state_dirs_are_never_the_legacy_state_dirs():
    paths = config()["arm_relative_paths"]
    assert paths["legacy_state_dir"] == "checkpoint_cadence_q10_state"
    assert paths["hardened_state_dir"] == "checkpoint_cadence_q10_state_hardened_21e3015"
    assert paths["legacy_state_dir"] != paths["hardened_state_dir"]
    assert paths["legacy_worker_log"] != paths["hardened_worker_log"]
    assert paths["legacy_worker_sidecar"] != paths["hardened_worker_sidecar"]


def minimal_manifest(run_name: str, run_dir: Path) -> dict:
    jobs = []
    for milestone in (17000, 18000, 19000, 20000, 20998):
        jobs.append({
            "id": f"{run_name}_{milestone}_clean_q10",
            "run_dir": str(run_dir),
            "checkpoint": str(run_dir / f"model_{milestone}.pt"),
            "gpu": 0,
            "seed": 0,
            "noise_scales": "0.0",
            "extra_args": [
                "--schedule-k", "20", "--exam-extra", "--allow-inexact-contract"
            ],
            "screen_only": True,
            "evaluation_role": "causal_continuation_diagnostic",
            "expected_evaluation_contract_exact": False,
            "formal_target": False,
        })
    return {
        "schema_version": 1,
        "screen_policy": {
            "seed": 0,
            "noise_scales": "0.0",
            "schedule_k": 20,
            "attempts_per_side": 10,
            "screen_only": True,
            "stop_or_promote_allowed": False,
        },
        "jobs": jobs,
    }


def test_manifest_validation_requires_hardened_screen_policy(tmp_path):
    arm = {"run_name": "arm", "first_job_id": "arm_17000_clean_q10"}
    manifest = minimal_manifest("arm", tmp_path)
    first = replacement.validate_manifest(manifest, arm)
    assert first["id"] == "arm_17000_clean_q10"
    manifest["screen_policy"]["stop_or_promote_allowed"] = True
    with pytest.raises(replacement.ContractError, match="lacks the hardened screen policy"):
        replacement.validate_manifest(manifest, arm)


def test_idle_audit_fails_on_any_child_without_signalling(monkeypatch):
    command = ["/usr/bin/python3", "/external/legacy_worker.py"]
    monkeypatch.setattr(replacement, "process_alive", lambda _pid: True)
    monkeypatch.setattr(replacement, "parse_proc_cmdline", lambda _pid: command)
    monkeypatch.setattr(replacement, "proc_children", lambda _pid: [901])
    monkeypatch.setattr(
        replacement,
        "process_table",
        lambda: [
            {"pid": 800, "pgid": 800, "ppid": 1, "args": "worker"},
            {"pid": 901, "pgid": 901, "ppid": 800, "args": "judge.sh"},
        ],
    )
    calls = []
    monkeypatch.setattr(replacement.os, "killpg", lambda *args: calls.append(args))
    with pytest.raises(replacement.ContractError, match="has child/judge"):
        replacement.assert_idle_exact_worker(800, command)
    assert calls == []


def test_exact_term_targets_only_the_two_verified_worker_pgids(monkeypatch):
    commands = {
        801: ["python3", "worker.py", "--manifest", "m1"],
        802: ["python3", "worker.py", "--manifest", "m2"],
    }
    audits = [
        {"arm": {"run_name": "a"}, "old_sidecar": {"pid": 801}},
        {"arm": {"run_name": "b"}, "old_sidecar": {"pid": 802}},
    ]
    snapshots = {
        "a": {"pid": 801, "pgid": 801, "command": commands[801]},
        "b": {"pid": 802, "pgid": 802, "command": commands[802]},
    }
    alive = {801, 802}
    calls = []
    monkeypatch.setattr(replacement, "process_alive", lambda pid: pid in alive)
    monkeypatch.setattr(replacement, "parse_proc_cmdline", lambda pid: commands[pid])

    def killpg(pid, sig):
        calls.append((pid, sig))
        alive.remove(pid)

    monkeypatch.setattr(replacement.os, "killpg", killpg)
    stopped = replacement.exact_term_verified_workers(audits, snapshots, 1)
    assert calls == [(801, signal.SIGTERM), (802, signal.SIGTERM)]
    assert stopped["a"]["signal"] == "SIGTERM"
    assert stopped["b"]["signal"] == "SIGTERM"


def test_hardened_command_changes_only_worker_and_state_dir(tmp_path):
    old = [
        "/usr/bin/python3", "/eval/legacy.py",
        "--manifest", "/runs/arm/checkpoint_cadence_q10.json",
        "--judge-script", "/eval/judge.sh",
        "--state-dir", "/runs/arm/old_state",
        "--wait-for-checkpoints",
    ]
    audit = {"old_sidecar": {"command": old}}
    new = replacement.derive_hardened_command(
        audit, Path("/external/hardened.py"), tmp_path / "new_state"
    )
    assert new[1] == "/external/hardened.py"
    assert replacement.option_value(new, "--state-dir") == str(tmp_path / "new_state")
    assert replacement.option_value(new, "--manifest") == replacement.option_value(
        old, "--manifest"
    )
    changed = [index for index, pair in enumerate(zip(old, new)) if pair[0] != pair[1]]
    assert changed == [1, 7]


def test_audit_binds_launch_contract_manifest_sidecar_and_legacy_17k(tmp_path, monkeypatch):
    data = config()
    arm = dict(data["arms"][0])
    arm["artifact_run_dir"] = str(tmp_path / arm["run_name"])
    root = Path(arm["artifact_run_dir"])
    root.mkdir()
    paths = replacement.arm_paths(data, arm)
    paths["legacy_state_dir"].mkdir()
    training_run = tmp_path / "training_run"
    training_run.mkdir()
    checkpoint = training_run / "model_17000.pt"
    checkpoint.write_bytes(b"checkpoint")
    manifest = minimal_manifest(arm["run_name"], training_run)
    paths["manifest"].write_text(json.dumps(manifest))
    legacy_worker = tmp_path / "legacy_worker.py"
    legacy_worker.write_text("# legacy\n")
    judge = tmp_path / "judge.sh"
    judge.write_text("#!/bin/sh\n")
    command = [
        data["runtime"]["worker_python"], str(legacy_worker),
        "--manifest", str(paths["manifest"]), "--judge-script", str(judge),
        "--state-dir", str(paths["legacy_state_dir"]), "--wait-for-checkpoints",
    ]
    worker = {
        "pid": 777,
        "pgid": 777,
        "command": command,
        "command_sha256": replacement.canonical_sha256(command),
        "state_path": str(paths["legacy_worker_sidecar"]),
    }
    paths["legacy_worker_sidecar"].write_text(json.dumps(worker))
    paths["legacy_worker_log"].write_text("legacy result log\n")
    paths["launch_contract"].write_text(json.dumps({
        "checkpoint_cadence_q10": {
            "path": str(paths["manifest"]), "sha256": file_sha(paths["manifest"]),
        },
        "q10_worker": worker,
    }))
    old_state_path = paths["legacy_state_dir"] / f"{arm['first_job_id']}.json"
    old_state_path.write_text(json.dumps({
        "status": "complete", "returncode": 0,
        "checkpoint_sha256": file_sha(checkpoint),
    }))
    monkeypatch.setattr(
        replacement,
        "assert_idle_exact_worker",
        lambda pid, expected: {
            "pid": pid, "pgid": pid, "command": expected,
            "children": [], "process_group_members": [pid],
        },
    )
    audit = replacement.audit_arm(
        data, arm, {"legacy_worker": legacy_worker, "judge": judge}
    )
    assert audit["manifest_sha256"] == file_sha(paths["manifest"])
    assert audit["immutable_old"]["first_state"]["sha256"] == file_sha(old_state_path)
    assert "worker_log" not in audit["immutable_old"]
    assert audit["legacy_worker_log_path"] == str(paths["legacy_worker_log"])
    assert all(key not in audit["old_first_state"] for key in replacement.HARD_STATE_KEYS)


def test_legacy_wait_log_is_frozen_only_after_term(tmp_path):
    log = tmp_path / "legacy.worker.log"
    log.write_text("waiting\n")
    audit = {"legacy_worker_log_path": str(log), "immutable_old": {}}
    # This append represents a normal legacy-worker wait note after preflight.
    log.write_text("waiting\nstill waiting\n")
    frozen = replacement.freeze_final_legacy_log(audit)
    replacement.verify_old_artifacts_unchanged(audit, frozen)
    log.write_text("waiting\nstill waiting\nunexpected post-TERM write\n")
    with pytest.raises(replacement.ContractError, match="changed after exact TERM"):
        replacement.verify_old_artifacts_unchanged(audit, frozen)


def test_hardened_17k_state_requires_all_three_exact_hashes(tmp_path):
    data = config()
    run_name = "arm"
    manifest = minimal_manifest(run_name, tmp_path / "training")
    first = manifest["jobs"][0]
    state_dir = tmp_path / "new_state"
    state_dir.mkdir()
    arm = {"run_name": run_name, "first_job_id": first["id"]}
    audit = {
        "arm": arm,
        "manifest": manifest,
        "manifest_sha256": "a" * 64,
        "first_job": first,
        "paths": {"hardened_state_dir": state_dir},
    }
    state_path = state_dir / f"{first['id']}.json"
    state = {
        "status": "complete", "returncode": 0,
        "manifest_sha256": audit["manifest_sha256"],
        "job_spec_sha256": replacement.canonical_sha256(first),
        "job_contract_sha256": replacement.canonical_sha256({
            "screen_policy": manifest["screen_policy"], "job": first,
        }),
        "judge_script_sha256": data["runtime"]["judge_sha256"],
        "eval_commit": data["runtime"]["expected_eval_commit"],
        "training_commit": data["runtime"]["expected_training_commit"],
    }
    state_path.write_text(json.dumps(state))
    result = replacement.validate_hard_first_state(audit, data)
    assert result and result["state"]["manifest_sha256"] == "a" * 64
    del state["job_contract_sha256"]
    state_path.write_text(json.dumps(state))
    with pytest.raises(replacement.ContractError, match="job_contract_sha256 mismatch"):
        replacement.validate_hard_first_state(audit, data)


def test_validate_mode_is_read_only(monkeypatch, capsys):
    monkeypatch.setattr(
        replacement,
        "preflight",
        lambda *_args, **_kwargs: ([{"old_process": {"pid": 1, "pgid": 1}}], {
            "transaction_path": "/external/not-created.json"
        }),
    )
    monkeypatch.setattr(
        replacement,
        "replace",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("validate replaced")),
    )
    monkeypatch.setattr(
        replacement,
        "atomic_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("validate wrote")),
    )
    assert replacement.main([
        "--config", str(CONFIG),
        "--expected-config-sha256", file_sha(CONFIG),
        "--expected-tool-sha256", file_sha(TOOL),
        "--pod", "pod1", "validate",
    ]) == 0
    assert "validated_read_only" in capsys.readouterr().out


def test_tool_never_uses_broad_kill_or_targets_trainers_or_judges():
    source = TOOL.read_text(encoding="utf-8")
    assert "pkill" not in source
    assert "SIGKILL" not in source
    assert "git pull" not in source
    assert "git switch" not in source
    assert "os.killpg(pid, signal.SIGTERM)" in source
    assert "start_new_session=True" in source
    assert "never a child/judge/trainer" in source
