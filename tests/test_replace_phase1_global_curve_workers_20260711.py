from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import signal

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/phase1_global_curve_worker_hardening_20260711.json"
TOOL = ROOT / "scripts/replace_phase1_global_curve_workers_20260711.py"
CURRENT_WORKER = (
    ROOT / "hope_training/whole_body_tracking/scripts/phase1_checkpoint_curve_worker.py"
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


replacement = load_module(TOOL, "phase1_global_curve_worker_hardening")


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config():
    return replacement.load_config(CONFIG)


def test_config_binds_exact_reviewed_seven_workers_and_tracked_manifests():
    data = config()
    assert data["pods"]["pod1"]["queues"] == [
        "cadence_causal_pod1",
        "cadence_fresh_pod1",
        "scaleout_causal_pod1",
        "scaleout_fresh_pod1",
    ]
    assert data["pods"]["pod2"]["queues"] == [
        "cadence_fresh_pod2",
        "scaleout_causal_pod2",
        "scaleout_fresh_pod2",
    ]
    assert {queue["legacy_pid_hint"] for queue in data["queues"]} == {
        1394150, 1394810, 1380340, 1397266, 194276, 192815, 195085
    }
    for queue in data["queues"]:
        source = ROOT / queue["source_repo_manifest"]
        assert file_sha(source) == queue["expected_manifest_sha256"]
    evidence = data["tracked_evidence"]
    assert "does not create a launch-contract or worker.launch sidecar" in evidence[
        "manual_launch_contract"
    ]
    assert "exact live /proc command" in evidence["sidecar_shape_rule"]


def test_standalone_worker_is_exact_hardened_source_and_legacy_sha_differs():
    data = config()
    assert file_sha(CURRENT_WORKER) == data["runtime"][
        "standalone_hardened_worker_sha256"
    ]
    assert data["runtime"]["legacy_worker_sha256"] != file_sha(CURRENT_WORKER)
    source = CURRENT_WORKER.read_text(encoding="utf-8")
    for marker in (
        "screen_policy", "manifest_sha256", "job_spec_sha256", "job_contract_sha256"
    ):
        assert marker in source


def test_all_seven_manifests_pass_screen_and_milestone_barrier_contracts():
    data = config()
    expected_counts = {
        "cadence_causal_pod1": 4,
        "cadence_fresh_pod1": 8,
        "scaleout_causal_pod1": 8,
        "scaleout_fresh_pod1": 63,
        "cadence_fresh_pod2": 7,
        "scaleout_causal_pod2": 8,
        "scaleout_fresh_pod2": 63,
    }
    for queue in data["queues"]:
        manifest = json.loads((ROOT / queue["source_repo_manifest"]).read_text())
        summary = replacement.validate_manifest(manifest, queue)
        assert summary["job_count"] == expected_counts[queue["queue_id"]]
        milestones = [group["iteration"] for group in summary["milestone_groups"]]
        assert milestones == sorted(set(milestones))
        if queue["queue_id"].startswith("scaleout_"):
            assert all(group["barrier_id"] for group in summary["milestone_groups"])


def test_manifest_rejects_partial_or_wrong_milestone_barrier():
    data = config()
    queue = next(q for q in data["queues"] if q["queue_id"] == "scaleout_causal_pod1")
    manifest = json.loads((ROOT / queue["source_repo_manifest"]).read_text())
    broken = copy.deepcopy(manifest)
    broken["jobs"][1]["barrier_id"] = "causal_19000"
    with pytest.raises(replacement.ContractError, match="barrier/milestone mismatch|part"):
        replacement.validate_manifest(broken, queue)
    broken = copy.deepcopy(manifest)
    broken["jobs"][2], broken["jobs"][4] = broken["jobs"][4], broken["jobs"][2]
    with pytest.raises(replacement.ContractError, match="milestone groups"):
        replacement.validate_manifest(broken, queue)


def test_worker_command_is_semantically_checked_then_exactly_hash_bound(tmp_path):
    data = config()
    queue = dict(data["queues"][0])
    legacy = tmp_path / "legacy.py"
    judge = tmp_path / "judge.sh"
    command = [
        "python3", str(legacy),
        "--manifest", queue["runtime_manifest"],
        "--judge-script", str(judge),
        "--state-dir", queue["legacy_state_dir"],
        "--max-active-cpu", "6",
        "--wait-for-checkpoints",
    ]
    runtime_paths = {"legacy_worker": legacy.resolve(), "judge": judge.resolve()}
    options = replacement.validate_worker_command(command, queue, data, runtime_paths)
    assert options["--wait-for-checkpoints"] is True
    assert replacement.canonical_sha256(command) == replacement.canonical_sha256(
        list(command)
    )
    command.append("--dry-run")
    with pytest.raises(replacement.ContractError, match="unregistered"):
        replacement.validate_worker_command(command, queue, data, runtime_paths)


def test_idle_worker_with_child_judge_fails_before_any_signal(monkeypatch, tmp_path):
    data = config()
    queue = dict(data["queues"][0])
    queue["legacy_pid_hint"] = 800
    legacy = tmp_path / "legacy.py"
    judge = tmp_path / "judge.sh"
    command = [
        "python3", str(legacy),
        "--manifest", queue["runtime_manifest"],
        "--judge-script", str(judge),
        "--state-dir", queue["legacy_state_dir"],
        "--max-active-cpu", "6", "--wait-for-checkpoints",
    ]
    runtime_paths = {"legacy_worker": legacy.resolve(), "judge": judge.resolve()}
    monkeypatch.setattr(replacement, "process_alive", lambda _pid: True)
    monkeypatch.setattr(replacement, "parse_proc_cmdline", lambda _pid: command)
    monkeypatch.setattr(
        replacement, "proc_executable", lambda _pid: Path(data["runtime"]["worker_python"])
    )
    monkeypatch.setattr(replacement, "proc_children", lambda _pid: [901])
    monkeypatch.setattr(
        replacement,
        "process_table",
        lambda: [
            {"pid": 800, "pgid": 800, "ppid": 1, "args": "worker"},
            {"pid": 901, "pgid": 901, "ppid": 800, "args": "judge"},
        ],
    )
    signals = []
    monkeypatch.setattr(replacement.os, "killpg", lambda *args: signals.append(args))
    with pytest.raises(replacement.ContractError, match="child/judge"):
        replacement.assert_idle_exact_worker(queue, data, runtime_paths)
    assert signals == []


def test_pod_atomic_final_preflight_fails_second_worker_with_zero_signals(monkeypatch):
    data = config()
    queues = [dict(data["queues"][0]), dict(data["queues"][1])]
    audits = [
        {"queue": queue, "process": {"command": ["python3", f"{index}.py"]}}
        for index, queue in enumerate(queues)
    ]
    calls = []

    def audit_worker(queue, *_args, **_kwargs):
        calls.append(queue["queue_id"])
        if len(calls) == 2:
            raise replacement.ContractError("worker has child/judge")
        return {"pid": queue["legacy_pid_hint"], "command": audits[0]["process"]["command"]}

    signals = []
    monkeypatch.setattr(replacement, "assert_idle_exact_worker", audit_worker)
    monkeypatch.setattr(replacement.os, "killpg", lambda *args: signals.append(args))
    with pytest.raises(replacement.ContractError, match="child/judge"):
        replacement.exact_term_verified_workers(audits, data, {})
    assert calls == [queues[0]["queue_id"], queues[1]["queue_id"]]
    assert signals == []


def test_exact_term_targets_only_registered_worker_pgids_and_never_kills(monkeypatch):
    data = config()
    queues = [dict(data["queues"][0]), dict(data["queues"][1])]
    commands = {
        queue["legacy_pid_hint"]: ["python3", f"{queue['queue_id']}.py"] for queue in queues
    }
    audits = [
        {"queue": queue, "process": {"command": commands[queue["legacy_pid_hint"]]}}
        for queue in queues
    ]
    alive = set(commands)

    def audit_worker(queue, *_args, **_kwargs):
        pid = queue["legacy_pid_hint"]
        return {"pid": pid, "pgid": pid, "command": commands[pid]}

    monkeypatch.setattr(replacement, "assert_idle_exact_worker", audit_worker)
    monkeypatch.setattr(replacement, "process_alive", lambda pid: pid in alive)
    monkeypatch.setattr(replacement, "parse_proc_cmdline", lambda pid: commands[pid])
    signals = []

    def killpg(pid, sig):
        signals.append((pid, sig))
        alive.remove(pid)

    monkeypatch.setattr(replacement.os, "killpg", killpg)
    stopped = replacement.exact_term_verified_workers(audits, data, {})
    assert signals == [
        (queues[0]["legacy_pid_hint"], signal.SIGTERM),
        (queues[1]["legacy_pid_hint"], signal.SIGTERM),
    ]
    assert all(item["signal"] == "SIGTERM" for item in stopped.values())


def make_completed_fixture(tmp_path: Path):
    data = config()
    manifest = json.loads(
        (ROOT / "configs/phase1_checkpoint_curve_cadence_fresh_pod1_20260711.json").read_text()
    )
    job = copy.deepcopy(manifest["jobs"][0])
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "model_4000.pt"
    checkpoint.write_bytes(b"finite checkpoint")
    job["run_dir"] = str(run_dir)
    job["checkpoint"] = str(checkpoint)
    judge = tmp_path / "judge.sh"
    judge.write_text("#!/bin/sh\n")
    state_dir = tmp_path / "legacy_state"
    state_dir.mkdir()
    state_path = state_dir / f"{job['id']}.json"
    log_path = state_dir / f"{job['id']}.log"
    log_path.write_text("rc=0\n")
    state = {
        "id": job["id"],
        "status": "complete",
        "returncode": 0,
        "pid": 456,
        "pgid": 456,
        "command": replacement.expected_judge_command(job, judge.resolve()),
        "run_dir": job["run_dir"],
        "checkpoint": job["checkpoint"],
        "checkpoint_sha256": file_sha(checkpoint),
        "judge_script_sha256": data["runtime"]["judge_sha256"],
        "eval_root": data["runtime"]["eval_checkout"],
        "eval_commit": data["runtime"]["expected_eval_commit"],
        "training_commit": data["runtime"]["expected_training_commit"],
    }
    state_path.write_text(json.dumps(state))
    runtime_paths = {"judge": judge.resolve()}
    return data, manifest, job, state_dir, state_path, log_path, runtime_paths


def test_completed_legacy_state_is_strictly_validated_and_migrated(tmp_path, monkeypatch):
    data, manifest, job, _state_dir, state_path, log_path, runtime_paths = (
        make_completed_fixture(tmp_path)
    )
    monkeypatch.setattr(replacement, "process_alive", lambda _pid: False)
    manifest_sha = "a" * 64
    source = replacement.validate_completed_state(
        state_path, log_path, job, manifest, manifest_sha, data, runtime_paths
    )
    new_state_dir = tmp_path / "new_state"
    new_state_dir.mkdir()
    audit = {
        "queue": {"expected_manifest_sha256": manifest_sha},
        "manifest": manifest,
        "outputs": {"new_state_dir": new_state_dir},
    }
    migrated = replacement.migrate_completed_states(
        audit, {"completed": [source]}, data
    )
    assert len(migrated) == 1
    state = json.loads(Path(migrated[0]["path"]).read_text())
    assert state["manifest_sha256"] == manifest_sha
    assert state["job_spec_sha256"] == replacement.canonical_sha256(job)
    assert state["job_contract_sha256"] == replacement.canonical_sha256({
        "screen_policy": manifest["screen_policy"], "job": job,
    })
    assert state["source_state"]["sha256"] == file_sha(state_path)
    assert state["source_log"]["sha256"] == file_sha(log_path)
    assert state["provenance_mode"] == "strict_legacy_state_attestation"


def test_completed_state_with_partial_hard_fields_fails_closed(tmp_path, monkeypatch):
    data, manifest, job, _state_dir, state_path, log_path, runtime_paths = (
        make_completed_fixture(tmp_path)
    )
    state = json.loads(state_path.read_text())
    state["manifest_sha256"] = "a" * 64
    state_path.write_text(json.dumps(state))
    monkeypatch.setattr(replacement, "process_alive", lambda _pid: False)
    with pytest.raises(replacement.ContractError, match="partially hardened"):
        replacement.validate_completed_state(
            state_path, log_path, job, manifest, "a" * 64, data, runtime_paths
        )


def test_attestation_binds_exact_evidence_and_rejects_live_change(tmp_path):
    data = config()
    audit = {
        "queue": {"queue_id": "q"},
        "evidence": {"process": {"pid": 1, "command_sha256": "a" * 64}},
    }
    summary = {"pod": "pod1"}
    document = replacement.attestation_document(
        data, "pod1", [audit], summary, config_sha="b" * 64, tool_sha="c" * 64
    )
    path = tmp_path / "attestation.json"
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    digest = file_sha(path)
    replacement.bind_attestation(
        path, digest, data, "pod1", [audit], config_sha="b" * 64, tool_sha="c" * 64
    )
    changed = copy.deepcopy(audit)
    changed["evidence"]["process"]["pid"] = 2
    with pytest.raises(replacement.ContractError, match="evidence changed"):
        replacement.bind_attestation(
            path, digest, data, "pod1", [changed],
            config_sha="b" * 64, tool_sha="c" * 64,
        )


def test_hardened_command_changes_only_worker_and_new_state_dir(tmp_path):
    old = [
        "python3", "/eval/legacy.py",
        "--manifest", "/external/manifest.json",
        "--judge-script", "/eval/judge.sh",
        "--state-dir", "/external/legacy_state",
        "--max-active-cpu", "6", "--wait-for-checkpoints",
    ]
    new = replacement.derive_hardened_command(
        old, Path("/external/hardened.py"), tmp_path / "new_state"
    )
    assert new[1] == "/external/hardened.py"
    assert replacement.parse_worker_options(new)["--manifest"] == "/external/manifest.json"
    assert replacement.parse_worker_options(new)["--state-dir"] == str(tmp_path / "new_state")
    changes = [index for index, pair in enumerate(zip(old, new)) if pair[0] != pair[1]]
    assert changes == [1, 7]


def test_tool_has_no_broad_or_real_hardware_actions():
    source = TOOL.read_text(encoding="utf-8")
    assert "pkill" not in source
    assert "SIGKILL" not in source
    assert "git pull" not in source
    assert "git switch" not in source
    assert "os.killpg(pid, signal.SIGTERM)" in source
    assert "start_new_session=True" in source
    assert "real hardware" in source
    assert "child/judge" in source
