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


def test_config_binds_exact_reviewed_six_live_workers_and_natural_exit_exclusion():
    data = config()
    assert data["contract_id"] == "phase1-global-curve-worker-hardening-20260711-v3"
    assert data["pods"]["pod1"]["queues"] == [
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
        1394810, 1380340, 1397266, 194276, 192815, 195085
    }
    assert len(data["queues"]) == 6
    by_id = {queue["queue_id"]: queue for queue in data["queues"]}
    assert by_id["scaleout_causal_pod1"]["legacy_worker_log"].endswith(
        "/scaleout_causal_pod1_46a0ce2/worker_terminal20998.log"
    )
    assert by_id["scaleout_causal_pod2"]["legacy_worker_log"].endswith(
        "/scaleout_causal_pod2_46a0ce2/worker_terminal20998.log"
    )
    observed_legacy = {
        "cadence_fresh_pod1": "51bbbef708673efd804da47580cc2477c9eb42f3acae138e8d55adba72ff784e",
        "scaleout_causal_pod1": "c8564617872259bdb04fba9b0887d149a6b555b159feb64a3d55b7007f1010bd",
        "scaleout_fresh_pod1": "ec5573d5341cfceca8c5b4d0098fd16ddeef3df1aab096af4b21651d6abaa654",
        "cadence_fresh_pod2": "92193cecc4811f97516de046e48be7749bc2ee939e668c7812d6b11769c21046",
        "scaleout_causal_pod2": "cf3faf27a5a10bcf9f48d4c04de4943df02f7a7685350975d9c8f87319f6ed33",
        "scaleout_fresh_pod2": "84ba5933dafd70e2ed82278e3a174a3a7b1788f570b24bc0c856cf8fad02bb1b",
    }
    for queue in data["queues"]:
        source = ROOT / queue["source_repo_hardened_manifest"]
        assert file_sha(source) == queue["expected_hardened_manifest_sha256"]
        assert queue["expected_legacy_manifest_sha256"] == observed_legacy[
            queue["queue_id"]
        ]
        assert queue["legacy_runtime_manifest"] != queue["hardened_runtime_manifest"]
        assert "/control/global_curve_hardening_v1/manifests/" in queue[
            "hardened_runtime_manifest"
        ]
    exclusion = data["out_of_live_replacement_scope"]
    assert exclusion == [{
        "queue_id": "cadence_causal_pod1",
        "pod": "pod1",
        "former_legacy_pid": 1394150,
        "disposition": "naturally_exited_after_M3_terminal_completion",
        "source_repo_manifest": "configs/phase1_checkpoint_curve_cadence_pod1_20260711.json",
        "expected_manifest_sha256": "b51ddaa50eba3b06893740c2764e98c96d5fbb8751993e95e3f934602c6a36de",
        "legacy_state_dir": "/workspace/codexschema/phase1_fresh_20260711/checkpoint_curves/cadence_pod1_46a0ce2",
        "signal_policy": "No live worker remains; this completed queue must never be required or signalled by this transaction. Preserve and audit its terminal evidence separately.",
    }]
    assert file_sha(ROOT / exclusion[0]["source_repo_manifest"]) == exclusion[0][
        "expected_manifest_sha256"
    ]
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


def test_all_six_live_manifests_pass_screen_and_milestone_barrier_contracts():
    data = config()
    expected_counts = {
        "cadence_fresh_pod1": 8,
        "scaleout_causal_pod1": 8,
        "scaleout_fresh_pod1": 63,
        "cadence_fresh_pod2": 7,
        "scaleout_causal_pod2": 8,
        "scaleout_fresh_pod2": 63,
    }
    for queue in data["queues"]:
        manifest = json.loads(
            (ROOT / queue["source_repo_hardened_manifest"]).read_text()
        )
        summary = replacement.validate_manifest(manifest, queue)
        assert summary["job_count"] == expected_counts[queue["queue_id"]]
        milestones = [group["iteration"] for group in summary["milestone_groups"]]
        assert milestones == sorted(set(milestones))
        if queue["queue_id"].startswith("scaleout_"):
            assert all(group["barrier_id"] for group in summary["milestone_groups"])


def test_manifest_rejects_partial_or_wrong_milestone_barrier():
    data = config()
    queue = next(q for q in data["queues"] if q["queue_id"] == "scaleout_causal_pod1")
    manifest = json.loads((ROOT / queue["source_repo_hardened_manifest"]).read_text())
    broken = copy.deepcopy(manifest)
    broken["jobs"][1]["barrier_id"] = "causal_19000"
    with pytest.raises(replacement.ContractError, match="barrier/milestone mismatch|part"):
        replacement.validate_manifest(broken, queue)
    broken = copy.deepcopy(manifest)
    broken["jobs"][2], broken["jobs"][4] = broken["jobs"][4], broken["jobs"][2]
    with pytest.raises(replacement.ContractError, match="milestone groups"):
        replacement.validate_manifest(broken, queue)


def compact_legacy_from_hardened(
    hardened: dict, *, drop_inexact_escape: bool, drop_metadata: bool = True
) -> dict:
    legacy = copy.deepcopy(hardened)
    if drop_metadata:
        legacy.pop("screen_policy", None)
    for job in legacy["jobs"]:
        if drop_metadata:
            for key in (
                "barrier_id",
                "screen_only",
                "evaluation_role",
                "expected_evaluation_contract_exact",
                "formal_target",
                "training_seed",
                "training_kind",
                "training_family",
                "face_command_pairing",
                "zero_joint_friction",
                "cell",
            ):
                job.pop(key, None)
        if drop_inexact_escape:
            job["extra_args"] = ["--schedule-k", "20"]
    return legacy


def test_compact_cadence_metadata_delta_is_compatible_but_still_rejudged():
    data = config()
    queue = next(q for q in data["queues"] if q["queue_id"] == "cadence_fresh_pod1")
    hardened = json.loads((ROOT / queue["source_repo_hardened_manifest"]).read_text())
    legacy = compact_legacy_from_hardened(hardened, drop_inexact_escape=False)
    result = replacement.validate_legacy_manifest_compatibility(legacy, hardened, queue)
    assert all(item["command_equivalent"] is True for item in result["jobs"])
    assert {
        item["completed_job_policy"] for item in result["jobs"]
    } == {"preserve_legacy_then_rejudge_hardened_exactness_unproven"}


def test_causal_missing_inexact_escape_is_compatible_only_with_hardened_rejudge():
    data = config()
    queue = next(q for q in data["queues"] if q["queue_id"] == "scaleout_causal_pod1")
    hardened = json.loads((ROOT / queue["source_repo_hardened_manifest"]).read_text())
    legacy = compact_legacy_from_hardened(
        hardened, drop_inexact_escape=True, drop_metadata=False
    )
    assert hashlib.sha256(
        (json.dumps(legacy, indent=2) + "\n").encode("utf-8")
    ).hexdigest() == queue["expected_legacy_manifest_sha256"]
    result = replacement.validate_legacy_manifest_compatibility(legacy, hardened, queue)
    assert all(item["command_equivalent"] is False for item in result["jobs"])
    assert {
        item["completed_job_policy"] for item in result["jobs"]
    } == {"preserve_legacy_then_rejudge_hardened_inexact"}
    broken = copy.deepcopy(legacy)
    broken["jobs"][0]["checkpoint"] = broken["jobs"][1]["checkpoint"]
    with pytest.raises(replacement.ContractError, match="not exactly the missing inexact escape"):
        replacement.validate_legacy_manifest_compatibility(broken, hardened, queue)
    broken = copy.deepcopy(legacy)
    broken["jobs"][0]["extra_args"] = ["--schedule-k", "10"]
    with pytest.raises(replacement.ContractError, match="not exactly the missing inexact escape"):
        replacement.validate_legacy_manifest_compatibility(broken, hardened, queue)


def test_worker_command_is_semantically_checked_then_exactly_hash_bound(tmp_path):
    data = config()
    queue = dict(data["queues"][0])
    legacy = tmp_path / "legacy.py"
    judge = tmp_path / "judge.sh"
    command = [
        "python3", str(legacy),
        "--manifest", queue["legacy_runtime_manifest"],
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


def test_relative_worker_argv_is_resolved_only_against_target_proc_cwd(tmp_path):
    data = config()
    queue = dict(data["queues"][0])
    target_cwd = tmp_path / "eval_checkout"
    target_cwd.mkdir()
    relative_worker = Path("hope_training/whole_body_tracking/scripts/phase1_checkpoint_curve_worker.py")
    legacy = (target_cwd / relative_worker).resolve()
    judge = tmp_path / "judge.sh"
    command = [
        "python3", str(relative_worker),
        "--manifest", queue["legacy_runtime_manifest"],
        "--judge-script", str(judge),
        "--state-dir", queue["legacy_state_dir"],
        "--max-active-cpu", "6", "--wait-for-checkpoints",
    ]
    runtime_paths = {"legacy_worker": legacy, "judge": judge.resolve()}
    replacement.validate_worker_command(
        command, queue, data, runtime_paths, process_cwd=target_cwd
    )
    with pytest.raises(replacement.ContractError, match="does not use the legacy worker"):
        replacement.validate_worker_command(
            command, queue, data, runtime_paths, process_cwd=tmp_path / "changed_cwd"
        )
    with pytest.raises(replacement.ContractError, match="requires the target process /proc cwd"):
        replacement.validate_worker_command(command, queue, data, runtime_paths)


def test_idle_worker_with_child_judge_fails_before_any_signal(monkeypatch, tmp_path):
    data = config()
    queue = dict(data["queues"][0])
    queue["legacy_pid_hint"] = 800
    legacy = tmp_path / "legacy.py"
    judge = tmp_path / "judge.sh"
    command = [
        "python3", str(legacy),
        "--manifest", queue["legacy_runtime_manifest"],
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
    monkeypatch.setattr(replacement, "proc_cwd", lambda _pid: tmp_path.resolve())
    monkeypatch.setattr(
        replacement,
        "proc_stdio_log",
        lambda _pid, _path: {
            "path": str(Path(queue["legacy_worker_log"])),
            "device": 1,
            "inode": 2,
            "fd1_path": str(Path(queue["legacy_worker_log"])),
            "fd2_path": str(Path(queue["legacy_worker_log"])),
        },
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
        {
            "queue": queue,
            "process": {
                "command": ["python3", f"{index}.py"],
                "cwd": "/workspace",
                "stdio_log": {"path": f"/logs/{index}.log", "device": 1, "inode": index},
            },
        }
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
        {
            "queue": queue,
            "process": {
                "command": commands[queue["legacy_pid_hint"]], "cwd": "/workspace",
                "stdio_log": {
                    "path": queue["legacy_worker_log"], "device": 1,
                    "inode": queue["legacy_pid_hint"],
                },
            },
        }
        for queue in queues
    ]
    alive = set(commands)

    def audit_worker(queue, *_args, **_kwargs):
        pid = queue["legacy_pid_hint"]
        return {
            "pid": pid, "pgid": pid, "command": commands[pid], "cwd": "/workspace",
            "stdio_log": {"path": queue["legacy_worker_log"], "device": 1, "inode": pid},
        }

    monkeypatch.setattr(replacement, "assert_idle_exact_worker", audit_worker)
    monkeypatch.setattr(replacement, "process_alive", lambda pid: pid in alive)
    monkeypatch.setattr(replacement, "parse_proc_cmdline", lambda pid: commands[pid])
    monkeypatch.setattr(replacement, "proc_cwd", lambda _pid: Path("/workspace"))
    monkeypatch.setattr(
        replacement,
        "proc_stdio_log",
        lambda pid, path: {"path": str(path), "device": 1, "inode": pid},
    )
    monkeypatch.setattr(replacement, "proc_children", lambda _pid: [])
    monkeypatch.setattr(
        replacement,
        "process_table",
        lambda: [
            {"pid": pid, "pgid": pid, "ppid": 1, "args": "worker"} for pid in commands
        ],
    )
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


def test_cwd_race_before_term_fails_with_zero_signals(monkeypatch):
    data = config()
    queue = dict(data["queues"][0])
    pid = queue["legacy_pid_hint"]
    command = ["python3", "relative_worker.py"]
    audit = {
        "queue": queue,
        "process": {
            "command": command,
            "cwd": "/workspace/original",
            "stdio_log": {"path": queue["legacy_worker_log"], "device": 1, "inode": pid},
        },
    }
    monkeypatch.setattr(
        replacement,
        "assert_idle_exact_worker",
        lambda *_args, **_kwargs: {
            "pid": pid, "pgid": pid, "command": command, "cwd": "/workspace/original",
            "stdio_log": {"path": queue["legacy_worker_log"], "device": 1, "inode": pid},
        },
    )
    monkeypatch.setattr(replacement, "process_alive", lambda _pid: True)
    monkeypatch.setattr(replacement, "parse_proc_cmdline", lambda _pid: command)
    monkeypatch.setattr(replacement, "proc_cwd", lambda _pid: Path("/workspace/changed"))
    signals = []
    monkeypatch.setattr(replacement.os, "killpg", lambda *args: signals.append(args))
    with pytest.raises(replacement.ContractError, match="identity changed"):
        replacement.exact_term_verified_workers([audit], data, {})
    assert signals == []


def test_stdio_log_rotation_race_before_term_fails_with_zero_signals(monkeypatch):
    data = config()
    queue = dict(data["queues"][0])
    pid = queue["legacy_pid_hint"]
    command = ["python3", "worker.py"]
    original_log = {
        "path": queue["legacy_worker_log"], "device": 7, "inode": 11,
    }
    audit = {
        "queue": queue,
        "process": {"command": command, "cwd": "/workspace", "stdio_log": original_log},
    }
    monkeypatch.setattr(
        replacement,
        "assert_idle_exact_worker",
        lambda *_args, **_kwargs: {
            "pid": pid, "pgid": pid, "command": command,
            "cwd": "/workspace", "stdio_log": original_log,
        },
    )
    monkeypatch.setattr(replacement, "process_alive", lambda _pid: True)
    monkeypatch.setattr(replacement, "parse_proc_cmdline", lambda _pid: command)
    monkeypatch.setattr(replacement, "proc_cwd", lambda _pid: Path("/workspace"))
    monkeypatch.setattr(
        replacement,
        "proc_stdio_log",
        lambda _pid, path: {"path": str(path), "device": 7, "inode": 12},
    )
    signals = []
    monkeypatch.setattr(replacement.os, "killpg", lambda *args: signals.append(args))
    with pytest.raises(replacement.ContractError, match="identity changed"):
        replacement.exact_term_verified_workers([audit], data, {})
    assert signals == []


def test_frozen_worker_log_rejects_same_path_new_inode(tmp_path):
    worker_log = tmp_path / "worker.log"
    worker_log.write_text("legacy\n")
    manifest_legacy = tmp_path / "legacy.json"
    manifest_hard = tmp_path / "hard.json"
    manifest_legacy.write_text("{}\n")
    manifest_hard.write_text("{}\n")
    identity = replacement.regular_file_identity(worker_log, "test worker log")
    frozen = {
        "evidence": {
            "legacy_worker_log": str(worker_log),
            "legacy_worker_log_identity": identity,
            "legacy_worker_log_sha256": file_sha(worker_log),
            "completed_jobs": [],
            "summary": None,
            "manifests": {
                "legacy": {"path": str(manifest_legacy), "sha256": file_sha(manifest_legacy)},
                "hardened": {"path": str(manifest_hard), "sha256": file_sha(manifest_hard)},
            },
        }
    }
    replacement.verify_frozen_legacy(frozen)
    replacement_path = tmp_path / "replacement.log"
    replacement_path.write_text("rotated\n")
    replacement_path.replace(worker_log)
    with pytest.raises(replacement.ContractError, match="inode changed"):
        replacement.verify_frozen_legacy(frozen)


def make_completed_fixture(tmp_path: Path, *, causal: bool = False):
    data = config()
    queue_id = "scaleout_causal_pod1" if causal else "cadence_fresh_pod1"
    queue = copy.deepcopy(next(q for q in data["queues"] if q["queue_id"] == queue_id))
    hardened_manifest = json.loads(
        (ROOT / queue["source_repo_hardened_manifest"]).read_text()
    )
    hardened_job = hardened_manifest["jobs"][0]
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    iteration = replacement.checkpoint_iteration(hardened_job)
    checkpoint = run_dir / f"model_{iteration}.pt"
    checkpoint.write_bytes(b"finite checkpoint")
    hardened_job["run_dir"] = str(run_dir)
    hardened_job["checkpoint"] = str(checkpoint)
    legacy_manifest = compact_legacy_from_hardened(
        hardened_manifest, drop_inexact_escape=causal, drop_metadata=not causal
    )
    legacy_job = legacy_manifest["jobs"][0]
    compatibility = replacement.validate_legacy_manifest_compatibility(
        legacy_manifest, hardened_manifest, queue
    )["jobs"][0]
    judge = tmp_path / "judge.sh"
    judge.write_text("#!/bin/sh\n")
    state_dir = tmp_path / "legacy_state"
    state_dir.mkdir()
    state_path = state_dir / f"{legacy_job['id']}.json"
    log_path = state_dir / f"{legacy_job['id']}.log"
    # The real legacy shape is an opaque judge log plus state identity; the
    # hardening transaction binds the bytes and does not infer report exactness.
    log_path.write_text("legacy judge completed rc=0\n")
    state = {
        "id": legacy_job["id"],
        "status": "complete",
        "returncode": 0,
        "pid": 456,
        "pgid": 456,
        "command": replacement.expected_judge_command(legacy_job, judge.resolve()),
        "run_dir": legacy_job["run_dir"],
        "checkpoint": legacy_job["checkpoint"],
        "checkpoint_sha256": file_sha(checkpoint),
        "judge_script_sha256": data["runtime"]["judge_sha256"],
        "eval_root": data["runtime"]["eval_checkout"],
        "eval_commit": data["runtime"]["expected_eval_commit"],
        "training_commit": data["runtime"]["expected_training_commit"],
    }
    state_path.write_text(json.dumps(state))
    runtime_paths = {"judge": judge.resolve()}
    return (
        data,
        queue,
        legacy_manifest,
        legacy_job,
        hardened_manifest,
        hardened_job,
        compatibility,
        state_path,
        log_path,
        runtime_paths,
    )


def test_completed_cadence_state_is_attested_but_not_laundered_to_hard_complete(
    tmp_path, monkeypatch
):
    (
        data, queue, legacy_manifest, legacy_job, hardened_manifest, hardened_job,
        compatibility, state_path, log_path, runtime_paths,
    ) = make_completed_fixture(tmp_path)
    monkeypatch.setattr(replacement, "process_alive", lambda _pid: False)
    source = replacement.validate_completed_state(
        state_path,
        log_path,
        legacy_job,
        queue["expected_legacy_manifest_sha256"],
        hardened_job,
        hardened_manifest,
        queue["expected_hardened_manifest_sha256"],
        compatibility,
        data,
        runtime_paths,
    )
    new_state_dir = tmp_path / "new_state"
    new_state_dir.mkdir()
    audit = {
        "queue": queue,
        "hardened_manifest": hardened_manifest,
        "outputs": {"new_state_dir": new_state_dir},
    }
    plan = replacement.migrate_completed_states(audit, {"completed": [source]}, data)
    assert plan[0]["action"] == "preserve_legacy_evidence_and_rejudge_hardened"
    assert plan[0]["legacy_hard_reuse_revoked"] is True
    assert plan[0]["command_equivalent"] is True
    assert plan[0]["source_state"]["sha256"] == file_sha(state_path)
    assert plan[0]["source_log"]["sha256"] == file_sha(log_path)
    assert not (new_state_dir / f"{hardened_job['id']}.json").exists()


def test_completed_causal_18k_without_escape_is_forced_to_hardened_rejudge(
    tmp_path, monkeypatch
):
    (
        data, queue, _legacy_manifest, legacy_job, hardened_manifest, hardened_job,
        compatibility, state_path, log_path, runtime_paths,
    ) = make_completed_fixture(tmp_path, causal=True)
    monkeypatch.setattr(replacement, "process_alive", lambda _pid: False)
    source = replacement.validate_completed_state(
        state_path,
        log_path,
        legacy_job,
        queue["expected_legacy_manifest_sha256"],
        hardened_job,
        hardened_manifest,
        queue["expected_hardened_manifest_sha256"],
        compatibility,
        data,
        runtime_paths,
    )
    assert source["state"]["command"][-2:] == ["--schedule-k", "20"]
    assert replacement.expected_judge_command(hardened_job, runtime_paths["judge"])[-2:] == [
        "--exam-extra", "--allow-inexact-contract"
    ]
    new_state_dir = tmp_path / "new_state"
    new_state_dir.mkdir()
    audit = {
        "queue": queue,
        "hardened_manifest": hardened_manifest,
        "outputs": {"new_state_dir": new_state_dir},
    }
    plan = replacement.migrate_completed_states(audit, {"completed": [source]}, data)
    assert plan[0]["command_equivalent"] is False
    assert plan[0]["legacy_completed_job_policy"] == (
        "preserve_legacy_then_rejudge_hardened_inexact"
    )
    assert not (new_state_dir / f"{hardened_job['id']}.json").exists()


def test_completed_state_with_partial_hard_fields_fails_closed(tmp_path, monkeypatch):
    (
        data, queue, _legacy_manifest, legacy_job, hardened_manifest, hardened_job,
        compatibility, state_path, log_path, runtime_paths,
    ) = make_completed_fixture(tmp_path)
    state = json.loads(state_path.read_text())
    state["manifest_sha256"] = "a" * 64
    state_path.write_text(json.dumps(state))
    monkeypatch.setattr(replacement, "process_alive", lambda _pid: False)
    with pytest.raises(replacement.ContractError, match="unexpectedly emitted hard fields"):
        replacement.validate_completed_state(
            state_path,
            log_path,
            legacy_job,
            queue["expected_legacy_manifest_sha256"],
            hardened_job,
            hardened_manifest,
            queue["expected_hardened_manifest_sha256"],
            compatibility,
            data,
            runtime_paths,
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


def test_hardened_command_changes_only_worker_manifest_and_new_state_dir(tmp_path):
    old = [
        "python3", "/eval/legacy.py",
        "--manifest", "/external/manifest.json",
        "--judge-script", "/eval/judge.sh",
        "--state-dir", "/external/legacy_state",
        "--max-active-cpu", "6", "--wait-for-checkpoints",
    ]
    new = replacement.derive_hardened_command(
        old,
        Path("/external/hardened.py"),
        Path("/external/hardened_manifest.json"),
        tmp_path / "new_state",
    )
    assert new[1] == "/external/hardened.py"
    assert replacement.parse_worker_options(new)["--manifest"] == (
        "/external/hardened_manifest.json"
    )
    assert replacement.parse_worker_options(new)["--state-dir"] == str(tmp_path / "new_state")
    changes = [index for index, pair in enumerate(zip(old, new)) if pair[0] != pair[1]]
    assert changes == [1, 3, 7]


def test_tool_has_no_broad_or_real_hardware_actions():
    source = TOOL.read_text(encoding="utf-8")
    assert "pkill" not in source
    assert "SIGKILL" not in source
    assert "git pull" not in source
    assert "git switch" not in source
    assert "os.killpg(pid, signal.SIGTERM)" in source
    assert "start_new_session=True" in source
    assert 'cwd=audit["process"]["cwd"]' in source
    assert "real hardware" in source
    assert "child/judge" in source
