from __future__ import annotations

import importlib.util
import fcntl
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_phase1_task_revision_0p5_exam.py"
QUEUE = ROOT / "configs" / "phase1_task_revision_supercombo_20260716.yaml"
ACTIVATION = ROOT / "configs" / "phase1_task_revision_0p5_exam_activation_v1_20260717.json"


def _load_module():
    spec = importlib.util.spec_from_file_location("task_revision_0p5_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


E = _load_module()


def _remote_namespace():
    namespace = {"__name__": "taskrev_remote_under_test", "__file__": "embedded.py"}
    exec(compile(E.REMOTE_PROGRAM, "embedded.py", "exec"), namespace)
    if namespace["_RENAMEAT2"] is None:
        def test_only_renameat2(old_fd, old_name, new_fd, new_name, _flags):
            old = os.fsdecode(old_name)
            new = os.fsdecode(new_name)
            try:
                os.stat(new, dir_fd=new_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                namespace["ctypes"].set_errno(namespace["errno"].EEXIST)
                return -1
            os.rename(old, new, src_dir_fd=old_fd, dst_dir_fd=new_fd)
            return 0
        namespace["_RENAMEAT2"] = test_only_renameat2
    return namespace


def _committed_spec(tmp_path, name="state"):
    return {
        "state_dir": str(tmp_path / name),
        "output_dir": str(tmp_path / f"{name}.output"),
        "job_id": "taskrev_p2_equal_reward",
        "milestone": 5700,
        "activation": {
            "activation_id": "fixed", "path": "fixed.json", "sha256": "a" * 64,
        },
        "catastrophic_cleanup": {
            "requires_delegated_cgroup_v2": True,
            "guardian_required": True,
            "cgroup_kill_required": True,
            "trusted_single_operator_filesystem": True,
            "unsupported_result": "NO_LAUNCH",
        },
        "supervision": {"heartbeat_seconds": 30.0},
    }


def _publish_committed_chain(remote, spec, *, include_ack=True, wrong_ack=False):
    state = Path(spec["state_dir"])
    state.mkdir()
    (state / "supervisor.log").write_bytes(b"")
    guard = remote["open_directory_guard"](state)
    pid = 4242
    hello = {
        "schema_version": 1,
        "artifact_kind": "taskrev_0p5_supervisor_hello",
        "plan_sha256": remote["canonical"](spec),
        "activation": spec["activation"],
        "job_id": spec["job_id"],
        "milestone": spec["milestone"],
        "pid": pid,
        "pgid": pid,
        "proc_start_ticks": 12345,
        "argv_sha256": remote["canonical"](["exact-supervisor"]),
        "commit_deadline_monotonic_ns": 2_000_000,
        "automatic_retry": False,
    }
    remote["guarded_publish_json"](guard, "child_hello.json", hello)
    hello_sha = remote["guarded_sha"](guard, "child_hello.json", "hello")
    ledger = {
        "schema_version": 1,
        "artifact_kind": "taskrev_0p5_launch_ledger",
        "plan_sha256": remote["canonical"](spec),
        "activation": spec["activation"],
        "job_id": spec["job_id"],
        "milestone": spec["milestone"],
        "pid": pid,
        "pgid": pid,
        "proc_start_ticks": 12345,
        "hello_sha256": hello_sha,
        "output_dir": spec["output_dir"],
        "state_dir": spec["state_dir"],
        "resource_samples_before_fork": [],
        "committed_utc": "now",
        "automatic_retry": False,
    }
    remote["guarded_publish_json"](guard, "launch_ledger.json", ledger)
    ledger_sha = remote["guarded_sha"](guard, "launch_ledger.json", "ledger")
    token = {
        "schema_version": 1,
        "artifact_kind": "taskrev_0p5_commit_token",
        "pid": pid,
        "pgid": pid,
        "proc_start_ticks": 12345,
        "hello_sha256": hello_sha,
        "ledger_sha256": ledger_sha,
        "nonce": "b" * 64,
        "published_utc": "now",
        "retry_authorized": False,
    }
    remote["guarded_publish_json"](guard, "commit_token.json", token)
    token_sha = remote["guarded_sha"](guard, "commit_token.json", "token")
    decision = {
        "schema_version": 1,
        "artifact_kind": "taskrev_0p5_launch_decision",
        "decision": "commit",
        "plan_sha256": remote["canonical"](spec),
        "pid": pid,
        "pgid": pid,
        "proc_start_ticks": 12345,
        "hello_sha256": hello_sha,
        "ledger_sha256": ledger_sha,
        "token_sha256": token_sha,
        "commit_deadline_monotonic_ns": 2_000_000,
        "decided_monotonic_ns": 1_000_000,
        "decided_utc": "now",
        "retry_authorized": False,
    }
    remote["guarded_publish_json"](guard, "launch_decision.json", decision)
    decision_sha = remote["guarded_sha"](
        guard, "launch_decision.json", "decision")
    if include_ack:
        ack = {
            "schema_version": 1,
            "artifact_kind": "taskrev_0p5_supervisor_commit_ack",
            "plan_sha256": remote["canonical"](spec),
            "pid": pid,
            "pgid": pid,
            "proc_start_ticks": 12345,
            "hello_sha256": hello_sha,
            "ledger_sha256": ledger_sha,
            "token_sha256": token_sha,
            "decision_sha256": "0" * 64 if wrong_ack else decision_sha,
            "acknowledged_utc": "now",
            "kit_lock_held": True,
            "resource_samples": [],
            "catastrophic_cleanup": {
                "contract": spec["catastrophic_cleanup"],
                "cgroup_path": "/sys/fs/cgroup/exact",
                "cgroup_exact_members": [pid],
                "supervisor_contained": True,
                "guardian": {
                    "pid": 4343, "pgid": 4343, "start_ticks": 54321,
                    "argv": ["guardian"],
                },
                "guardian_live_exact": True,
            },
            "automatic_retry": False,
        }
        remote["guarded_publish_json"](guard, "commit_ack.json", ack)
    return guard, hello


def test_build_plan_is_fixed_to_pod2_equal_reward_model5700():
    plan = E.build_plan(QUEUE, activation_path=ACTIVATION, eval_gpu=2)

    assert plan["job_id"] == "taskrev_p2_equal_reward"
    assert plan["milestone"] == 5700
    assert plan["milestone_offset_from_parent"] == 1000
    assert plan["pod"] == "pod2"
    assert plan["training_gpu"] == 0
    assert plan["eval_gpu"] == 2
    assert plan["expected_claim_content_sha256"] == (
        "e10d2c248d90daa3172ea80147a394dad64ce326eb4052889c25bfb9d3df420b"
    )
    assert plan["output_dir"].endswith("/p2_equal_reward/timing_exam_0p5/model_5700")
    assert plan["behavior_receipt"].endswith("/behavior_milestones/model_5700.json")
    assert plan["automatic_retry"] is False
    assert plan["formal_evidence_eligible"] is False
    assert plan["evaluation_contract_exact"] is False
    assert plan["catastrophic_cleanup"] == {
        "requires_delegated_cgroup_v2": True,
        "guardian_required": True,
        "cgroup_kill_required": True,
        "trusted_single_operator_filesystem": True,
        "unsupported_result": "NO_LAUNCH",
    }
    assert plan["source_closure"]["evaluator"] == (
        "67300ba2faae0f3443496219f1c6cf3fcc16afa182b45e6f95d4fbb82c60c094"
    )


@pytest.mark.parametrize("gpu", [-1, 3, 99])
def test_build_plan_rejects_unapproved_eval_gpu(gpu):
    with pytest.raises(E.ExamError, match="activation-authorized"):
        E.build_plan(QUEUE, activation_path=ACTIVATION, eval_gpu=gpu)


def test_build_plan_rejects_execution_from_copied_harness(tmp_path):
    copied = tmp_path / SCRIPT.name
    copied.write_bytes(SCRIPT.read_bytes())
    spec = importlib.util.spec_from_file_location("copied_taskrev_harness", copied)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    with pytest.raises(module.ExamError, match="activation-approved harness path"):
        module.build_plan(QUEUE, activation_path=ACTIVATION, eval_gpu=0)


def test_activation_rejects_candidate_or_retry_drift(tmp_path):
    value = json.loads(ACTIVATION.read_text())
    value["selection"]["job_id"] = "taskrev_p2_velocity_reward"
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(value))
    with pytest.raises(E.ExamError, match="fixed equal-reward"):
        E.build_plan(QUEUE, activation_path=changed, eval_gpu=0)

    value = json.loads(ACTIVATION.read_text())
    value["authority"]["automatic_retry"] = True
    changed.write_text(json.dumps(value))
    with pytest.raises(E.ExamError, match="authority differs"):
        E.build_plan(QUEUE, activation_path=changed, eval_gpu=0)

    value = json.loads(ACTIVATION.read_text())
    value["catastrophic_cleanup"]["guardian_required"] = False
    changed.write_text(json.dumps(value))
    with pytest.raises(E.ExamError, match="catastrophic-cleanup"):
        E.build_plan(QUEUE, activation_path=changed, eval_gpu=0)


def test_plan_and_nonexecuted_launch_are_local_dry_runs(monkeypatch, capsys):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("dry-run attempted SSH")

    monkeypatch.setattr(E, "remote_action", forbidden)
    assert E.main(["--queue", str(QUEUE), "--activation", str(ACTIVATION),
                   "--eval-gpu", "1", "plan"]) == 0
    assert json.loads(capsys.readouterr().out)["dry_run"] is True
    assert E.main(["--queue", str(QUEUE), "--activation", str(ACTIVATION),
                   "--eval-gpu", "1", "launch"]) == 0
    assert json.loads(capsys.readouterr().out)["dry_run"] is True


def test_launch_requires_exact_confirmation_before_ssh(monkeypatch, capsys):
    monkeypatch.setattr(
        E, "remote_action", lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("bad confirmation reached SSH")))
    rc = E.main(["--queue", str(QUEUE), "--activation", str(ACTIVATION),
                 "--eval-gpu", "1", "launch", "--execute", "--confirm", "WRONG"])
    assert rc == 2
    assert "confirmation token mismatch" in capsys.readouterr().err


def test_bounded_ssh_timeout_marks_launch_unknown_never_retry(monkeypatch):
    plan = E.build_plan(QUEUE, activation_path=ACTIVATION, eval_gpu=1)

    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd="ssh", timeout=60)

    monkeypatch.setattr(E.subprocess, "run", timeout)
    with pytest.raises(E.ExamError, match="UNKNOWN.*never replayed"):
        E.remote_action(plan, action="launch")


def test_remote_contract_is_persistent_bounded_receipt_gated_and_no_retry():
    source = E.REMOTE_PROGRAM
    assert "runtime.attest_milestone" not in source
    assert "pre-existing checkpoint milestone receipt is required" in source
    assert "planner_initial_tts_exact_0p5_count" in source
    assert "stable_resource_gate(spec)" in source
    assert "fcntl.LOCK_EX | fcntl.LOCK_NB" in source
    assert "os.fork()" in source
    assert "os.setsid()" in source
    assert "child_hello.json" in source
    assert "launch_ledger.json" in source
    assert "commit_token.json" in source
    assert "commit_ack.json" in source
    assert "heartbeat.jsonl" in source
    assert "stdout=evaluator_fd" in source
    assert "subprocess.PIPE" not in source
    assert "evaluator_total_timeout_seconds" in source
    assert "forced_after_complete_handshake" in source
    assert '"automatic_retry": False' in source
    assert '"trainer_or_robot_signals": []' in source
    assert "pkill" not in source
    assert "killall" not in source
    assert source.count("os.kill(pid, signal.SIGKILL)") == 1
    assert "os.killpg(" in source
    assert "waitid(WNOWAIT)" in source
    assert "group_empty_confirmed" in source
    assert "revalidate_directory_guard" in source
    assert "cleanup_unproven_quarantine" in source
    assert "cleanup_quarantine.json" in source
    assert "prepare_owned_cgroup(spec)" in source
    assert "cgroup.kill" in source
    assert "cgroup.events" in source
    assert "start_cgroup_guardian" in source
    assert "guardian_live_exact" in source
    assert "pass_fds=(cgroup[\"child\"][\"fd\"],)" in source
    assert "child_preexec(expected_parent_pid, cgroup_dir_fd)" in source


def test_exact_signal_refuses_identity_drift_without_signalling(monkeypatch):
    remote = _remote_namespace()
    calls = []
    remote["exact_live"] = lambda _identity: None
    remote["leader_exited_unreaped"] = lambda _proc: False
    remote["os"].killpg = lambda *args: calls.append(args)

    class Proc:
        returncode = None

        def poll(self):
            return None

    with pytest.raises(RuntimeError, match="identity changed"):
        remote["exact_signal_evaluator"](
            {"pid": 101, "pgid": 101, "sid": 101, "ppid": 99,
             "start_ticks": 9, "argv": ["python"]},
            Proc(), 1.0)
    assert calls == []


@pytest.mark.parametrize("prctl_rc,parent_matches,expected_exit", [
    (1, True, 126),
    (0, False, 125),
])
def test_child_preexec_fails_closed_on_prctl_or_parent_race(
        monkeypatch, prctl_rc, parent_matches, expected_exit):
    remote = _remote_namespace()

    class ChildExit(BaseException):
        def __init__(self, code):
            self.code = code

    monkeypatch.setattr(remote["os"], "setsid", lambda: None)
    monkeypatch.setattr(remote["os"], "getppid", lambda: 77 if parent_matches else 78)
    monkeypatch.setattr(remote["os"], "_exit", lambda code: (_ for _ in ()).throw(ChildExit(code)))
    remote["_PRCTL"] = lambda *_args: prctl_rc
    with pytest.raises(ChildExit) as caught:
        remote["child_preexec"](77, -1)()
    assert caught.value.code == expected_exit


def test_child_preexec_moves_child_into_held_cgroup_before_exec(monkeypatch, tmp_path):
    remote = _remote_namespace()
    cgroup = tmp_path / "owned"
    cgroup.mkdir()
    (cgroup / "cgroup.procs").write_bytes(b"")
    cgroup_fd = os.open(cgroup, os.O_RDONLY)

    class ChildExit(BaseException):
        pass

    monkeypatch.setattr(remote["os"], "setsid", lambda: None)
    monkeypatch.setattr(remote["os"], "getppid", lambda: 77)
    monkeypatch.setattr(
        remote["os"], "_exit",
        lambda code: (_ for _ in ()).throw(ChildExit(code)))
    remote["_PRCTL"] = lambda *_args: 0
    remote["child_preexec"](77, cgroup_fd)()
    assert (cgroup / "cgroup.procs").read_bytes() == b"0\n"


def test_guardian_parent_death_kills_cgroup_before_removal_and_lock_release(monkeypatch):
    remote = _remote_namespace()
    events = []
    writes = []
    parent_fd, child_fd = 71, 72
    cgroup = {
        "parent": {"fd": parent_fd}, "child": {"fd": child_fd},
        "name": "owned", "path": "/sys/fs/cgroup/owned",
    }
    monkeypatch.setattr(remote["os"], "setsid", lambda: None)
    monkeypatch.setattr(remote["os"], "getppid", lambda: 99)
    monkeypatch.setattr(remote["signal"], "signal", lambda *_args: None)
    monkeypatch.setitem(remote, "close_fds_except", lambda _keep: None)
    monkeypatch.setattr(remote["select"], "select", lambda *_args: ([73], [], []))
    monkeypatch.setattr(remote["os"], "read", lambda _fd, _count: b"")
    monkeypatch.setattr(
        remote["os"], "write",
        lambda fd, raw: (writes.append((fd, raw)) or len(raw)))
    population = iter([True, False])
    monkeypatch.setitem(remote, "cgroup_populated", lambda _child: next(population))
    monkeypatch.setattr(remote["os"], "open", lambda *_args, **_kwargs: 76)
    monkeypatch.setattr(
        remote["os"], "fstat",
        lambda _fd: type("Info", (), {"st_mode": 0o100600})())
    monkeypatch.setattr(remote["os"], "close", lambda _fd: None)
    monkeypatch.setitem(
        remote, "guardian_kill_cgroup_and_wait",
        lambda _kill_fd, _child: events.append("kill_and_zero"))
    monkeypatch.setitem(
        remote, "remove_owned_cgroup", lambda _cgroup: events.append("remove"))
    remote["_PRCTL"] = lambda *_args: 0

    assert remote["guardian_main"](99, 73, 74, cgroup, 75) == 0
    assert writes == [(74, b"R"), (74, b"K0")]
    assert events == ["kill_and_zero", "remove"]


def test_guardian_normal_finish_refuses_silent_population(monkeypatch):
    remote = _remote_namespace()
    events = []
    writes = []
    cgroup = {
        "parent": {"fd": 71}, "child": {"fd": 72},
        "name": "owned", "path": "/sys/fs/cgroup/owned",
    }
    monkeypatch.setattr(remote["os"], "setsid", lambda: None)
    monkeypatch.setattr(remote["os"], "getppid", lambda: 99)
    monkeypatch.setattr(remote["signal"], "signal", lambda *_args: None)
    monkeypatch.setitem(remote, "close_fds_except", lambda _keep: None)
    monkeypatch.setattr(remote["select"], "select", lambda *_args: ([73], [], []))
    monkeypatch.setattr(remote["os"], "read", lambda _fd, _count: b"F")
    monkeypatch.setattr(
        remote["os"], "write",
        lambda fd, raw: (writes.append((fd, raw)) or len(raw)))
    population = iter([True, False])
    monkeypatch.setitem(remote, "cgroup_populated", lambda _child: next(population))
    monkeypatch.setattr(remote["os"], "open", lambda *_args, **_kwargs: 76)
    monkeypatch.setattr(
        remote["os"], "fstat",
        lambda _fd: type("Info", (), {"st_mode": 0o100600})())
    monkeypatch.setattr(remote["os"], "close", lambda _fd: None)
    monkeypatch.setitem(
        remote, "guardian_kill_cgroup_and_wait",
        lambda _kill_fd, _child: events.append("kill_and_zero"))
    monkeypatch.setitem(
        remote, "remove_owned_cgroup", lambda _cgroup: events.append("remove"))
    remote["_PRCTL"] = lambda *_args: 0

    assert remote["guardian_main"](99, 73, 74, cgroup, 75) == 0
    assert writes == [(74, b"R"), (74, b"K0")]
    assert events == ["kill_and_zero", "remove"]


def test_exact_signal_waits_for_entire_owned_group_before_reaping(monkeypatch):
    remote = _remote_namespace()
    leader = {"pid": 101, "pgid": 101, "sid": 101, "ppid": 77,
              "state": "S", "start_ticks": 9, "argv": ["python", "exam.py"]}
    child = {"pid": 102, "pgid": 101, "sid": 101, "ppid": 101,
             "state": "S", "start_ticks": 10, "argv": ["kit-child"]}
    zombie = dict(leader, state="Z")
    snapshots = iter([[leader, child], [zombie, child], [zombie], []])
    monkeypatch.setitem(remote, "refresh_owned_group", lambda *_args: next(snapshots))
    monkeypatch.setitem(remote, "exact_live", lambda _identity: leader)
    exited = iter([False, True])
    monkeypatch.setitem(remote, "leader_exited_unreaped", lambda _proc: next(exited))
    times = iter([0.0, 2.0, 2.0, 2.1])
    monkeypatch.setattr(remote["time"], "monotonic", lambda: next(times))
    calls = []
    monkeypatch.setattr(remote["os"], "killpg", lambda pgid, sig: calls.append((pgid, sig)))

    class Proc:
        pid = 101
        returncode = None

        def wait(self, timeout=None):
            self.returncode = -9
            return self.returncode

    receipt = remote["exact_signal_evaluator"](
        leader, Proc(), 1.0, owned={101: remote["_owned_fingerprint"](leader)})
    assert calls == [(101, remote["signal"].SIGTERM), (101, remote["signal"].SIGKILL)]
    assert receipt["group_empty_confirmed"] is True
    assert [row["pid"] for row in receipt["members_before_term"]] == [101, 102]
    assert [row["pid"] for row in receipt["members_before_kill"]] == [101, 102]


def test_owned_group_rejects_pid_reuse_and_unproven_member(monkeypatch):
    remote = _remote_namespace()
    leader = {"pid": 101, "pgid": 101, "sid": 101, "ppid": 77,
              "state": "S", "start_ticks": 9, "argv": ["python", "exam.py"]}
    child = {"pid": 102, "pgid": 101, "sid": 101, "ppid": 101,
             "state": "S", "start_ticks": 10, "argv": ["kit-child"]}
    owned = {
        101: remote["_owned_fingerprint"](leader),
        102: remote["_owned_fingerprint"](child),
    }
    reused = dict(child, start_ticks=11)
    monkeypatch.setitem(remote, "_scan_process_group", lambda _pgid: [leader, reused])
    with pytest.raises(RuntimeError, match="identity drifted"):
        remote["refresh_owned_group"](leader, dict(owned))

    foreign = {"pid": 103, "pgid": 101, "sid": 101, "ppid": 1,
               "state": "S", "start_ticks": 12, "argv": ["foreign"]}
    monkeypatch.setitem(remote, "_scan_process_group", lambda _pgid: [leader, foreign])
    with pytest.raises(RuntimeError, match="cannot be proven"):
        remote["refresh_owned_group"](
            leader, {101: remote["_owned_fingerprint"](leader)})


def test_converter_binding_error_runs_local_owned_cleanup(monkeypatch, tmp_path):
    remote = _remote_namespace()
    output = tmp_path / "output"
    output.mkdir()
    (output / "isaac_timing_0p5.json").write_text("{}")
    (output / "isaac_timing_0p5.csv").write_text("x\n")
    (output / "evaluator.log").write_text("done\n")
    guard = remote["open_directory_guard"](output)
    fake = type("FakePopen", (), {"pid": 444, "returncode": None})()
    monkeypatch.setattr(remote["subprocess"], "Popen", lambda *_a, **_k: fake)
    monkeypatch.setitem(remote, "source_environment", lambda _source: {})
    monkeypatch.setitem(remote, "complete_handshake", lambda *_a, **_k: True)
    monkeypatch.setitem(remote, "guardian_live_exact", lambda _guardian: {})
    monkeypatch.setitem(remote, "cgroup_populated", lambda _child: False)
    monkeypatch.setitem(remote, "require_owned_cgroup_members", lambda *_a, **_k: None)
    identity = {"pid": 444, "pgid": 444, "sid": 444, "ppid": 77,
                "state": "S", "start_ticks": 12, "argv": ["wrong"]}

    def fail_bind(_proc, _command, _parent, seconds=5.0, identity_sink=None):
        identity_sink["identity"] = identity
        raise RuntimeError("spawned child argv differs")

    cleaned = []
    monkeypatch.setitem(remote, "bind_owned_leader", fail_bind)
    monkeypatch.setitem(
        remote, "close_owned_child",
        lambda proc, bound, owned, signals, grace, target: cleaned.append(
            (proc, bound, dict(owned), target)))
    spec = {
        "output_dir": str(output),
        "source_closure": {"spec": "a" * 64},
        "paper": {"path": str(tmp_path / "paper.json"), "file_sha256": "b" * 64},
        "supervision": {"converter_timeout_seconds": 10.0,
                        "exact_term_grace_seconds": 1.0},
    }
    context = {
        "source": tmp_path, "schedule": tmp_path / "schedule.json",
        "checkpoint": tmp_path / "model.pt", "checkpoint_sha": "c" * 64,
        "hard_path": tmp_path / "hard.json", "hard_sha": "d" * 64,
    }
    try:
        with pytest.raises(RuntimeError, match="argv differs"):
            remote["validate_and_convert"](
                spec, context, evaluator_state={}, signals=[], gpu_samples=[],
                started="now", output_guard=guard, stop_requested={"signal": None},
                cgroup={"child": {"fd": -1}, "path": "/cg"}, guardian={})
        assert len(cleaned) == 1
        assert cleaned[0][0] is fake
        assert cleaned[0][1] == identity
        assert cleaned[0][3] == "owned_converter_pgid"
    finally:
        remote["close_directory_guard"](guard)


def test_converter_cleanup_uncertainty_is_typed_to_withhold_terminal(monkeypatch, tmp_path):
    remote = _remote_namespace()
    output = tmp_path / "output"
    output.mkdir()
    for name, raw in (("isaac_timing_0p5.json", "{}"),
                      ("isaac_timing_0p5.csv", "x\n"),
                      ("evaluator.log", "done\n")):
        (output / name).write_text(raw)
    guard = remote["open_directory_guard"](output)
    fake = type("FakePopen", (), {"pid": 445, "returncode": None})()
    monkeypatch.setattr(remote["subprocess"], "Popen", lambda *_a, **_k: fake)
    monkeypatch.setitem(remote, "source_environment", lambda _source: {})
    monkeypatch.setitem(remote, "complete_handshake", lambda *_a, **_k: True)
    monkeypatch.setitem(remote, "guardian_live_exact", lambda _guardian: {})
    monkeypatch.setitem(remote, "cgroup_populated", lambda _child: False)
    monkeypatch.setitem(remote, "require_owned_cgroup_members", lambda *_a, **_k: None)
    identity = {"pid": 445, "pgid": 445, "sid": 445, "ppid": 77,
                "state": "S", "start_ticks": 13, "argv": ["wrong"]}

    def fail_bind(_proc, _command, _parent, seconds=5.0, identity_sink=None):
        identity_sink["identity"] = identity
        raise RuntimeError("spawned child argv differs")

    monkeypatch.setitem(remote, "bind_owned_leader", fail_bind)
    monkeypatch.setitem(
        remote, "close_owned_child",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("group still live")))
    spec = {
        "output_dir": str(output), "source_closure": {"spec": "a" * 64},
        "paper": {"path": str(tmp_path / "paper.json"), "file_sha256": "b" * 64},
        "supervision": {"converter_timeout_seconds": 10.0,
                        "exact_term_grace_seconds": 1.0},
    }
    context = {
        "source": tmp_path, "schedule": tmp_path / "schedule.json",
        "checkpoint": tmp_path / "model.pt", "checkpoint_sha": "c" * 64,
        "hard_path": tmp_path / "hard.json", "hard_sha": "d" * 64,
    }
    try:
        with pytest.raises(remote["OwnedCleanupUnproven"], match="group still live"):
            remote["validate_and_convert"](
                spec, context, evaluator_state={}, signals=[], gpu_samples=[],
                started="now", output_guard=guard, stop_requested={"signal": None},
                cgroup={"child": {"fd": -1}, "path": "/cg"}, guardian={})
    finally:
        remote["close_directory_guard"](guard)


def test_directory_guard_rejects_parent_symlink_and_replacement(tmp_path):
    remote = _remote_namespace()
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(OSError):
        remote["open_directory_guard"](alias / "child", create_missing=True)

    guarded = tmp_path / "guarded"
    guarded.mkdir()
    guard = remote["open_directory_guard"](guarded)
    moved = tmp_path / "moved"
    guarded.rename(moved)
    guarded.mkdir()
    try:
        with pytest.raises(RuntimeError, match="replaced"):
            remote["revalidate_directory_guard"](guard)
    finally:
        remote["close_directory_guard"](guard)


def test_guarded_leaf_read_uses_held_fd_and_rejects_symlink(monkeypatch, tmp_path):
    remote = _remote_namespace()
    output = tmp_path / "output"
    output.mkdir()
    leaf = output / "score.json"
    leaf.write_bytes(b"trusted-score")
    guard = remote["open_directory_guard"](output)
    original_read = remote["os"].read
    swapped = {"done": False}

    def swap_after_open(fd, count):
        row = original_read(fd, count)
        if row and not swapped["done"]:
            swapped["done"] = True
            leaf.rename(output / "score.original")
            leaf.write_bytes(b"poison-score")
        return row

    try:
        monkeypatch.setattr(remote["os"], "read", swap_after_open)
        with pytest.raises(RuntimeError, match="entry was replaced"):
            remote["guarded_stable_bytes"](guard, "score.json", "score")
        assert leaf.read_bytes() == b"poison-score"
        monkeypatch.setattr(remote["os"], "read", original_read)
        (output / "score.link").symlink_to(leaf)
        with pytest.raises(OSError):
            remote["guarded_stable_bytes"](guard, "score.link", "symlink score")
    finally:
        remote["close_directory_guard"](guard)


def test_guarded_leaf_publish_fails_if_entry_is_replaced_after_open(monkeypatch, tmp_path):
    remote = _remote_namespace()
    output = tmp_path / "output"
    output.mkdir()
    guard = remote["open_directory_guard"](output)
    original_write = remote["os"].write
    swapped = {"done": False}

    def swap_after_write(fd, raw):
        count = original_write(fd, raw)
        if count and not swapped["done"]:
            swapped["done"] = True
            (output / "terminal.json").write_bytes(b"poison-terminal")
        return count

    try:
        monkeypatch.setattr(remote["os"], "write", swap_after_write)
        with pytest.raises(FileExistsError):
            remote["guarded_publish_bytes"](
                guard, "terminal.json", b"trusted-terminal\n")
        assert (output / "terminal.json").read_bytes() == b"poison-terminal"
    finally:
        remote["close_directory_guard"](guard)


def test_guarded_publish_exposes_only_complete_final_name(monkeypatch, tmp_path):
    remote = _remote_namespace()
    output = tmp_path / "output"
    output.mkdir()
    guard = remote["open_directory_guard"](output)
    raw = b"complete-terminal\n"
    original_rename = remote["_RENAMEAT2"]
    observed = {"called": False}

    def inspect_before_atomic_publish(old_fd, old_name, new_fd, new_name, flags):
        observed["called"] = True
        assert os.fsdecode(new_name) == "terminal.json"
        with pytest.raises(FileNotFoundError):
            os.stat("terminal.json", dir_fd=new_fd, follow_symlinks=False)
        temporary = os.stat(
            os.fsdecode(old_name), dir_fd=old_fd, follow_symlinks=False)
        assert temporary.st_size == len(raw)
        return original_rename(old_fd, old_name, new_fd, new_name, flags)

    try:
        monkeypatch.setitem(remote, "_RENAMEAT2", inspect_before_atomic_publish)
        remote["guarded_publish_bytes"](guard, "terminal.json", raw)
        assert observed["called"] is True
        assert (output / "terminal.json").read_bytes() == raw
    finally:
        remote["close_directory_guard"](guard)


def test_resource_gate_samples_all_pod_gpus_twice_and_fails_on_other_gpu(monkeypatch):
    remote = _remote_namespace()
    gate = {
        "allowed_eval_gpus": [0, 1, 2],
        "minimum_selected_gpu_free_mib": 6000,
        "minimum_other_gpu_free_mib": 3000,
        "stable_samples": 2,
        "sample_interval_seconds": 0.1,
    }
    spec = {"eval_gpu": 1, "resource_gate": gate}
    calls = []
    monkeypatch.setitem(remote, "other_evaluators", lambda: [])
    monkeypatch.setattr(remote["time"], "sleep", lambda _seconds: None)
    monkeypatch.setitem(remote, "parse_gpu", lambda gpu: (
        calls.append(gpu) or {"free_mib": 7000 if gpu == 1 else 4000}))
    samples = remote["stable_resource_gate"](spec)
    assert calls == [0, 1, 2, 0, 1, 2]
    assert [sorted(row["gpus"]) for row in samples] == [["0", "1", "2"], ["0", "1", "2"]]

    monkeypatch.setitem(remote, "parse_gpu", lambda gpu: {
        "free_mib": 2999 if gpu == 2 else 7000})
    with pytest.raises(RuntimeError, match="cross-GPU reserve GPU 2"):
        remote["stable_resource_gate"](spec)


def test_source_environment_preserves_physical_gpu_order(monkeypatch, tmp_path):
    remote = _remote_namespace()
    source = tmp_path / "source"
    expected = (
        source / "hope_training/whole_body_tracking/source/whole_body_tracking"
    ).resolve()

    def setup_with(visible):
        rows = [f"HOPE_WBT_PYTHONPATH={expected}"]
        if visible is not None:
            rows.append(f"CUDA_VISIBLE_DEVICES={visible}")
        return ("\0".join(rows) + "\0").encode()

    monkeypatch.setattr(
        remote["subprocess"], "check_output", lambda *_a, **_k: setup_with("2,1,0"))
    with pytest.raises(RuntimeError, match="physical Pod GPU order"):
        remote["source_environment"](source)

    monkeypatch.setattr(
        remote["subprocess"], "check_output", lambda *_a, **_k: setup_with(None))
    env = remote["source_environment"](source)
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1,2"
    assert env["PYTHONPATH"].split(os.pathsep)[0] == str(expected)


def test_last_heartbeat_tolerates_only_partial_final_row(tmp_path):
    remote = _remote_namespace()
    path = tmp_path / "heartbeat.jsonl"
    path.write_bytes(b'{"phase":"running"}\n{"phase":')
    assert remote["last_heartbeat"](path) == {"phase": "running"}


def test_last_heartbeat_rejects_complete_malformed_row(tmp_path):
    remote = _remote_namespace()
    path = tmp_path / "heartbeat.jsonl"
    path.write_bytes(b'{"phase":"running"}\n{"phase":}\n')
    with pytest.raises(Exception, match="Expecting value"):
        remote["last_heartbeat"](path)


def test_heartbeat_append_rejects_short_record(monkeypatch, tmp_path):
    remote = _remote_namespace()
    path = tmp_path / "heartbeat.jsonl"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND, 0o600)
    original_write = remote["os"].write
    calls = []

    def short_write(target, raw):
        calls.append(bytes(raw))
        return original_write(target, raw[:-1])

    try:
        monkeypatch.setattr(remote["os"], "write", short_write)
        with pytest.raises(RuntimeError, match="heartbeat append write was incomplete"):
            remote["heartbeat"](fd, {"phase": "running"})
        assert len(calls) == 1
        assert not path.read_bytes().endswith(b"\n")
    finally:
        os.close(fd)


def test_inspect_requires_fresh_heartbeat_and_exact_ack(monkeypatch, tmp_path):
    remote = _remote_namespace()
    spec = _committed_spec(tmp_path)
    guard, hello = _publish_committed_chain(remote, spec)
    try:
        remote["guarded_publish_bytes"](
            guard, "heartbeat.jsonl", b'{"phase":"evaluator_running"}\n')
    finally:
        remote["close_directory_guard"](guard)

    supervisor = {
        "pid": hello["pid"], "pgid": hello["pid"], "sid": hello["pid"],
        "ppid": 1, "state": "S", "start_ticks": hello["proc_start_ticks"],
        "argv": ["exact-supervisor"],
    }
    monkeypatch.setitem(remote, "proc_identity", lambda pid: supervisor if pid == 4242 else None)
    monkeypatch.setitem(remote, "exact_live", lambda _identity: {"state": "S"})
    heartbeat = Path(spec["state_dir"]) / "heartbeat.jsonl"
    stale = time.time() - 200
    os.utime(heartbeat, (stale, stale))
    assert remote["inspect"](spec)["status"] == "running_stale_heartbeat"
    os.utime(heartbeat, None)
    assert remote["inspect"](spec)["status"] == "running_exact"

    wrong_spec = _committed_spec(tmp_path, "wrong_ack")
    wrong_guard, _ = _publish_committed_chain(remote, wrong_spec, wrong_ack=True)
    remote["close_directory_guard"](wrong_guard)
    with pytest.raises(RuntimeError, match="commit acknowledgment"):
        remote["inspect"](wrong_spec)


def test_fake_committed_terminal_without_ack_is_rejected(monkeypatch, tmp_path):
    remote = _remote_namespace()
    spec = _committed_spec(tmp_path)
    guard, _ = _publish_committed_chain(remote, spec, include_ack=False)
    terminal_content = {
        "status": "complete_inexact_isaac_k100",
        "job_id": spec["job_id"],
        "milestone": spec["milestone"],
        "retry_authorized": False,
        "trainer_or_robot_signals": [],
        "catastrophic_cleanup": {
            "contract": spec["catastrophic_cleanup"],
            "guardian_finish_result": "D0",
            "cgroup_populated_zero_acknowledged": True,
            "cgroup_removed_after_populated_zero": True,
        },
    }
    remote["guarded_publish_json"](
        guard, "terminal.json", {
            "schema_version": 1,
            "content": terminal_content,
            "content_sha256": remote["canonical"](terminal_content),
        })
    remote["close_directory_guard"](guard)
    monkeypatch.setitem(remote, "proc_identity", lambda _pid: None)
    with pytest.raises(FileNotFoundError):
        remote["inspect"](spec)


def test_launch_two_phase_commit_is_no_clobber_and_returns_after_ack(tmp_path):
    remote = _remote_namespace()
    state = tmp_path / "state"
    output = tmp_path / "output"
    lock = tmp_path / "kit.lock"
    spec = {
        "state_dir": str(state),
        "output_dir": str(output),
        "kit_lock": str(lock),
        "job_id": "taskrev_p2_equal_reward",
        "milestone": 5700,
        "activation": {"activation_id": "fixed", "path": "fixed.json", "sha256": "a" * 64},
        "supervision": {
            "hello_timeout_seconds": 2.0,
            "commit_timeout_seconds": 2.0,
            "ack_observation_seconds": 2.0,
        },
    }
    remote["validate_inputs"] = lambda _spec, validate_process=True: {}
    remote["stable_resource_gate"] = lambda _spec: [{"free_mib": 9999}]
    remote["prepare_owned_cgroup"] = lambda _spec: {
        "parent": {"fd": -1}, "child": {"fd": -1},
        "path": "/fake/cgroup", "name": "fake",
    }
    remote["close_owned_cgroup_guards"] = lambda _cgroup: None
    remote["_PRCTL"] = lambda *_args: 0
    remote["proc_identity"] = lambda pid: {
        "pid": pid, "pgid": pid, "sid": pid, "ppid": remote["os"].getppid(),
        "state": "S", "start_ticks": 123, "argv": ["fake-supervisor"]
    }

    def fake_child(child_spec, lock_fd, log_fd, state_guard, _output_parent_guard,
                   _cgroup):
        remote["os"].setsid()
        identity = remote["proc_identity"](remote["os"].getpid())
        hello = {
            "schema_version": 1,
            "artifact_kind": "taskrev_0p5_supervisor_hello",
            "plan_sha256": remote["canonical"](child_spec),
            "activation": child_spec["activation"],
            "job_id": child_spec["job_id"],
            "milestone": child_spec["milestone"],
            "pid": identity["pid"], "pgid": identity["pgid"],
            "proc_start_ticks": identity["start_ticks"],
            "argv_sha256": remote["canonical"](identity["argv"]),
            "commit_deadline_monotonic_ns": remote["time"].monotonic_ns() + 2_000_000_000,
            "automatic_retry": False,
        }
        remote["guarded_publish_json"](state_guard, "child_hello.json", hello)
        assert remote["guarded_wait_for"](state_guard, "commit_token.json", 2.0)
        token, _ = remote["guarded_stable_json"](
            state_guard, "commit_token.json", "token")
        ledger_raw = remote["guarded_stable_bytes"](
            state_guard, "launch_ledger.json", "ledger")
        token_raw = remote["guarded_stable_bytes"](
            state_guard, "commit_token.json", "token")
        decision = {
            "schema_version": 1,
            "artifact_kind": "taskrev_0p5_launch_decision",
            "decision": "commit",
            "plan_sha256": remote["canonical"](child_spec),
            "pid": identity["pid"], "pgid": identity["pgid"],
            "proc_start_ticks": identity["start_ticks"],
            "hello_sha256": remote["guarded_sha"](
                state_guard, "child_hello.json", "hello"),
            "ledger_sha256": remote["hashlib"].sha256(ledger_raw).hexdigest(),
            "token_sha256": remote["hashlib"].sha256(token_raw).hexdigest(),
            "commit_deadline_monotonic_ns": hello["commit_deadline_monotonic_ns"],
            "decided_monotonic_ns": remote["time"].monotonic_ns(),
            "decided_utc": "now", "retry_authorized": False,
        }
        remote["guarded_publish_json"](
            state_guard, "launch_decision.json", decision)
        ack = {
            "pid": identity["pid"], "proc_start_ticks": identity["start_ticks"],
            "plan_sha256": remote["canonical"](child_spec),
            "token_sha256": remote["guarded_sha"](
                state_guard, "commit_token.json", "token"),
        }
        remote["guarded_publish_json"](state_guard, "commit_ack.json", ack)
        remote["os"].close(log_fd)
        remote["os"].close(lock_fd)
        return 0

    remote["supervisor_child"] = fake_child
    remote["validate_committed_chain"] = lambda *_a, **_k: {}
    result = remote["launch"](spec)
    assert result["status"] == "running_or_committed_exact"
    assert result["retry_authorized"] is False
    remote["os"].waitpid(result["supervisor_pid"], 0)
    assert (state / "child_hello.json").is_file()
    assert (state / "launch_ledger.json").is_file()
    assert (state / "commit_token.json").is_file()
    assert (state / "commit_ack.json").is_file()
    with pytest.raises(RuntimeError, match="no-clobber supervisor state"):
        remote["launch"](spec)


def test_launch_cgroup_preflight_fails_before_state_or_output_namespace(
        monkeypatch, tmp_path):
    remote = _remote_namespace()
    state = tmp_path / "states" / "state"
    output = tmp_path / "outputs" / "output"
    spec = {
        "state_dir": str(state), "output_dir": str(output),
        "kit_lock": str(tmp_path / "kit.lock"),
        "job_id": "taskrev_p2_equal_reward", "milestone": 5700,
        "activation": {"sha256": "a" * 64},
        "supervision": {},
    }
    remote["validate_inputs"] = lambda _spec, validate_process=True: {}
    remote["stable_resource_gate"] = lambda _spec: []
    remote["_PRCTL"] = lambda *_args: 0
    remote["proc_identity"] = lambda pid: {
        "pid": pid, "pgid": pid, "sid": pid, "ppid": 1,
        "state": "S", "start_ticks": 1, "argv": ["launcher"],
    }
    monkeypatch.setitem(
        remote, "prepare_owned_cgroup",
        lambda _spec: (_ for _ in ()).throw(
            RuntimeError("delegated cgroup v2 unavailable")))

    with pytest.raises(RuntimeError, match="delegated cgroup v2 unavailable"):
        remote["launch"](spec)
    assert not state.exists()
    assert not output.exists()
    assert not (tmp_path / "kit.lock").exists()


def test_cgroup_migration_rejects_partial_control_write(monkeypatch, tmp_path):
    remote = _remote_namespace()
    cgroup_dir = tmp_path / "cgroup"
    cgroup_dir.mkdir()
    (cgroup_dir / "cgroup.procs").write_bytes(b"")
    guard = remote["open_directory_guard"](cgroup_dir)
    original_write = remote["os"].write
    writes = []

    def partial_once(fd, raw):
        writes.append(bytes(raw))
        return original_write(fd, raw[:1])

    try:
        monkeypatch.setattr(remote["os"], "write", partial_once)
        with pytest.raises(RuntimeError, match="control write was incomplete"):
            remote["move_current_to_owned_cgroup"]({"child": guard})
        assert writes == [b"0\n"]
        assert (cgroup_dir / "cgroup.procs").read_bytes() == b"0"
    finally:
        remote["close_directory_guard"](guard)


@pytest.mark.skipif(sys.platform != "linux", reason="requires Linux cgroup v2")
def test_sigkill_supervisor_guardian_empties_descendants_before_lock_release(tmp_path):
    """Exercise the real parent-death path when this test has cgroup delegation.

    Environments without a writable delegated cgroup-v2 child skip before any
    evaluator is created.  A capable Linux CI/Pod exercises a leader that forks
    a TERM-ignoring helper, then loses the supervisor to SIGKILL.
    """
    remote = _remote_namespace()
    ready_read, ready_write = os.pipe()
    supervisor_pid = os.fork()
    if supervisor_pid == 0:  # pragma: no cover - exercised only on delegated Linux
        os.close(ready_read)
        cgroup = None
        lock_fd = -1
        try:
            spec = {
                "job_id": f"guardian_integration_{os.getpid()}",
                "milestone": os.getpid(),
                "activation": {"sha256": f"{os.getpid():064x}"[-64:]},
            }
            try:
                cgroup = remote["prepare_owned_cgroup"](spec)
            except BaseException as exc:
                os.write(ready_write, f"SKIP:{type(exc).__name__}:{exc}".encode())
                os._exit(77)
            lock_fd = os.open(tmp_path / "kit.lock", os.O_RDWR | os.O_CREAT, 0o600)
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            remote["start_cgroup_guardian"](cgroup, lock_fd, 5.0)
            eval_read, eval_write = os.pipe()
            code = """
import os, signal, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
read_fd, write_fd = os.pipe()
helper = os.fork()
if helper == 0:
    os.close(read_fd)
    os.setsid()
    grandchild = os.fork()
    if grandchild == 0:
        time.sleep(600)
    os.write(write_fd, f"{os.getpid()} {grandchild}".encode())
    time.sleep(600)
os.close(write_fd)
descendants = os.read(read_fd, 128).decode()
os.close(read_fd)
os.write(int(os.environ["READY_FD"]),
         f"{os.getpid()} {descendants}".encode())
time.sleep(600)
"""
            env = dict(os.environ)
            env["READY_FD"] = str(eval_write)
            leader = subprocess.Popen(
                [sys.executable, "-c", code], env=env, close_fds=True,
                pass_fds=(cgroup["child"]["fd"], eval_write),
                preexec_fn=remote["child_preexec"](
                    os.getpid(), cgroup["child"]["fd"]),
            )
            os.close(eval_write)
            readable, _, _ = select.select([eval_read], [], [], 5.0)
            if not readable:
                raise RuntimeError("evaluator/helper readiness timed out")
            pids = os.read(eval_read, 128).decode()
            os.close(eval_read)
            os.write(
                ready_write,
                json.dumps({"cgroup": cgroup["path"], "pids": pids,
                            "leader": leader.pid}).encode())
            while True:
                time.sleep(60)
        except BaseException as exc:
            try:
                os.write(ready_write, f"ERROR:{type(exc).__name__}:{exc}".encode())
            except OSError:
                pass
            os._exit(78)

    os.close(ready_write)
    readable, _, _ = select.select([ready_read], [], [], 10.0)
    if not readable:
        os.kill(supervisor_pid, signal.SIGKILL)
        os.waitpid(supervisor_pid, 0)
        pytest.fail("guardian integration supervisor did not become ready")
    raw = os.read(ready_read, 4096).decode()
    os.close(ready_read)
    if raw.startswith("SKIP:"):
        os.waitpid(supervisor_pid, 0)
        pytest.skip(raw)
    if raw.startswith("ERROR:"):
        os.waitpid(supervisor_pid, 0)
        pytest.fail(raw)
    info = json.loads(raw)
    leader_pid, helper_pid, grandchild_pid = [
        int(value) for value in info["pids"].split()]
    assert leader_pid == info["leader"]
    cgroup_members = {
        int(value) for value in
        (Path(info["cgroup"]) / "cgroup.procs").read_text().splitlines()
    }
    assert {leader_pid, helper_pid, grandchild_pid}.issubset(cgroup_members)

    os.kill(supervisor_pid, signal.SIGKILL)
    os.waitpid(supervisor_pid, 0)
    contender = os.open(tmp_path / "kit.lock", os.O_RDWR)
    acquired = False
    deadline = time.monotonic() + 15.0
    try:
        while time.monotonic() < deadline:
            try:
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except BlockingIOError:
                time.sleep(0.05)
        assert acquired, "guardian never released Kit lock after exact cgroup cleanup"
        assert not Path(info["cgroup"]).exists()
        for pid in (leader_pid, helper_pid, grandchild_pid):
            current = remote["proc_identity"](pid)
            assert current is None or current.get("state") == "Z"
        assert not [
            row for row in remote["_scan_process_group"](leader_pid)
            if row.get("state") != "Z"
        ]
    finally:
        if acquired:
            fcntl.flock(contender, fcntl.LOCK_UN)
        os.close(contender)


def test_remote_command_embeds_one_action_and_no_ssh():
    plan = E.build_plan(QUEUE, activation_path=ACTIVATION, eval_gpu=0)
    command = E._remote_command(plan, action="inspect")
    assert command.startswith("/workspace/hope_isaac_venv/bin/python -B -c ")
    assert "ssh" not in command
    assert E.REMOTE_PROGRAM not in command
    with pytest.raises(E.ExamError, match="launch or inspect"):
        E._remote_command(plan, action="stop")
