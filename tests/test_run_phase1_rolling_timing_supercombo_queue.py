from __future__ import annotations

import base64
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_phase1_rolling_timing_supercombo_queue.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("rolling_queue_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


Q = _load_module()


def _parent(name: str, iteration: int) -> dict:
    root = f"/workspace/parents/{name}"
    run = f"/workspace/parent-runs/{name}"
    return {
        "preferred_candidate_reason": "unit fixture",
        "original_job_id": f"original_{name}",
        "original_run_name": f"original_{name}",
        "original_run_dir": run,
        "original_queue_claim_path": f"{run}/queue_claim.json",
        "original_queue_claim_sha256": "1" * 64,
        "original_run_binding_path": f"{run}/run_binding.json",
        "original_run_binding_sha256": "2" * 64,
        "selected_rsl_log_dir": root,
        "ranking_snapshot_checkpoint_filename": f"model_{iteration}.pt",
        "selected_checkpoint_path": f"{root}/model_{iteration}.pt",
        "selected_embedded_iteration": iteration,
        "selected_checkpoint_sha256": "3" * 64,
        "selected_hard_contract_path": f"{root}/params/training_contract.json",
        "selected_hard_contract_sha256": "4" * 64,
        "immutable_stop_receipt_still_required": False,
        "selection_is_final": True,
    }


def _asset() -> dict:
    return {
        "target_relative_path": (
            "hope_training/whole_body_tracking/source/whole_body_tracking/"
            "whole_body_tracking/assets/agibot_a3"
        ),
        "donor": {
            "checkout": "/workspace/donor",
            "commit": "d" * 40,
            "relative_path": (
                "hope_training/whole_body_tracking/source/whole_body_tracking/"
                "whole_body_tracking/assets/agibot_a3"
            ),
        },
        "file_count": 46,
        "total_file_bytes": 100,
        "tree_content_sha256": "a" * 64,
        "symlinks_forbidden": True,
        "target_must_be_gitignored": True,
    }


def _job(index: int, launch_round: int, slot: str, parent: str) -> dict:
    action = "signed_face_v4rg_shared_face"
    return {
        "id": f"rolling_job_{index}",
        "human_name": f"rolling continuation {index}",
        "launch_round": launch_round,
        "action": action,
        "status": "ready",
        "blocker": None,
        "motion": {
            "action": action,
            "bindings": {
                "motion_file": "/workspace/assets/fh.npz",
                "motion_file_2": "/workspace/assets/bh.npz",
            },
        },
        "bank": {
            "action": action,
            "train_path": "/workspace/assets/bank.npz",
            "train_arg": "++task.racket.question_bank",
        },
        "exam": {
            "action": action,
            "path": "/workspace/assets/exam.json",
            "family": "signed_exam",
        },
        "source": {
            "checkout": "/workspace/source",
            "commit": "c" * 40,
            "ignored_runtime_asset": _asset(),
        },
        "runtime_binding": True,
        "warm_start": {
            "parent": parent,
            "checkpoint_path": _parent(
                parent,
                {"pod1_parent": 1600, "pod2_parent": 4700, "pod2_continuous": 4500}[
                    parent
                ],
            )[
                "selected_checkpoint_path"
            ],
            "transfer_mode": "strict_full_state_preserve_optimizer",
            "checkpoint_tolerant": False,
            "allow_missing_contract": False,
            "allow_contract_mismatch": True,
            "descendant_exact_eligible": False,
        },
        "recipe": {
            "base": [
                "task=HOPEPingPongVirtualBall",
                "algo=ppo",
                "task.motion.speed_scale_per_clip=[1.32,1.0]",
                "task.racket.target_delay_steps=2",
            ],
            "delta": [
                "task.motion.speed_scale_per_clip=[1.886,1.286]",
                f"task.rewards.foot_orientation_weight=-0.{index % 7}",
            ],
        },
        "seed": 3,
        "budget": {
            "num_envs": 4096,
            "max_iterations": 2001,
            "save_interval": 100,
            "iteration_semantics": "additional_updates_after_full_state_resume",
        },
        "milestones": [200, 500, 1000, 2000],
        "milestone_semantics": "offsets_from_attested_parent",
        "resource": {
            "policy": "dispatch_gpu_round_robin",
            "required_slot": slot,
        },
        "formal_evidence_eligible": False,
        "run_name": f"rolling_run_{index}",
        "run_dir": f"/workspace/rolling/runs/job_{index}",
    }


def _queue() -> dict:
    parents = {
        "pod1_parent": _parent("pod1_parent", 1600),
        "pod2_parent": _parent("pod2_parent", 4700),
        "pod2_continuous": _parent("pod2_continuous", 4500),
    }
    slots = [
        "pod1/gpu0",
        "pod1/gpu1",
        "pod1/gpu2",
        "pod2/gpu0",
        "pod2/gpu1",
        "pod2/gpu2",
    ]
    jobs = []
    for launch_round in range(1, 5):
        for slot in slots:
            index = len(jobs)
            if slot.startswith("pod1/"):
                parent = "pod1_parent"
            elif launch_round <= 2:
                parent = "pod2_parent"
            else:
                parent = "pod2_continuous"
            jobs.append(_job(index, launch_round, slot, parent))
    return {
        "schema_version": 1,
        "simulation_only": True,
        "launch_authorized": True,
        "preregistration_status": "activated_demo_only_inexact",
        "formal_evidence_eligible": False,
        "ssh": {"key": "/key"},
        "pods": {
            "pod1": {
                "host": "pod1",
                "port": 1,
                "gpus": [0, 1, 2],
                "max_trainers_per_gpu": 4,
            },
            "pod2": {
                "host": "pod2",
                "port": 2,
                "gpus": [0, 1, 2],
                "max_trainers_per_gpu": 4,
            },
        },
        "dispatch_pods": ["pod1", "pod2"],
        "blocking_contract": {
            "source_checkout": "/workspace/source",
            "source_commit": "c" * 40,
            "source_full_scene_probe_evidence": {
                "training_runtime_status": "passed_natural_exit_rc0",
                "first_iteration_observed": True,
                "checkpoint_iteration": 1,
                "checkpoint_sha256": "7" * 64,
                "hard_contract_sha256": "8" * 64,
                "tensor_nonfinite_count": 0,
                "fatal_count": 0,
                "training_contract_lineage_exact": 1,
                "process_group_naturally_empty": True,
            },
            "hotstart_harness": {
                "runner_script_sha256": Q._runner_payload()[1],
                "reviewed_tests_passed": True,
                "reviewed_test_count": 80,
            },
        },
        "predecessor_stop_contract": {
            "evidence_state": "immutable_receipts_bound",
            "pod1": {
                "stop_receipt_path": "/workspace/evidence/pod1-stop.json",
                "stop_receipt_sha256": "5" * 64,
            },
            "pod2": {
                "stop_receipt_path": "/workspace/evidence/pod2-stop.json",
                "stop_receipt_sha256": "6" * 64,
            },
        },
        "parent_selection": {
            "policy": "same_pod_only",
            "selection_state": "final",
            **parents,
        },
        "jobs": jobs,
    }


def _write(tmp_path: Path, queue: dict) -> Path:
    path = tmp_path / "queue.yaml"
    path.write_text(yaml.safe_dump(queue, sort_keys=False), encoding="utf-8")
    return path


def _loaded(tmp_path: Path, queue: dict | None = None) -> dict:
    value = Q.load_queue(_write(tmp_path, _queue() if queue is None else queue))
    Q._bind_parent_context(value)
    return value


def test_exact_24_job_four_round_layout_and_absolute_plan(tmp_path):
    queue = _loaded(tmp_path)
    plan = Q.cmd_plan(queue)
    assert plan["activation_ready"] is True
    assert len(plan["jobs"]) == 24
    assert [row["required_slot"] for row in plan["jobs"][:6]] == list(Q.EXPECTED_SLOTS)
    assert [row["launch_round"] for row in plan["jobs"]] == [
        launch_round for launch_round in range(1, 5) for _ in range(6)
    ]
    pod1 = plan["jobs"][0]
    assert pod1["parent_iteration"] == 1600
    assert pod1["absolute_iteration_exclusive_bound"] == 3601
    assert pod1["milestones"] == [1800, 2100, 2600, 3600]
    assert pod1["absolute_checkpoint_filenames"][-1] == "model_3600.pt"
    pod2 = plan["jobs"][3]
    assert pod2["parent_iteration"] == 4700
    assert pod2["absolute_iteration_exclusive_bound"] == 6701
    assert pod2["milestones"] == [4900, 5200, 5700, 6700]


def test_milestone_attestor_dry_run_uses_registered_absolute_schedule_and_no_ssh(
    tmp_path, monkeypatch
):
    queue = _loaded(tmp_path)
    calls = []
    monkeypatch.setattr(Q.lean, "_run_ssh", lambda *_a, **_k: calls.append("ssh"))
    result = Q.cmd_attest_milestone(
        queue,
        job_id="rolling_job_0",
        milestone=3600,
        execute=False,
        confirm=None,
    )
    assert result["dry_run"] is True
    assert result["pod"] == "pod1"
    assert result["required_slot"] == "pod1/gpu0"
    assert result["registered_absolute_milestones"] == [1800, 2100, 2600, 3600]
    assert result["binding_path"] == "/workspace/rolling/runs/job_0/run_binding.json"
    assert result["receipt_path"] == (
        "/workspace/rolling/runs/job_0/milestones/model_3600.json"
    )
    assert result["lean_queue_runtime_sha256"] in result["remote_script"]
    assert Q.lean.ATTESTOR_RUNTIME_ROOT in result["remote_script"]
    assert "lean_queue_runtime.py attest" in result["remote_script"]
    assert "--expected-claim-content-sha256" in result["remote_script"]
    assert "--expected-job-id rolling_job_0" in result["remote_script"]
    assert "--expected-runtime-sha256" in result["remote_script"]
    assert "model_3600.pt" not in result["remote_script"]
    assert (
        "/workspace/source/hope_training/whole_body_tracking/scripts/lean_queue_runtime.py"
        not in result["remote_script"]
    )
    assert calls == []


@pytest.mark.parametrize(
    ("job_id", "milestone", "message"),
    [
        ("missing", 3600, "unknown queue job"),
        ("rolling_job_0", 200, "registered absolute milestones"),
        ("rolling_job_3", 5100, "registered absolute milestones"),
    ],
)
def test_milestone_attestor_rejects_unknown_job_or_unregistered_iteration_before_ssh(
    tmp_path, monkeypatch, job_id, milestone, message
):
    queue = _loaded(tmp_path)
    calls = []
    monkeypatch.setattr(Q.lean, "_run_ssh", lambda *_a, **_k: calls.append("ssh"))
    with pytest.raises(Q.ContinuationQueueError, match=message):
        Q.cmd_attest_milestone(
            queue,
            job_id=job_id,
            milestone=milestone,
            execute=True,
            confirm=Q.ATTEST_CONFIRM,
        )
    assert calls == []


def test_milestone_attestor_execute_requires_dedicated_confirmation_before_ssh(
    tmp_path, monkeypatch
):
    queue = _loaded(tmp_path)
    calls = []
    monkeypatch.setattr(Q.lean, "_run_ssh", lambda *_a, **_k: calls.append("ssh"))
    with pytest.raises(Q.ContinuationQueueError, match=Q.ATTEST_CONFIRM):
        Q.cmd_attest_milestone(
            queue,
            job_id="rolling_job_3",
            milestone=5200,
            execute=True,
            confirm=Q.CONFIRM,
        )
    assert calls == []


def test_milestone_attestor_execute_uses_yaml_pod_and_one_no_signal_ssh(
    tmp_path, monkeypatch
):
    queue = _loaded(tmp_path)
    calls = []

    def fake_ssh(queue_arg, pod, remote, **kwargs):
        calls.append((queue_arg, pod, remote, kwargs))
        return '{"receipt_path":"/workspace/rolling/runs/job_3/milestones/model_5200.json"}'

    monkeypatch.setattr(Q.lean, "_run_ssh", fake_ssh)
    result = Q.cmd_attest_milestone(
        queue,
        job_id="rolling_job_3",
        milestone=5200,
        execute=True,
        confirm=Q.ATTEST_CONFIRM,
    )
    assert result["dry_run"] is False
    assert result["pod"] == "pod2"
    assert len(calls) == 1
    _queue_arg, pod, remote, kwargs = calls[0]
    assert pod == "pod2"
    assert kwargs == {
        "timeout": 120,
        "phase": "rolling-attest-milestone:rolling_job_3:5200",
    }
    assert "/workspace/rolling/runs/job_3/run_binding.json" in remote
    assert result["lean_queue_runtime_sha256"] in remote
    assert remote.index("claim_preflight_passed") < remote.rindex(
        "lean_queue_runtime.py attest"
    )
    assert all(token not in remote for token in ("kill ", "pkill", "killall", "signal"))


@pytest.mark.parametrize("drift", ["run_dir", "source", "recipe", "slot"])
def test_milestone_attestor_remote_preflight_rejects_old_claim_after_job_drift(
    tmp_path, drift
):
    queue = _loaded(tmp_path)
    original_job = queue["jobs"][0]
    original_job["run_dir"] = str(tmp_path / "old-run")
    Path(original_job["run_dir"]).mkdir()
    original_slot = Q._slots(queue)[original_job["resource"]["required_slot"]]
    launch_runner_sha = queue["blocking_contract"]["hotstart_harness"][
        "runner_script_sha256"
    ]
    old_claim, _argv, _absolute = Q._launch_contract(
        queue,
        original_job,
        original_slot,
        runner_script_sha256=launch_runner_sha,
    )
    claim_path = Path(original_job["run_dir"]) / "queue_claim.json"
    claim_path.write_text(json.dumps(old_claim), encoding="utf-8")
    original = Q._attestor_claim_spec(queue, original_job, original_slot)

    def run_preflight(spec):
        encoded = base64.b64encode(
            json.dumps(spec, separators=(",", ":"), sort_keys=True).encode()
        ).decode()
        return subprocess.run(
            [sys.executable, "-c", Q._ATTEST_CLAIM_PREFLIGHT, encoded],
            capture_output=True,
            text=True,
            check=False,
        )

    assert run_preflight(original).returncode == 0

    changed = copy.deepcopy(queue)
    job = changed["jobs"][0]
    if drift == "run_dir":
        job["run_dir"] = "/workspace/rolling/runs/drifted"
    elif drift == "source":
        job["source"]["checkout"] = "/workspace/drifted-source"
        job["source"]["commit"] = "e" * 40
    elif drift == "recipe":
        job["recipe"]["delta"].append("task.racket.target_delay_steps=4")
    else:
        job["resource"]["required_slot"] = "pod1/gpu1"
    slot = Q._slots(changed)[job["resource"]["required_slot"]]
    replacement = Q._attestor_claim_spec(changed, job, slot)

    assert replacement["content_sha256"] != original["content_sha256"]
    rejected = run_preflight(replacement)
    assert rejected.returncode != 0
    assert "claim_preflight_passed" not in rejected.stdout


def test_milestone_attestor_cli_defaults_to_dry_run_surface():
    args = Q._parser().parse_args(
        ["attest-milestone", "--job-id", "rolling_job_0", "--milestone", "3600"]
    )
    assert args.mode == "attest-milestone"
    assert args.execute is False
    assert args.confirm is None


def test_pending_queue_reports_blockers_without_ssh(tmp_path, monkeypatch):
    queue = _queue()
    queue["launch_authorized"] = False
    queue["preregistration_status"] = "blocked_pending_probe"
    queue["blocking_contract"]["source_full_scene_probe_evidence"] = "PENDING_PROBE"
    queue["blocking_contract"]["hotstart_harness"] = "PENDING_HARNESS"
    for job in queue["jobs"]:
        job["status"] = "blocked"
        job["blocker"] = "global source gate pending"
    loaded = _loaded(tmp_path, queue)
    monkeypatch.setattr(Q.lean, "_run_ssh", lambda *_a, **_k: pytest.fail("SSH called"))
    result = Q.validate_queue(loaded)
    assert result["schema_valid"] is True
    assert result["activation_ready"] is False
    assert "launch_authorized is false" in result["blockers"]
    assert "24 jobs remain blocked" in result["blockers"]
    assert "source_full_scene_probe_evidence must be a pass mapping" in result["blockers"]
    assert "hotstart_harness must be a reviewed pass mapping" in result["blockers"]


@pytest.mark.parametrize("status", ["blocked", "blocked_pending_probe", "launch_ready", "garbage", None])
def test_preregistration_status_is_an_exact_allowlist(tmp_path, status):
    queue = _loaded(tmp_path)
    queue["preregistration_status"] = status
    blockers = Q.activation_blockers(queue)
    assert any("preregistration_status must be" in blocker for blocker in blockers)


def test_full_scene_evidence_and_reviewed_harness_fail_closed(tmp_path):
    queue = _loaded(tmp_path)
    assert Q.activation_blockers(queue) == []

    wrong_runtime = copy.deepcopy(queue)
    wrong_runtime["blocking_contract"]["source_full_scene_probe_evidence"][
        "training_runtime_status"
    ] = "failed"
    assert any("training_runtime_status" in value for value in Q.activation_blockers(wrong_runtime))

    missing_evidence = copy.deepcopy(queue)
    missing_evidence["blocking_contract"].pop("source_full_scene_probe_evidence")
    assert any("pass mapping" in value for value in Q.activation_blockers(missing_evidence))

    wrong_runner = copy.deepcopy(queue)
    wrong_runner["blocking_contract"]["hotstart_harness"]["runner_script_sha256"] = "0" * 64
    assert any("runner bytes" in value for value in Q.activation_blockers(wrong_runner))

    false_tests = copy.deepcopy(queue)
    false_tests["blocking_contract"]["hotstart_harness"]["reviewed_tests_passed"] = False
    assert any("reviewed_tests_passed" in value for value in Q.activation_blockers(false_tests))

    low_count = copy.deepcopy(queue)
    low_count["blocking_contract"]["hotstart_harness"]["reviewed_test_count"] = 79
    assert any("reviewed_test_count" in value for value in Q.activation_blockers(low_count))


def test_same_key_delta_is_single_override_but_duplicate_final_key_is_rejected(tmp_path):
    queue = _loaded(tmp_path)
    compiled = Q._compile_recipe(queue["jobs"][0], queue["jobs"][0]["id"])
    speed = [value for value in compiled if value.startswith("task.motion.speed_scale_per_clip=")]
    assert speed == ["task.motion.speed_scale_per_clip=[1.886,1.286]"]

    broken = _queue()
    broken["jobs"][0]["recipe"]["delta"].append(
        "task.motion.speed_scale_per_clip=[2.64,1.8]"
    )
    with pytest.raises(Q.ContinuationQueueError, match="sets final Hydra key.*more than once"):
        Q.load_queue(_write(tmp_path, broken))


@pytest.mark.parametrize(
    "override",
    [
        "checkpoint_path=/workspace/parent.pt",
        "checkpoint_tolerant=false",
        "checkpoint_allow_missing_contract=false",
        "checkpoint_allow_contract_mismatch=true",
        "seed=9",
        "num_envs=1",
        "max_iterations=1",
        "algo.runner.save_interval=1",
        "run_name=stolen",
        "device=cuda:2",
        "++training_queue_claim_path=/workspace/stolen.json",
        "++training_run_binding_path=/workspace/stolen.json",
    ],
)
def test_harness_owned_overrides_cannot_be_injected(tmp_path, override):
    queue = _queue()
    queue["jobs"][0]["recipe"]["delta"].append(override)
    with pytest.raises(Q.ContinuationQueueError, match="harness-owned key"):
        Q.load_queue(_write(tmp_path, queue))


def test_wrong_parent_slot_checkpoint_and_sha_fail_closed(tmp_path):
    cross_pod = _queue()
    cross_pod["jobs"][0]["warm_start"]["parent"] = "pod2_parent"
    cross_pod["jobs"][0]["warm_start"]["checkpoint_path"] = cross_pod[
        "parent_selection"
    ]["pod2_parent"]["selected_checkpoint_path"]
    with pytest.raises(Q.ContinuationQueueError, match="different Pods"):
        Q.load_queue(_write(tmp_path, cross_pod))

    wrong_checkpoint = _queue()
    wrong_checkpoint["jobs"][0]["warm_start"]["checkpoint_path"] = (
        "/workspace/parents/pod1_parent/model_1500.pt"
    )
    with pytest.raises(Q.ContinuationQueueError, match="differs from parent selection"):
        Q.load_queue(_write(tmp_path, wrong_checkpoint))

    wrong_sha = _queue()
    wrong_sha["parent_selection"]["pod1_parent"]["selected_checkpoint_sha256"] = "bad"
    with pytest.raises(Q.ContinuationQueueError, match="64 lowercase hex"):
        Q.load_queue(_write(tmp_path, wrong_sha))


def test_wrong_capacity_or_round_layout_is_rejected(tmp_path):
    capacity = _queue()
    capacity["pods"]["pod2"]["max_trainers_per_gpu"] = 3
    with pytest.raises(Q.ContinuationQueueError, match="capacity must be exactly four"):
        Q.load_queue(_write(tmp_path, capacity))

    layout = _queue()
    layout["jobs"][0]["resource"]["required_slot"] = "pod1/gpu1"
    with pytest.raises(Q.ContinuationQueueError, match="each of the six GPUs exactly once"):
        Q.load_queue(_write(tmp_path, layout))


def test_global_source_and_stop_receipt_bindings_are_exact(tmp_path):
    source = _queue()
    source["jobs"][0]["source"]["commit"] = "e" * 40
    with pytest.raises(Q.ContinuationQueueError, match="source commit differs"):
        Q.load_queue(_write(tmp_path, source))

    receipt = _queue()
    receipt["predecessor_stop_contract"]["pod1"]["stop_receipt_sha256"] = "bad"
    with pytest.raises(Q.ContinuationQueueError, match="64 lowercase hex"):
        Q.load_queue(_write(tmp_path, receipt))


def test_dry_fill_is_six_gpu_round_robin_and_claim_is_no_clobber(tmp_path):
    queue = _loaded(tmp_path)
    result = Q.cmd_fill(queue, count=12, execute=False, confirm=None)
    assert [job["required_slot"] for job in result["jobs"]] == [
        *Q.EXPECTED_SLOTS,
        *Q.EXPECTED_SLOTS,
    ]
    assert [job["launch_round"] for job in result["jobs"]] == [1] * 6 + [2] * 6
    first = queue["jobs"][0]
    slot = Q._slots(queue)[first["resource"]["required_slot"]]
    remote = Q._launch_script(queue, first, slot)
    assert f'test "$count" -lt 4' in remote
    assert f"mkdir {first['run_dir']}" in remote
    assert "set -o noclobber" in remote
    assert "test ! -e" in remote and "run_binding.json" in remote
    assert "Learning iteration 1601/3601" in remote


def test_execute_needs_exact_confirmation_before_live_snapshot(tmp_path, monkeypatch):
    queue = _loaded(tmp_path)
    calls = []
    monkeypatch.setattr(Q.lean, "live_snapshot", lambda *_a: calls.append("snapshot"))
    with pytest.raises(Q.ContinuationQueueError, match=Q.CONFIRM):
        Q.cmd_fill(queue, count=1, execute=True, confirm="wrong")
    assert calls == []


def test_execute_fill_launches_only_one_per_pod_concurrently_and_counts_attempts(
    tmp_path, monkeypatch
):
    queue = _loaded(tmp_path)
    monkeypatch.setattr(Q.lean, "GLOBAL_SCHEDULER_LOCK", tmp_path / "scheduler.lock")
    lock = threading.Lock()
    barrier = threading.Barrier(2)
    claims = {}
    snapshot_calls = 0
    active_total = 0
    max_active_total = 0
    active_by_pod = {"pod1": 0, "pod2": 0}
    max_active_by_pod = {"pod1": 0, "pod2": 0}
    job_slots = {
        job["id"]: job["resource"]["required_slot"] for job in queue["jobs"]
    }

    def fake_snapshot(_queue):
        nonlocal snapshot_calls
        with lock:
            snapshot_calls += 1
            return (
                {slot: 0 for slot in Q.EXPECTED_SLOTS},
                copy.deepcopy(claims),
            )

    def fake_ssh(_queue, pod, _remote, **kwargs):
        nonlocal active_total, max_active_total
        job_id = kwargs["phase"].removeprefix("rolling-continuation-launch:")
        slot = job_slots[job_id]
        with lock:
            active_total += 1
            active_by_pod[pod] += 1
            max_active_total = max(max_active_total, active_total)
            max_active_by_pod[pod] = max(max_active_by_pod[pod], active_by_pod[pod])
        try:
            barrier.wait(timeout=2)
            with lock:
                claims[job_id] = {
                    "pod": pod,
                    "gpu": int(slot.removeprefix(f"{pod}/gpu")),
                    "state": "launched",
                }
            return f"launched {job_id}"
        finally:
            with lock:
                active_by_pod[pod] -= 1
                active_total -= 1

    monkeypatch.setattr(Q.lean, "live_snapshot", fake_snapshot)
    monkeypatch.setattr(Q.lean, "_run_ssh", fake_ssh)
    result = Q.cmd_fill(queue, count=4, execute=True, confirm=Q.CONFIRM)

    assert result["attempted_count"] == 4
    assert len(result["launched"]) == 4
    assert snapshot_calls == 2
    assert max_active_total == 2
    assert max_active_by_pod == {"pod1": 1, "pod2": 1}
    assert [row["required_slot"].split("/", 1)[0] for row in result["launched"]] == [
        "pod1",
        "pod2",
        "pod1",
        "pod2",
    ]


def test_execute_fill_never_replays_attempted_jobs_when_next_snapshot_omits_claims(
    tmp_path, monkeypatch
):
    queue = _loaded(tmp_path)
    monkeypatch.setattr(Q.lean, "GLOBAL_SCHEDULER_LOCK", tmp_path / "scheduler.lock")
    barrier = threading.Barrier(2)
    calls = []

    def empty_snapshot(_queue):
        return {slot: 0 for slot in Q.EXPECTED_SLOTS}, {}

    def fake_ssh(_queue, pod, _remote, **kwargs):
        job_id = kwargs["phase"].removeprefix("rolling-continuation-launch:")
        calls.append((pod, job_id))
        barrier.wait(timeout=2)
        return f"launched {job_id}"

    monkeypatch.setattr(Q.lean, "live_snapshot", empty_snapshot)
    monkeypatch.setattr(Q.lean, "_run_ssh", fake_ssh)
    result = Q.cmd_fill(queue, count=4, execute=True, confirm=Q.CONFIRM)

    assert result["attempted_count"] == 4
    attempted_ids = [row["job_id"] for row in result["launched"]]
    assert attempted_ids == ["rolling_job_0", "rolling_job_3", "rolling_job_1", "rolling_job_4"]
    assert len(attempted_ids) == len(set(attempted_ids))
    assert {job_id for _pod, job_id in calls} == set(attempted_ids)


def test_execute_fill_waits_for_cross_pod_sibling_and_reports_partial_success(
    tmp_path, monkeypatch
):
    queue = _loaded(tmp_path)
    monkeypatch.setattr(Q.lean, "GLOBAL_SCHEDULER_LOCK", tmp_path / "scheduler.lock")
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    calls = []
    snapshot_calls = 0
    pod2_finished = threading.Event()

    def fake_snapshot(_queue):
        nonlocal snapshot_calls
        snapshot_calls += 1
        return {slot: 0 for slot in Q.EXPECTED_SLOTS}, {}

    def fake_ssh(_queue, pod, _remote, **kwargs):
        job_id = kwargs["phase"].removeprefix("rolling-continuation-launch:")
        with lock:
            calls.append((pod, job_id))
        barrier.wait(timeout=2)
        if pod == "pod1":
            raise Q.lean.QueueError("pod1 deliberate launch failure")
        pod2_finished.set()
        return "pod2 launched successfully"

    monkeypatch.setattr(Q.lean, "live_snapshot", fake_snapshot)
    monkeypatch.setattr(Q.lean, "_run_ssh", fake_ssh)
    with pytest.raises(Q.ContinuationLaunchBatchError) as caught:
        Q.cmd_fill(queue, count=4, execute=True, confirm=Q.CONFIRM)

    error = caught.value
    assert pod2_finished.is_set()
    assert snapshot_calls == 1
    assert sorted(pod for pod, _job_id in calls) == ["pod1", "pod2"]
    assert error.result["attempted_count"] == 2
    assert [row["required_slot"] for row in error.result["launched"]] == ["pod2/gpu0"]
    assert [row["required_slot"] for row in error.result["failed"]] == ["pod1/gpu0"]
    assert error.result["failed"][0]["error_kind"] == "QueueError"
    assert "successful=['rolling_job_3']" in str(error)
    assert "failed=['rolling_job_0']" in str(error)


def test_main_prints_machine_readable_partial_batch_result(tmp_path, monkeypatch, capsys):
    queue_path = _write(tmp_path, _queue())
    partial = {
        "mode": "fill",
        "dry_run": False,
        "count_limit": 2,
        "attempted_count": 2,
        "scheduler_lock": "/tmp/lock",
        "launched": [{"job_id": "ok", "required_slot": "pod2/gpu0"}],
        "failed": [
            {
                "job_id": "bad",
                "required_slot": "pod1/gpu0",
                "error_kind": "QueueError",
                "error": "failed",
            }
        ],
    }

    def fail_fill(*_args, **_kwargs):
        raise Q.ContinuationLaunchBatchError(partial)

    monkeypatch.setattr(Q, "cmd_fill", fail_fill)
    rc = Q.main(
        [
            "--queue",
            str(queue_path),
            "fill",
            "--count",
            "2",
            "--execute",
            "--confirm",
            Q.CONFIRM,
        ]
    )
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    lines = captured.err.splitlines()
    assert lines[0].startswith("BATCH_RESULT=")
    assert json.loads(lines[0].removeprefix("BATCH_RESULT=")) == partial
    assert lines[1].startswith("ERROR: cross-Pod launch batch failed")


def test_existing_claim_must_match_required_slot(tmp_path):
    queue = _loaded(tmp_path)
    job = queue["jobs"][0]
    with pytest.raises(Q.ContinuationQueueError, match="expected required_slot"):
        Q._validate_live_claim_slots(
            queue,
            {job["id"]: {"pod": "pod1", "gpu": 2}},
        )


class _FakeScalar:
    def __init__(self, value: int):
        self._value = value

    def item(self) -> int:
        return self._value


class _FakeTensor:
    def __init__(self, values: list[float]):
        self.values = values

    def numel(self) -> int:
        return len(self.values)


class _FakeTorch:
    Tensor = _FakeTensor

    @staticmethod
    def is_floating_point(_value):
        return True

    @staticmethod
    def is_complex(_value):
        return False

    @staticmethod
    def isfinite(value):
        return SimpleNamespace(sum=lambda: _FakeScalar(sum(x == x for x in value.values)))


def _checkpoint(*, optimizer: bool = True) -> dict:
    value = {
        "iter": 1600,
        "model_state_dict": {
            "actor.0.weight": _FakeTensor([1.0, 2.0]),
            "critic.0.weight": _FakeTensor([3.0]),
        },
        "infos": {
            "training_contract_schema_version": 3,
            "training_contract_sha256": "4" * 64,
            "training_contract_lineage_exact": 0,
        },
    }
    if optimizer:
        value["optimizer_state_dict"] = {
            "state": {0: {"moment": _FakeTensor([0.1])}},
            "param_groups": [{"params": [0]}],
        }
    return value


def test_parent_checkpoint_requires_nonempty_optimizer_and_param_groups():
    with pytest.raises(Q.ContinuationQueueError, match="optimizer_state_dict"):
        Q._validate_checkpoint_payload(
            _checkpoint(optimizer=False),
            expected_iteration=1600,
            expected_hard_sha256="4" * 64,
            torch_module=_FakeTorch,
        )
    missing_groups = _checkpoint()
    missing_groups["optimizer_state_dict"]["param_groups"] = []
    with pytest.raises(Q.ContinuationQueueError, match="param_groups must be nonempty"):
        Q._validate_checkpoint_payload(
            missing_groups,
            expected_iteration=1600,
            expected_hard_sha256="4" * 64,
            torch_module=_FakeTorch,
        )


def test_parent_checkpoint_finite_hard_binding_and_full_state_pass():
    audit = Q._validate_checkpoint_payload(
        _checkpoint(),
        expected_iteration=1600,
        expected_hard_sha256="4" * 64,
        torch_module=_FakeTorch,
    )
    assert audit == {
        "floating_tensor_count": 3,
        "floating_elements": 4,
        "nonfinite_floating_elements": 0,
        "actor_model_key_count": 1,
        "critic_model_key_count": 1,
        "optimizer_resume_eligible": 1,
    }


@pytest.mark.parametrize("missing_prefix", ["actor.", "critic."])
def test_parent_checkpoint_requires_actor_and_critic_model_keys(missing_prefix):
    checkpoint = _checkpoint()
    checkpoint["model_state_dict"] = {
        key: value
        for key, value in checkpoint["model_state_dict"].items()
        if not key.startswith(missing_prefix)
    }
    with pytest.raises(Q.ContinuationQueueError, match=r"both actor\.\* and critic\.\*"):
        Q._validate_checkpoint_payload(
            checkpoint,
            expected_iteration=1600,
            expected_hard_sha256="4" * 64,
            torch_module=_FakeTorch,
        )


def test_claim_uses_absolute_budget_and_one_harness_checkpoint(tmp_path):
    queue = _loaded(tmp_path)
    job = queue["jobs"][0]
    slot = Q._slots(queue)[job["resource"]["required_slot"]]
    claim, argv, absolute = Q._launch_contract(queue, job, slot)
    assert absolute["absolute_iteration_exclusive_bound"] == 3601
    assert claim["content"]["budget"]["milestones"] == [1800, 2100, 2600, 3600]
    assert claim["content"]["formal_evidence_eligible"] is False
    assert claim["content"]["continuation"]["descendant_exact_eligible"] is False
    continuation = claim["content"]["continuation"]
    parent = queue["parent_selection"]["pod1_parent"]
    assert continuation["parent_original_queue_claim_path"] == parent[
        "original_queue_claim_path"
    ]
    assert continuation["parent_original_queue_claim_sha256"] == parent[
        "original_queue_claim_sha256"
    ]
    assert continuation["parent_original_run_binding_path"] == parent[
        "original_run_binding_path"
    ]
    assert continuation["parent_original_run_binding_sha256"] == parent[
        "original_run_binding_sha256"
    ]
    assert continuation["parent_rsl_log_dir"] == parent["selected_rsl_log_dir"]
    runner_raw, runner_sha = Q._runner_payload()
    assert hashlib.sha256(runner_raw).hexdigest() == runner_sha
    assert (
        claim["content"]["continuation"]["continuation_runner_script_sha256"]
        == runner_sha
    )
    keys = [Q.lean._override_key(value, "argv") for value in argv[2:]]
    assert len(keys) == len(set(keys))
    assert keys.count("checkpoint_path") == 1
    assert f"max_iterations=2001" in argv
    assert "max_iterations=3601" not in argv
    assert claim["content"]["budget"]["trainer_max_iterations_arg"] == 2001
    assert claim["content"]["budget"]["absolute_iteration_exclusive_bound"] == 3601
    optimizer_flags = {
        "checkpoint_tolerant=false",
        "checkpoint_allow_missing_contract=false",
        "checkpoint_allow_contract_mismatch=true",
    }
    assert optimizer_flags.issubset(set(argv))
    assert claim["training_argv"][-1].startswith("++training_launch_claim_sha256=")


def test_parent_validator_is_transmitted_and_does_not_require_runner_in_source(tmp_path):
    queue = _loaded(tmp_path)
    job = queue["jobs"][0]
    raw, digest = Q._runner_payload()
    command = Q._parent_validation_command(job, raw, digest)
    assert digest in command
    assert "embedded_rolling_parent_validator" in command
    assert "from run_phase1_rolling_timing_supercombo_queue import" not in command
    assert len(command.encode()) < 64 * 1024


def test_inspect_parents_uses_one_read_only_connection_per_pod_while_blocked(
    tmp_path, monkeypatch
):
    source = _queue()
    source["launch_authorized"] = False
    source["preregistration_status"] = "blocked_pending_parent_inspection"
    for job in source["jobs"]:
        job["status"] = "blocked"
        job["blocker"] = "parent inspection pending"
    queue = _loaded(tmp_path, source)
    calls = []

    def fake_ssh(_queue, pod, remote, **kwargs):
        calls.append((pod, kwargs["phase"], kwargs["timeout"], remote))
        assert "nvidia-smi" not in remote
        assert "launch_kit_training_locked" not in remote
        names = ["pod1_parent"] if pod == "pod1" else ["pod2_parent", "pod2_continuous"]
        return "\n".join(
            json.dumps(
                {
                    "parent": name,
                    "embedded_iteration": {
                        "pod1_parent": 1600,
                        "pod2_parent": 4700,
                        "pod2_continuous": 4500,
                    }[name],
                    "optimizer_state_entries": 74,
                    "optimizer_param_groups": 1,
                    "actor_model_key_count": 7,
                    "critic_model_key_count": 7,
                    "optimizer_resume_eligible": 1,
                    "floating_tensor_count": 74,
                    "floating_elements": 1762715,
                    "nonfinite_floating_elements": 0,
                },
                sort_keys=True,
            )
            for name in names
        )

    monkeypatch.setattr(Q.lean, "_run_ssh", fake_ssh)
    result = Q.cmd_inspect_parents(queue)
    assert result["read_only"] is True
    assert result["unique_parent_count"] == 3
    assert result["ssh_connections"] == {"pod1": 1, "pod2": 1}
    assert {row["parent"] for row in result["parents"]} == {
        "pod1_parent",
        "pod2_parent",
        "pod2_continuous",
    }
    assert sorted(pod for pod, _phase, _timeout, _remote in calls) == ["pod1", "pod2"]
    assert all(phase == "rolling-continuation-inspect-parents" for _, phase, _, _ in calls)
    assert all(timeout == 180 for _, _, timeout, _ in calls)


def test_inspect_parent_timeout_is_not_replayed(tmp_path, monkeypatch):
    queue = _loaded(tmp_path)
    calls = {"pod1": 0, "pod2": 0}

    def fail_once(_queue, pod, _remote, **_kwargs):
        calls[pod] += 1
        raise Q.lean.QueueError(f"{pod} timed out")

    monkeypatch.setattr(Q.lean, "_run_ssh", fail_once)
    with pytest.raises(Q.ContinuationQueueError, match="timed out"):
        Q.cmd_inspect_parents(queue)
    assert calls["pod1"] <= 1
    assert calls["pod2"] <= 1


def test_parent_file_sha_mismatch_fails_before_checkpoint_load(tmp_path):
    claim_content = {"schema_version": 1}
    claim = {
        "schema_version": 2,
        "content": claim_content,
        "content_sha256": Q._canonical_sha256(claim_content),
    }
    claim_path = tmp_path / "queue_claim.json"
    claim_path.write_text(json.dumps(claim), encoding="utf-8")
    binding_content = {
        "schema_version": 1,
        "claim_path": str(claim_path),
        "binding_path": str(tmp_path / "run_binding.json"),
        "claim_content_sha256": claim["content_sha256"],
        "rsl_log_dir": str(tmp_path / "rsl"),
        "job_id": "parent",
    }
    binding = {
        "schema_version": 1,
        "content": binding_content,
        "content_sha256": Q._canonical_sha256(binding_content),
    }
    binding_path = tmp_path / "run_binding.json"
    binding_path.write_text(json.dumps(binding), encoding="utf-8")
    rsl = tmp_path / "rsl"
    (rsl / "params").mkdir(parents=True)
    checkpoint = rsl / "model_1600.pt"
    checkpoint.write_bytes(b"checkpoint")
    hard = rsl / "params" / "training_contract.json"
    hard.write_text(json.dumps({"schema_version": 3}), encoding="utf-8")
    # _validate_parent_spec deliberately insists on /workspace paths in real
    # launches.  This fixture reaches the stable-file SHA branch by replacing
    # only that path gate; no filesystem mutation is performed by validation.
    original = Q._workspace_path
    Q._workspace_path = lambda value, _label, allow_pending=False: str(value)
    try:
        spec = {
            "parent_name": "pod1_parent",
            "original_job_id": "parent",
            "claim_path": str(claim_path),
            "claim_sha256": "0" * 64,
            "binding_path": str(binding_path),
            "binding_sha256": hashlib.sha256(binding_path.read_bytes()).hexdigest(),
            "rsl_log_dir": str(rsl),
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "hard_contract_path": str(hard),
            "hard_contract_sha256": hashlib.sha256(hard.read_bytes()).hexdigest(),
            "embedded_iteration": 1600,
        }
        with pytest.raises(Q.ContinuationQueueError, match="queue claim SHA mismatch"):
            Q._validate_parent_spec(spec, torch_module=_FakeTorch)
    finally:
        Q._workspace_path = original
