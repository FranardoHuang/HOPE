from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_lean_training_queue.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("lean_queue_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


Q = _load_module()


def _job(index: int, *, status: str = "ready") -> dict:
    return {
        "id": f"job{index}",
        "human_name": f"action-specific pilot {index}",
        "action": f"action{index}",
        "status": status,
        "blocker": "offline gate open" if status == "blocked" else None,
        "motion": {"action": f"action{index}", "bindings": {"motion_file": f"/workspace/motion/{index}.npz"}},
        "bank": {"action": f"action{index}", "train_path": f"/workspace/bank/{index}.npz", "train_arg": "++task.racket.question_bank"},
        "exam": {"action": f"action{index}", "path": f"/workspace/exam/{index}.json", "family": f"exam{index}"},
        "source": {"checkout": "/workspace/source", "commit": "1" * 40},
        "recipe": {"base": ["task=Task", "algo=ppo"], "delta": [f"x={index}"]},
        "seed": index,
        "budget": {"num_envs": 512, "max_iterations": 1000, "save_interval": 100},
        "milestones": [200, 500, 1000],
        "resource": {"policy": "six_gpu_round_robin"},
        "run_name": f"run{index}",
        "run_dir": f"/workspace/runs/{index}",
    }


def _queue(job_count: int = 1) -> dict:
    return {
        "schema_version": 1,
        "simulation_only": True,
        "ssh": {"key": "/key"},
        "pods": {
            "pod1": {"host": "pod1", "port": 1, "gpus": [0, 1, 2], "max_trainers_per_gpu": 4},
            "pod2": {"host": "pod2", "port": 2, "gpus": [0, 1, 2], "max_trainers_per_gpu": 3},
        },
        "jobs": [_job(index) for index in range(job_count)],
    }


def _write(tmp_path: Path, queue: dict) -> Path:
    path = tmp_path / "queue.yaml"
    path.write_text(yaml.safe_dump(queue, sort_keys=False), encoding="utf-8")
    return path


def test_yaml_requires_action_specific_bindings_and_three_milestones(tmp_path):
    queue = _queue()
    loaded = Q.load_queue(_write(tmp_path, queue))
    assert loaded["jobs"][0]["motion"]["bindings"]["motion_file"] == "/workspace/motion/0.npz"
    for missing in ("motion", "bank", "exam", "source", "recipe", "seed", "budget", "milestones", "resource"):
        broken = _queue()
        del broken["jobs"][0][missing]
        try:
            Q.load_queue(_write(tmp_path, broken))
        except Q.QueueError:
            pass
        else:
            raise AssertionError(f"missing binding was accepted: {missing}")
    wrong = _queue()
    wrong["jobs"][0]["bank"]["action"] = "different_action"
    try:
        Q.load_queue(_write(tmp_path, wrong))
    except Q.QueueError:
        pass
    else:
        raise AssertionError("cross-action bank binding was accepted")


def test_round_robin_fills_six_gpus_one_round_at_a_time_and_honors_caps():
    queue = _queue(22)
    empty = {slot.name: 0 for slot in Q.slots(queue)}
    assignments = Q._assign(queue, empty)
    names = [slot.name for _, slot in assignments]
    one_round = [
        "pod1/gpu0", "pod1/gpu1", "pod1/gpu2",
        "pod2/gpu0", "pod2/gpu1", "pod2/gpu2",
    ]
    assert names[:6] == one_round
    assert names[6:12] == one_round
    assert len(assignments) == 21
    assert names.count("pod1/gpu0") == 4
    assert names.count("pod2/gpu0") == 3


def test_runner_entrypoints_are_source_pinned_and_yaml_override_is_rejected(tmp_path):
    queue = _queue()
    queue["runner"] = {
        "entrypoint_relative": "../../robot/broad_kill.sh",
        "setup_relative": "/tmp/setup.sh",
    }
    try:
        Q.load_queue(_write(tmp_path, queue))
    except Q.QueueError:
        pass
    else:
        raise AssertionError("queue-controlled runner path was accepted")


def test_ready_placeholder_and_parent_traversal_fail_before_any_ssh(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(Q, "_run_ssh", lambda *args, **kwargs: calls.append(args))
    for field, value in (
        ("zero_commit", "0" * 40),
        ("placeholder_motion", "/workspace/path/to/motion.npz"),
        ("traversal_source", "/workspace/source/../robot"),
    ):
        queue = _queue()
        if field == "zero_commit":
            queue["jobs"][0]["source"]["commit"] = value
        elif field == "placeholder_motion":
            queue["jobs"][0]["motion"]["bindings"]["motion_file"] = value
        else:
            queue["jobs"][0]["source"]["checkout"] = value
        rc = Q.main(
            [
                "--queue", str(_write(tmp_path, queue)), "launch-next",
                "--execute", "--confirm", Q.CONFIRM,
            ]
        )
        assert rc == 2
    assert calls == []


def test_blocked_job_is_never_assigned_or_rendered_for_launch():
    queue = _queue(3)
    queue["jobs"][0] = _job(0, status="blocked")
    assignments = Q._assign(queue, {slot.name: 0 for slot in Q.slots(queue)})
    assert [job["id"] for job, _ in assignments] == ["job1", "job2"]
    plan = Q.cmd_plan(queue, live=False)
    assert plan["blocked"] == [{"job_id": "job0", "reason": "offline gate open"}]


def test_existing_claim_is_skipped_so_launch_next_advances():
    queue = _queue(3)
    empty = {slot.name: 0 for slot in Q.slots(queue)}
    assignments = Q._assign(queue, empty, {"job0"})
    assert [job["id"] for job, _ in assignments] == ["job1", "job2"]


def test_launch_next_defaults_to_dry_run_and_includes_only_minimal_preflight():
    queue = _queue()
    result = Q.cmd_launch_next(queue, execute=False, confirm=None)
    assert result["dry_run"] is True
    rendered = " ".join(result["ssh_argv"])
    assert "git -C" in rendered
    assert "status --porcelain" in rendered
    assert "nvidia-smi" in rendered
    assert "/workspace/motion/0.npz" in rendered
    assert "/workspace/bank/0.npz" in rendered
    assert "/workspace/exam/0.json" in rendered
    assert "device=cuda%3A0" in rendered or "device=cuda:0" in rendered
    assert "/workspace/source/hope_training/whole_body_tracking/scripts/launch_kit_training_locked.sh" in rendered
    assert "pip freeze" not in rendered
    assert "sha256sum" not in rendered
    assert "killall" not in rendered
    assert "pkill" not in rendered


def test_execute_locks_before_resampling_and_selecting_slot(tmp_path, monkeypatch):
    queue = _queue(2)
    order = []
    monkeypatch.setattr(Q, "GLOBAL_SCHEDULER_LOCK", tmp_path / "scheduler.lock")

    def fake_flock(_fd, operation):
        assert operation == Q.fcntl.LOCK_EX
        order.append("lock")

    def fake_snapshot(_queue):
        assert order == ["lock"]
        order.append("snapshot")
        occupancy = {slot.name: 0 for slot in Q.slots(queue)}
        occupancy["pod1/gpu0"] = 1
        return occupancy, {}

    def fake_ssh(_queue, pod, _remote, **_kwargs):
        assert order == ["lock", "snapshot"]
        order.append("launch")
        assert pod == "pod1"
        return "launched"

    monkeypatch.setattr(Q.fcntl, "flock", fake_flock)
    monkeypatch.setattr(Q, "live_snapshot", fake_snapshot)
    monkeypatch.setattr(Q, "_run_ssh", fake_ssh)
    result = Q.cmd_launch_next(queue, execute=True, confirm=Q.CONFIRM)
    assert order == ["lock", "snapshot", "launch"]
    assert result["resource"] == "pod1/gpu1"


def test_example_is_valid_and_safely_blocked():
    queue = Q.load_queue(ROOT / "configs" / "lean_training_queue.example.yaml")
    assert queue["jobs"][0]["status"] == "blocked"
    assert Q.cmd_plan(queue, live=False)["assignments"] == []
