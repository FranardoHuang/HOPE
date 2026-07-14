from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import shlex
import subprocess
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


def test_dispatch_pods_excludes_reserved_pod_but_keeps_round_robin():
    queue = _queue(8)
    queue["dispatch_pods"] = ["pod2"]
    for job in queue["jobs"]:
        job["resource"] = {"policy": "dispatch_gpu_round_robin"}
    empty = {slot.name: 0 for slot in Q.slots(queue)}
    assignments = Q._assign(queue, empty)
    names = [slot.name for _, slot in assignments]
    assert names == [
        "pod2/gpu0", "pod2/gpu1", "pod2/gpu2",
        "pod2/gpu0", "pod2/gpu1", "pod2/gpu2",
        "pod2/gpu0", "pod2/gpu1",
    ]
    assert all(slot.pod == "pod2" for _, slot in assignments)


def test_six_gpu_policy_rejects_restricted_dispatch_set(tmp_path):
    queue = _queue()
    queue["dispatch_pods"] = ["pod2"]
    try:
        Q.load_queue(_write(tmp_path, queue))
    except Q.QueueError as exc:
        assert "six_gpu_round_robin requires both" in str(exc)
    else:
        raise AssertionError("reserved Pod was silently admitted by six-GPU policy")


def test_duplicate_nvidia_rows_count_one_unique_numeric_pid_per_gpu():
    snapshot = {
        "compute_rows": [
            "GPU-A, 1881321",
            "GPU-A, 1881321",
            "GPU-A, 1881444",
            "GPU-B, 2999000",
            "GPU-B, 2999000",
        ],
        "gpu_rows": ["0, GPU-A", "1, GPU-B", "2, GPU-C"],
    }
    assert Q._parse_gpu_occupancy("pod1", snapshot) == {
        "pod1/gpu0": 2,
        "pod1/gpu1": 1,
        "pod1/gpu2": 0,
    }


def test_remote_final_capacity_awk_deduplicates_and_ignores_non_numeric_rows():
    completed = subprocess.run(
        ["awk", Q.UNIQUE_NUMERIC_PID_AWK],
        input="1881321\n1881321\nnot-a-pid\n 1881444 \n",
        text=True,
        check=True,
        stdout=subprocess.PIPE,
    )
    assert completed.stdout.strip() == "2"


def test_child_env_builder_makes_setup_exported_pythonpath_effective(tmp_path):
    source = tmp_path / "source"
    package = source / "lean_queue_fake_module"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("VALUE = 7\n", encoding="utf-8")
    setup = tmp_path / "setup.sh"
    setup.write_text(
        f"export HOPE_WBT_PYTHONPATH={shlex.quote(str(source))}\n",
        encoding="utf-8",
    )
    probe = [sys.executable, "-c", "import lean_queue_fake_module"]
    raw_env = dict(os.environ)
    raw_env.pop("PYTHONPATH", None)
    raw = subprocess.run(probe, env=raw_env, check=False)
    assert raw.returncode != 0
    child = Q._child_env_command(probe, 2)
    completed = subprocess.run(
        ["bash", "-lc", f"source {shlex.quote(str(setup))}; {child}"],
        check=False,
    )
    assert completed.returncode == 0


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


def test_recipe_compiler_rejects_ambiguous_or_harness_owned_overrides_before_ssh(
    tmp_path, monkeypatch
):
    calls = []
    monkeypatch.setattr(Q, "_run_ssh", lambda *args, **kwargs: calls.append(args))
    attacks = {
        "duplicate_normalized_key": ["+task=OtherTask"],
        "harness_owned_seed": ["seed=4"],
        "harness_owned_motion": ["++motion_file=/tmp/other.npz"],
        "harness_owned_bank": ["task.racket.question_bank=/tmp/other.npz"],
        "harness_owned_budget": ["num_envs=1"],
        "harness_owned_run": ["run_name=collision"],
        "harness_owned_device": ["device=cpu"],
        "harness_owned_claim": [f"++training_launch_claim_sha256={'0' * 64}"],
        "hydra_flag": ["--multirun"],
        "hydra_interpolation": ["x=${oc.env:HOME}"],
        "hydra_delete": ["~x=1"],
    }
    for name, delta in attacks.items():
        queue = _queue()
        queue["jobs"][0]["recipe"]["delta"] = delta
        try:
            Q.load_queue(_write(tmp_path, queue))
        except Q.QueueError:
            pass
        else:
            raise AssertionError(f"unsafe recipe was accepted: {name}")
    assert calls == []


def test_run_directory_is_globally_unique_and_outside_ready_sources(tmp_path):
    duplicate = _queue(2)
    duplicate["jobs"][1]["run_dir"] = duplicate["jobs"][0]["run_dir"]
    try:
        Q.load_queue(_write(tmp_path, duplicate))
    except Q.QueueError as exc:
        assert "duplicate run_dir" in str(exc)
    else:
        raise AssertionError("duplicate run_dir was accepted")

    for run_dir in ("/workspace/source", "/workspace/source/runs/job0"):
        nested = _queue()
        nested["jobs"][0]["run_dir"] = run_dir
        try:
            Q.load_queue(_write(tmp_path, nested))
        except Q.QueueError as exc:
            assert "must not equal or be inside" in str(exc)
        else:
            raise AssertionError(f"source-contained run_dir was accepted: {run_dir}")


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
    assert 'PYTHONPATH="${HOPE_WBT_PYTHONPATH}"' in rendered
    assert "find_spec" in rendered
    assert "--cfg job --resolve" in rendered
    assert "training_launch_claim_sha256=" in rendered
    assert rendered.index("find_spec") < rendered.index("queue_claim.json")
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


def test_pending_claim_reservations_spread_five_independent_resamples():
    queue = _queue(5)
    occupancy = {slot.name: 0 for slot in Q.slots(queue)}
    claims = {}
    resources = []
    for _ in range(5):
        effective = Q._effective_occupancy(queue, occupancy, claims)
        job, slot = Q._assign(queue, effective, set(claims))[0]
        resources.append(slot.name)
        claims[job["id"]] = {
            "pod": slot.pod,
            "gpu": slot.gpu,
            "state": "claimed",
            "claim_path": f"{job['run_dir']}/queue_claim.json",
        }
    assert resources == [
        "pod1/gpu0", "pod1/gpu1", "pod1/gpu2", "pod2/gpu0", "pod2/gpu1"
    ]


def test_doctor_and_launch_share_exact_claim_bound_argv_before_fresh_claim():
    queue = _queue()
    job = queue["jobs"][0]
    slot = Q.slots(queue)[0]
    claim, training_argv = Q._launch_contract(queue, job, slot)
    digest = Q._canonical_sha256(claim["content"])
    assert digest == claim["content_sha256"]
    assert claim["training_argv"] == [
        *claim["content"]["training_argv_without_claim"],
        f"++training_launch_claim_sha256={digest}",
    ]
    assert claim["training_argv"] == training_argv
    assert claim["content"]["source"] == job["source"]
    assert claim["content"]["run_name"] == job["run_name"]
    assert claim["content"]["budget"]["max_iterations"] == 1000
    assert claim["content"]["inputs"]["motion"]["bindings"] == job["motion"]["bindings"]
    assert claim["content"]["inputs"]["bank"]["train_path"] == job["bank"]["train_path"]
    assert claim["content"]["inputs"]["exam"]["path"] == job["exam"]["path"]

    compose = Q._child_env_command(Q._hydra_compose_argv(training_argv), slot.gpu)
    trainer = Q._child_env_command(training_argv, slot.gpu)
    doctor = Q._doctor_script(queue, job, slot)
    launch_body = shlex.split(Q._launch_script(queue, job, slot))[-1]
    assert compose in doctor
    assert compose in launch_body
    assert trainer in launch_body
    assert "find_spec" in doctor
    assert "HOPE_WBT_PYTHONPATH" in doctor
    assert "queue_claim.json" not in doctor
    assert job["run_dir"] not in doctor
    assert "mkdir" not in doctor

    run_parent = str(Path(job["run_dir"]).parent)
    assert f"mkdir -p {run_parent}" in launch_body
    assert f"mkdir {job['run_dir']}" in launch_body
    assert f"mkdir -p {job['run_dir']}" not in launch_body
    assert launch_body.index(compose) < launch_body.index("queue_claim.json")
    assert launch_body.index(f"mkdir {job['run_dir']}") < launch_body.index(
        "queue_claim.json"
    )
    assert launch_body.index("queue_claim.json") < launch_body.rindex(trainer)


def test_fill_is_one_scheduler_sequence_and_stops_before_claim_on_doctor_failure(
    tmp_path, monkeypatch
):
    queue = _queue(2)
    monkeypatch.setattr(Q, "GLOBAL_SCHEDULER_LOCK", tmp_path / "scheduler.lock")
    occupancy = {slot.name: 0 for slot in Q.slots(queue)}
    claims = {}
    calls = []

    def fake_snapshot(_queue):
        return dict(occupancy), dict(claims)

    def fake_ssh(_queue, pod, _remote, **kwargs):
        calls.append((pod, kwargs["phase"]))
        raise Q.QueueError("doctor failed")

    monkeypatch.setattr(Q, "live_snapshot", fake_snapshot)
    monkeypatch.setattr(Q, "_run_ssh", fake_ssh)
    try:
        Q.cmd_fill(queue, execute=True, confirm=Q.CONFIRM, count=2)
    except Q.QueueError as exc:
        assert "doctor failed" in str(exc)
    else:
        raise AssertionError("doctor failure was ignored")
    assert calls == [("pod1", "doctor:job0")]
    assert claims == {}


def test_fill_rescans_after_first_iteration_before_next_job(tmp_path, monkeypatch):
    queue = _queue(3)
    monkeypatch.setattr(Q, "GLOBAL_SCHEDULER_LOCK", tmp_path / "scheduler.lock")
    occupancy = {slot.name: 0 for slot in Q.slots(queue)}
    claims = {}
    calls = []

    def fake_snapshot(_queue):
        calls.append("snapshot")
        return dict(occupancy), dict(claims)

    def fake_ssh(_queue, pod, _remote, **kwargs):
        phase = kwargs["phase"]
        calls.append(phase)
        if phase.startswith("launch-first-iteration:"):
            job_id = phase.split(":", 1)[1]
            index = int(job_id.removeprefix("job"))
            slot = Q.slots(queue)[index]
            occupancy[slot.name] += 1
            claims[job_id] = {
                "pod": slot.pod,
                "gpu": slot.gpu,
                "state": "launched",
                "claim_path": f"{queue['jobs'][index]['run_dir']}/queue_claim.json",
            }
        return "ok"

    monkeypatch.setattr(Q, "live_snapshot", fake_snapshot)
    monkeypatch.setattr(Q, "_run_ssh", fake_ssh)
    result = Q.cmd_fill(queue, execute=True, confirm=Q.CONFIRM, count=2)
    assert [item["resource"] for item in result["launched"]] == [
        "pod1/gpu0", "pod1/gpu1"
    ]
    assert calls == [
        "snapshot",
        "doctor:job0",
        "launch-first-iteration:job0",
        "snapshot",
        "doctor:job1",
        "launch-first-iteration:job1",
    ]


def test_example_is_valid_and_safely_blocked():
    queue = Q.load_queue(ROOT / "configs" / "lean_training_queue.example.yaml")
    assert queue["jobs"][0]["status"] == "blocked"
    assert Q.cmd_plan(queue, live=False)["assignments"] == []


def test_active_fresh_c_queue_is_one_seed_one_mechanism_per_ready_cell():
    queue = Q.load_queue(
        ROOT / "configs" / "phase1_fresh_c_mechanism_queue_20260714.yaml"
    )
    ready = [job for job in queue["jobs"] if job["status"] == "ready"]
    blocked = [job for job in queue["jobs"] if job["status"] == "blocked"]
    rejected = [job for job in queue["jobs"] if job["status"] == "rejected"]
    assert len(ready) == 6
    assert len(rejected) == 6
    assert blocked == []
    assert queue["dispatch_pods"] == ["pod2"]
    axis_prefixes = (
        "++task.rewards.free_wrist_vel_mimic=",
        "++task.rewards.motion_scale_in_window=",
        "task.rewards.base_decel_weight=",
        "task.motion.post_swing_start_prob=",
    )
    expected = {
        "fresh_c_v1_free_wrist_velocity_retry_v2": ("true", "1.0", "0.0", "0.25"),
        "fresh_c_v2_motion_window_scale_retry_v2": ("false", "0.25", "0.0", "0.25"),
        "fresh_c_v1_v2_combined_retry_v2": ("true", "0.25", "0.0", "0.25"),
        "fresh_c_base_deceleration_retry_v2": ("false", "1.0", "1.0", "0.25"),
        "fresh_c_post_swing_replay_half_retry_v2": ("false", "1.0", "0.0", "0.5"),
        "fresh_c_qdot_limit_hinge_w5_retry_v2": ("false", "1.0", "0.0", "0.25"),
    }
    for job in ready:
        assert job["seed"] == 3
        assert job["budget"] == {
            "num_envs": 4096, "max_iterations": 1001, "save_interval": 100
        }
        assert job["milestones"] == [200, 500, 1000]
        expected_source = (
            "a6ccdc7a1c696ff37878039f1e1d83dea28a2bfa"
            if job["id"] == "fresh_c_qdot_limit_hinge_w5_retry_v2"
            else "4467d79f1ed425a4263f0caaad2f661e1ec737ad"
        )
        assert job["source"]["commit"] == expected_source
        base = job["recipe"]["base"]
        delta = job["recipe"]["delta"]
        assert not any(item.startswith(axis_prefixes) for item in base)
        actual = []
        for prefix in axis_prefixes:
            matches = [item.removeprefix(prefix) for item in delta if item.startswith(prefix)]
            assert len(matches) == 1
            actual.append(matches[0])
        assert tuple(actual) == expected[job["id"]]
        qdot_weight = [
            item for item in delta
            if item.startswith("task.rewards.joint_velocity_limit_hinge_weight=")
        ]
        qdot_margin = [
            item for item in delta
            if item.startswith("task.rewards.joint_velocity_limit_hinge_margin=")
        ]
        if job["id"] == "fresh_c_qdot_limit_hinge_w5_retry_v2":
            assert qdot_weight == ["task.rewards.joint_velocity_limit_hinge_weight=-5.0"]
            assert qdot_margin == ["task.rewards.joint_velocity_limit_hinge_margin=0.85"]
        else:
            assert qdot_weight == []
            assert qdot_margin == []
    plan = Q.cmd_plan(queue, live=False)
    assert len(plan["assignments"]) == 6
    assert [item["resource"] for item in plan["assignments"]] == [
        "pod2/gpu0", "pod2/gpu1", "pod2/gpu2",
        "pod2/gpu0", "pod2/gpu1", "pod2/gpu2",
    ]
    retry_ready = [job for job in ready if job["id"].endswith("_retry_v2")]
    assert len(retry_ready) == 6
    for attempt1, retry in zip(rejected, retry_ready, strict=True):
        assert retry["recipe"] == attempt1["recipe"]
        assert retry["motion"] == attempt1["motion"]
        assert retry["bank"] == attempt1["bank"]
        assert retry["exam"] == attempt1["exam"]
        assert retry["run_dir"] != attempt1["run_dir"]
