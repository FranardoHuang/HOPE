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
        "runtime_binding": True,
        "recipe": {"base": ["task=Task", "algo=ppo"], "delta": [f"x={index}"]},
        "seed": index,
        "budget": {"num_envs": 512, "max_iterations": 1001, "save_interval": 100},
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
    off_by_one = _queue()
    off_by_one["jobs"][0]["budget"]["max_iterations"] = 1000
    try:
        Q.load_queue(_write(tmp_path, off_by_one))
    except Q.QueueError as exc:
        assert "max_iterations-1" in str(exc)
    else:
        raise AssertionError("unreachable terminal checkpoint was accepted")
    continuation = _queue()
    continuation["jobs"][0]["recipe"]["delta"].append(
        "checkpoint_path=/workspace/model_100.pt"
    )
    try:
        Q.load_queue(_write(tmp_path, continuation))
    except Q.QueueError as exc:
        assert "supports fresh runs only" in str(exc)
    else:
        raise AssertionError("unbound continuation start iteration was accepted")


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


def test_preferred_slot_is_used_until_capacity_then_round_robin_falls_back():
    queue = _queue(4)
    queue["dispatch_pods"] = ["pod2"]
    for job in queue["jobs"]:
        job["resource"] = {
            "policy": "dispatch_gpu_round_robin",
            "preferred_slot": "pod2/gpu1",
        }
    empty = {slot.name: 0 for slot in Q.slots(queue)}
    names = [slot.name for _, slot in Q._assign(queue, empty)]
    assert names == ["pod2/gpu1", "pod2/gpu1", "pod2/gpu1", "pod2/gpu0"]


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
        "harness_owned_claim_path": ["++training_queue_claim_path=/tmp/claim"],
        "harness_owned_binding_path": ["++training_run_binding_path=/tmp/binding"],
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
    assert claim["content"]["budget"]["max_iterations"] == 1001
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
    assert f"++training_queue_claim_path={job['run_dir']}/queue_claim.json" in doctor
    assert f"++training_run_binding_path={job['run_dir']}/run_binding.json" in doctor
    assert "set -o noclobber" not in doctor
    assert "mkdir" not in doctor
    assert f"exec {Q.GPU_LAUNCH_LOCK_FD}>/tmp/hope_lean_queue_gpu{slot.gpu}.lock" in launch_body
    assert f"flock -n {Q.GPU_LAUNCH_LOCK_FD}" in launch_body
    assert f" {Q.GPU_LAUNCH_LOCK_FD}>&-" in launch_body
    assert f"flock -n /tmp/hope_lean_queue_gpu{slot.gpu}.lock bash" not in launch_body

    run_parent = str(Path(job["run_dir"]).parent)
    assert f"mkdir -p {run_parent}" in launch_body
    assert f"mkdir {job['run_dir']}" in launch_body
    assert f"mkdir {job['run_dir']}/milestones" in launch_body
    assert f"mkdir -p {job['run_dir']}" not in launch_body
    claim_write = launch_body.index("( set -o noclobber")
    capacity_check = launch_body.index("count=$(nvidia-smi")
    assert launch_body.index(compose) < claim_write
    assert launch_body.index(compose) < capacity_check
    assert capacity_check < launch_body.index(f"mkdir {job['run_dir']}")
    assert launch_body.index(f"mkdir {job['run_dir']}") < claim_write
    assert claim_write < launch_body.rindex(trainer)
    launcher_call = launch_body.index("launch_kit_training_locked.sh")
    first_iter_phase = launch_body.index("phase=first_iter")
    assert launcher_call < first_iter_phase


def test_boot_warmup_is_tiny_claim_bound_and_never_reuses_science_namespace():
    queue = _queue()
    job = queue["jobs"][0]
    slot = Q.slots(queue)[2]
    original_budget = dict(job["budget"])
    claim, argv, run_dir = Q._boot_warmup_contract(
        queue, job, slot, "conditional_gpu2_a1"
    )
    assert claim["content"]["purpose"] == "boot_warmup_not_science"
    assert claim["content"]["budget"] == {
        "num_envs": 1, "max_iterations": 2, "save_interval": 1
    }
    assert claim["content_sha256"] == Q._canonical_sha256(claim["content"])
    assert argv[-1] == f"++training_launch_claim_sha256={claim['content_sha256']}"
    assert "num_envs=1" in argv
    assert "max_iterations=2" in argv
    assert "algo.runner.save_interval=1" in argv
    assert any(item.startswith("run_name=boot_warmup_") for item in argv)
    assert not any("training_queue_claim_path" in item for item in argv)
    assert not any("training_run_binding_path" in item for item in argv)
    assert job["run_dir"] not in " ".join(argv)
    assert f"/{job['source']['commit']}/{slot.pod}/gpu{slot.gpu}/" in run_dir
    assert run_dir != job["run_dir"] and not run_dir.startswith(job["source"]["checkout"])
    assert job["budget"] == original_budget

    rendered = Q._boot_warmup_script(queue, job, slot, "conditional_gpu2_a1")
    assert "warmup_claim.json" in rendered
    assert "KIT_BOOT_TIMEOUT_S=180" in rendered
    assert "num_envs=1" in rendered and "max_iterations=2" in rendered
    warmup_body = shlex.split(rendered)[-1]
    assert f"exec {Q.GPU_LAUNCH_LOCK_FD}>/tmp/hope_lean_queue_gpu{slot.gpu}.lock" in warmup_body
    assert f"flock -n {Q.GPU_LAUNCH_LOCK_FD}" in warmup_body
    assert f" {Q.GPU_LAUNCH_LOCK_FD}>&-" in warmup_body
    assert "training_queue_claim_path" not in rendered
    assert "training_run_binding_path" not in rendered
    assert "pkill" not in rendered and "killall" not in rendered


def test_full_scene_probe_preserves_complete_scene_and_scale_in_isolated_namespace():
    queue = _queue()
    job = queue["jobs"][0]
    job["budget"]["num_envs"] = 4096
    slot = Q.slots(queue)[4]
    frozen_job = yaml.safe_dump(job, sort_keys=True)
    science_argv = Q._training_argv(
        queue, job, slot.gpu, include_run_binding=False
    )
    claim, argv, run_dir = Q._full_scene_probe_contract(
        queue, job, slot, "formal_scale_a1"
    )

    content = claim["content"]
    assert content["purpose"] == "full_scene_probe_not_science"
    assert content["not_science"] is True
    assert content["attestable"] is False
    assert content["promotable"] is False
    assert content["source_job_budget"] == {
        "num_envs": 4096, "max_iterations": 1001, "save_interval": 100
    }
    assert content["budget"] == {
        "num_envs": 4096, "max_iterations": 2, "save_interval": 1
    }
    assert content["source"] == job["source"]
    assert content["pod"] == slot.pod and content["gpu"] == slot.gpu
    assert content["inputs"] == {
        "motion": job["motion"], "bank": job["bank"], "exam": job["exam"]
    }
    assert claim["content_sha256"] == Q._canonical_sha256(content)
    assert argv[-1] == f"++training_launch_claim_sha256={claim['content_sha256']}"

    probe_argv = argv[:-1]
    assert len(probe_argv) == len(science_argv)
    changed_keys = {
        "max_iterations", "algo.runner.save_interval", "run_name"
    }
    for science_arg, probe_arg in zip(science_argv[2:], probe_argv[2:]):
        key = Q._override_key(science_arg, "science argv")
        if key not in changed_keys:
            assert probe_arg == science_arg
    assert "num_envs=4096" in probe_argv
    assert "max_iterations=2" in probe_argv
    assert "algo.runner.save_interval=1" in probe_argv
    assert any(
        item.startswith("run_name=full_scene_probe_not_science_job0_")
        for item in probe_argv
    )
    assert not any("training_queue_claim_path" in item for item in probe_argv)
    assert not any("training_run_binding_path" in item for item in probe_argv)
    assert "/_full_scene_probes/job0/" in run_dir
    assert job["run_dir"] not in run_dir
    assert not run_dir.startswith(job["source"]["checkout"])
    assert yaml.safe_dump(job, sort_keys=True) == frozen_job


def test_full_scene_probe_fails_closed_on_num_envs_drift(monkeypatch):
    queue = _queue()
    job = queue["jobs"][0]
    job["budget"]["num_envs"] = 4096
    slot = Q.slots(queue)[0]
    original = Q._training_argv

    def drifted(*args, **kwargs):
        argv = original(*args, **kwargs)
        return ["num_envs=1" if item == "num_envs=4096" else item for item in argv]

    monkeypatch.setattr(Q, "_training_argv", drifted)
    try:
        Q._full_scene_probe_contract(queue, job, slot, "drift_a1")
    except Q.QueueError as exc:
        assert "preserve the source job num_envs" in str(exc)
    else:
        raise AssertionError("a 4096-to-1 full-scene probe drift was accepted")


def test_full_scene_probe_script_is_no_clobber_first_iteration_only():
    queue = _queue()
    job = queue["jobs"][0]
    job["budget"]["num_envs"] = 4096
    slot = Q.slots(queue)[1]
    _claim, _argv, run_dir = Q._full_scene_probe_contract(
        queue, job, slot, "reuse_guard_a1"
    )
    rendered = Q._full_scene_probe_script(
        queue, job, slot, "reuse_guard_a1"
    )
    body = shlex.split(rendered)[-1]
    assert f"mkdir {run_dir}" in body
    assert f"mkdir -p {run_dir}" not in body
    assert "set -o noclobber" in body
    assert "full_scene_probe_claim.json" in body
    assert "queue_claim.json" not in body
    assert "run_binding.json" not in body
    assert "/milestones" not in body
    assert "KIT_BOOT_MARKER='Learning iteration'" in body or "KIT_BOOT_MARKER=Learning" in body
    assert "KIT_BOOT_TIMEOUT_S=900" in body
    assert "KIT_BOOT_STALE_TIMEOUT_S=180" in body
    assert "phase=first_iter not_science=true" in body
    assert f" {Q.GPU_LAUNCH_LOCK_FD}>&-" in body
    assert "pkill" not in rendered and "killall" not in rendered


def test_full_scene_probe_requires_exact_job_confirmation_and_dispatch(monkeypatch):
    queue = _queue()
    calls = []
    monkeypatch.setattr(Q, "live_snapshot", lambda *_args: calls.append("live"))
    for wrong_confirm in (Q.CONFIRM, Q.WARMUP_CONFIRM, Q.ATTEST_CONFIRM):
        try:
            Q.cmd_full_scene_probe(
                queue, job_id="job0", pod="pod1", gpu=0, attempt_id="a1",
                execute=True, confirm=wrong_confirm,
            )
        except Q.QueueError as exc:
            assert Q.FULL_SCENE_PROBE_CONFIRM in str(exc)
        else:
            raise AssertionError(
                f"foreign confirmation {wrong_confirm} authorized a full-scene probe"
            )
    assert calls == []

    queue["dispatch_pods"] = ["pod2"]
    try:
        Q.cmd_full_scene_probe(
            queue, job_id="job0", pod="pod1", gpu=0, attempt_id="a1",
            execute=False, confirm=None,
        )
    except Q.QueueError as exc:
        assert "not dispatch-enabled" in str(exc)
    else:
        raise AssertionError("reserved Pod received a full-scene probe")

    queue["jobs"][0]["resource"]["preferred_slot"] = "pod2/gpu1"
    try:
        Q.cmd_full_scene_probe(
            queue, job_id="job0", pod="pod2", gpu=2, attempt_id="a1",
            execute=False, confirm=None,
        )
    except Q.QueueError as exc:
        assert "must use preferred_slot pod2/gpu1" in str(exc)
    else:
        raise AssertionError("probe drifted from the job's bound GPU")


def test_full_scene_probe_execute_reads_only_selected_dispatch_pod(monkeypatch):
    queue = _queue()
    queue["dispatch_pods"] = ["pod2"]
    queue["jobs"][0]["resource"] = {
        "policy": "dispatch_gpu_round_robin",
        "preferred_slot": "pod2/gpu1",
    }
    calls = []

    def fake_ssh(_queue, pod, _remote, *, timeout=30, phase="remote-command"):
        calls.append((pod, phase, timeout))
        if phase.startswith("slot-occupancy:"):
            return "0\n"
        return "KIT_BOOT_READY\n"

    monkeypatch.setattr(Q, "_run_ssh", fake_ssh)
    monkeypatch.setattr(
        Q,
        "live_snapshot",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("full-scene probe must not call all-Pod live_snapshot")
        ),
    )
    result = Q.cmd_full_scene_probe(
        queue,
        job_id="job0",
        pod="pod2",
        gpu=1,
        attempt_id="selected_only_a1",
        execute=True,
        confirm=Q.FULL_SCENE_PROBE_CONFIRM,
    )
    assert result["first_iteration_observed"] is True
    assert [pod for pod, _phase, _timeout in calls] == ["pod2", "pod2"]
    assert calls[0][1] == "slot-occupancy:pod2/gpu1"
    assert calls[1][1].startswith("full-scene-probe:job0:")


def test_live_slot_occupancy_binds_requested_pod_gpu_and_fails_closed(monkeypatch):
    queue = _queue()
    calls = []

    def fake_ssh(_queue, pod, remote, **_kwargs):
        calls.append((pod, remote))
        return "2\n" if pod == "pod1" else "1\n"

    monkeypatch.setattr(Q, "_run_ssh", fake_ssh)
    pod1_slot = next(slot for slot in Q.slots(queue) if slot.name == "pod1/gpu2")
    pod2_slot = next(slot for slot in Q.slots(queue) if slot.name == "pod2/gpu0")
    assert Q.live_slot_occupancy(queue, pod1_slot) == 2
    assert Q.live_slot_occupancy(queue, pod2_slot) == 1
    assert calls[0][0] == "pod1" and "nvidia-smi -i 2" in calls[0][1]
    assert calls[1][0] == "pod2" and "nvidia-smi -i 0" in calls[1][1]

    monkeypatch.setattr(Q, "_run_ssh", lambda *_args, **_kwargs: "UNKNOWN\n")
    try:
        Q.live_slot_occupancy(queue, pod2_slot)
    except Q.QueueError as exc:
        assert "invalid compute occupancy" in str(exc)
    else:
        raise AssertionError("unknown selected-slot occupancy did not fail closed")


def test_full_scene_probe_capacity_and_empty_dispatch_fail_before_launch(monkeypatch):
    queue = _queue()
    queue["dispatch_pods"] = ["pod2"]
    queue["jobs"][0]["resource"] = {"policy": "dispatch_gpu_round_robin"}
    phases = []

    def at_capacity(_queue, pod, _remote, *, phase="remote-command", **_kwargs):
        phases.append((pod, phase))
        return "3\n"

    monkeypatch.setattr(Q, "_run_ssh", at_capacity)
    try:
        Q.cmd_full_scene_probe(
            queue, job_id="job0", pod="pod2", gpu=1, attempt_id="full_a1",
            execute=True, confirm=Q.FULL_SCENE_PROBE_CONFIRM,
        )
    except Q.QueueError as exc:
        assert "slot is at capacity" in str(exc)
    else:
        raise AssertionError("full selected slot launched at capacity")
    assert phases == [("pod2", "slot-occupancy:pod2/gpu1")]

    queue["dispatch_pods"] = []
    phases.clear()
    try:
        Q.cmd_full_scene_probe(
            queue, job_id="job0", pod="pod2", gpu=1, attempt_id="none_a1",
            execute=False, confirm=None,
        )
    except Q.QueueError as exc:
        assert "not dispatch-enabled" in str(exc)
    else:
        raise AssertionError("empty dispatch set selected a probe slot")
    assert phases == []


def test_full_scene_probe_rejects_terminal_or_placeholder_and_avoids_cross_job_collision():
    queue = _queue(2)
    slot = Q.slots(queue)[0]
    _a, _argv_a, run_a = Q._full_scene_probe_contract(
        queue, queue["jobs"][0], slot, "same_attempt"
    )
    _b, _argv_b, run_b = Q._full_scene_probe_contract(
        queue, queue["jobs"][1], slot, "same_attempt"
    )
    assert run_a != run_b
    assert "/job0/" in run_a and "/job1/" in run_b

    queue["jobs"][0]["status"] = "rejected"
    queue["jobs"][0]["blocker"] = "retired"
    try:
        Q.cmd_full_scene_probe(
            queue, job_id="job0", pod="pod1", gpu=0, attempt_id="a1",
            execute=False, confirm=None,
        )
    except Q.QueueError as exc:
        assert "ready or blocked" in str(exc)
    else:
        raise AssertionError("terminal job was used for a full-scene probe")

    queue["jobs"][0]["status"] = "blocked"
    queue["jobs"][0]["source"]["commit"] = Q.ZERO_COMMIT
    try:
        Q.cmd_full_scene_probe(
            queue, job_id="job0", pod="pod1", gpu=0, attempt_id="a2",
            execute=False, confirm=None,
        )
    except Q.QueueError as exc:
        assert "all-zero placeholder" in str(exc)
    else:
        raise AssertionError("placeholder blocked row was used as an exact probe")


def test_legacy_source_capability_does_not_require_or_inject_p1_runtime():
    queue = _queue()
    job = queue["jobs"][0]
    job["runtime_binding"] = False
    slot = Q.slots(queue)[0]
    _claim, argv = Q._launch_contract(queue, job, slot)
    assert not any("training_queue_claim_path" in item for item in argv)
    assert not any("training_run_binding_path" in item for item in argv)
    assert Q.QUEUE_RUNTIME_RELATIVE not in Q._doctor_body(queue, job, slot)


def test_boot_warmup_requires_dedicated_confirmation_and_dispatch_slot():
    queue = _queue()
    try:
        Q.cmd_boot_warmup(
            queue,
            job_id="job0",
            pod="pod1",
            gpu=0,
            attempt_id="a1",
            execute=True,
            confirm=Q.CONFIRM,
        )
    except Q.QueueError as exc:
        assert Q.WARMUP_CONFIRM in str(exc)
    else:
        raise AssertionError("science confirmation token authorized a boot warmup")

    queue["dispatch_pods"] = ["pod2"]
    try:
        Q.cmd_boot_warmup(
            queue,
            job_id="job0",
            pod="pod1",
            gpu=0,
            attempt_id="a1",
            execute=False,
            confirm=None,
        )
    except Q.QueueError as exc:
        assert "not dispatch-enabled" in str(exc)
    else:
        raise AssertionError("reserved Pod received a boot warmup")


def test_fill_uses_one_atomic_launch_ssh_and_stops_on_embedded_preflight_failure(
    tmp_path, monkeypatch
):
    queue = _queue(2)
    monkeypatch.setattr(Q, "GLOBAL_SCHEDULER_LOCK", tmp_path / "scheduler.lock")
    occupancy = {slot.name: 0 for slot in Q.slots(queue)}
    claims = {}
    calls = []

    def fake_snapshot(_queue):
        return dict(occupancy), dict(claims)

    def fake_ssh(_queue, pod, remote, **kwargs):
        calls.append((pod, kwargs["phase"]))
        rendered = shlex.split(remote)[-1]
        assert "find_spec" in rendered
        assert "--cfg job --resolve" in rendered
        assert rendered.index("--cfg job --resolve") < rendered.index("count=$(nvidia-smi")
        assert rendered.index("count=$(nvidia-smi") < rendered.index("( set -o noclobber")
        raise Q.QueueError("embedded preflight failed")

    monkeypatch.setattr(Q, "live_snapshot", fake_snapshot)
    monkeypatch.setattr(Q, "_run_ssh", fake_ssh)
    try:
        Q.cmd_fill(queue, execute=True, confirm=Q.CONFIRM, count=2)
    except Q.QueueError as exc:
        assert "embedded preflight failed" in str(exc)
    else:
        raise AssertionError("embedded launch preflight failure was ignored")
    assert calls == [("pod1", "launch-first-iteration:job0")]
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
    assert result["result_schema_version"] == 2
    assert [item["resource"] for item in result["launched"]] == [
        "pod1/gpu0", "pod1/gpu1"
    ]
    assert all(
        item["preflight_mode"] == "embedded_in_atomic_launch"
        and "doctor_output" not in item
        for item in result["launched"]
    )
    assert calls == [
        "snapshot",
        "launch-first-iteration:job0",
        "snapshot",
        "launch-first-iteration:job1",
    ]


def test_milestone_attestor_dry_run_follows_only_binding_and_does_not_ssh(monkeypatch):
    queue = _queue()
    calls = []
    monkeypatch.setattr(Q, "_run_ssh", lambda *args, **kwargs: calls.append(args))
    result = Q.cmd_attest_milestone(
        queue,
        job_id="job0",
        milestone=200,
        execute=False,
        confirm=None,
    )
    assert result["dry_run"] is True
    assert result["binding_path"] == "/workspace/runs/0/run_binding.json"
    assert result["receipt_path"] == "/workspace/runs/0/milestones/model_200.json"
    assert "lean_queue_runtime.py attest" in result["remote_script"]
    assert "model_200.pt" not in result["remote_script"]
    assert calls == []


def test_milestone_attestor_execute_requires_its_own_confirmation_before_ssh(monkeypatch):
    queue = _queue()
    calls = []
    monkeypatch.setattr(Q, "live_snapshot", lambda *_args: calls.append("live"))
    try:
        Q.cmd_attest_milestone(
            queue,
            job_id="job0",
            milestone=200,
            execute=True,
            confirm=None,
        )
    except Q.QueueError as exc:
        assert Q.ATTEST_CONFIRM in str(exc)
    else:
        raise AssertionError("execute without attestor confirmation did not fail")
    assert calls == []


def test_milestone_attestor_rejects_mutable_source_drift_before_ssh(monkeypatch):
    queue = _queue()
    job = queue["jobs"][0]
    slot = Q.slots(queue)[0]
    immutable_claim, _argv = Q._launch_contract(queue, job, slot)
    claim_state = {
        "pod": slot.pod,
        "gpu": slot.gpu,
        "state": "launched",
        "claim_schema_version": 2,
        "claim_content_sha256": immutable_claim["content_sha256"],
        "claim_path": f"{job['run_dir']}/queue_claim.json",
    }
    job["source"] = {"checkout": "/workspace/attacker", "commit": "b" * 40}
    calls = []
    occupancy = {item.name: 0 for item in Q.slots(queue)}
    monkeypatch.setattr(
        Q, "live_snapshot", lambda *_args: (occupancy, {job["id"]: claim_state})
    )
    monkeypatch.setattr(Q, "_run_ssh", lambda *_args, **_kwargs: calls.append("ssh"))
    try:
        Q.cmd_attest_milestone(
            queue,
            job_id=job["id"],
            milestone=200,
            execute=True,
            confirm=Q.ATTEST_CONFIRM,
        )
    except Q.QueueError as exc:
        assert "immutable launch claim" in str(exc)
    else:
        raise AssertionError("mutable source selected a verifier for an old claim")
    assert calls == []


def test_milestone_attestor_rejects_legacy_capability_before_live_snapshot(monkeypatch):
    queue = _queue()
    queue["jobs"][0]["runtime_binding"] = False
    calls = []
    monkeypatch.setattr(Q, "live_snapshot", lambda *_args: calls.append("live"))
    try:
        Q.cmd_attest_milestone(
            queue,
            job_id="job0",
            milestone=200,
            execute=True,
            confirm=Q.ATTEST_CONFIRM,
        )
    except Q.QueueError as exc:
        assert "runtime_binding=true" in str(exc)
    else:
        raise AssertionError("legacy job was allowed to infer a missing binding")
    assert calls == []


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
    assert len(ready) == 7
    assert len(rejected) == 8
    assert [job["id"] for job in blocked] == [
        "fresh_c_conditional_face_w04",
        "fresh_c_conditional_face_matched_control_p1r1",
        "fresh_c_conditional_face_w04_p1r1",
    ]
    assert queue["dispatch_pods"] == ["pod2"]
    conditional = {
        job["id"]: job for job in queue["jobs"]
        if job["id"].startswith("fresh_c_conditional_face_")
    }
    assert {
        job_id: job["status"] for job_id, job in conditional.items()
    } == {
        "fresh_c_conditional_face_matched_control": "rejected",
        "fresh_c_conditional_face_w04": "blocked",
        "fresh_c_conditional_face_matched_control_p1r1": "blocked",
        "fresh_c_conditional_face_w04_p1r1": "blocked",
    }
    assert all(
        job["resource"]["preferred_slot"] == "pod2/gpu1"
        for job in conditional.values()
    )
    p1_pair = [
        conditional["fresh_c_conditional_face_matched_control_p1r1"],
        conditional["fresh_c_conditional_face_w04_p1r1"],
    ]
    assert all(job["runtime_binding"] is True for job in p1_pair)
    assert all(job["source"] == {
        "checkout": "/workspace/codexschema/nohope_p1_077e70c",
        "commit": "077e70cfd89cfe21cdc24dc928e62b3fc2a8820f",
    } for job in p1_pair)
    def delta_map(job):
        return {
            item.split("=", 1)[0].lstrip("+"): item.split("=", 1)[1]
            for item in job["recipe"]["delta"]
        }

    control_delta = delta_map(p1_pair[0])
    treatment_delta = delta_map(p1_pair[1])
    assert {
        key: (control_delta[key], treatment_delta[key])
        for key in control_delta if control_delta[key] != treatment_delta[key]
    } == {
        "task.rewards.racket_face_conditional_guidance_weight": ("0.0", "-0.4")
    }
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
        "fresh_c_qdot_limit_hinge_matched_control_retry_v2": ("false", "1.0", "0.0", "0.25"),
        "fresh_c_conditional_face_matched_control": ("false", "1.0", "0.0", "0.25"),
        "fresh_c_conditional_face_w04": ("false", "1.0", "0.0", "0.25"),
    }
    for job in ready:
        assert job["seed"] == 3
        assert job["budget"] == {
            "num_envs": 4096, "max_iterations": 1001, "save_interval": 100
        }
        assert job["milestones"] == [200, 500, 1000]
        if job["id"].startswith("fresh_c_conditional_face_"):
            expected_source = "61007e93879f35677e4c7d38cf7f681f324f9571"
        elif job["id"].startswith("fresh_c_qdot_limit_hinge_"):
            expected_source = "a6ccdc7a1c696ff37878039f1e1d83dea28a2bfa"
        else:
            expected_source = "4467d79f1ed425a4263f0caaad2f661e1ec737ad"
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
        elif job["id"] in {
            "fresh_c_qdot_limit_hinge_matched_control_retry_v2",
            "fresh_c_conditional_face_matched_control",
            "fresh_c_conditional_face_w04",
        }:
            assert qdot_weight == ["task.rewards.joint_velocity_limit_hinge_weight=0.0"]
            assert qdot_margin == ["task.rewards.joint_velocity_limit_hinge_margin=0.85"]
        else:
            assert qdot_weight == []
            assert qdot_margin == []
        conditional_weight = [
            item for item in delta
            if item.startswith("++task.rewards.racket_face_conditional_guidance_weight=")
        ]
        if job["id"] == "fresh_c_conditional_face_matched_control":
            assert conditional_weight == [
                "++task.rewards.racket_face_conditional_guidance_weight=0.0"
            ]
        elif job["id"] == "fresh_c_conditional_face_w04":
            assert conditional_weight == [
                "++task.rewards.racket_face_conditional_guidance_weight=-0.4"
            ]
        else:
            assert conditional_weight == []
    plan = Q.cmd_plan(queue, live=False)
    assert len(plan["assignments"]) == 7
    assert [item["resource"] for item in plan["assignments"]] == [
        "pod2/gpu0", "pod2/gpu1", "pod2/gpu2",
        "pod2/gpu0", "pod2/gpu1", "pod2/gpu2",
        "pod2/gpu0",
    ]
    retry_ready = [job for job in ready if job["id"].endswith("_retry_v2")]
    assert len(retry_ready) == 7
    retry_by_id = {job["id"]: job for job in retry_ready}
    retryable_rejected = [
        job for job in rejected if f"{job['id']}_retry_v2" in retry_by_id
    ]
    assert len(retryable_rejected) == 7
    for attempt1 in retryable_rejected:
        retry = retry_by_id[f"{attempt1['id']}_retry_v2"]
        assert retry["recipe"] == attempt1["recipe"]
        assert retry["motion"] == attempt1["motion"]
        assert retry["bank"] == attempt1["bank"]
        assert retry["exam"] == attempt1["exam"]
        assert retry["run_dir"] != attempt1["run_dir"]
