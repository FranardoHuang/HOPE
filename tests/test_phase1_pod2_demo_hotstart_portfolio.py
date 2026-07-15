from __future__ import annotations

import base64
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_phase1_demo_hotstart_queue.py"
QUEUE = ROOT / "configs" / "phase1_pod2_demo_hotstart_portfolio_20260716.yaml"


def _module():
    spec = importlib.util.spec_from_file_location("demo_hotstart_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


D = _module()


def _raw() -> dict:
    return yaml.safe_load(QUEUE.read_text(encoding="utf-8"))


def _write(tmp_path: Path, value: dict) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "queue.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def _values(job: dict) -> dict[str, str]:
    return {
        D.Q._override_key(raw, job["id"]): raw.partition("=")[2]
        for raw in [*job["recipe"]["base"], *job["recipe"]["delta"]]
    }


def _activated(tmp_path: Path) -> tuple[dict, Path]:
    raw = copy.deepcopy(_raw())
    raw["launch_authorized"] = True
    raw["preregistration_status"] = "activated_demo_only_inexact"
    raw["activation_contract"]["state"] = "activated"
    raw["activation_contract"]["receipt_file_sha256"] = "a" * 64
    for parent in raw["parents"].values():
        parent["checkpoint_sha256"] = "b" * 64
        parent["hard_contract_sha256"] = "c" * 64
        parent["training_launch_claim_sha256"] = "d" * 64
        parent["queue_claim_content_sha256"] = "d" * 64
        parent["queue_claim_file_sha256"] = "e" * 64
        parent["run_binding_file_sha256"] = "f" * 64
        parent["run_binding_content_sha256"] = "1" * 64
    for job in raw["jobs"]:
        job["status"] = "ready"
        job["blocker"] = None
    path = _write(tmp_path, raw)
    return D.load_queue(path), path


def test_pending_queue_is_six_blocked_pod2_rows_and_plan_is_nonlaunching():
    queue = D.load_queue(QUEUE)
    plan = D.cmd_plan(queue)
    assert queue["dispatch_pods"] == ["pod2"]
    assert queue["launch_authorized"] is False
    assert len(queue["jobs"]) == 6
    assert queue["pods"]["pod2"]["max_trainers_per_gpu"] == 4
    assert "parent_snapshot_receipt_v2.json" in queue["activation_contract"]["receipt_path"]
    assert plan["assignments"] == []
    assert len(plan["blocked"]) == 6
    assert all(job["status"] == "blocked" for job in queue["jobs"])
    assert all(job["runtime_binding"] is True for job in queue["jobs"])
    assert all(job["source"]["commit"] == D.EXPECTED_SOURCE for job in queue["jobs"])


def test_six_recipes_match_the_frozen_demo_portfolio():
    queue = D.load_queue(QUEUE)
    jobs = {job["id"]: job for job in queue["jobs"]}
    actual = {}
    for job_id, job in jobs.items():
        value = _values(job)
        actual[job_id] = (
            job["warm_start"]["parent"],
            value["task.rewards.free_wrist_vel_mimic"],
            value["task.rewards.motion_scale_in_window"],
            value["task.rewards.joint_velocity_limit_hinge_weight"],
            value["task.rewards.racket_face_conditional_guidance_weight"],
            value["task.rewards.foot_orientation_weight"],
            value["task.rewards.free_non_striking_arm_mimic"],
        )
        assert value["task.env.episode_length_s"] == "10.0"
        assert (
            value["task.rewards.racket_position_weight"],
            value["task.rewards.racket_velocity_weight"],
            value["task.rewards.racket_normal_weight"],
        ) == ("14.0", "10.0", "5.0")
        assert job["budget"] == {
            "num_envs": 4096, "max_iterations": 5001, "save_interval": 100
        }
        assert job["milestones"] == [3700, 4000, 4500, 5500, 7500]
    assert actual == {
        "demo_qdot_v1v2_face_w0p4": ("qdot", "true", "0.25", "-5.0", "-0.4", "-0.3", "false"),
        "demo_qdot_v1v2_face_w0p2": ("qdot", "true", "0.25", "-5.0", "-0.2", "-0.3", "false"),
        "demo_v1v2_qdot_w5_face_w0p4": ("v1v2", "true", "0.25", "-5.0", "-0.4", "-0.3", "false"),
        "demo_v1v2_qdot_w2p5_face_w0p4_free_arm": ("v1v2", "true", "0.25", "-2.5", "-0.4", "-0.3", "true"),
        "demo_control_qdot_w5_face_w0p4": ("control", "false", "1.0", "-5.0", "-0.4", "-0.3", "false"),
        "demo_control_full_stack_free_arm_foot_w0p6": ("control", "true", "0.25", "-5.0", "-0.4", "-0.6", "true"),
    }


def test_activated_queue_round_robins_and_claim_binds_inexact_parent_receipt(tmp_path):
    queue, _path = _activated(tmp_path)
    plan = D.cmd_plan(queue)
    assert [row["resource"] for row in plan["assignments"]] == D.EXPECTED_SLOTS
    first = queue["jobs"][0]
    slot = next(slot for slot in D.Q.slots(queue) if slot.name == "pod2/gpu0")
    claim, argv = D._demo_claim(queue, first, slot)
    content = claim["content"]
    assert content["formal_exact_eligible"] is False
    assert content["demo_warm_start"]["checkpoint_sha256"] == "b" * 64
    assert content["demo_warm_start"]["hard_contract_sha256"] == "c" * 64
    assert content["demo_warm_start"]["training_launch_claim_sha256"] == "d" * 64
    assert content["demo_warm_start"]["queue_claim_content_sha256"] == "d" * 64
    assert content["demo_warm_start"]["run_binding_content_sha256"] == "1" * 64
    assert content["activation_receipt"]["file_sha256"] == "a" * 64
    assert "checkpoint_tolerant=false" in argv
    assert "checkpoint_allow_missing_contract=false" in argv
    assert "checkpoint_allow_contract_mismatch=true" in argv
    assert any(
        item.startswith("checkpoint_path=") and "parent_snapshots_v2" in item
        for item in argv
    )
    assert content["source_contract_files"] == D.EXPECTED_SOURCE_CONTRACT_FILES
    assert argv[-1] == f"++training_launch_claim_sha256={claim['content_sha256']}"


def test_fourth_slot_activation_only_selects_jobs_one_and_two(tmp_path):
    queue, _path = _activated(tmp_path)
    occupancy = {"pod2/gpu0": 3, "pod2/gpu1": 3, "pod2/gpu2": 0}
    assignments = D.Q._assign(queue, occupancy)
    assert [(job["id"], slot.name) for job, slot in assignments] == [
        ("demo_qdot_v1v2_face_w0p4", "pod2/gpu0"),
        ("demo_qdot_v1v2_face_w0p2", "pod2/gpu1"),
    ]


def test_generic_fresh_queue_guard_remains_fresh_only(tmp_path):
    raw = _raw()
    raw["pods"]["pod2"]["max_trainers_per_gpu"] = 3
    with pytest.raises(D.Q.QueueError, match="supports fresh runs only"):
        D.Q.load_queue(_write(tmp_path, raw))


def test_activation_and_exactness_flags_fail_closed(tmp_path):
    raw = _raw()
    raw["jobs"][0]["warm_start"]["descendant_exact_eligible"] = True
    with pytest.raises(D.DemoQueueError, match="exact-ineligible"):
        D.load_queue(_write(tmp_path, raw))

    raw = _raw()
    raw["launch_authorized"] = True
    with pytest.raises(D.DemoQueueError, match="exactly follow activation"):
        D.load_queue(_write(tmp_path, raw))


def test_parent_attestation_is_separate_one_pod_no_retry_dry_run():
    queue = D.load_queue(QUEUE)
    result = D.cmd_parent_attest(queue, execute=False, confirm=None)
    assert result["automatic_activation"] is False
    assert result["automatic_retry"] is False
    command = " ".join(result["ssh_argv"])
    preflight = " ".join(result["preflight_ssh_argv"])
    assert "162.43.172.181" in command
    assert "162.43.172.181" in preflight
    assert "162.43.172.171" not in command
    assert "pkill" not in command
    assert "killall" not in command


def test_parent_inspect_is_explicitly_read_only_and_pod2_only():
    queue = D.load_queue(QUEUE)
    result = D.cmd_parent_inspect(queue, execute=False, confirm=None)
    assert result["read_only"] is True
    assert result["creates_snapshots"] is False
    assert result["creates_receipt"] is False
    command = " ".join(result["ssh_argv"])
    assert "162.43.172.181" in command
    assert "162.43.172.171" not in command
    spec = D._parent_spec(queue, mode="inspect")
    assert spec["mode"] == "inspect"
    assert spec["receipt_path"].endswith("parent_snapshot_receipt_v2.json")


def test_parent_spec_binds_exact_original_claim_binding_and_descendant_hard_changes():
    queue = D.load_queue(QUEUE)
    spec = D._parent_spec(queue, mode="attest")
    assert list(spec["parents"]) == ["qdot", "v1v2", "control"]
    assert spec["parents"]["qdot"]["original_job_id"] == "p1_long_no_replay_qdot_w5_seed3"
    for name, parent in spec["parents"].items():
        assert parent["original_pod"] == "pod2"
        assert parent["original_gpu"] == 2
        assert parent["live_queue_claim_path"].endswith("queue_claim.json")
        assert parent["live_run_binding_path"].endswith("run_binding.json")
        assert "parent_snapshots_v2" in parent["snapshot_checkpoint_path"]
        assert len(parent["descendant_contract_values"]) == 2


def test_parent_program_is_fd_snapshot_based_and_strict_about_full_state():
    program = D.PARENT_PROGRAM
    for required in (
        "io.BytesIO(raw)", "O_NOFOLLOW", "O_EXCL", "snapshot_queue_claim_path",
        "snapshot_run_binding_path", "optimizer.get(\"state\")",
        "optimizer.get(\"param_groups\")", "key.startswith(\"actor.\")",
        "key.startswith(\"critic.\")", "queue claim canonical SHA mismatch",
        "run binding canonical SHA mismatch", "mode not in {\"inspect\", \"attest\", \"verify\"}",
    ):
        assert required in program
    assert "torch.load(checkpoint_path" not in program
    assert "parent_model3500_finite_receipt.json" not in D.EXPECTED_RECEIPT_PATH


def test_launch_requires_source_hashes_strict_resume_and_exact_failure_identity(tmp_path):
    queue, _path = _activated(tmp_path)
    job = queue["jobs"][0]
    slot = next(slot for slot in D.Q.slots(queue) if slot.name == "pod2/gpu0")
    script = D._launch_script(queue, job, slot)
    assert D.EXPECTED_SOURCE_CONTRACT_FILES[
        "hope_training/whole_body_tracking/scripts/train.py"
    ] in script
    assert "strict_full_state_resume_proven=true" in script
    assert "failure_path" in D.FIRST_ITER_PROGRAM
    assert "manual_exact_pgid_disposition_required" in D.FIRST_ITER_PROGRAM
    assert "pkill" not in script
    assert "killall" not in script
    assert "checkpoint_path=/workspace/codexschema/phase1_demo_hotstart_20260716/activation/parent_snapshots_v2/qdot/model_3500.pt" in script
    proof = D.FIRST_ITER_PROGRAM
    assert "explicit hard-contract mismatch override" in proof
    assert "continuing at iteration 3500, optimizer=resumed" in proof
    assert "joint_velocity_limit_hinge_reward" in proof
    assert "conditional_signed_face" in proof


def test_all_recipe_identity_and_contract_mutations_fail_closed(tmp_path):
    mutations = []
    for section in ("base", "delta"):
        mutations.extend([
            (f"{section}-change", lambda raw, s=section: raw["jobs"][0]["recipe"][s].__setitem__(0, raw["jobs"][0]["recipe"][s][0] + "_drift")),
            (f"{section}-missing", lambda raw, s=section: raw["jobs"][0]["recipe"][s].pop()),
            (f"{section}-extra", lambda raw, s=section: raw["jobs"][0]["recipe"][s].append("task.fake=1")),
        ])
    mutations.extend([
        ("id", lambda raw: raw["jobs"][0].__setitem__("id", "demo_changed")),
        ("parent", lambda raw: raw["jobs"][0]["warm_start"].__setitem__("parent", "control")),
        ("run-name", lambda raw: raw["jobs"][0].__setitem__("run_name", "changed")),
        ("run-dir", lambda raw: raw["jobs"][0].__setitem__("run_dir", "/workspace/changed")),
        ("slot", lambda raw: raw["jobs"][0]["resource"].__setitem__("required_slot", "pod2/gpu2")),
        ("host", lambda raw: raw["pods"]["pod2"].__setitem__("host", "127.0.0.1")),
        ("port", lambda raw: raw["pods"]["pod2"].__setitem__("port", 22)),
        ("capacity", lambda raw: raw["pods"]["pod2"].__setitem__("max_trainers_per_gpu", 3)),
        ("source", lambda raw: raw["jobs"][0]["source"].__setitem__("commit", "0" * 40)),
        ("source-hash", lambda raw: raw["source_contract_files"].__setitem__("hope_training/whole_body_tracking/scripts/train.py", "0" * 64)),
        ("parent-id", lambda raw: raw["parents"]["qdot"].__setitem__("original_job_id", "wrong")),
        ("parent-claim", lambda raw: raw["parents"]["qdot"].__setitem__("live_queue_claim_path", "/workspace/wrong/queue_claim.json")),
        ("snapshot", lambda raw: raw["parents"]["qdot"].__setitem__("snapshot_checkpoint_path", "/workspace/wrong/model_3500.pt")),
        ("receipt-v1", lambda raw: raw["activation_contract"].__setitem__("receipt_path", "/workspace/codexschema/phase1_demo_hotstart_20260716/activation/parent_model3500_finite_receipt.json")),
        ("release-rule", lambda raw: raw["activation_contract"].__setitem__("gpu_release_rule", "drift")),
    ])
    for index, (label, mutate) in enumerate(mutations):
        raw = _raw()
        mutate(raw)
        with pytest.raises(D.DemoQueueError):
            D.load_queue(_write(tmp_path / f"case_{index}_{label}", raw))


def test_activated_parent_sha_closure_rejects_missing_or_split_claim(tmp_path):
    raw = _raw()
    raw["launch_authorized"] = True
    raw["preregistration_status"] = "activated_demo_only_inexact"
    raw["activation_contract"]["state"] = "activated"
    raw["activation_contract"]["receipt_file_sha256"] = "a" * 64
    for job in raw["jobs"]:
        job["status"] = "ready"
        job["blocker"] = None
    for parent in raw["parents"].values():
        for key in (
            "checkpoint_sha256", "hard_contract_sha256", "queue_claim_file_sha256",
            "queue_claim_content_sha256", "run_binding_file_sha256",
            "run_binding_content_sha256", "training_launch_claim_sha256",
        ):
            parent[key] = "b" * 64
    raw["parents"]["qdot"]["training_launch_claim_sha256"] = "c" * 64
    with pytest.raises(D.DemoQueueError, match="launch claim differs"):
        D.load_queue(_write(tmp_path, raw))


def _embedded_definitions(program: str) -> dict:
    prefix = program.split("\n\ntry:\n    main()", 1)[0]
    namespace: dict = {}
    exec(compile(prefix, "<embedded-test>", "exec"), namespace)
    return namespace


def test_snapshot_changed_after_verify_is_rejected_by_in_lock_recheck(tmp_path):
    root = tmp_path / "snapshots"
    root.mkdir()
    files = []
    for index, label in enumerate(("checkpoint", "hard", "claim", "binding")):
        path = root / f"{label}.bin"
        payload = f"original-{index}".encode()
        path.write_bytes(payload)
        os.chmod(path, 0o444)
        files.append({
            "label": label, "path": str(path),
            "sha256": hashlib.sha256(payload).hexdigest(),
        })
    os.chmod(root / "checkpoint.bin", 0o644)
    (root / "checkpoint.bin").write_bytes(b"tampered-after-verify")
    os.chmod(root / "checkpoint.bin", 0o444)
    allowed_prefix = str(tmp_path) + "/"
    program = D.SNAPSHOT_RECHECK_PROGRAM.replace('"/workspace/"', repr(allowed_prefix))
    encoded = base64.b64encode(json.dumps({
        "job_id": "attack", "files": files,
    }).encode()).decode()
    completed = subprocess.run(
        [sys.executable, "-c", program, encoded], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    assert completed.returncode == 2
    assert "SHA differs from activated queue" in completed.stderr


def test_resume_without_first_post_resume_learning_step_is_rejected():
    namespace = _embedded_definitions(D.FIRST_ITER_PROGRAM)
    first = namespace["first_post_resume_iteration"]
    resume = (
        "[train.py] WARNING: explicit hard-contract mismatch override:\n"
        "[train.py] RESUMED from checkpoint: snapshot/model_3500.pt "
        "(continuing at iteration 3500, optimizer=resumed)\n"
    )
    assert first(resume) is None
    assert first(resume + "Learning iteration 3500/8501\n") is None
    assert first(resume + "Learning iteration 3501/8501\n") == 3501


def test_stale_reused_or_exited_bound_pid_is_rejected(tmp_path):
    namespace = _embedded_definitions(D.FIRST_ITER_PROGRAM)
    identity = namespace["proc_identity"]
    error = namespace["ProofError"]
    proc = tmp_path / "proc" / "123"
    proc.mkdir(parents=True)
    fields = ["S", *(["0"] * 18), "456"]
    (proc / "stat").write_text("123 (trainer) " + " ".join(fields), encoding="utf-8")
    (proc / "cmdline").write_bytes(b"python\0train.py\0")
    expected = {
        "pid": 123, "pgid": 123, "starttime_ticks": 456,
        "argv": ["python", "train.py"],
    }
    assert identity(expected, proc_root=tmp_path / "proc", getpgid=lambda _pid: 123) == {
        "pid": 123, "pgid": 123, "starttime_ticks": 456,
    }
    stale = {**expected, "starttime_ticks": 455}
    with pytest.raises(error, match="drifted or was reused"):
        identity(stale, proc_root=tmp_path / "proc", getpgid=lambda _pid: 123)
    with pytest.raises(error, match="drifted or was reused"):
        identity(expected, proc_root=tmp_path / "proc", getpgid=lambda _pid: 999)
    (proc / "stat").unlink()
    with pytest.raises(error, match="exited before"):
        identity(expected, proc_root=tmp_path / "proc", getpgid=lambda _pid: 123)
