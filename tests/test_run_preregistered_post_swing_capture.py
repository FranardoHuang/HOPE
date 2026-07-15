"""Dependency-light red-team tests for the post-swing one-shot controller."""

from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import types

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_preregistered_post_swing_capture.py"
SPEC = importlib.util.spec_from_file_location("post_swing_capture_runner", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)
BUILDER_SCRIPT = ROOT / "scripts/build_post_swing_capture_plan_v2.py"
BUILDER_SPEC = importlib.util.spec_from_file_location("post_swing_capture_builder", BUILDER_SCRIPT)
assert BUILDER_SPEC is not None and BUILDER_SPEC.loader is not None
BUILDER = importlib.util.module_from_spec(BUILDER_SPEC)
BUILDER_SPEC.loader.exec_module(BUILDER)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _file_row(path: Path, *, absolute=True) -> dict:
    raw = path.read_bytes()
    return {
        "path": str(path) if absolute else path.name,
        "bytes": len(raw),
        "sha256": _sha(raw),
    }


def _document(path: Path, content: dict, *, schema=1) -> tuple[dict, dict]:
    document = {
        "schema_version": schema,
        "content": content,
        "content_sha256": _sha(_canonical(content)),
    }
    raw = _canonical(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return document, {
        "path": str(path),
        "file_sha256": _sha(raw),
        "content_sha256": document["content_sha256"],
    }


def _patch_artifact_paths(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    monkeypatch.setattr(RUNNER, "ARTIFACT_ROOT", root)
    monkeypatch.setattr(RUNNER, "CAPTURE_PARENT", root / "capture")
    monkeypatch.setattr(RUNNER, "LAUNCH_PARENT", root / "launch")
    monkeypatch.setattr(RUNNER, "GPU_LEASE_PATH", tmp_path / "hope_lean_queue_gpu2.lock")
    monkeypatch.setattr(RUNNER, "ISAAC_PYTHON", tmp_path / "venv/bin/python")


def _plan(tmp_path: Path, monkeypatch=None) -> dict:
    if monkeypatch is not None:
        _patch_artifact_paths(monkeypatch, tmp_path)
    source = tmp_path / "source"
    source.mkdir(parents=True, exist_ok=True)
    (source / "hope_training/whole_body_tracking").mkdir(parents=True, exist_ok=True)
    plan_id = "case-v2"
    requested_python = RUNNER.ISAAC_PYTHON
    train = source / "hope_training/whole_body_tracking/scripts/train.py"
    return {
        "schema_version": 2,
        "plan_id": plan_id,
        "status": "preregistered_capture_not_started",
        "simulation_only": True,
        "capture_source": {
            "checkout": str(source),
            "commit": "c" * 40,
            "clean_required": True,
            "files": {
                "controller": {"path": "scripts/controller.py", "bytes": 1, "sha256": "1" * 64},
                "inference_runner": {
                    "path": "hope_training/whole_body_tracking/scripts/play.py",
                    "bytes": 1,
                    "sha256": "2" * 64,
                },
                "lean_queue_runtime": {
                    "path": "hope_training/whole_body_tracking/scripts/lean_queue_runtime.py",
                    "bytes": 1,
                    "sha256": "3" * 64,
                },
                "producer": {
                    "path": "hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py",
                    "bytes": 1,
                    "sha256": "e" * 64,
                },
            },
            "ignored_runtime_asset": {
                "relative_path": "asset",
                "file_count": 1,
                "total_file_bytes": 1,
                "tree_content_sha256": "4" * 64,
                "symlinks_forbidden": True,
            },
            "full_tree": {
                "file_count": 1,
                "total_file_bytes": 1,
                "tree_content_sha256": "5" * 64,
            },
        },
        "teacher_checkpoint": {
            "path": str(tmp_path / "teacher/model_500.pt"),
            "sha256": "6" * 64,
            "embedded_iteration": 500,
            "floating_elements": 10,
            "nonfinite_floating_elements": 0,
            "fresh_lineage": 1,
            "training_source_commit": "d" * 40,
            "hard_contract": {
                "path": str(tmp_path / "teacher/params/training_contract.json"),
                "sha256": "7" * 64,
                "schema_version": 3,
            },
            "launch_claim": {"path": str(tmp_path / "claim.json"), "file_sha256": "8" * 64, "content_sha256": "9" * 64},
            "run_binding": {"path": str(tmp_path / "binding.json"), "file_sha256": "a" * 64, "content_sha256": "b" * 64},
            "milestone_receipt": {"path": str(tmp_path / "receipt.json"), "file_sha256": "c" * 64, "content_sha256": "d" * 64},
        },
        "ordered_motion_inputs": [
            {"path": str(tmp_path / "motions/f.npz"), "sha256": "e" * 64},
            {"path": str(tmp_path / "motions/b.npz"), "sha256": "f" * 64},
        ],
        "question_bank": {"path": str(tmp_path / "bank/train.npz"), "sha256": "0" * 64},
        "capture_contract": {
            "pod": "pod2",
            "gpu": 2,
            "gpu_uuid": "GPU-exact-1",
            "cuda_visible_devices": "GPU-exact-1",
            "runtime_device": "cuda:0",
            "num_envs": 4096,
            "target_count": 4096,
            "max_inference_steps": 20000,
            "seed": 3,
            "wrap_teleport": False,
            "post_swing_start_prob": 0.25,
            "root_linear_velocity_limit_mps": 2.0,
            "root_angular_velocity_limit_radps": 4.0,
            "namespace_id": plan_id,
            "output_directory": str(RUNNER.CAPTURE_PARENT / plan_id),
            "launch_root": str(RUNNER.LAUNCH_PARENT / plan_id),
            "output_must_be_absent_before_one_shot": True,
            "capture_is_inference_only": True,
            "ppo_updates": 0,
            "natural_wrap_only": True,
            "timeout_or_failure_reset_states_forbidden": True,
            "launch_handoff": "execve_same_pid_v1",
        },
        "runtime_environment": {
            "node": {
                "hostname": "pod2-test",
                "machine_id_path": "/etc/machine-id",
                "machine_id_sha256": "1" * 64,
                "boot_id_path": "/proc/sys/kernel/random/boot_id",
                "boot_id_sha256": "2" * 64,
            },
            "gpu": {"physical_index": 2, "uuid": "GPU-exact-1", "lease_path": str(RUNNER.GPU_LEASE_PATH)},
            "python": {
                "requested_path": str(requested_python),
                "resolved_path": "/usr/bin/python3.10",
                "symlink_chain": [
                    {"kind": "symlink", "path": str(requested_python), "target": "python3.10"},
                    {"kind": "symlink", "path": str(requested_python.parent / "python3.10"), "target": "/usr/bin/python3.10"},
                    {"kind": "regular", "path": "/usr/bin/python3.10", "bytes": 1, "sha256": "2" * 64},
                ],
                "pyvenv_cfg": {"path": str(requested_python.parents[1] / "pyvenv.cfg"), "bytes": 1, "sha256": "3" * 64},
            },
            "runtime_trees": [
                {
                    "label": label,
                    "path": str(tmp_path / f"tree-{label}"),
                    "on_pythonpath": True,
                    "file_count": 1,
                    "total_file_bytes": 1,
                    "tree_content_sha256": "5" * 64,
                }
                for label in sorted(RUNNER.RUNTIME_TREE_LABELS)
            ],
            "tools": {
                "git": {"path": "/usr/bin/git", "bytes": 1, "sha256": "6" * 64},
                "nvidia_smi": {"path": "/usr/bin/nvidia-smi", "bytes": 1, "sha256": "7" * 64},
            },
            "environment": {"exact": {"PATH": "/usr/bin:/bin", "HOME": "/root"}},
            "compose_timeout_s": 30,
        },
        "runtime_recipe_derivation": {
            "source": "teacher_checkpoint.run_binding exact training_argv",
            "keep_all_task_motion_bank_seed_num_env_overrides": True,
            "deduplicate_identical_hydra_keys": True,
            "replace_executable_train_with_play": True,
            "remove_keys": sorted(RUNNER.EXPECTED_REMOVE_KEYS),
            "add_keys": sorted(RUNNER.EXPECTED_ADD_KEYS),
            "runtime_hard_contract_must_equal_teacher_checkpoint_hard_contract_before_first_state": True,
            "seed_must_be_applied_by_play": True,
        },
        "authorization": {
            "capture_authorized": True,
            "attestation_authorized_only_after_complete_capture": True,
            "first_reset_probe_authorized": False,
            "scientific_training_authorized": False,
            "second_seed_authorized": False,
            "judge_authorized": False,
            "promotion_authorized": False,
            "hardware_authorized": False,
        },
        "failure_policy": {
            "preserve_partial_namespace": True,
            "same_namespace_retry_forbidden": True,
            "automatic_retry_forbidden": True,
            "exact_numeric_process_group_only": True,
            "pod1_and_pod2_gpu0_forbidden": True,
        },
    }


def _binding(plan: dict, extra=None) -> dict:
    args = [
        plan["runtime_environment"]["python"]["requested_path"],
        "/training/source/hope_training/whole_body_tracking/scripts/train.py",
        "task=HOPEPingPongVirtualBall", "algo=ppo", "headless=true", "device=cuda:0",
        "num_envs=4096", "seed=3", "task.motion.wrap_teleport=false",
        "task.motion.post_swing_start_prob=0.25",
        f"motion_file={plan['ordered_motion_inputs'][0]['path']}",
        f"motion_file_2={plan['ordered_motion_inputs'][1]['path']}",
        f"++task.racket.question_bank={plan['question_bank']['path']}",
        "logger=tensorboard", "video=false", "checkpoint_path=null",
        "checkpoint_tolerant=false", "checkpoint_allow_missing_contract=false",
        "checkpoint_allow_contract_mismatch=false", "max_iterations=1001",
        "algo.runner.save_interval=100", "run_name=old",
        "++training_queue_claim_path=/claim", "++training_run_binding_path=/binding",
        "++training_launch_claim_sha256=" + "a" * 64,
    ]
    args.extend(extra or [])
    return {
        "content": {
            "training_argv": args,
            "source": {"checkout": "/training/source", "commit": plan["teacher_checkpoint"]["training_source_commit"]},
        }
    }


def test_derivation_removes_train_only_keys_and_preserves_seed(tmp_path, monkeypatch):
    plan = _plan(tmp_path, monkeypatch)
    argv = RUNNER._derive_argv(plan, _binding(plan))
    normalized = {RUNNER._normal_key(value): value.split("=", 1)[1] for value in argv[2:]}
    assert not RUNNER.EXPECTED_REMOVE_KEYS.intersection(normalized)
    assert normalized["seed"] == "3"
    assert normalized["checkpoint"] == plan["teacher_checkpoint"]["path"]
    assert argv[0] == str(RUNNER.ISAAC_PYTHON)
    assert argv[1].endswith("hope_training/whole_body_tracking/scripts/play.py")


def test_derivation_rejects_conflicting_duplicate_and_arbitrary_executable(tmp_path, monkeypatch):
    plan = _plan(tmp_path, monkeypatch)
    with pytest.raises(RUNNER.CaptureContractError, match="conflicting duplicate"):
        RUNNER._derive_argv(plan, _binding(plan, ["seed=4"]))
    binding = _binding(plan)
    binding["content"]["training_argv"][0] = "/usr/bin/ssh"
    with pytest.raises(RUNNER.CaptureContractError, match="exact venv Python"):
        RUNNER._derive_argv(plan, binding)


def test_plan_rejects_namespace_escape_uint32_and_unsafe_environment(tmp_path, monkeypatch):
    plan = _plan(tmp_path, monkeypatch)
    plan["plan_id"] = "../escape"
    with pytest.raises(RUNNER.CaptureContractError, match="filesystem-safe"):
        RUNNER._validate_plan(plan)
    plan = _plan(tmp_path, monkeypatch)
    plan["capture_contract"]["seed"] = 2**32
    with pytest.raises(RUNNER.CaptureContractError, match="uint32"):
        RUNNER._validate_plan(plan)
    plan = _plan(tmp_path, monkeypatch)
    plan["runtime_environment"]["environment"]["exact"]["PYTHONPATH"] = "/evil"
    with pytest.raises(RUNNER.CaptureContractError, match="allowlist"):
        RUNNER._validate_plan(plan)
    plan = _plan(tmp_path, monkeypatch)
    plan["runtime_environment"]["node"]["machine_id_path"] = str(tmp_path / "fake-machine-id")
    with pytest.raises(RUNNER.CaptureContractError, match="fixed /etc/machine-id"):
        RUNNER._validate_plan(plan)
    plan = _plan(tmp_path, monkeypatch)
    plan["authorization"]["promotion_authorized"] = True
    with pytest.raises(RUNNER.CaptureContractError, match="promotion_authorized"):
        RUNNER._validate_plan(plan)


def test_plan_reader_rejects_symlink_duplicate_keys_and_nonfinite(tmp_path, monkeypatch):
    _patch_artifact_paths(monkeypatch, tmp_path)
    target = tmp_path / "plan.json"
    target.write_text('{"schema_version":2,"schema_version":2}', encoding="utf-8")
    link = tmp_path / "plan-link.json"
    link.symlink_to(target)
    with pytest.raises(RUNNER.CaptureContractError, match="regular non-symlink"):
        RUNNER._load_plan(link, _sha(target.read_bytes()))
    with pytest.raises(RUNNER.CaptureContractError, match="duplicate JSON key"):
        RUNNER._load_plan(target, _sha(target.read_bytes()))
    target.write_text('{"x":NaN}', encoding="utf-8")
    with pytest.raises(RUNNER.CaptureContractError, match="non-finite"):
        RUNNER._load_plan(target, _sha(target.read_bytes()))


def test_inventory_covers_untracked_ignored_bytes_and_rejects_symlink(tmp_path):
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "tracked.py").write_bytes(b"tracked")
    first = RUNNER._inventory(tree)
    (tree / "ignored.py").write_bytes(b"ignored")
    second = RUNNER._inventory(tree)
    assert second["file_count"] == first["file_count"] + 1
    assert second["tree_content_sha256"] != first["tree_content_sha256"]
    (tree / "link").symlink_to(tree / "tracked.py")
    with pytest.raises(RUNNER.CaptureContractError, match="invalid file entry"):
        RUNNER._inventory(tree)


def test_external_tools_use_exact_executable_bytes_and_ignore_parent_path(tmp_path, monkeypatch):
    plan = _plan(tmp_path, monkeypatch)
    nvidia = tmp_path / "bin/nvidia-smi"
    git = tmp_path / "bin/git"
    nvidia.parent.mkdir(parents=True)
    for path in (nvidia, git):
        path.write_bytes(b"#!/bin/sh\nexit 99\n")
        path.chmod(0o755)
    plan["runtime_environment"]["tools"] = {
        "git": _file_row(git),
        "nvidia_smi": _file_row(nvidia),
    }
    calls = []

    def output(command, **kwargs):
        calls.append((command, kwargs))
        assert kwargs["env"] == {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"}
        if "--query-gpu=index,uuid" in command:
            return "2, GPU-exact-1\n"
        if "--query-compute-apps=gpu_uuid,pid,process_name" in command:
            return ""
        return "c" * 40

    monkeypatch.setenv("PATH", str(tmp_path / "evil"))
    monkeypatch.setattr(RUNNER.subprocess, "check_output", output)
    state = RUNNER._gpu_state(plan)
    assert state["gpus"] == [{"index": 2, "uuid": "GPU-exact-1"}]
    assert all(call[0][0] == str(nvidia) for call in calls)
    assert RUNNER._git_output(git, tmp_path, "rev-parse", "HEAD") == "c" * 40
    assert calls[-1][0][0] == str(git)


def test_no_follow_context_does_not_relabel_leaf_exclusive_collision(tmp_path):
    path = tmp_path / "receipt.json"
    RUNNER._exclusive_write(path, b"one")
    with pytest.raises(FileExistsError):
        RUNNER._exclusive_write(path, b"two")


def test_exact_python_symlink_chain_and_pyvenv_are_bound(tmp_path, monkeypatch):
    _patch_artifact_paths(monkeypatch, tmp_path)
    real = tmp_path / "usr/bin/python3.10"
    real.parent.mkdir(parents=True)
    real.write_bytes(b"python-binary")
    real.chmod(0o755)
    requested = RUNNER.ISAAC_PYTHON
    requested.parent.mkdir(parents=True)
    requested.symlink_to("python3.10")
    (requested.parent / "python3.10").symlink_to(real)
    pyvenv = requested.parents[1] / "pyvenv.cfg"
    pyvenv.write_bytes(b"home = exact\n")
    train = tmp_path / "train.py"
    train.write_bytes(b"# train\n")
    runtime = {
        "python": {
            "requested_path": str(requested),
            "resolved_path": str(real),
            "symlink_chain": [
                {"kind": "symlink", "path": str(requested), "target": "python3.10"},
                {"kind": "symlink", "path": str(requested.parent / "python3.10"), "target": str(real)},
                {"kind": "regular", **_file_row(real)},
            ],
            "pyvenv_cfg": _file_row(pyvenv),
        },
    }
    proof = RUNNER._verify_python_runtime(runtime)
    assert proof["symlink_chain"][-1]["sha256"] == _sha(real.read_bytes())
    (requested.parent / "python3.10").unlink()
    (requested.parent / "python3.10").symlink_to("/bin/false")
    with pytest.raises(RUNNER.CaptureContractError, match="target drifted"):
        RUNNER._verify_python_runtime(runtime)


def _lineage_fixture(tmp_path: Path, monkeypatch):
    plan = _plan(tmp_path, monkeypatch)
    teacher = plan["teacher_checkpoint"]
    checkpoint = Path(teacher["path"])
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    teacher["sha256"] = _sha(checkpoint.read_bytes())
    hard = Path(teacher["hard_contract"]["path"])
    hard.parent.mkdir(parents=True)
    hard.write_bytes(_canonical({"schema_version": 3}))
    teacher["hard_contract"]["sha256"] = _sha(hard.read_bytes())
    claim_content = {"schema_version": 1, "job_id": "teacher-job"}
    claim, teacher["launch_claim"] = _document(tmp_path / "claim.json", claim_content, schema=2)
    binding_content = {
        "schema_version": 1,
        "job_id": "teacher-job",
        "source": {"checkout": "/training/source", "commit": teacher["training_source_commit"]},
        "pod": "pod2",
        "gpu": 1,
        "claim_content_sha256": claim["content_sha256"],
        "training_argv": _binding(plan)["content"]["training_argv"],
    }
    binding, teacher["run_binding"] = _document(tmp_path / "binding.json", binding_content)
    receipt_content = {
        "schema_version": 1,
        "job_id": "teacher-job",
        "binding_path": teacher["run_binding"]["path"],
        "binding_content_sha256": binding["content_sha256"],
        "claim_content_sha256": claim["content_sha256"],
        "milestone": 500,
        "process_state_at_attestation": "exited",
        "checkpoint": {
            "path": str(checkpoint), "sha256": teacher["sha256"],
            "filename_iteration": 500, "embedded_iteration": 500,
            "tensor_count": 2, "floating_tensor_count": 1,
            "floating_elements": teacher["floating_elements"],
            "nonfinite_floating_elements": 0,
        },
        "hard_contract": {
            "path": str(hard), "schema_version": 3,
            "sha256": teacher["hard_contract"]["sha256"], "lineage_exact": 1,
        },
    }
    receipt, teacher["milestone_receipt"] = _document(tmp_path / "receipt.json", receipt_content)
    source = Path(plan["capture_source"]["checkout"])
    lean = source / plan["capture_source"]["files"]["lean_queue_runtime"]["path"]
    lean.parent.mkdir(parents=True, exist_ok=True)
    lean.write_bytes(b"# exact lean runtime")
    plan["capture_source"]["files"]["lean_queue_runtime"] = {
        "path": str(lean.relative_to(source)), "bytes": len(lean.read_bytes()), "sha256": _sha(lean.read_bytes())
    }
    stub = types.SimpleNamespace(
        _load_binding=lambda _path: (binding, binding_content, claim, claim_content)
    )
    monkeypatch.setattr(RUNNER, "_load_lean_runtime", lambda _path: stub)
    return plan, receipt_content


def test_lean_binding_and_milestone_lineage_close_exactly(tmp_path, monkeypatch):
    plan, _ = _lineage_fixture(tmp_path, monkeypatch)
    binding, proof = RUNNER._verify_teacher_lineage(plan, Path(plan["capture_source"]["checkout"]))
    assert proof["claim_content_sha256"] == binding["content"]["claim_content_sha256"]
    assert proof["hard_contract_sha256"] == plan["teacher_checkpoint"]["hard_contract"]["sha256"]


def test_mixed_milestone_lineage_is_rejected(tmp_path, monkeypatch):
    plan, content = _lineage_fixture(tmp_path, monkeypatch)
    content["claim_content_sha256"] = "f" * 64
    _receipt, plan["teacher_checkpoint"]["milestone_receipt"] = _document(tmp_path / "mixed.json", content)
    with pytest.raises(RUNNER.CaptureContractError, match="rebound"):
        RUNNER._verify_teacher_lineage(plan, Path(plan["capture_source"]["checkout"]))


def test_shared_gpu_lease_is_mutually_exclusive(tmp_path, monkeypatch):
    _patch_artifact_paths(monkeypatch, tmp_path)
    with RUNNER._gpu_lease() as descriptor:
        assert os.get_inheritable(descriptor)
        with pytest.raises(RUNNER.CaptureContractError, match="already held"):
            with RUNNER._gpu_lease():
                pass
    with RUNNER._gpu_lease():
        pass


def _fake_runtime(plan):
    return _binding(plan), {
        "source_commit": plan["capture_source"]["commit"],
        "asset_inventory": {"file_count": 1, "total_file_bytes": 1, "tree_content_sha256": "a" * 64},
        "gpu": {"index": 2, "uuid": "GPU-exact-1"},
        "verification_elapsed_ms": 1,
    }


def test_plan_composes_with_exact_launch_context_and_writes_no_namespace(tmp_path, monkeypatch):
    plan = _plan(tmp_path, monkeypatch)
    calls = []
    runtime_calls = []

    def verify(candidate, _script):
        runtime_calls.append(candidate)
        return _fake_runtime(candidate)

    def compose(command, **kwargs):
        calls.append((command, kwargs))
        return types.SimpleNamespace(returncode=0, stdout=b"resolved-config")

    monkeypatch.setattr(RUNNER, "_verify_runtime", verify)
    monkeypatch.setattr(RUNNER.subprocess, "run", compose)
    monkeypatch.setattr(
        RUNNER.os,
        "execve",
        lambda *_args: pytest.fail("plan mode must not start the capture process"),
    )
    raw = _canonical(plan)
    result = RUNNER._plan_summary(plan, raw, Path("controller"))

    assert len(runtime_calls) == 2
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command[-3:] == ["--cfg", "job", "--resolve"]
    assert kwargs["cwd"] == Path(plan["capture_source"]["checkout"]) / "hope_training/whole_body_tracking"
    assert kwargs["env"] == RUNNER._environment(plan)
    assert kwargs["timeout"] == plan["runtime_environment"]["compose_timeout_s"]
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["stdout"] is subprocess.PIPE
    assert kwargs["stderr"] is subprocess.STDOUT
    assert result["hydra_compose"]["output_sha256"] == _sha(b"resolved-config")
    assert result["hydra_compose"]["output_bytes"] == len(b"resolved-config")
    assert type(result["hydra_compose"]["elapsed_ms"]) is int
    assert not RUNNER.ARTIFACT_ROOT.exists()
    assert not (RUNNER.CAPTURE_PARENT / plan["plan_id"]).exists()
    assert not (RUNNER.LAUNCH_PARENT / plan["plan_id"]).exists()


def test_plan_compose_failure_spends_no_namespace_or_claim(tmp_path, monkeypatch):
    plan = _plan(tmp_path, monkeypatch)
    monkeypatch.setattr(RUNNER, "_verify_runtime", lambda p, _s: _fake_runtime(p))
    monkeypatch.setattr(
        RUNNER.subprocess,
        "run",
        lambda *_args, **_kwargs: types.SimpleNamespace(returncode=9, stdout=b"bad recipe"),
    )
    with pytest.raises(RUNNER.CaptureContractError, match="rc=9"):
        RUNNER._plan_summary(plan, _canonical(plan), Path("controller"))
    assert not RUNNER.ARTIFACT_ROOT.exists()
    assert not (RUNNER.CAPTURE_PARENT / plan["plan_id"]).exists()
    assert not (RUNNER.LAUNCH_PARENT / plan["plan_id"]).exists()


def test_plan_post_compose_drift_fails_without_spending_namespace(tmp_path, monkeypatch):
    plan = _plan(tmp_path, monkeypatch)
    proofs = [_fake_runtime(plan), _fake_runtime(plan)]
    proofs[1][1]["source_commit"] = "e" * 40
    monkeypatch.setattr(RUNNER, "_verify_runtime", lambda *_args: proofs.pop(0))
    monkeypatch.setattr(
        RUNNER.subprocess,
        "run",
        lambda *_args, **_kwargs: types.SimpleNamespace(returncode=0, stdout=b"resolved"),
    )
    with pytest.raises(RUNNER.CaptureContractError, match="drifted"):
        RUNNER._plan_summary(plan, _canonical(plan), Path("controller"))
    assert not RUNNER.ARTIFACT_ROOT.exists()


def test_compose_failure_records_all_evidence_without_capture_namespace(tmp_path, monkeypatch):
    plan = _plan(tmp_path, monkeypatch)
    monkeypatch.setattr(RUNNER, "_verify_runtime", lambda p, _s: _fake_runtime(p))
    commands = []

    def compose(command, **kwargs):
        commands.append((command, kwargs))
        return types.SimpleNamespace(returncode=9, stdout=b"compose failed")

    monkeypatch.setattr(RUNNER.subprocess, "run", compose)
    with pytest.raises(RUNNER.CaptureContractError, match="rc=9"):
        RUNNER._launch(plan, _canonical(plan), Path("controller"))
    assert commands[0][0][-3:] == ["--cfg", "job", "--resolve"]
    assert commands[0][1]["timeout"] == 30
    launch = RUNNER.LAUNCH_PARENT / plan["plan_id"]
    assert (launch / "launch_intent.json").is_file()
    assert (launch / "hydra_compose.log").read_bytes() == b"compose failed"
    assert json.loads((launch / "failure.json").read_text())["stage"] == "hydra_compose"
    assert not (RUNNER.CAPTURE_PARENT / plan["plan_id"]).exists()


def test_post_compose_drift_spends_launch_but_not_capture(tmp_path, monkeypatch):
    plan = _plan(tmp_path, monkeypatch)
    proofs = [_fake_runtime(plan), _fake_runtime(plan)]
    proofs[1][1]["source_commit"] = "e" * 40
    monkeypatch.setattr(RUNNER, "_verify_runtime", lambda *_args: proofs.pop(0))
    monkeypatch.setattr(
        RUNNER.subprocess,
        "run",
        lambda *_args, **_kwargs: types.SimpleNamespace(returncode=0, stdout=b"resolved"),
    )
    with pytest.raises(RUNNER.CaptureContractError, match="drifted"):
        RUNNER._launch(plan, _canonical(plan), Path("controller"))
    launch = RUNNER.LAUNCH_PARENT / plan["plan_id"]
    assert json.loads((launch / "failure.json").read_text())["stage"] == "runtime_verification_after_compose"
    receipt = json.loads((launch / "hydra_compose_receipt.json").read_text())
    assert receipt["output_sha256"] == _sha(b"resolved")
    assert not (RUNNER.CAPTURE_PARENT / plan["plan_id"]).exists()


def test_same_pid_exec_intent_exists_before_exec_and_no_child_is_spawned(tmp_path, monkeypatch):
    plan = _plan(tmp_path, monkeypatch)
    (Path(plan["capture_source"]["checkout"]) / "hope_training/whole_body_tracking").mkdir(
        parents=True, exist_ok=True
    )
    monkeypatch.setattr(RUNNER, "_verify_runtime", lambda p, _s: _fake_runtime(p))
    monkeypatch.setattr(
        RUNNER.subprocess,
        "run",
        lambda *_args, **_kwargs: types.SimpleNamespace(returncode=0, stdout=b"resolved"),
    )
    monkeypatch.setattr(RUNNER.os, "setsid", lambda: None)
    monkeypatch.setattr(
        RUNNER,
        "_proc_identity",
        lambda _pid: {
            "pid": os.getpid(), "pgid": os.getpid(), "sid": os.getpid(),
            "starttime_ticks": 123, "state": "S", "argv": [],
        },
    )
    observed = {}

    def fake_exec(path, argv, environment):
        intent = json.loads((RUNNER.LAUNCH_PARENT / plan["plan_id"] / "exec_intent.json").read_text())
        observed.update(path=path, argv=argv, environment=environment, intent=intent)
        raise OSError("sentinel exec failure")

    monkeypatch.setattr(RUNNER.os, "execve", fake_exec)
    with pytest.raises(RUNNER.CaptureContractError, match="execve_same_pid"):
        RUNNER._launch(plan, _canonical(plan), Path("controller"))
    assert observed["path"] == str(RUNNER.ISAAC_PYTHON)
    assert observed["intent"]["pid"] == os.getpid()
    assert observed["intent"]["handoff"] == "execve_same_pid_v1"
    assert observed["environment"]["CUDA_VISIBLE_DEVICES"] == "GPU-exact-1"
    assert (RUNNER.CAPTURE_PARENT / plan["plan_id"]).is_dir()


def _write_proc(proc_root: Path, pid: int, *, state: str, pgid: int, sid: int, start: int, argv: list[str]):
    root = proc_root / str(pid)
    root.mkdir(parents=True)
    fields = [state, "1", str(pgid), str(sid), *("0" for _ in range(15)), str(start)]
    (root / "stat").write_text(f"{pid} (capture worker) " + " ".join(fields), encoding="utf-8")
    (root / "cmdline").write_bytes(b"\0".join(value.encode() for value in argv) + b"\0")


def test_status_rejects_symlink_artifact_and_zombie_identity(tmp_path, monkeypatch):
    plan = _plan(tmp_path, monkeypatch)
    raw = _canonical(plan)
    launch = RUNNER.LAUNCH_PARENT / plan["plan_id"]
    output = RUNNER.CAPTURE_PARENT / plan["plan_id"]
    launch.mkdir(parents=True)
    output.mkdir(parents=True)
    argv = ["python", "play.py"]
    argv_sha = _sha(_canonical(argv))
    pid = 43210
    runtime = {"schema_version": 1, "plan_sha256": _sha(raw), "argv": argv, "argv_sha256": argv_sha}
    intent = {
        "schema_version": 1, "started_utc": "now", "pid": pid, "pgid": pid, "sid": pid,
        "leader_starttime_ticks": 55, "plan_sha256": _sha(raw), "argv_sha256": argv_sha,
        "environment_sha256": "a" * 64, "source_commit": plan["capture_source"]["commit"],
        "capture_output": str(output), "run_log": str(launch / "run.log"),
        "gpu_lease_path": str(RUNNER.GPU_LEASE_PATH), "handoff": "execve_same_pid_v1",
    }
    (launch / "runtime_argv.json").write_bytes(_canonical(runtime))
    (launch / "exec_intent.json").write_bytes(_canonical(intent))
    target = tmp_path / "foreign.npz"
    target.write_bytes(b"foreign")
    (output / "natural_wrap_states.npz").symlink_to(target)
    proc = tmp_path / "proc"
    _write_proc(proc, pid, state="Z", pgid=pid, sid=pid, start=55, argv=argv)
    monkeypatch.setattr(RUNNER, "_gpu_state", lambda _plan: {"gpus": [], "compute_apps": []})
    monkeypatch.setattr(RUNNER, "_lease_is_held", lambda: False)
    result = RUNNER._status(plan, raw, proc)
    assert not result["leader_alive"]
    assert not result["leader_identity_exact"]
    assert result["artifacts"]["natural_wrap_states.npz"]["kind"] == "symlink_rejected"


def test_status_rejects_rebound_teacher_receipt(tmp_path, monkeypatch):
    plan = _plan(tmp_path, monkeypatch)
    raw = _canonical(plan)
    output = RUNNER.CAPTURE_PARENT / plan["plan_id"]
    output.mkdir(parents=True)
    (output / "natural_wrap_capture.json").write_bytes(b"result")
    (output / "natural_wrap_capture.claim.json").write_bytes(b"claim")
    receipt = {
        "teacher": {
            "source_commit": plan["teacher_checkpoint"]["training_source_commit"],
            "checkpoint_sha256": "0" * 64,
            "training_contract_sha256": plan["teacher_checkpoint"]["hard_contract"]["sha256"],
            "training_contract_schema_version": 3,
            "fresh_lineage": True,
        },
        "motion_clips": plan["ordered_motion_inputs"],
        "attestation": {},
    }
    (output / "teacher_receipt.json").write_bytes(_canonical(receipt))
    monkeypatch.setattr(RUNNER, "_gpu_state", lambda _plan: {"gpus": [], "compute_apps": []})
    monkeypatch.setattr(RUNNER, "_lease_is_held", lambda: False)
    result = RUNNER._status(plan, raw, tmp_path / "proc")
    assert result["teacher_receipt_binding_exact"] is False


def _write_status_receipt(
    plan: dict, output: Path, attestor_source: dict, retry_binding: dict
) -> dict:
    result_raw = b"immutable-capture-result"
    claim_raw = b"immutable-capture-claim"
    (output / "natural_wrap_capture.json").write_bytes(result_raw)
    (output / "natural_wrap_capture.claim.json").write_bytes(claim_raw)
    receipt = {
        "teacher": {
            "source_commit": plan["teacher_checkpoint"]["training_source_commit"],
            "checkpoint_sha256": plan["teacher_checkpoint"]["sha256"],
            "training_contract_sha256": plan["teacher_checkpoint"]["hard_contract"]["sha256"],
            "training_contract_schema_version": 3,
            "fresh_lineage": True,
        },
        "motion_clips": [
            {"index": index, "sha256": row["sha256"]}
            for index, row in enumerate(plan["ordered_motion_inputs"])
        ],
        "attestation": {
            "schema_version": 2,
            "artifact_kind": "hope_post_swing_teacher_capture_attestation",
            "capture_result_sha256": _sha(result_raw),
            "capture_result_relative_path": "natural_wrap_capture.json",
            "capture_claim_sha256": _sha(claim_raw),
            "capture_claim_relative_path": "natural_wrap_capture.claim.json",
            "checkpoint": {
                "sha256": plan["teacher_checkpoint"]["sha256"],
                "training_contract_schema_version": 3,
                "training_contract_sha256": plan["teacher_checkpoint"]["hard_contract"]["sha256"],
                "training_contract_lineage_exact": True,
                "training_launch_claim_sha256": plan["teacher_checkpoint"]["launch_claim"]["content_sha256"],
            },
            "hard_contract": {
                "sha256": plan["teacher_checkpoint"]["hard_contract"]["sha256"],
                "schema_version": 3,
            },
            "checkpoint_source": {
                "commit": plan["teacher_checkpoint"]["training_source_commit"],
                "launch_claim_content_sha256": plan["teacher_checkpoint"]["launch_claim"]["content_sha256"],
            },
            "capture_source": {
                "commit": plan["capture_source"]["commit"],
                "clean": True,
                "producer_source_sha256": plan["capture_source"]["files"]["producer"]["sha256"],
            },
            "attestor_source": dict(attestor_source),
            "retry_authorization": dict(retry_binding),
        },
    }
    (output / "teacher_receipt.json").write_bytes(_canonical(receipt) + b"\n")
    return receipt


def test_status_accepts_split_capture_and_attestor_lineage(tmp_path, monkeypatch):
    plan = _plan(tmp_path, monkeypatch)
    raw = _canonical(plan)
    output = RUNNER.CAPTURE_PARENT / plan["plan_id"]
    output.mkdir(parents=True)
    attestor_source = {
        "commit": "f" * 40,
        "clean": True,
        "attestor_source_sha256": "a" * 64,
    }
    retry_binding = {
        "authorization_id": "test-v3-attestor-attempt2",
        "file_sha256": "b" * 64,
        "v3_plan_file_sha256": _sha(raw),
    }
    _write_status_receipt(plan, output, attestor_source, retry_binding)
    monkeypatch.setattr(
        RUNNER,
        "_status_retry_authorization",
        lambda *_args: {
            "attestor_source": attestor_source,
            "receipt_binding": retry_binding,
            "status_source_commit": "1" * 40,
        },
    )
    monkeypatch.setattr(RUNNER, "_gpu_state", lambda _plan: {"gpus": [], "compute_apps": []})
    monkeypatch.setattr(RUNNER, "_lease_is_held", lambda: False)
    status = RUNNER._status(plan, raw, tmp_path / "proc")
    assert plan["capture_source"]["commit"] != attestor_source["commit"]
    assert status["teacher_receipt_binding_exact"] is True


@pytest.mark.parametrize(
    "mutation",
    (
        "capture_commit",
        "producer_sha",
        "attestor_commit",
        "attestor_sha",
        "swap_sources",
        "dirty_capture_checkout",
        "dirty_attestor_checkout",
    ),
)
def test_status_rejects_rebound_swapped_or_dirty_source_lineage(
    tmp_path, monkeypatch, mutation
):
    plan = _plan(tmp_path, monkeypatch)
    raw = _canonical(plan)
    output = RUNNER.CAPTURE_PARENT / plan["plan_id"]
    output.mkdir(parents=True)
    live_attestor = {
        "commit": "f" * 40,
        "clean": True,
        "attestor_source_sha256": "a" * 64,
    }
    retry_binding = {
        "authorization_id": "test-v3-attestor-attempt2",
        "file_sha256": "b" * 64,
        "v3_plan_file_sha256": _sha(raw),
    }
    receipt = _write_status_receipt(plan, output, live_attestor, retry_binding)
    if mutation == "capture_commit":
        receipt["attestation"]["capture_source"]["commit"] = "0" * 40
    elif mutation == "producer_sha":
        receipt["attestation"]["capture_source"]["producer_source_sha256"] = "0" * 64
    elif mutation == "attestor_commit":
        receipt["attestation"]["attestor_source"]["commit"] = "0" * 40
    elif mutation == "attestor_sha":
        receipt["attestation"]["attestor_source"]["attestor_source_sha256"] = "0" * 64
    elif mutation == "swap_sources":
        receipt["attestation"]["capture_source"], receipt["attestation"]["attestor_source"] = (
            receipt["attestation"]["attestor_source"],
            receipt["attestation"]["capture_source"],
        )
    elif mutation == "dirty_capture_checkout":
        receipt["attestation"]["capture_source"]["clean"] = False
    (output / "teacher_receipt.json").write_bytes(_canonical(receipt) + b"\n")

    if mutation == "dirty_attestor_checkout":
        def dirty(*_args):
            raise RUNNER.CaptureContractError("status authorization source has tracked changes")

        monkeypatch.setattr(RUNNER, "_status_retry_authorization", dirty)
    else:
        monkeypatch.setattr(
            RUNNER,
            "_status_retry_authorization",
            lambda *_args: {
                "attestor_source": live_attestor,
                "receipt_binding": retry_binding,
                "status_source_commit": "1" * 40,
            },
        )
    monkeypatch.setattr(RUNNER, "_gpu_state", lambda _plan: {"gpus": [], "compute_apps": []})
    monkeypatch.setattr(RUNNER, "_lease_is_held", lambda: False)
    status = RUNNER._status(plan, raw, tmp_path / "proc")
    assert status["teacher_receipt_binding_exact"] is False


def test_status_retry_authorization_is_tracked_clean_and_binds_v3_and_attestor(
    tmp_path, monkeypatch
):
    plan = _plan(tmp_path, monkeypatch)
    plan_sha = _sha(_canonical(plan))
    output = RUNNER.CAPTURE_PARENT / plan["plan_id"]
    output.mkdir(parents=True)
    capture_claim = b"capture claim"
    states = b"states"
    result = b"capture result"
    (output / "natural_wrap_capture.claim.json").write_bytes(capture_claim)
    (output / "natural_wrap_states.npz").write_bytes(states)
    (output / "natural_wrap_capture.json").write_bytes(result)
    source = tmp_path / "fixed-attestor-source"
    controller = source / "scripts/run_preregistered_post_swing_capture.py"
    attestor = source / "scripts/attest_post_swing_teacher.py"
    controller.parent.mkdir(parents=True)
    controller.write_bytes(b"fixed controller\n")
    attestor.write_bytes(b"fixed attestor\n")
    authorization_path = source / RUNNER.RETRY_AUTHORIZATION_RELATIVE
    authorization_path.parent.mkdir(parents=True)
    authorization = {
        "schema_version": 1,
        "artifact_kind": RUNNER.RETRY_AUTHORIZATION_KIND,
        "authorization_id": "test-v3-attestor-attempt2",
        "v3_plan": {"plan_id": plan["plan_id"], "file_sha256": plan_sha},
        "capture": {
            "output_directory": str(output),
            "output_receipt": str(output / "teacher_receipt.json"),
            "capture_claim_sha256": _sha(capture_claim),
            "states_sha256": _sha(states),
            "result_sha256": _sha(result),
            "state_count": plan["capture_contract"]["target_count"],
        },
        "teacher": {
            "checkpoint_sha256": plan["teacher_checkpoint"]["sha256"],
            "hard_contract_sha256": plan["teacher_checkpoint"]["hard_contract"]["sha256"],
            "launch_claim_content_sha256": plan["teacher_checkpoint"]["launch_claim"]["content_sha256"],
        },
        "capture_source": {
            "commit": plan["capture_source"]["commit"],
            "producer_source_sha256": plan["capture_source"]["files"]["producer"]["sha256"],
        },
        "attestor_source": {
            "commit": "f" * 40,
            "attestor_source_sha256": _sha(attestor.read_bytes()),
        },
        "decision": {
            "capture_retry_authorized": False,
            "attestor_attempt2_authorized": True,
            "first_reset_probe_authorized": False,
            "scientific_training_authorized": False,
        },
    }
    authorization_path.write_bytes(_canonical(authorization) + b"\n")
    monkeypatch.setattr(RUNNER, "_verify_executable_row", lambda *args: {})
    monkeypatch.setattr(
        RUNNER,
        "RETRY_AUTHORIZATION_SHA256",
        _sha(authorization_path.read_bytes()),
    )

    dirty = False
    def git_output(_git, checkout, *args):
        assert checkout == source
        if args[:2] == ("rev-parse", "HEAD"):
            return "f" * 40
        if args[:2] == ("status", "--porcelain=v1"):
            return " M configs/retry.json" if dirty else ""
        assert args[:2] == ("ls-files", "--error-unmatch")
        return RUNNER.RETRY_AUTHORIZATION_RELATIVE.as_posix()

    monkeypatch.setattr(RUNNER, "_git_output", git_output)
    proof = RUNNER._status_retry_authorization(plan, output, controller, plan_sha)
    assert proof["attestor_source"] == {
        "commit": authorization["attestor_source"]["commit"],
        "clean": True,
        "attestor_source_sha256": authorization["attestor_source"]["attestor_source_sha256"],
    }
    assert proof["receipt_binding"]["file_sha256"] == _sha(authorization_path.read_bytes())
    frozen_authorization_raw = authorization_path.read_bytes()
    authorization["authorization_id"] = "rebound-clean-source"
    authorization_path.write_bytes(_canonical(authorization) + b"\n")
    with pytest.raises(RUNNER.CaptureContractError, match="bytes differ from source gate"):
        RUNNER._status_retry_authorization(plan, output, controller, plan_sha)
    authorization_path.write_bytes(frozen_authorization_raw)
    dirty = True
    with pytest.raises(RUNNER.CaptureContractError, match="tracked changes"):
        RUNNER._status_retry_authorization(plan, output, controller, plan_sha)


def test_committed_retry_authorization_matches_source_gate_and_fixed_attestor():
    raw = (ROOT / RUNNER.RETRY_AUTHORIZATION_RELATIVE).read_bytes()
    assert _sha(raw) == RUNNER.RETRY_AUTHORIZATION_SHA256
    authorization = RUNNER._strict_json_loads(raw, "committed retry authorization")
    assert authorization["attestor_source"] == {
        "commit": "a38b7e9e693db407795d9a5f3af144b8f8e293cf",
        "attestor_source_sha256": "03611b565a539fa81811ac76c4631484a60679adfd11c1f1e07599081f46310f",
    }
    assert authorization["decision"] == {
        "capture_retry_authorized": False,
        "attestor_attempt2_authorized": True,
        "first_reset_probe_authorized": False,
        "scientific_training_authorized": False,
    }


def test_inventory_small_tree_is_subsecond(tmp_path):
    tree = tmp_path / "inventory"
    tree.mkdir()
    for index in range(128):
        (tree / f"file-{index:03d}.bin").write_bytes(bytes([index % 251]) * 4096)
    start = __import__("time").monotonic()
    proof = RUNNER._inventory(tree)
    elapsed = __import__("time").monotonic() - start
    assert proof["file_count"] == 128
    assert elapsed < 2.0


def test_schema2_builder_roundtrips_through_controller_validator(tmp_path, monkeypatch):
    source = tmp_path / "exact-source"
    paths = [
        "scripts/run_preregistered_post_swing_capture.py",
        "scripts/attest_post_swing_teacher.py",
        "hope_training/whole_body_tracking/scripts/play.py",
        "hope_training/whole_body_tracking/scripts/train.py",
        "hope_training/whole_body_tracking/scripts/lean_queue_runtime.py",
        "hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py",
    ]
    for relative in paths:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"# {relative}\n".encode())
    asset = source / "asset"
    asset.mkdir()
    (asset / "robot.usd").write_bytes(b"asset")
    template = _plan(tmp_path)
    template["capture_source"]["ignored_runtime_asset"]["relative_path"] = "asset"
    template_path = tmp_path / "template.json"
    template_path.write_bytes(_canonical(template))
    python = tmp_path / "venv/bin/python"
    real_python = tmp_path / "usr/bin/python3.10"
    python.parent.mkdir(parents=True)
    real_python.parent.mkdir(parents=True)
    real_python.write_bytes(b"python")
    real_python.chmod(0o755)
    python.symlink_to(real_python)
    (tmp_path / "venv/pyvenv.cfg").write_bytes(b"home=exact\n")
    machine = tmp_path / "machine-id"
    boot = tmp_path / "boot-id"
    machine.write_bytes(b"machine\n")
    boot.write_bytes(b"boot\n")
    git = tmp_path / "tools/git"
    nvidia = tmp_path / "tools/nvidia-smi"
    git.parent.mkdir()
    for tool in (git, nvidia):
        tool.write_bytes(b"#!/bin/sh\n")
        tool.chmod(0o755)
    tree_values = []
    for label in sorted(BUILDER.contract.RUNTIME_TREE_LABELS):
        tree = tmp_path / f"tree-{label}"
        tree.mkdir()
        (tree / "module.py").write_bytes(label.encode())
        tree_values.append(f"{label}={tree}:on")
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setattr(BUILDER.contract, "ARTIFACT_ROOT", artifact_root)
    monkeypatch.setattr(BUILDER.contract, "CAPTURE_PARENT", artifact_root / "capture")
    monkeypatch.setattr(BUILDER.contract, "LAUNCH_PARENT", artifact_root / "launch")
    monkeypatch.setattr(BUILDER.contract, "GPU_LEASE_PATH", tmp_path / "gpu.lock")
    monkeypatch.setattr(BUILDER.contract, "ISAAC_PYTHON", python)
    monkeypatch.setattr(BUILDER.contract, "MACHINE_ID_PATH", machine)
    monkeypatch.setattr(BUILDER.contract, "BOOT_ID_PATH", boot)
    git_calls = iter(["e" * 40, ""])
    monkeypatch.setattr(BUILDER.contract, "_git_output", lambda *_args: next(git_calls))
    output = tmp_path / "generated/plan.json"
    args = types.SimpleNamespace(
        template_plan=template_path,
        capture_source_checkout=source,
        plan_id="generated-v2",
        gpu_uuid="GPU-exact-1",
        hostname="pod2-exact",
        python=python,
        git=git,
        nvidia_smi=nvidia,
        runtime_tree=tree_values,
        env=["PATH=/usr/bin:/bin", "HOME=/root"],
        seed=3,
        compose_timeout_s=30,
        output=output,
    )
    generated = BUILDER.build(args)
    BUILDER.contract._validate_plan(generated)
    raw = _canonical(generated) + b"\n"
    output.parent.mkdir()
    output.write_bytes(raw)
    loaded, loaded_raw = BUILDER.contract._load_plan(output, _sha(raw))
    assert loaded == generated
    assert loaded_raw == raw
    assert generated["capture_source"]["full_tree"]["file_count"] >= len(paths) + 1
