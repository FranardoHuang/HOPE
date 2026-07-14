"""Dependency-light adversarial tests for the full-scene terminal gate."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R = _load("lean_queue_runtime", SCRIPTS / "lean_queue_runtime.py")
P = _load("full_scene_probe_runtime_under_test", SCRIPTS / "full_scene_probe_runtime.py")


class FakeTensor:
    def __init__(self, values):
        self.values = list(values)

    def numel(self):
        return len(self.values)


class _Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class _Mask:
    def __init__(self, values):
        self.values = values

    def sum(self):
        return _Scalar(sum(self.values))


class FakeTorch:
    Tensor = FakeTensor

    @staticmethod
    def is_floating_point(_value):
        return True

    @staticmethod
    def is_complex(_value):
        return False

    @staticmethod
    def isfinite(value):
        return _Mask([math.isfinite(item) for item in value.values])


def _write_proc(proc_root: Path, pid: int, pgid: int, starttime: int, argv: list[str]):
    root = proc_root / str(pid)
    root.mkdir(parents=True, exist_ok=True)
    rest = ["S", *( ["0"] * 18), str(starttime)]
    rest[2] = str(pgid)
    (root / "stat").write_text(
        f"{pid} (probe process) " + " ".join(rest) + "\n", encoding="utf-8"
    )
    (root / "cmdline").write_bytes(
        b"\0".join(item.encode("utf-8") for item in argv) + b"\0"
    )
    return pgid


def _envelope(content):
    return {
        "schema_version": 1,
        "content": content,
        "content_sha256": R.canonical_sha256(content),
    }


def _write_json(path: Path, value):
    path.write_text(
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _fixture(tmp_path: Path, *, live=False, wrong_supervisor_argv=False):
    root = tmp_path / "case"
    source = root / "source"
    train = source / R.TRAIN_ENTRY_RELATIVE
    train.parent.mkdir(parents=True)
    train.write_text("# exact train entry\n", encoding="utf-8")
    run_dir = root / "runs/probe"
    run_dir.mkdir(parents=True)
    claim_path = run_dir / R.PROBE_CLAIM_NAME
    binding_path = run_dir / R.PROBE_BINDING_NAME
    run_name = "full_scene_probe_not_science_case"
    log_dir = (
        source
        / R.WBT_RELATIVE
        / "logs/rsl_rl/experiment"
        / f"2026-07-14_12-34-56_{run_name}"
    )
    (log_dir / "params").mkdir(parents=True)
    assets = root / "assets"
    assets.mkdir()
    motion0 = assets / "forehand.npz"
    motion1 = assets / "backhand.npz"
    bank = assets / "train.npz"
    motion0.write_bytes(b"motion-forehand")
    motion1.write_bytes(b"motion-backhand")
    bank.write_bytes(b"bank")
    asset_contract = {
        "target_relative_path": "runtime_assets/agibot_a3",
        "donor": {
            "checkout": str(root / "donor"),
            "commit": "b" * 40,
            "relative_path": "assets/agibot_a3",
        },
        "file_count": 46,
        "total_file_bytes": 15378264,
        "tree_content_sha256": "c" * 64,
        "symlinks_forbidden": True,
        "target_must_be_gitignored": True,
    }
    source_asset_receipt = root / "source_asset_receipts/receipt.json"
    source_asset_receipt.parent.mkdir(parents=True)
    argv_without_claim = [
        "/exact/python",
        str(train),
        "task=Task",
        "algo=ppo",
        f"motion_file={motion0}",
        f"motion_file_2={motion1}",
        f"++task.racket.question_bank={bank}",
        f"run_name={run_name}",
        f"++training_queue_claim_path={claim_path}",
        f"++training_run_binding_path={binding_path}",
    ]
    supervisor_prefix = [
        "/exact/python",
        str(SCRIPTS / "full_scene_probe_runtime.py"),
        "supervise",
        "--run-dir",
        str(run_dir),
        "--log",
        str(run_dir / "run.log"),
        "--",
    ]
    content = {
        "schema_version": 1,
        "purpose": R.PROBE_PURPOSE,
        "not_science": True,
        "attestable": False,
        "promotable": False,
        "job_id": "job0",
        "pod": "pod2",
        "gpu": 1,
        "source": {
            "checkout": str(source),
            "commit": "a" * 40,
            "ignored_runtime_asset": asset_contract,
        },
        "source_asset_receipt_path": str(source_asset_receipt),
        "supervisor_argv_prefix": supervisor_prefix,
        "expected_training_contract_lineage_exact": 1,
        "run_name": run_name,
        "run_dir": str(run_dir),
        "budget": {
            "num_envs": 4096,
            "max_iterations": 2,
            "save_interval": 1,
            "milestones": [1],
        },
        "inputs": {
            "motion": {
                "action": "paired",
                "bindings": {"motion_file": str(motion0), "motion_file_2": str(motion1)},
            },
            "bank": {"train_path": str(bank), "train_arg": "++task.racket.question_bank"},
            "exam": {"path": str(assets / "exam.npz")},
        },
        "training_argv_without_claim": argv_without_claim,
    }
    claim_digest = R.canonical_sha256(content)
    full_argv = [*argv_without_claim, f"++training_launch_claim_sha256={claim_digest}"]
    claim = {
        "schema_version": 2,
        "content": content,
        "content_sha256": claim_digest,
        "training_argv": full_argv,
    }
    _write_json(claim_path, claim)
    source_asset_content = {
        "schema_version": 1,
        "pod": "pod2",
        "source": {"checkout": str(source), "commit": "a" * 40},
        "ignored_runtime_asset": asset_contract,
        "ignored_runtime_asset_sha256": R.canonical_sha256(asset_contract),
        "target_path": str(source / asset_contract["target_relative_path"]),
        "inventory": {
            "file_count": asset_contract["file_count"],
            "total_file_bytes": asset_contract["total_file_bytes"],
            "tree_content_sha256": asset_contract["tree_content_sha256"],
        },
        "urdf_reference_closure": {
            "mesh_reference_occurrences": 43,
            "unique_mesh_references": 43,
            "resolved_regular_meshes": 43,
        },
        "target_gitignored": True,
        "symlinks_present": False,
    }
    _write_json(source_asset_receipt, _envelope(source_asset_content))
    proc_root = root / "proc"
    supervisor_pid = 41000
    trainer_pid = 41001
    supervisor_start = 7000
    trainer_start = 7001
    supervisor_argv = (
        ["/exact/unbound-wrapper", "probe"]
        if wrong_supervisor_argv
        else [*supervisor_prefix, *full_argv]
    )
    _write_proc(proc_root, supervisor_pid, supervisor_pid, supervisor_start, supervisor_argv)
    _write_proc(proc_root, trainer_pid, supervisor_pid, trainer_start, full_argv)
    pgids = {supervisor_pid: supervisor_pid, trainer_pid: supervisor_pid}
    binding = R.publish_run_binding(
        claim_path=claim_path,
        binding_path=binding_path,
        log_dir=log_dir,
        claim_digest=claim_digest,
        actual_argv=full_argv,
        pid=trainer_pid,
        proc_root=proc_root,
        getpgid=lambda pid: pgids[pid],
        environ={"CUDA_VISIBLE_DEVICES": "1"},
        source_verifier=lambda _path, commit: {"head": commit, "clean": True},
    )
    hard = {
        "schema_version": 3,
        "motion_clips": [
            {"index": 0, "basename": motion0.name, "sha256": hashlib.sha256(motion0.read_bytes()).hexdigest()},
            {"index": 1, "basename": motion1.name, "sha256": hashlib.sha256(motion1.read_bytes()).hexdigest()},
        ],
        "question_bank": {"sha256": hashlib.sha256(bank.read_bytes()).hexdigest()},
    }
    hard_path = log_dir / "params/training_contract.json"
    hard_path.write_text(json.dumps(hard, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    hard_sha = hashlib.sha256(hard_path.read_bytes()).hexdigest()
    log_lines = [
        P.PHASE_PREFIX + json.dumps({"phase": "scene_import_start"}, separators=(",", ":")),
        P.PHASE_PREFIX + json.dumps({"phase": "scene_import_done"}, separators=(",", ":")),
        P.PHASE_PREFIX + json.dumps(
            {"phase": "hard_contract_written", "sha256": hard_sha}, separators=(",", ":")
        ),
        "Learning iteration 1/2",
        P.SUPERVISOR_PHASE_PREFIX + json.dumps(
            {"phase": "first_iteration_observed"}, separators=(",", ":")
        ),
    ]
    (run_dir / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    checkpoint_path = log_dir / "model_1.pt"
    checkpoint_path.write_bytes(b"stable model one")
    checkpoint = {
        "iter": 1,
        "model_state_dict": {"weight": FakeTensor([1.0, 2.0])},
        "infos": {
            "training_contract_schema_version": 3,
            "training_contract_sha256": hard_sha,
            "training_contract_lineage_exact": 1,
            "training_launch_claim_sha256": claim_digest,
        },
    }
    process = binding["content"]["process"]
    supervisor = binding["content"]["supervisor_process"]
    exit_content = {
        "schema_version": 1,
        "purpose": R.PROBE_PURPOSE,
        "claim_path": str(claim_path),
        "claim_content_sha256": claim_digest,
        "binding_path": str(binding_path),
        "binding_content_sha256": binding["content_sha256"],
        "log_path": str(run_dir / "run.log"),
        "supervisor_process": supervisor,
        "trainer_process": process,
        "first_iteration_observed": True,
        "termination": {"kind": "normal_exit", "exit_code": 0},
    }
    _write_json(run_dir / P.EXIT_NAME, _envelope(exit_content))
    if not live:
        shutil.rmtree(proc_root / str(supervisor_pid))
        shutil.rmtree(proc_root / str(trainer_pid))
    return {
        "run_dir": run_dir,
        "source": source,
        "proc_root": proc_root,
        "pgids": pgids,
        "supervisor_pid": supervisor_pid,
        "trainer_pid": trainer_pid,
        "supervisor_start": supervisor_start,
        "trainer_start": trainer_start,
        "supervisor_argv": supervisor_argv,
        "full_argv": full_argv,
        "checkpoint": checkpoint,
        "hard_path": hard_path,
        "log_path": run_dir / "run.log",
        "exit_path": run_dir / P.EXIT_NAME,
        "claim_digest": claim_digest,
        "source_asset_receipt": source_asset_receipt,
    }


def _finalize(fixture, **kwargs):
    return P.finalize(
        fixture["run_dir"],
        expected_claim_digest=fixture["claim_digest"],
        source_asset_receipt=fixture["source_asset_receipt"],
        checkpoint_loader=lambda _path: fixture["checkpoint"],
        torch_module=FakeTorch,
        proc_root=fixture["proc_root"],
        getpgid=lambda pid: fixture["pgids"].get(pid, pid),
        source_verifier=kwargs.pop(
            "source_verifier", lambda _path, commit: {"head": commit, "clean": True}
        ),
        **kwargs,
    )


def _rewrite_exit(fixture, mutate):
    value = json.loads(fixture["exit_path"].read_text(encoding="utf-8"))
    mutate(value["content"])
    value["content_sha256"] = R.canonical_sha256(value["content"])
    _write_json(fixture["exit_path"], value)


def test_happy_terminal_pass_and_identical_repeat(tmp_path):
    fixture = _fixture(tmp_path)
    first = _finalize(fixture)
    assert first["result"]["content"]["status"] == "passed"
    assert first["result"]["content"]["unlock_authorized"] is True
    assert first["result"]["content"]["not_science"] is True
    assert first["repeated_identical"] is False
    frozen = (fixture["run_dir"] / P.RESULT_NAME).read_bytes()
    second = _finalize(fixture)
    assert second["repeated_identical"] is True
    assert (fixture["run_dir"] / P.RESULT_NAME).read_bytes() == frozen


def test_binding_rejects_unclaimed_supervisor_wrapper(tmp_path):
    with pytest.raises(R.LeanQueueRuntimeError, match="supervisor argv differs"):
        _fixture(tmp_path, wrong_supervisor_argv=True)


def test_still_live_is_not_ready_and_writes_no_result(tmp_path):
    fixture = _fixture(tmp_path, live=True)
    with pytest.raises(P.FullSceneProbeNotReady, match="still live"):
        _finalize(fixture)
    assert not (fixture["run_dir"] / P.RESULT_NAME).exists()


def test_orphan_in_original_process_group_is_not_ready(tmp_path):
    fixture = _fixture(tmp_path)
    orphan_pid = 42000
    _write_proc(
        fixture["proc_root"],
        orphan_pid,
        fixture["supervisor_pid"],
        8800,
        ["/exact/orphan-gpu-child"],
    )
    with pytest.raises(P.FullSceneProbeNotReady, match="still has live members"):
        _finalize(fixture)
    assert not (fixture["run_dir"] / P.RESULT_NAME).exists()


def test_reused_pid_is_terminal_failure_not_false_absence(tmp_path):
    fixture = _fixture(tmp_path)
    _write_proc(
        fixture["proc_root"], fixture["trainer_pid"], fixture["trainer_pid"],
        fixture["trainer_start"] + 1, fixture["full_argv"],
    )
    result = _finalize(fixture)
    assert result["result"]["content"]["status"] == "failed"
    assert "PID was reused" in result["result"]["content"]["failure_reason"]


@pytest.mark.parametrize(
    "termination",
    [
        {"kind": "normal_exit", "exit_code": 7},
        {"kind": "signal", "signal": 9},
    ],
)
def test_nonzero_or_signal_exit_cannot_pass(tmp_path, termination):
    fixture = _fixture(tmp_path)
    _rewrite_exit(fixture, lambda content: content.__setitem__("termination", termination))
    result = _finalize(fixture)
    content = result["result"]["content"]
    assert content["status"] == "failed" and content["unlock_authorized"] is False
    assert content["automatic_retry_authorized"] is False
    assert content["terminal_evidence"]["exit_receipt"]["termination"] == termination


def test_fatal_log_marker_cannot_pass(tmp_path):
    fixture = _fixture(tmp_path)
    with fixture["log_path"].open("a", encoding="utf-8") as stream:
        stream.write("Traceback (most recent call last):\n")
    result = _finalize(fixture)
    assert "fatal markers" in result["result"]["content"]["failure_reason"]


def test_missing_model_cannot_pass(tmp_path):
    fixture = _fixture(tmp_path)
    next(fixture["hard_path"].parent.parent.glob("model_1.pt")).unlink()
    result = _finalize(fixture)
    assert result["result"]["content"]["status"] == "failed"
    assert "checkpoint is missing" in result["result"]["content"]["failure_reason"]


def test_nan_checkpoint_cannot_pass(tmp_path):
    fixture = _fixture(tmp_path)
    fixture["checkpoint"]["model_state_dict"]["weight"] = FakeTensor([float("nan")])
    result = _finalize(fixture)
    assert "non-finite" in result["result"]["content"]["failure_reason"]


def test_embedded_iteration_mismatch_cannot_pass(tmp_path):
    fixture = _fixture(tmp_path)
    fixture["checkpoint"]["iter"] = 0
    result = _finalize(fixture)
    assert "filename iteration differs" in result["result"]["content"]["failure_reason"]


def test_contract_sha_mismatch_cannot_pass(tmp_path):
    fixture = _fixture(tmp_path)
    fixture["checkpoint"]["infos"]["training_contract_sha256"] = "0" * 64
    result = _finalize(fixture)
    assert "hard-contract SHA" in result["result"]["content"]["failure_reason"]


def test_claim_binding_mismatch_cannot_pass(tmp_path):
    fixture = _fixture(tmp_path)
    _rewrite_exit(
        fixture,
        lambda content: content.__setitem__("claim_content_sha256", "0" * 64),
    )
    result = _finalize(fixture)
    assert "claim_content_sha256 differs" in result["result"]["content"]["failure_reason"]


def test_terminal_source_drift_cannot_pass(tmp_path):
    fixture = _fixture(tmp_path)
    result = _finalize(
        fixture,
        source_verifier=lambda _path, commit: {"head": commit, "clean": False},
    )
    assert "exact clean source" in result["result"]["content"]["failure_reason"]


def test_ignored_source_asset_receipt_drift_cannot_pass(tmp_path):
    fixture = _fixture(tmp_path)
    receipt = json.loads(fixture["source_asset_receipt"].read_text(encoding="utf-8"))
    receipt["content"]["inventory"]["file_count"] -= 1
    receipt["content_sha256"] = R.canonical_sha256(receipt["content"])
    _write_json(fixture["source_asset_receipt"], receipt)
    result = _finalize(fixture)
    assert "inventory mismatch" in result["result"]["content"]["failure_reason"]


def test_causal_lineage_cannot_unlock_fresh_probe(tmp_path):
    fixture = _fixture(tmp_path)
    fixture["checkpoint"]["infos"]["training_contract_lineage_exact"] = 0
    result = _finalize(fixture)
    assert "lineage must equal 1" in result["result"]["content"]["failure_reason"]


def test_current_queue_claim_drift_refuses_without_freezing_result(tmp_path):
    fixture = _fixture(tmp_path)
    with pytest.raises(P.FullSceneProbeError, match="selected queue row differs"):
        P.finalize(
            fixture["run_dir"],
            expected_claim_digest="0" * 64,
            source_asset_receipt=fixture["source_asset_receipt"],
            checkpoint_loader=lambda _path: fixture["checkpoint"],
            torch_module=FakeTorch,
            proc_root=fixture["proc_root"],
            getpgid=lambda pid: fixture["pgids"].get(pid, pid),
            source_verifier=lambda _path, commit: {"head": commit, "clean": True},
        )
    assert not (fixture["run_dir"] / P.RESULT_NAME).exists()


def test_corrupt_checkpoint_loader_becomes_auditable_terminal_failure(tmp_path):
    fixture = _fixture(tmp_path)

    def corrupt(_path):
        raise RuntimeError("corrupt checkpoint archive")

    result = P.finalize(
        fixture["run_dir"],
        expected_claim_digest=fixture["claim_digest"],
        source_asset_receipt=fixture["source_asset_receipt"],
        checkpoint_loader=corrupt,
        torch_module=FakeTorch,
        proc_root=fixture["proc_root"],
        getpgid=lambda pid: fixture["pgids"].get(pid, pid),
        source_verifier=lambda _path, commit: {"head": commit, "clean": True},
    )
    assert result["result"]["content"]["status"] == "failed"
    assert "corrupt checkpoint archive" in result["result"]["content"]["failure_reason"]


@pytest.mark.parametrize("marker", ["NaN", "Inf", "Killed"])
def test_additional_fatal_markers_cannot_pass(tmp_path, marker):
    fixture = _fixture(tmp_path)
    with fixture["log_path"].open("a", encoding="utf-8") as stream:
        stream.write(marker + "\n")
    result = _finalize(fixture)
    assert "fatal markers" in result["result"]["content"]["failure_reason"]


def test_existing_result_rejects_different_recomputed_bytes(tmp_path):
    fixture = _fixture(tmp_path)
    _finalize(fixture)
    with fixture["log_path"].open("a", encoding="utf-8") as stream:
        stream.write("Traceback (most recent call last):\n")
    with pytest.raises(P.FullSceneProbeError, match="different bytes"):
        _finalize(fixture)


def test_supervisor_and_finalizer_source_have_no_signal_operation():
    source = (SCRIPTS / "full_scene_probe_runtime.py").read_text(encoding="utf-8")
    assert "os.kill" not in source
    assert ".kill(" not in source
    assert ".terminate(" not in source
    assert "automatic_retry_authorized" in source


def test_ordinary_attestor_refuses_probe_binding(tmp_path):
    fixture = _fixture(tmp_path, live=True)
    with pytest.raises(R.LeanQueueRuntimeError, match="refuses non-science"):
        R.attest_milestone(
            fixture["run_dir"] / R.PROBE_BINDING_NAME,
            1,
            checkpoint_loader=lambda _path: fixture["checkpoint"],
            torch_module=FakeTorch,
            proc_root=fixture["proc_root"],
            getpgid=lambda pid: fixture["pgids"][pid],
        )
