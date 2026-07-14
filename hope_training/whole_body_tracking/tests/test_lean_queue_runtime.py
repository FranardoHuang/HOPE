"""Dependency-light attacks for lean-queue run binding and milestones."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/lean_queue_runtime.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "lean_queue_runtime_under_test", MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R = _load_module()


class FakeTensor:
    def __init__(self, values, *, floating=True, complex_value=False):
        self.values = list(values)
        self.floating = floating
        self.complex_value = complex_value

    def numel(self):
        return len(self.values)


class _FakeScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class _FakeMask:
    def __init__(self, values):
        self.values = values

    def sum(self):
        return _FakeScalar(sum(self.values))


class FakeTorch:
    Tensor = FakeTensor

    @staticmethod
    def is_floating_point(value):
        return value.floating

    @staticmethod
    def is_complex(value):
        return value.complex_value

    @staticmethod
    def isfinite(value):
        return _FakeMask(
            [
                math.isfinite(item.real) and math.isfinite(item.imag)
                if isinstance(item, complex)
                else math.isfinite(item)
                for item in value.values
            ]
        )


def _write_proc(proc_root: Path, pid: int, starttime: int, argv: list[str]) -> None:
    root = proc_root / str(pid)
    root.mkdir(parents=True, exist_ok=True)
    # Fields after ``comm`` begin with field 3 (state); field 22 starttime is index 19.
    rest = ["S", *(["0"] * 18), str(starttime)]
    (root / "stat").write_text(
        f"{pid} (python queue trainer) " + " ".join(rest) + "\n",
        encoding="utf-8",
    )
    (root / "cmdline").write_bytes(b"\0".join(item.encode() for item in argv) + b"\0")


def _fixture(tmp_path: Path, name: str = "case") -> dict:
    root = tmp_path / name
    source = root / "source"
    train = source / R.TRAIN_ENTRY_RELATIVE
    train.parent.mkdir(parents=True)
    train.write_text("# exact train entry\n", encoding="utf-8")
    run_dir = root / "runs/arm"
    (run_dir / R.MILESTONE_DIR_NAME).mkdir(parents=True)
    claim_path = run_dir / R.CLAIM_NAME
    binding_path = run_dir / R.BINDING_NAME
    run_name = "queue_arm_seed0"
    log_dir = (
        source
        / R.WBT_RELATIVE
        / "logs/rsl_rl/experiment"
        / f"2026-07-14_12-34-56_{run_name}"
    )
    log_dir.mkdir(parents=True)
    argv_without_claim = [
        "/exact/python",
        str(train),
        "task=Task",
        "algo=ppo",
        f"run_name={run_name}",
        "device=cuda:0",
        f"++training_queue_claim_path={claim_path}",
        f"++training_run_binding_path={binding_path}",
    ]
    content = {
        "schema_version": 1,
        "job_id": "job0",
        "action": "action0",
        "pod": "pod1",
        "gpu": 0,
        "source": {"checkout": str(source), "commit": "a" * 40},
        "run_name": run_name,
        "run_dir": str(run_dir),
        "seed": 0,
        "budget": {
            "num_envs": 512,
            # RSL's fresh loop emits model_N only when max_iterations > N.
            "max_iterations": 1001,
            "save_interval": 100,
            "milestones": [200, 500, 1000],
        },
        "inputs": {},
        "training_argv_without_claim": argv_without_claim,
    }
    claim_digest = R.canonical_sha256(content)
    full_argv = [
        *argv_without_claim,
        f"++training_launch_claim_sha256={claim_digest}",
    ]
    claim = {
        "schema_version": 2,
        "content": content,
        "content_sha256": claim_digest,
        "training_argv": full_argv,
    }
    claim_path.write_text(
        json.dumps(claim, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    proc_root = root / "proc"
    pid = 43210
    starttime = 987654
    _write_proc(proc_root, pid, starttime, full_argv)
    return {
        "source": source,
        "run_dir": run_dir,
        "claim_path": claim_path,
        "binding_path": binding_path,
        "log_dir": log_dir,
        "claim_digest": claim_digest,
        "full_argv": full_argv,
        "proc_root": proc_root,
        "pid": pid,
        "starttime": starttime,
    }


def _publish(fixture: dict):
    return R.publish_run_binding(
        claim_path=fixture["claim_path"],
        binding_path=fixture["binding_path"],
        log_dir=fixture["log_dir"],
        claim_digest=fixture["claim_digest"],
        actual_argv=fixture["full_argv"],
        pid=fixture["pid"],
        proc_root=fixture["proc_root"],
        getpgid=lambda _pid: fixture["pid"],
        environ={"CUDA_VISIBLE_DEVICES": "0"},
        source_verifier=lambda _source, commit: {"head": commit, "clean": True},
    )


def _checkpoint(fixture: dict, *, embedded=200, values=(1.0, 2.0)):
    hard = {"schema_version": 3, "contract": "exact"}
    hard_path = fixture["log_dir"] / "params/training_contract.json"
    hard_path.parent.mkdir(parents=True)
    hard_path.write_text(
        json.dumps(hard, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    hard_sha = hashlib.sha256(hard_path.read_bytes()).hexdigest()
    path = fixture["log_dir"] / "model_200.pt"
    path.write_bytes(b"stable checkpoint fixture")
    value = {
        "iter": embedded,
        "model_state_dict": {"weight": FakeTensor(values)},
        "infos": {
            "training_contract_schema_version": 3,
            "training_contract_sha256": hard_sha,
            "training_contract_lineage_exact": 1,
            "training_launch_claim_sha256": fixture["claim_digest"],
        },
    }
    return path, value


def _attest(fixture: dict, checkpoint: dict):
    return R.attest_milestone(
        fixture["binding_path"],
        200,
        checkpoint_loader=lambda _path: checkpoint,
        torch_module=FakeTorch,
        proc_root=fixture["proc_root"],
        getpgid=lambda _pid: fixture["pid"],
    )


def test_binding_rejects_fake_log_dir_and_is_atomic_no_clobber(tmp_path):
    fixture = _fixture(tmp_path)
    fake = tmp_path / "foreign/experiment/2026-07-14_12-34-56_queue_arm_seed0"
    with pytest.raises(R.LeanQueueRuntimeError, match="outside the source-owned log root"):
        R.publish_run_binding(
            claim_path=fixture["claim_path"],
            binding_path=fixture["binding_path"],
            log_dir=fake,
            claim_digest=fixture["claim_digest"],
            actual_argv=fixture["full_argv"],
            pid=fixture["pid"],
            proc_root=fixture["proc_root"],
            getpgid=lambda _pid: fixture["pid"],
            environ={"CUDA_VISIBLE_DEVICES": "0"},
            source_verifier=lambda _source, commit: {"head": commit, "clean": True},
        )
    binding = _publish(fixture)
    assert binding["content"]["rsl_log_dir"] == str(fixture["log_dir"])
    assert binding["content"]["process"] == {
        "pid": fixture["pid"],
        "pgid": fixture["pid"],
        "starttime_ticks": fixture["starttime"],
        "argv": fixture["full_argv"],
    }
    frozen = fixture["binding_path"].read_bytes()
    with pytest.raises(R.LeanQueueRuntimeError, match="overwrite is forbidden"):
        _publish(fixture)
    assert fixture["binding_path"].read_bytes() == frozen


def test_attestor_rejects_pid_reuse_before_reading_checkpoint(tmp_path):
    fixture = _fixture(tmp_path)
    _publish(fixture)
    _path, checkpoint = _checkpoint(fixture)
    _write_proc(
        fixture["proc_root"],
        fixture["pid"],
        fixture["starttime"] + 1,
        fixture["full_argv"],
    )
    with pytest.raises(R.LeanQueueRuntimeError, match="PID was reused"):
        _attest(fixture, checkpoint)
    assert not (fixture["run_dir"] / "milestones/model_200.json").exists()


def test_process_identity_rejects_pid_reuse_during_one_read():
    def stat_text(starttime):
        return "7 (trainer) " + " ".join(["S", *(["0"] * 18), str(starttime)])

    class StatFile:
        calls = 0

        def read_text(self, **_kwargs):
            self.calls += 1
            return stat_text(100 if self.calls == 1 else 101)

    class CmdlineFile:
        def read_bytes(self):
            return b"python\0train.py\0"

    stat_file = StatFile()

    class ProcDir:
        def __truediv__(self, name):
            return stat_file if name == "stat" else CmdlineFile()

    class ProcRoot:
        def __truediv__(self, _pid):
            return ProcDir()

    with pytest.raises(R.LeanQueueRuntimeError, match="changed while reading identity"):
        R._process_identity(7, proc_root=ProcRoot(), getpgid=lambda _pid: 7)


def test_attestor_rejects_filename_embedded_iteration_mismatch(tmp_path):
    fixture = _fixture(tmp_path)
    _publish(fixture)
    _path, checkpoint = _checkpoint(fixture, embedded=199)
    with pytest.raises(R.LeanQueueRuntimeError, match="filename iteration differs"):
        _attest(fixture, checkpoint)


def test_attestor_rejects_nan_in_any_nested_floating_tensor(tmp_path):
    fixture = _fixture(tmp_path)
    _publish(fixture)
    _path, checkpoint = _checkpoint(fixture, values=(1.0, float("nan")))
    with pytest.raises(R.LeanQueueRuntimeError, match="non-finite floating tensors"):
        _attest(fixture, checkpoint)


def test_attestor_rejects_nan_in_nested_complex_tensor(tmp_path):
    fixture = _fixture(tmp_path)
    _publish(fixture)
    _path, checkpoint = _checkpoint(fixture)
    checkpoint["complex"] = FakeTensor(
        [complex(float("nan"), 0.0)], floating=False, complex_value=True
    )
    with pytest.raises(R.LeanQueueRuntimeError, match="non-finite floating tensors"):
        _attest(fixture, checkpoint)


def test_source_verifier_rejects_drift_after_launcher_preflight(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(source), "config", "user.name", "Test"], check=True)
    tracked = source / "tracked.txt"
    tracked.write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(source), "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", str(source), "commit", "-qm", "fixture"], check=True)
    commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    assert R._verify_git_source(source, commit) == {"head": commit, "clean": True}
    tracked.write_text("dirty\n", encoding="utf-8")
    with pytest.raises(R.LeanQueueRuntimeError, match="dirty at binding time"):
        R._verify_git_source(source, commit)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("training_contract_sha256", "0" * 64, "hard-contract SHA"),
        ("training_launch_claim_sha256", "1" * 64, "launch-claim lineage"),
    ],
)
def test_attestor_rejects_hard_contract_or_claim_lineage_mismatch(
    tmp_path, field, value, message
):
    fixture = _fixture(tmp_path)
    _publish(fixture)
    _path, checkpoint = _checkpoint(fixture)
    checkpoint["infos"][field] = value
    with pytest.raises(R.LeanQueueRuntimeError, match=message):
        _attest(fixture, checkpoint)


def test_attestor_writes_one_immutable_exact_lineage_receipt(tmp_path):
    fixture = _fixture(tmp_path)
    binding = _publish(fixture)
    checkpoint_path, checkpoint = _checkpoint(fixture)
    result = _attest(fixture, checkpoint)
    receipt_path = Path(result["receipt_path"])
    assert receipt_path == fixture["run_dir"] / "milestones/model_200.json"
    receipt = result["receipt"]
    assert receipt["content_sha256"] == R.canonical_sha256(receipt["content"])
    content = receipt["content"]
    assert content["binding_content_sha256"] == binding["content_sha256"]
    assert content["claim_content_sha256"] == fixture["claim_digest"]
    assert content["checkpoint"]["path"] == str(checkpoint_path)
    assert content["checkpoint"]["embedded_iteration"] == 200
    assert content["checkpoint"]["nonfinite_floating_elements"] == 0
    assert content["hard_contract"]["lineage_exact"] == 1
    frozen = receipt_path.read_bytes()
    with pytest.raises(R.LeanQueueRuntimeError, match="overwrite is forbidden"):
        _attest(fixture, checkpoint)
    assert receipt_path.read_bytes() == frozen


def test_train_publishes_binding_after_exact_log_selection_before_kit_env(tmp_path):
    source = (ROOT / "scripts/train.py").read_text(encoding="utf-8")
    log_selection = source.index("log_dir = os.path.join(log_root_path, log_dir)")
    publish = source.index("_publish_lean_queue_binding_if_requested(cfg, log_dir)")
    env_build = source.index("env = gym.make(task_id")
    assert log_selection < publish < env_build
    assert "_ORIGINAL_TRAINING_ARGV" in source
    assert 'pathlib.Path("/proc/self/cmdline")' in source
    assert "training_queue_claim_path and training_run_binding_path must be supplied together" in source
    assert source.index('_emit_lean_queue_phase(cfg, "scene_import_start")') < env_build
    assert env_build < source.index('_emit_lean_queue_phase(cfg, "scene_import_done")')
    assert '_emit_lean_queue_phase(cfg, "hydra_resolved")' in source
    assert '_emit_lean_queue_phase(cfg, "app_started")' in source
