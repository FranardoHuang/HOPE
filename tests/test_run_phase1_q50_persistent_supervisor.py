from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_phase1_q50_persistent_supervisor.py"
PRODUCTION_CONFIG = (
    ROOT
    / "configs"
    / "phase1_fresh_SZ_model4000_seed_stability_q50_persistent_supervisor_20260713.json"
)
BOOT_ID = "12345678-1234-1234-1234-123456789abc"


def _load_module():
    spec = importlib.util.spec_from_file_location("q50_supervisor_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


S = _load_module()


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def _sha(path: Path) -> str:
    return S._sha256_file(path)


def _runner_source() -> str:
    return r'''import json
import time

if __name__ == "__main__":
    print("[fake-bound-runner] started", flush=True)
    time.sleep(0.35)
else:
    def load_execution_config(path):
        return json.loads(path.read_text())

    def _resolve_bound_sources(config_path, execution):
        return config_path, {}, config_path, {}

    def _validate_activation_document(path, expected_sha, queue_path, queue, prereg_path):
        return {"path": path, "sha256": expected_sha}

    def _validate_pod_result(path, expected_sha, execution, prereg, activation,
                             expected_config_sha, *, pod):
        document = json.loads(path.read_text())
        content = document["content"]
        if content.get("full_evidence") != "accepted_by_exact_bound_runner":
            raise RuntimeError("full evidence absent")
        if content.get("pod") != pod:
            raise RuntimeError("wrong pod")
        return content
'''


def _fixture(tmp_path: Path, *, suffix: str = "a") -> tuple[Path, dict]:
    source = tmp_path / f"source_{suffix}"
    runner = source / "scripts" / "run_phase1_fresh_sz_model4000_q50.py"
    runner.parent.mkdir(parents=True)
    runner.write_text(_runner_source(), encoding="utf-8")
    execution = source / "configs" / "execution.json"
    activation = source / "activation.json"
    runtime1 = source / "pod1.runtime.json"
    runtime2 = source / "pod2.runtime.json"
    _write(execution, {"contract_id": "test-model4000-q50"})
    _write(activation, {"content": {"barrier_id": "test-barrier"}})
    _write(runtime1, {"pod": "pod1", "status": "prepared_not_started"})
    _write(runtime2, {"pod": "pod2", "status": "prepared_not_started"})
    state_root = tmp_path / f"supervisor_state_{suffix}"
    result_root = tmp_path / f"result_state_{suffix}"
    python = {
        "path": sys.executable,
        "resolved_path": str(Path(sys.executable).resolve()),
        "sha256": _sha(Path(sys.executable)),
    }
    environment = {
        "HOME": str(tmp_path),
        "LANG": "C",
        "PATH": "/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "TMPDIR": str(tmp_path),
    }
    config = {
        "schema_version": 1,
        "contract_id": "test-q50-persistent-supervisor-v1",
        "status": "manual_launch_only",
        "auto_start": False,
        "real_robot_authorized": False,
        "supervisor_source": {"path": str(SCRIPT), "sha256": _sha(SCRIPT)},
        "environment": environment,
        "runner": {"path": str(runner), "sha256": _sha(runner)},
        "execution_config": {"path": str(execution), "sha256": _sha(execution)},
        "activation": {"path": str(activation), "sha256": _sha(activation)},
        "handshake": {
            "hello_timeout_seconds": 1,
            "commit_timeout_seconds": 1,
            "poll_seconds": 0.01,
        },
        "pods": {
            "pod1": {
                "launch_authorized": True,
                "blocker": "",
                "python": python,
                "runtime_contract": {"path": str(runtime1), "sha256": _sha(runtime1)},
                "state_dir": str(state_root / "pod1_v1"),
                "result_path": str(result_root / "pod1_result.json"),
                "arm_order": ["seed1", "seed3"],
            },
            "pod2": {
                "launch_authorized": True,
                "blocker": "",
                "python": python,
                "runtime_contract": {"path": str(runtime2), "sha256": _sha(runtime2)},
                "state_dir": str(state_root / "pod2_v1"),
                "result_path": str(result_root / "pod2_result.json"),
                "arm_order": ["seed2", "seed4"],
            },
        },
    }
    config_path = source / "configs" / "supervisor.json"
    _write(config_path, config)
    return config_path, S.load_supervisor_config(config_path, _sha(config_path))


def _identity_reader(config: dict, pod: str):
    binding = config["pods"][pod]
    state_dir = Path(binding["state_dir"])

    def read(pid: int) -> dict | None:
        try:
            pgid = os.getpgid(pid)
        except ProcessLookupError:
            return None
        acknowledged = (state_dir / "commit_ack.json").is_file()
        return {
            "pid": pid,
            "pgid": pgid,
            "state": "S",
            "start_ticks": pid * 1009,
            "cmdline": S._runner_argv(config, pod) if acknowledged else [],
            "executable_realpath": binding["python"]["resolved_path"],
            "executable_sha256": binding["python"]["sha256"],
            "environment_sha256": (
                S._canonical_sha256(config["environment"])
                if acknowledged
                else "0" * 64
            ),
        }

    return read


def _boot_id() -> str:
    return BOOT_ID


def _launch(config: dict, pod: str = "pod1", **kwargs):
    return S._launch_loaded(
        config,
        pod,
        identity_reader=_identity_reader(config, pod),
        boot_id_reader=_boot_id,
        **kwargs,
    )


def _wait_for(path: Path, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.02)
    pytest.fail(f"timed out waiting for {path}")


def _terminal_result(config: dict, pod: str, *, full: bool) -> dict:
    content = {
        "pod": pod,
        "runtime_contract": config["pods"][pod]["runtime_contract"],
    }
    if full:
        content["full_evidence"] = "accepted_by_exact_bound_runner"
    return {
        "schema_version": 1,
        "artifact_kind": "phase1_fresh_sz_model4000_q50_pod",
        "content_sha256": S._canonical_sha256(content),
        "content": content,
    }


def test_public_surface_has_only_launch_and_inspect_and_source_has_no_control_channel():
    parser = S._parser()
    choices = None
    for action in parser._actions:
        if getattr(action, "choices", None):
            choices = set(action.choices)
    assert choices == {"launch", "inspect"}
    source = SCRIPT.read_text(encoding="utf-8").lower()
    forbidden = (
        "subprocess",
        "os." + "kill",
        "killall",
        "signal.",
        "paramiko",
        "fabric.connection",
    )
    assert all(token not in source for token in forbidden)
    assert "os.fork()" in source
    assert "os.setsid()" in source
    assert "os.execve(" in source
    assert "os.closerange(" in source


def test_public_invocation_requires_the_exact_fixed_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _, config = _fixture(tmp_path)
    with pytest.raises(S.SupervisorError, match="invoking environment differs"):
        S._require_invoking_environment(config)
    monkeypatch.setattr(os, "environ", dict(config["environment"]))
    S._require_invoking_environment(config)


def test_launch_observes_exact_exec_and_duplicate_is_no_clobber(tmp_path: Path):
    _, config = _fixture(tmp_path)
    launched = _launch(config)
    state_dir = Path(config["pods"]["pod1"]["state_dir"])
    assert launched["status"] == "running_exact"
    assert launched["pid"] == launched["pgid"]
    for name in (
        "child_hello.json",
        "launch_ledger.json",
        "commit_token.json",
        "commit_ack.json",
        "runner.stdout_stderr.log",
    ):
        assert (state_dir / name).is_file()
    with pytest.raises(S.SupervisorError, match="no-clobber"):
        _launch(config)
    os.waitpid(launched["pid"], 0)


def test_preexisting_result_rejected_before_state_reservation(tmp_path: Path):
    _, config = _fixture(tmp_path)
    result = Path(config["pods"]["pod1"]["result_path"])
    _write(result, _terminal_result(config, "pod1", full=True))
    with pytest.raises(S.SupervisorError, match="pre-existing terminal result"):
        _launch(config)
    assert not Path(config["pods"]["pod1"]["state_dir"]).exists()


def test_inspect_checks_exe_environment_and_full_terminal_result(tmp_path: Path):
    _, config = _fixture(tmp_path)
    launched = _launch(config)
    live = S._inspect_loaded(
        config,
        "pod1",
        identity_reader=_identity_reader(config, "pod1"),
        boot_id_reader=_boot_id,
    )
    assert live["status"] == "running_exact"
    os.waitpid(launched["pid"], 0)
    result_path = Path(config["pods"]["pod1"]["result_path"])
    _write(result_path, _terminal_result(config, "pod1", full=True))
    terminal = S._inspect_loaded(
        config,
        "pod1",
        identity_reader=lambda _pid: None,
        boot_id_reader=_boot_id,
    )
    assert terminal["status"] == "terminal_result_validated"
    assert terminal["result"]["sha256"] == _sha(result_path)


def test_minimal_terminal_result_is_rejected_by_exact_bound_runner(tmp_path: Path):
    _, config = _fixture(tmp_path)
    launched = _launch(config)
    os.waitpid(launched["pid"], 0)
    result_path = Path(config["pods"]["pod1"]["result_path"])
    _write(result_path, _terminal_result(config, "pod1", full=False))
    with pytest.raises(S.SupervisorError, match="exact bound-runner"):
        S._inspect_loaded(
            config,
            "pod1",
            identity_reader=lambda _pid: None,
            boot_id_reader=_boot_id,
        )


@pytest.mark.parametrize("field", ("start_ticks", "executable_sha256", "environment_sha256"))
def test_inspect_rejects_reused_or_different_live_process(tmp_path: Path, field: str):
    _, config = _fixture(tmp_path, suffix=field)
    launched = _launch(config, "pod2")
    exact = _identity_reader(config, "pod2")

    def wrong(pid: int) -> dict | None:
        value = exact(pid)
        assert value is not None
        if field == "start_ticks":
            value[field] += 1
        else:
            value[field] = "f" * 64
        return value

    with pytest.raises(S.SupervisorError):
        S._inspect_loaded(
            config,
            "pod2",
            identity_reader=wrong,
            boot_id_reader=_boot_id,
        )
    os.waitpid(launched["pid"], 0)


def test_parent_exit_before_commit_leaves_no_token_and_child_times_out(tmp_path: Path):
    _, config = _fixture(tmp_path)
    outer = os.fork()
    if outer == 0:
        _launch(config, after_hello_hook=lambda: os._exit(91))
        os._exit(92)
    _, status = os.waitpid(outer, 0)
    assert os.WEXITSTATUS(status) == 91
    state_dir = Path(config["pods"]["pod1"]["state_dir"])
    _wait_for(state_dir / "child_exit.json")
    child_exit = json.loads((state_dir / "child_exit.json").read_text())
    assert child_exit["status"] == "commit_token_timeout"
    assert not (state_dir / "launch_ledger.json").exists()
    assert not (state_dir / "commit_token.json").exists()


def test_parent_stall_past_child_deadline_cannot_publish_ledger(tmp_path: Path):
    _, config = _fixture(tmp_path)
    with pytest.raises(S.SupervisorError):
        _launch(config, after_hello_hook=lambda: time.sleep(1.1))
    state_dir = Path(config["pods"]["pod1"]["state_dir"])
    _wait_for(state_dir / "child_exit.json")
    hello = json.loads((state_dir / "child_hello.json").read_text())
    try:
        os.waitpid(hello["pid"], 0)
    except ChildProcessError:
        pass
    assert not (state_dir / "launch_ledger.json").exists()
    assert not (state_dir / "commit_token.json").exists()


def test_rehash_delay_past_deadline_never_starts_bound_runner(tmp_path: Path):
    _, config = _fixture(tmp_path)
    with pytest.raises(S.SupervisorError, match="exact exec was not observed"):
        _launch(config, child_after_rehash_hook=lambda: time.sleep(1.1))
    state_dir = Path(config["pods"]["pod1"]["state_dir"])
    _wait_for(state_dir / "child_exit.json")
    hello = json.loads((state_dir / "child_hello.json").read_text())
    try:
        os.waitpid(hello["pid"], 0)
    except ChildProcessError:
        pass
    child_exit = json.loads((state_dir / "child_exit.json").read_text())
    assert child_exit["status"] == "child_setup_or_exec_failed"
    assert "after rehash before acknowledgment" in child_exit["error"]
    assert (state_dir / "launch_ledger.json").is_file()
    assert (state_dir / "commit_token.json").is_file()
    assert not (state_dir / "commit_ack.json").exists()
    assert not Path(config["pods"]["pod1"]["result_path"]).exists()
    log = (state_dir / "runner.stdout_stderr.log").read_text(encoding="utf-8")
    assert "[fake-bound-runner] started" not in log


def test_binding_mismatch_fails_before_state_reservation(tmp_path: Path):
    _, config = _fixture(tmp_path)
    runtime = Path(config["pods"]["pod1"]["runtime_contract"]["path"])
    runtime.write_text("changed\n", encoding="utf-8")
    with pytest.raises(S.SupervisorError, match="bytes changed"):
        _launch(config)
    assert not Path(config["pods"]["pod1"]["state_dir"]).exists()


def test_atomic_json_is_no_clobber(tmp_path: Path):
    path = tmp_path / "ledger.json"
    S._atomic_json_no_clobber(path, {"value": 1})
    with pytest.raises(S.SupervisorError, match="no-clobber"):
        S._atomic_json_no_clobber(path, {"value": 2})
    assert json.loads(path.read_text()) == {"value": 1}


def test_production_config_parser_rejects_duplicate_key_and_nonfinite(tmp_path: Path):
    original = PRODUCTION_CONFIG.read_text(encoding="utf-8")
    duplicate = tmp_path / "duplicate-production-config.json"
    duplicate.write_text(original.replace("{", '{"schema_version":1,', 1), encoding="utf-8")
    with pytest.raises(S.SupervisorError, match="duplicate JSON key"):
        S.load_supervisor_config(duplicate, _sha(duplicate))

    nonfinite = tmp_path / "nonfinite-production-config.json"
    nonfinite.write_text(original.replace("{", '{"poison":NaN,', 1), encoding="utf-8")
    with pytest.raises(S.SupervisorError, match="non-finite JSON constant"):
        S.load_supervisor_config(nonfinite, _sha(nonfinite))


@pytest.mark.parametrize("artifact", ("launch_ledger.json", "commit_token.json"))
def test_preserved_ledger_and_token_parser_reject_duplicate_keys(
    tmp_path: Path, artifact: str
):
    _, config = _fixture(tmp_path, suffix=artifact)
    launched = _launch(config)
    state_dir = Path(config["pods"]["pod1"]["state_dir"])
    path = state_dir / artifact
    raw = path.read_text(encoding="utf-8")
    path.write_text(raw.replace("{", '{"schema_version":1,', 1), encoding="utf-8")
    with pytest.raises(S.SupervisorError, match="duplicate JSON key"):
        S._validate_preserved_ledger(config, "pod1", state_dir)
    os.waitpid(launched["pid"], 0)


def test_exact_production_config_loads_through_checked_in_evidence_mirror():
    data = json.loads(PRODUCTION_CONFIG.read_text(encoding="utf-8"))
    remote_to_local = {
        data["runner"]["path"]: ROOT / "scripts" / "run_phase1_fresh_sz_model4000_q50.py",
        data["execution_config"]["path"]: (
            ROOT
            / "configs"
            / "phase1_fresh_SZ_model4000_seed_stability_q50_execution_20260712.json"
        ),
        data["activation"]["path"]: (
            ROOT
            / "configs"
            / "phase1_fresh_SZ_model4000_seed_stability_q50_activation_20260713.json"
        ),
    }
    loaded = S.load_supervisor_config(
        PRODUCTION_CONFIG,
        _sha(PRODUCTION_CONFIG),
        validation_path_resolver=lambda path: remote_to_local[str(path)],
    )
    assert data["supervisor_source"]["sha256"] == _sha(SCRIPT)
    assert loaded["runner"]["sha256"] == (
        "de0abff6096efdea8ce78dbac6f3115d09e70be8ca0fc841a36be2cdbfbf6b85"
    )
    assert loaded["execution_config"]["sha256"] == (
        "3109acd41726ef1a3063637e2a565cb2f4abe8992bb96473940700981e7c4385"
    )
    assert loaded["activation"]["sha256"] == (
        "9dea76c2a9039dc35f8f996fa112e0e28ee320cb9b7c7ec877be942e021ce704"
    )
    assert all(
        entry["python"]["sha256"]
        == "06630724486efc9d97db03c62949511584b896c110097153ef970f9294fd3ba0"
        for entry in loaded["pods"].values()
    )
    assert loaded["auto_start"] is False
    assert loaded["real_robot_authorized"] is False
