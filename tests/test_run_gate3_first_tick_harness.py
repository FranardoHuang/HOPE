from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = ROOT / "scripts" / "run_gate3_first_tick_harness.py"
AUDIT_PATH = ROOT / "configs" / "gate3_legacy_process_audit_20260712.json"
LEGACY_SHELL = ROOT / "agi" / "a3_deploy_example" / "scripts" / "pp_gate3_rally.sh"
LEGACY_CONDUCTOR = ROOT / "agi" / "a3_deploy_example" / "scripts" / "pp_rally_conductor.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


H = _load_module("gate3_first_tick_harness_under_test", HARNESS_PATH)


def _write(path: Path, data: bytes, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    if executable:
        path.chmod(0o755)


def _artifact(path: Path, *, executable: bool) -> dict:
    return {"path": str(path), "sha256": H.sha256_file(path), "executable": executable}


def _contract(tmp_path: Path) -> dict:
    assets = tmp_path / "assets"
    bin_dir = tmp_path / "bin"
    work = tmp_path / "work"
    env_dir = tmp_path / "env"
    ledger = tmp_path / "ledger"
    train = tmp_path / "train"
    evaluation = tmp_path / "eval"
    for path in (work, env_dir, ledger, train, evaluation):
        path.mkdir(parents=True, exist_ok=True)
    paths = {
        "vendor_sim_binary": bin_dir / "vendor_sim",
        "vendor_sim_config": assets / "vendor_sim.yaml",
        "vendor_mjcf": assets / "a3_pingpong.xml",
        "planner_binary": bin_dir / "hope_planner_node",
        "planner_config": assets / "planner.yaml",
        "runner_binary": bin_dir / "a3_deploy_onnx_ref_pingpong",
        "runner_runtime_config": assets / "runner.yaml",
        "runner_model": assets / "policy.onnx",
        "kit_binary": bin_dir / "kit",
    }
    for name in H.EXECUTABLE_ARTIFACTS:
        _write(paths[name], f"#!/bin/sh\n# {name}\n".encode(), executable=True)
    _write(paths["vendor_mjcf"], b"<mujoco/>\n")
    _write(
        paths["vendor_sim_config"],
        f"mjcf: {paths['vendor_mjcf']}\n".encode(),
    )
    _write(paths["planner_config"], b"planner: exact\n")
    _write(paths["runner_runtime_config"], b"runner: exact\n")
    _write(paths["runner_model"], b"fake-onnx-for-contract-unit-test\n")
    artifacts = {
        name: _artifact(paths[name], executable=name in H.EXECUTABLE_ARTIFACTS)
        for name in H.ARTIFACT_KEYS
    }
    env = {
        "PATH": str(env_dir),
        "LD_LIBRARY_PATH": str(env_dir),
        "PYTHONPATH": str(env_dir),
        "AMENT_PREFIX_PATH": str(env_dir),
        "HOME": str(env_dir),
        "LANG": "C.UTF-8",
        "RMW_IMPLEMENTATION": "rmw_fastrtps_cpp",
        "ROS_DOMAIN_ID": "217",
        "ROS_LOCALHOST_ONLY": "1",
        "A3_SOURCE_ROBOT_ENV": "0",
        "A3_HARDWARE_ALLOWED": "0",
        "A3_TRANSPORT": "iceoryx",
        "MUJOCO_GL": "egl",
    }
    return {
        "schema_version": 1,
        "contract_id": "unit-test-gate3-first-tick-v1",
        "created_utc": "2026-07-12T02:00:00Z",
        "status": "preregistered_not_run",
        "scope": "vendor_gate3_first_tick_no_publish_only",
        "source_commit": "a" * 40,
        "harness_sha256": H.sha256_file(HARNESS_PATH),
        "hardware_authorized": False,
        "artifacts": artifacts,
        "read_only_checkouts": {
            "training": {"path": str(train), "commit": "1" * 40},
            "evaluation": {"path": str(evaluation), "commit": "2" * 40},
        },
        "formal_loader": {
            "required": True,
            "required_output_substrings": [
                "[pp PREFLIGHT] accepted", "backend_not_initialized=true", "obs_dim=179"
            ],
            "forbidden_output_substrings": [
                "backend cfg", "A3AimrtBackend initialised", "backend started"
            ],
            "requires_no_publish": True,
        },
        "first_tick_evidence": {
            "runner_flag": H.FIRST_TICK_OUTPUT_FLAG,
            "output_placeholder": H.FIRST_TICK_OUTPUT_PLACEHOLDER,
            "schema_version": 1,
            "required_vector_lengths": H.FIRST_TICK_VECTOR_LENGTHS,
            "required_joint_count": 31,
            "qpos_layout": H.FIRST_TICK_QPOS_LAYOUT,
            "qvel_layout": H.FIRST_TICK_QVEL_LAYOUT,
            "pose_quaternion_order": H.FIRST_TICK_POSE_QUATERNION_ORDER,
            "target_frame": H.FIRST_TICK_TARGET_FRAME,
            "obs_contract": H.FIRST_TICK_OBS_CONTRACT,
            "required_target_fields": list(H.FIRST_TICK_TARGET_FIELDS),
            "require_all_finite": True,
        },
        "runtime": {
            "ros_domain_id": 217,
            "ledger_root": str(ledger),
            "lock_path": str(ledger / ".gate3_first_tick.lock"),
            "conflict_locks": {
                "kit": str(tmp_path / "kit.lock"),
                "vendor_sim": str(tmp_path / "sim.lock"),
                "planner": str(tmp_path / "planner.lock"),
                "runner": str(tmp_path / "runner.lock"),
            },
            "conflict_artifact_keys": list(H.CONFLICT_ARTIFACT_KEYS),
            "environment": env,
            "timeouts_s": {
                "formal_loader": 30,
                "vendor_sim_ready": 30,
                "planner_ready": 30,
                "runner_first_tick": 30,
                "term": 5,
                "kill": 2,
            },
            "readiness_substrings": {
                "vendor_sim": "vendor sim ready",
                "planner": "planner ready",
                "runner": H.FIRST_TICK_MARKER,
            },
            "commands": {
                "vendor_sim": {
                    "argv": [str(paths["vendor_sim_binary"]), "--config", str(paths["vendor_sim_config"])],
                    "cwd": str(work),
                },
                "planner": {
                    "argv": [str(paths["planner_binary"]), "--config", str(paths["planner_config"])],
                    "cwd": str(work),
                },
                "runner": {
                    "argv": [
                        str(paths["runner_binary"]),
                        "--runtime-cfg", str(paths["runner_runtime_config"]),
                        "--model-path", str(paths["runner_model"]),
                        "--planner", "--no-publish", "--start", "passive",
                        H.FIRST_TICK_OUTPUT_FLAG, H.FIRST_TICK_OUTPUT_PLACEHOLDER,
                    ],
                    "cwd": str(work),
                },
            },
            "transport_scope": "vendor_sim_only_no_hardware",
            "body_command_publish_allowed": False,
        },
        "activation": {
            "default_mode": "plan",
            "run_cli_arming_phrase": H.ARMING_PHRASE,
            "no_publish_required": True,
            "real_robot_authorized": False,
        },
        "decision_policy": copy.deepcopy(H.DECISION_POLICY),
        "engine_gap_diagnostic_ladder": copy.deepcopy(H.ENGINE_GAP_DIAGNOSTIC_LADDER),
        "ready_state_diagnostic": copy.deepcopy(H.READY_STATE_DIAGNOSTIC),
    }


@pytest.fixture
def contract_env(tmp_path: Path, monkeypatch):
    contract = _contract(tmp_path)
    monkeypatch.setattr(H, "current_source_commit", lambda: "a" * 40)
    monkeypatch.setattr(
        H,
        "validate_read_only_checkout",
        lambda name, spec: {"path": spec["path"], "commit": spec["commit"], "clean": True},
    )
    return contract


def test_valid_contract_builds_plan_without_process_or_hardware_authority(
    tmp_path: Path, contract_env: dict
):
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract_env), encoding="utf-8")
    plan = H.build_plan(path, H.sha256_file(path), contract_env, proc_root=tmp_path / "no-proc")
    assert plan["status"] == "validated_plan_no_process_started"
    assert plan["component_order"] == ["vendor_sim", "planner", "runner"]
    assert plan["formal_loader"]["required_before_components"] is True
    assert plan["formal_loader"]["argv"][-3:] == [
        "--planner", "--no-publish", "--model-preflight-only"
    ]
    assert plan["runtime"]["environment"]["A3_HARDWARE_ALLOWED"] == "0"
    assert set(plan["runtime"]["conflict_locks"]) == {
        "kit", "vendor_sim", "planner", "runner"
    }
    assert plan["actions"]["processes_started"] == []
    assert plan["activation"]["armed"] is False
    assert plan["decision_policy"] == H.DECISION_POLICY
    assert plan["decision_policy"]["isaac_role"] == "training_and_diagnostic_only"
    assert all(not row["inference_allowed"] for row in plan["engine_gap_diagnostic_ladder"])
    assert plan["ready_state_diagnostic"]["formal_result_allowed"] is False


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value.__setitem__("hardware_authorized", True),
        lambda value: value["runtime"].__setitem__("body_command_publish_allowed", True),
        lambda value: value["runtime"]["environment"].__setitem__("A3_HARDWARE_ALLOWED", "1"),
        lambda value: value["runtime"]["environment"].__setitem__("ROS_LOCALHOST_ONLY", "0"),
        lambda value: value["runtime"]["environment"].__setitem__("ROS_DOMAIN_ID", "7"),
        lambda value: value["activation"].__setitem__("real_robot_authorized", True),
        lambda value: value["decision_policy"].__setitem__("isaac_role", "promotion_arbiter"),
        lambda value: value["engine_gap_diagnostic_ladder"][0].__setitem__(
            "inference_allowed", True
        ),
        lambda value: value["ready_state_diagnostic"].__setitem__("formal_result_allowed", True),
    ),
)
def test_contract_rejects_hardware_publish_domain_or_unearned_inference(
    contract_env: dict, mutation
):
    mutation(contract_env)
    with pytest.raises(H.HarnessError):
        H.validate_contract(contract_env)


@pytest.mark.parametrize(
    "mutation,match",
    (
        (
            lambda value: value["runtime"]["commands"]["runner"]["argv"].remove("--no-publish"),
            "no-publish",
        ),
        (
            lambda value: value["runtime"]["commands"]["runner"]["argv"].__setitem__(
                value["runtime"]["commands"]["runner"]["argv"].index("passive"), "motion"
            ),
            "start passive",
        ),
        (
            lambda value: value["runtime"]["commands"]["runner"]["argv"].__setitem__(
                value["runtime"]["commands"]["runner"]["argv"].index(
                    H.FIRST_TICK_OUTPUT_PLACEHOLDER
                ),
                "/tmp/unbound.json",
            ),
            "unbound absolute",
        ),
        (
            lambda value: value["runtime"]["commands"]["runner"]["argv"].append(
                "--model-preflight-only"
            ),
            "first tick",
        ),
    ),
)
def test_runner_runtime_command_is_passive_no_publish_and_structured_trace_bound(
    contract_env: dict, mutation, match: str
):
    mutation(contract_env)
    with pytest.raises(H.HarnessError, match=match):
        H.validate_contract(contract_env)


def test_artifact_requires_absolute_regular_nonsymlink_and_sha(contract_env: dict, tmp_path: Path):
    contract_env["artifacts"]["runner_model"]["sha256"] = "0" * 64
    with pytest.raises(H.HarnessError, match="SHA mismatch"):
        H.validate_contract(contract_env)
    contract_env = _contract(tmp_path / "second")
    model = Path(contract_env["artifacts"]["runner_model"]["path"])
    link = model.with_name("policy-link.onnx")
    link.symlink_to(model)
    contract_env["artifacts"]["runner_model"] = {
        "path": str(link), "sha256": H.sha256_file(model), "executable": False
    }
    with pytest.raises(H.HarnessError, match="symlink"):
        H.validate_artifact("runner_model", contract_env["artifacts"]["runner_model"])


def test_vendor_config_must_reference_exact_bound_mjcf(contract_env: dict):
    sim_cfg = Path(contract_env["artifacts"]["vendor_sim_config"]["path"])
    sim_cfg.write_text("mjcf: some-relative-file.xml\n", encoding="utf-8")
    contract_env["artifacts"]["vendor_sim_config"]["sha256"] = H.sha256_file(sim_cfg)
    with pytest.raises(H.HarnessError, match="exact bound MJCF"):
        H.validate_contract(contract_env)


def test_existing_exact_lock_fails_instead_of_cleanup(tmp_path: Path, contract_env: dict):
    lock = Path(contract_env["runtime"]["conflict_locks"]["vendor_sim"])
    lock.write_text("foreign", encoding="utf-8")
    accepted = H.validate_contract(contract_env)
    with pytest.raises(H.HarnessError, match="lock conflict"):
        H.preflight_conflicts(accepted, proc_root=tmp_path / "no-proc")
    assert lock.read_text(encoding="utf-8") == "foreign"


def _proc_stat(pid: int, ppid: int, pgid: int, session: int, starttime: int, state: str = "S") -> str:
    rest = [state, str(ppid), str(pgid), str(session)] + ["0"] * 15 + [str(starttime)]
    assert len(rest) == 20
    return f"{pid} (unit helper) " + " ".join(rest) + "\n"


def _make_proc(
    proc_root: Path,
    pid: int,
    *,
    ppid: int,
    pgid: int,
    session: int,
    starttime: int,
    token: str,
    argv: list[str],
    executable: Path,
) -> None:
    base = proc_root / str(pid)
    base.mkdir(parents=True)
    (base / "stat").write_text(_proc_stat(pid, ppid, pgid, session, starttime), encoding="utf-8")
    (base / "cmdline").write_bytes(b"\0".join(value.encode() for value in argv) + b"\0")
    (base / "environ").write_bytes(
        f"{H.OWNERSHIP_ENV_KEY}={token}\0A3_HARDWARE_ALLOWED=0\0".encode()
    )
    (base / "exe").symlink_to(executable)


def test_process_identity_binds_starttime_cmdline_group_and_token(tmp_path: Path):
    proc_root = tmp_path / "proc"
    exe = tmp_path / "runner"
    _write(exe, b"runner", executable=True)
    token = "owned-token"
    _make_proc(
        proc_root,
        4321,
        ppid=100,
        pgid=4321,
        session=4321,
        starttime=98765,
        token=token,
        argv=[str(exe), "--no-publish"],
        executable=exe,
    )
    identity = H.read_process_identity(4321, token, proc_root=proc_root)
    assert identity.starttime_ticks == 98765
    assert identity.pgid == identity.session == identity.pid == 4321
    assert identity.cmdline == (str(exe), "--no-publish")
    (proc_root / "4321" / "environ").write_bytes(b"FOREIGN=1\0")
    with pytest.raises(H.HarnessError, match="ownership token"):
        H.read_process_identity(4321, token, proc_root=proc_root)


def test_exact_signal_only_after_double_group_identity_validation(tmp_path: Path, monkeypatch):
    proc_root = tmp_path / "proc"
    exe = tmp_path / "sim"
    _write(exe, b"sim", executable=True)
    token = "group-token"
    _make_proc(
        proc_root,
        5000,
        ppid=1,
        pgid=5000,
        session=5000,
        starttime=100,
        token=token,
        argv=[str(exe)],
        executable=exe,
    )
    _make_proc(
        proc_root,
        5001,
        ppid=5000,
        pgid=5000,
        session=5000,
        starttime=101,
        token=token,
        argv=[str(exe), "child"],
        executable=exe,
    )
    identity = H.read_process_identity(5000, token, proc_root=proc_root)
    sent = []
    monkeypatch.setattr(H.os, "killpg", lambda pgid, sig: sent.append((pgid, sig)))
    managed = H.ManagedProcess(
        role="vendor_sim",
        popen=SimpleNamespace(),
        identity=identity,
        stdout_path=tmp_path / "stdout",
        stderr_path=tmp_path / "stderr",
        started_utc="now",
        cleanup=[],
    )
    H.exact_signal_owned_group(managed, H.signal.SIGTERM, proc_root=proc_root)
    assert sent == [(5000, H.signal.SIGTERM)]
    assert [row["pid"] for row in managed.cleanup[0]["validated_members"]] == [5000, 5001]
    (proc_root / "5001" / "cmdline").write_bytes(b"foreign\0")
    with pytest.raises(H.HarnessError, match="changed"):
        H.exact_signal_owned_group(managed, H.signal.SIGKILL, proc_root=proc_root)


def test_exact_conflict_scan_does_not_fuzzy_match_and_never_signals(tmp_path: Path):
    proc_root = tmp_path / "proc"
    target = tmp_path / "runner"
    other = tmp_path / "runner-helper"
    _write(target, b"runner", executable=True)
    _write(other, b"other", executable=True)
    _make_proc(
        proc_root,
        7000,
        ppid=1,
        pgid=7000,
        session=7000,
        starttime=200,
        token="not-relevant",
        argv=[str(other), f"prefix-{target.name}"],
        executable=other,
    )
    scan = H.scan_exact_process_conflicts({"runner": str(target)}, proc_root=proc_root)
    assert scan == {"supported": True, "conflicts": []}
    (proc_root / "7000" / "cmdline").write_bytes(str(target).encode() + b"\0")
    scan = H.scan_exact_process_conflicts({"runner": str(target)}, proc_root=proc_root)
    assert scan["conflicts"][0]["pid"] == 7000


def _first_tick() -> dict:
    return {
        "schema_version": 1,
        "source": "production_runner_first_tick",
        "tick": 0,
        "joint_names": [f"joint_{index:02d}" for index in range(31)],
        "qpos_layout": H.FIRST_TICK_QPOS_LAYOUT,
        "qvel_layout": H.FIRST_TICK_QVEL_LAYOUT,
        "pose_quaternion_order": H.FIRST_TICK_POSE_QUATERNION_ORDER,
        "target_frame": H.FIRST_TICK_TARGET_FRAME,
        "obs_contract": H.FIRST_TICK_OBS_CONTRACT,
        "qpos": [0.0] * 38,
        "qvel": [0.0] * 37,
        "base_pose": [0.0, 0.0, 1.0684, 1.0, 0.0, 0.0, 0.0],
        "racket_pose": [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        "target": {
            "position": [1.0, 0.0, 1.0],
            "velocity": [0.0, 0.0, 0.0],
            "normal": [1.0, 0.0, 0.0],
            "rho": 0.0,
            "time_to_strike": 0.0,
            "swing_type": 0,
            "valid": False,
        },
        "obs": [0.0] * 179,
    }


def test_first_tick_full_state_trace_requires_all_vectors_and_records_each_sha(tmp_path: Path):
    path = tmp_path / "first_tick.json"
    path.write_text(json.dumps(_first_tick()), encoding="utf-8")
    evidence = H.validate_first_tick_trace(path)
    assert evidence["vector_lengths"] == H.FIRST_TICK_VECTOR_LENGTHS
    assert set(evidence["per_field_canonical_sha256"]) == {
        "joint_names", "qpos", "qvel", "base_pose", "racket_pose", "target", "obs"
    }
    assert all(len(value) == 64 for value in evidence["per_field_canonical_sha256"].values())
    bad = _first_tick()
    bad["obs"].pop()
    path2 = tmp_path / "bad.json"
    path2.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(H.HarnessError, match="179"):
        H.validate_first_tick_trace(path2)


def test_no_clobber_plan_and_owned_lock(tmp_path: Path):
    output = tmp_path / "plan.json"
    H.atomic_json_no_clobber(output, {"plan": True})
    with pytest.raises(H.HarnessError, match="no-clobber"):
        H.atomic_json_no_clobber(output, {"plan": False})
    lock = tmp_path / "owned.lock"
    inode, _ = H.acquire_lock(lock, "abc")
    with pytest.raises(H.HarnessError, match="conflict"):
        H.acquire_lock(lock, "def")
    H.release_owned_lock(lock, "abc", inode)
    assert not lock.exists()


def test_run_requires_exact_arming_phrase_and_linux_proc(tmp_path: Path):
    plan = {
        "runtime": {"body_command_publish_allowed": False},
        "conflict_preflight": {"runtime_eligible_on_host": True},
    }
    with pytest.raises(H.HarnessError, match="arming phrase"):
        H.authorize_run("almost", plan)
    H.authorize_run(H.ARMING_PHRASE, plan)
    plan["conflict_preflight"]["runtime_eligible_on_host"] = False
    with pytest.raises(H.HarnessError, match="Linux /proc"):
        H.authorize_run(H.ARMING_PHRASE, plan)


def test_legacy_audit_binds_concrete_broad_process_and_boot_risks():
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    assert H.sha256_file(LEGACY_SHELL) == audit["legacy_shell"]["sha256"]
    assert H.sha256_file(LEGACY_CONDUCTOR) == audit["legacy_conductor"]["sha256"]
    risks = {row["id"]: row for row in audit["risks"]}
    assert len(risks) == 14
    assert risks["G3LEG-001"]["severity"] == "critical"
    assert risks["G3LEG-002"]["lines"] == [229, 230, 231, 311, 312, 313, 314, 315, 316, 317, 318]
    assert "post-loop assertion" in risks["G3LEG-012"]["evidence"]
    shell = LEGACY_SHELL.read_text(encoding="utf-8")
    conductor = LEGACY_CONDUCTOR.read_text(encoding="utf-8")
    assert shell.count("pkill -9") == 11
    assert '["pgrep", "-f", "hope_planner_node"]' in conductor


def test_new_harness_has_no_broad_search_signal_or_shell_execution():
    source = HARNESS_PATH.read_text(encoding="utf-8")
    for forbidden in ("pkill", "pgrep", "killall", "shell=True", "os.kill("):
        assert forbidden not in source
    assert "os.killpg(" in source
    assert "start_new_session=True" in source
    assert H.OWNERSHIP_ENV_KEY in source
    assert H.FIRST_TICK_OUTPUT_FLAG in source
