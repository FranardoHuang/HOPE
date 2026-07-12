from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat

import pytest


ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "agi/a3_deploy_example/scripts/gate3_first_tick_state_bridge.py"


def _module():
    spec = importlib.util.spec_from_file_location("gate3_first_tick_state_bridge", BRIDGE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wire_abi_and_kernel_locked_whole_record(tmp_path: Path) -> None:
    m = _module()
    assert m.WIRE.size == 256
    assert m.SEQUENCE_OFFSET == 16
    path = tmp_path.resolve() / "native-state"
    state = m.NativeStateFile(str(path))
    try:
        state.update(
            base_pose_stamp_ns=100,
            base_twist_stamp_ns=101,
            racket_pose_stamp_ns=102,
            base_pose_receive_monotonic_ns=200,
            base_twist_receive_monotonic_ns=201,
            racket_pose_receive_monotonic_ns=202,
            base_pose_receive_system_ns=300,
            base_twist_receive_system_ns=301,
            racket_pose_receive_system_ns=302,
            base_position_world=(0.0, 0.0, 1.0),
            base_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
            base_linear_velocity_world=(0.125, -0.25, 0.375),
            base_angular_velocity_world=(0.01, 0.02, 0.03),
            racket_position_world=(0.7, -0.4, 0.9),
            racket_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        )
        fd = os.open(path, os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_SH)
            raw = os.pread(fd, m.WIRE.size, 0)
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
        unpacked = m.WIRE.unpack(raw)
        assert unpacked[0] == m.MAGIC
        assert unpacked[1] == m.VERSION
        assert unpacked[2] == m.WIRE.size
        assert unpacked[3] > 0 and unpacked[3] % 2 == 0
        # Header (4) + nine int64 timestamps => base position starts at 13;
        # 3 pos + 4 quat => native base linear velocity starts at 20.
        assert unpacked[20:23] == (0.125, -0.25, 0.375)
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        with pytest.raises(ValueError, match="did not advance"):
            state.update(base_pose_stamp_ns=99)
    finally:
        state.close()
    assert not path.exists()


def test_bridge_is_exclusive_and_rejects_symlink_components(tmp_path: Path) -> None:
    m = _module()
    target = tmp_path.resolve() / "state"
    target.write_bytes(b"owned")
    with pytest.raises(FileExistsError):
        m.NativeStateFile(str(target))

    real = tmp_path.resolve() / "real"
    real.mkdir()
    link = tmp_path.resolve() / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="canonical|symlink"):
        m.NativeStateFile(str(link / "state"))


def test_bridge_source_is_subscription_only_and_no_velocity_fallback() -> None:
    source = BRIDGE.read_text()
    assert "create_publisher" not in source
    assert "/sim/a3/reset" not in source
    assert "base_linear_velocity_world=_finite" in source
    assert "message.twist.linear.x" in source
    assert "LOCK_EX" in source and "os.pwrite" in source
    assert "atexit.register(output.close)" in source
    assert "numpy.gradient" not in source
    assert "previous_position" not in source
    assert "base_linear_velocity_world=(0.0" not in source


def test_formal_capture_gate_waits_for_planner_owned_target() -> None:
    policy = (
        ROOT
        / "agi/a3_deploy_example/src/a3/a3_deploy_onnx_ref/include/a3_pingpong/pp_policy.hpp"
    ).read_text()
    main = (
        ROOT
        / "agi/a3_deploy_example/src/a3/a3_deploy_onnx_ref/src/a3_deploy/a3_pingpong_main.cpp"
    ).read_text()
    assert "PlannerActorCandidateEligible" in policy
    assert "planner_engaged_" in policy and "planner_have_hold_" in policy
    assert "if (ppp->CopyFirstTickCompute(capture))" in main
    assert "first_tick_requested && first_tick_output.empty()" in main
    assert "idle/waiting/recovery actor callback" in main
    assert "the first actor compute snapshot is unavailable" not in main


def test_source_contract_hashes_bytes_and_keeps_runtime_null() -> None:
    contract = json.loads(
        (ROOT / "configs/gate3_first_tick_source_contract_20260712.json").read_text()
    )
    required_reviewed_subset = {
        "production_build_definition",
        "production_runtime_config",
        "production_policy_capture",
        "production_onnx_loader",
        "production_sha256",
        "robot_state_interface",
        "robot_state_synchronizer",
        "production_policy_driver",
        "native_state_bridge",
        "native_json_contract",
        "production_runner_entrypoint",
    }
    assert required_reviewed_subset <= set(contract["tracked_artifacts"])
    exactness = contract["exactness"]
    assert exactness["evaluation_contract_exact"] is False
    assert exactness["planner_snapshot_exact"] is False
    assert exactness["native_sample_alignment_exact"] is False
    assert exactness["source_binary_binding_exact"] is False
    assert exactness["source_semantics_closure_exact"] is False
    assert exactness["runtime_artifact_closure_exact"] is False
    assert exactness["inexact_reasons"]
    for artifact in contract["tracked_artifacts"].values():
        data = (ROOT / artifact["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == artifact["sha256"]
    bindings = contract["runtime_bindings"]
    assert bindings["runtime_exact"] is False
    for key, value in bindings.items():
        if key != "runtime_exact":
            assert value is None
    assert contract["verification"]["vendor_first_tick"] == "not_run"
    assert contract["verification"]["gate3_gate3b"] == "not_run"
