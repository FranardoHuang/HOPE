from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_ready_to_strike_motion.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_ready_to_strike_motion_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M = _load_module()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _motion_arrays(*, frames: int, ready: bool) -> dict[str, np.ndarray]:
    fps = 50
    dt = 1.0 / fps
    joints = 3
    bodies = 2
    time = np.arange(frames, dtype=np.float64) * dt
    if ready:
        joint_pos = np.repeat(
            np.array([[-0.12, 0.08, 0.03]], dtype=np.float32), frames, axis=0
        )
        body_pos = np.zeros((frames, bodies, 3), dtype=np.float32)
        body_pos[:, 0, 2] = 1.0
        body_pos[:, 1, 0] = 0.25
        body_pos[:, 1, 2] = 1.2
    else:
        joint_pos = np.stack(
            [
                -0.05 + 0.18 * time + 0.03 * time**2,
                0.02 - 0.10 * time + 0.02 * time**2,
                0.04 + 0.07 * time,
            ],
            axis=1,
        ).astype(np.float32)
        body_pos = np.zeros((frames, bodies, 3), dtype=np.float32)
        body_pos[:, 0, 0] = (0.02 * time).astype(np.float32)
        body_pos[:, 0, 2] = 1.0
        body_pos[:, 1, 0] = (0.25 + 0.10 * time).astype(np.float32)
        body_pos[:, 1, 1] = (-0.03 * time**2).astype(np.float32)
        body_pos[:, 1, 2] = 1.2
    joint_vel = np.gradient(joint_pos, dt, axis=0).astype(np.float32)
    body_quat = np.zeros((frames, bodies, 4), dtype=np.float32)
    body_quat[..., 0] = 1.0
    body_lin = np.gradient(body_pos, dt, axis=0).astype(np.float32)
    body_ang = np.zeros((frames, bodies, 3), dtype=np.float32)
    return {
        "fps": np.array([fps], dtype=np.int64),
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "body_pos_w": body_pos,
        "body_quat_w": body_quat,
        "body_lin_vel_w": body_lin,
        "body_ang_vel_w": body_ang,
        "kinematics_schema_version": np.array([2], dtype=np.int64),
        "body_pos_point": np.array("link_origin"),
        "body_lin_vel_point": np.array("center_of_mass"),
        "body_names": np.array(["pelvis_link", "racket_link"]),
    }


def _write_motion(path: Path, *, frames: int = 40, ready: bool = False) -> dict[str, np.ndarray]:
    arrays = _motion_arrays(frames=frames, ready=ready)
    with path.open("wb") as stream:
        np.savez(stream, **arrays)
    return arrays


def _command(source: Path, ready: Path, output: Path, contract: Path) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT),
        "--source",
        str(source),
        "--ready-source",
        str(ready),
        "--ready-frame",
        "0",
        "--contact-frame",
        "30",
        "--join-frame",
        "16",
        "--hold-frames",
        "4",
        "--blend-intervals",
        "8",
        "--output-npz",
        str(output),
        "--output-contract",
        str(contract),
    ]


def test_quintic_boundary_conditions_are_c2_exact() -> None:
    p0 = np.array([0.1, -0.2])
    v0 = np.array([0.0, 0.0])
    a0 = np.array([0.0, 0.0])
    p1 = np.array([0.7, 0.3])
    v1 = np.array([0.4, -0.1])
    a1 = np.array([-0.2, 0.5])
    p, v, a = M.quintic_hermite(p0, v0, a0, p1, v1, a1, 0.2, [0.0, 0.2])
    np.testing.assert_allclose(p[0], p0, rtol=0.0, atol=1.0e-13)
    np.testing.assert_allclose(v[0], v0, rtol=0.0, atol=1.0e-13)
    np.testing.assert_allclose(a[0], a0, rtol=0.0, atol=1.0e-13)
    np.testing.assert_allclose(p[-1], p1, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(v[-1], v1, rtol=0.0, atol=1.0e-11)
    np.testing.assert_allclose(a[-1], a1, rtol=0.0, atol=1.0e-10)


def test_build_preserves_ready_contact_window_and_velocity_continuity(tmp_path: Path) -> None:
    source_path = tmp_path / "source.npz"
    ready_path = tmp_path / "ready.npz"
    output_path = tmp_path / "candidate.npz"
    contract_path = tmp_path / "candidate.contract.json"
    source = _write_motion(source_path)
    ready = _write_motion(ready_path, frames=12, ready=True)

    completed = subprocess.run(
        _command(source_path, ready_path, output_path, contract_path),
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["training_authorized"] is False
    assert summary["contact_time_from_frame0_s"] == 0.5

    with np.load(output_path, allow_pickle=False) as archive:
        output = {key: np.asarray(archive[key]).copy() for key in archive.files}
    assert output["joint_pos"].shape[0] == 35
    assert np.array_equal(output["joint_pos"][0], ready["joint_pos"][0])
    assert np.array_equal(output["body_pos_w"][0], ready["body_pos_w"][0])
    assert np.array_equal(output["body_quat_w"][0], ready["body_quat_w"][0])
    assert np.array_equal(output["joint_vel"][:2], np.zeros_like(output["joint_vel"][:2]))
    assert np.array_equal(
        output["body_lin_vel_w"][:2], np.zeros_like(output["body_lin_vel_w"][:2])
    )

    # Source join frame 16 maps to output frame 11.  The quintic endpoint
    # matches source position/velocity, so there is no pose jump and only the
    # expected finite-grid derivative error at the discrete splice.
    output_join = 11
    assert np.array_equal(output["joint_pos"][output_join], source["joint_pos"][16])
    assert float(np.max(np.abs(output["joint_vel"][output_join] - source["joint_vel"][16]))) < 0.05
    assert float(
        np.max(np.abs(output["joint_pos"][output_join] - output["joint_pos"][output_join - 1]))
    ) < 0.02
    assert np.isfinite(output["joint_vel"][output_join - 1 : output_join + 2]).all()

    # Five 50 Hz frames before contact plus the contact row are byte-identical
    # for every schema-2 time channel.  No cropped or synthesized contact data
    # can pass this assertion.
    for key in M.SCHEMA2_TIME_KEYS:
        assert np.array_equal(output[key][20:26], source[key][25:31]), key
        assert np.array_equal(output[key][25], source[key][30]), key
    assert np.isfinite(output["joint_pos"]).all()
    assert np.isfinite(output["body_quat_w"]).all()
    expected_joint_vel = np.gradient(output["joint_pos"], 1.0 / 50.0, axis=0).astype(np.float32)
    assert np.array_equal(output["joint_vel"], expected_joint_vel)

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert contract["tool"]["sha256"] == _sha(SCRIPT)
    assert contract["tool"]["binding_semantics"] == (
        "source_file_snapshot_at_main_entry_unchanged_before_publish"
    )
    assert contract["output"]["npz"]["sha256"] == _sha(output_path)
    assert contract["proof"]["protected_window_bitwise_equal"] is True
    assert contract["proof"]["output_contact_frame"] == 25
    assert contract["synthesis"]["simple_crop"] is False
    assert contract["synthesis"]["production_fk_rebuild_required"] is True
    assert contract["authorization"] == {
        "host_candidate_materialized": True,
        "topp_runup_0p5_pass": False,
        "l0_static_pass": False,
        "vendor_l1_pass": False,
        "self_hit_pass": False,
        "table_net_clearance_5mm_pass": False,
        "dynamics_pass": False,
        "training_authorized": False,
        "deployment_authorized": False,
        "hardware_authorized": False,
    }


def test_publication_is_no_clobber_and_preserves_first_bundle(tmp_path: Path) -> None:
    source = tmp_path / "source.npz"
    ready = tmp_path / "ready.npz"
    output = tmp_path / "candidate.npz"
    contract = tmp_path / "candidate.json"
    _write_motion(source)
    _write_motion(ready, frames=12, ready=True)
    command = _command(source, ready, output, contract)
    first = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    assert first.returncode == 0, first.stderr
    before = output.read_bytes(), contract.read_bytes()
    second = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    assert second.returncode == 2
    assert "no-clobber" in second.stderr
    assert (output.read_bytes(), contract.read_bytes()) == before


def test_symlinked_input_and_output_parent_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "source.npz"
    ready = tmp_path / "ready.npz"
    _write_motion(source)
    _write_motion(ready, frames=12, ready=True)
    source_link = tmp_path / "source-link.npz"
    source_link.symlink_to(source)
    output = tmp_path / "candidate.npz"
    contract = tmp_path / "candidate.json"
    linked_input = subprocess.run(
        _command(source_link, ready, output, contract),
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert linked_input.returncode == 2
    assert "symlink" in linked_input.stderr
    assert not output.exists() and not contract.exists()

    real_parent = tmp_path / "real"
    real_parent.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real_parent, target_is_directory=True)
    linked_parent = subprocess.run(
        _command(source, ready, alias / "candidate.npz", alias / "candidate.json"),
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert linked_parent.returncode == 2
    assert "symlink" in linked_parent.stderr
    assert not (real_parent / "candidate.npz").exists()


def test_nonfinite_and_late_join_are_rejected_without_outputs(tmp_path: Path) -> None:
    source = tmp_path / "source.npz"
    ready = tmp_path / "ready.npz"
    output = tmp_path / "candidate.npz"
    contract = tmp_path / "candidate.json"
    arrays = _motion_arrays(frames=40, ready=False)
    arrays["joint_pos"][4, 1] = np.nan
    arrays["joint_vel"] = np.gradient(arrays["joint_pos"], 1.0 / 50.0, axis=0).astype(np.float32)
    with source.open("wb") as stream:
        np.savez(stream, **arrays)
    _write_motion(ready, frames=12, ready=True)
    nonfinite = subprocess.run(
        _command(source, ready, output, contract),
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert nonfinite.returncode == 2
    assert "NaN/Inf" in nonfinite.stderr
    assert not output.exists() and not contract.exists()

    _write_motion(source)
    command = _command(source, ready, output, contract)
    command[command.index("--join-frame") + 1] = "26"
    late = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    assert late.returncode == 2
    assert "protected window" in late.stderr
    assert not output.exists() and not contract.exists()


def test_ready_uses_frame0_pose_and_explicitly_zeros_source_velocity(tmp_path: Path) -> None:
    source_path = tmp_path / "source.npz"
    ready_path = tmp_path / "moving-ready.npz"
    output_path = tmp_path / "candidate.npz"
    contract_path = tmp_path / "candidate.json"
    _write_motion(source_path)
    _write_motion(ready_path, frames=12, ready=False)
    completed = subprocess.run(
        _command(source_path, ready_path, output_path, contract_path),
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    with np.load(output_path, allow_pickle=False) as archive:
        output = {key: np.asarray(archive[key]).copy() for key in archive.files}
    ready = _motion_arrays(frames=12, ready=False)
    assert np.array_equal(output["joint_pos"][0], ready["joint_pos"][0])
    assert np.array_equal(output["joint_vel"][:3], np.zeros_like(output["joint_vel"][:3]))
    assert np.array_equal(
        output["body_lin_vel_w"][:4], np.zeros_like(output["body_lin_vel_w"][:4])
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert contract["proof"]["ready_source_velocity_channels_ignored"] is True
    assert contract["proof"]["ready_velocity_definition"] == "explicit_bitwise_zero"


def test_nonzero_ready_frame_is_rejected(tmp_path: Path) -> None:
    source_path = tmp_path / "source.npz"
    ready_path = tmp_path / "ready.npz"
    output_path = tmp_path / "candidate.npz"
    contract_path = tmp_path / "candidate.json"
    _write_motion(source_path)
    _write_motion(ready_path, frames=12, ready=True)
    command = _command(source_path, ready_path, output_path, contract_path)
    command[command.index("--ready-frame") + 1] = "1"
    completed = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, check=False
    )
    assert completed.returncode == 2
    assert "frame 0" in completed.stderr
    assert not output_path.exists() and not contract_path.exists()


def test_quaternion_sign_at_join_matches_bitwise_source_suffix(tmp_path: Path) -> None:
    source_path = tmp_path / "source.npz"
    ready_path = tmp_path / "ready.npz"
    output_path = tmp_path / "candidate.npz"
    contract_path = tmp_path / "candidate.json"
    source = _motion_arrays(frames=40, ready=False)
    source["body_quat_w"][16:] *= -1.0
    with source_path.open("wb") as stream:
        np.savez(stream, **source)
    _write_motion(ready_path, frames=12, ready=True)

    completed = subprocess.run(
        _command(source_path, ready_path, output_path, contract_path),
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    with np.load(output_path, allow_pickle=False) as archive:
        output_quat = np.asarray(archive["body_quat_w"]).copy()
    output_join = 11
    assert np.array_equal(output_quat[output_join], source["body_quat_w"][16])
    assert np.all(np.sum(output_quat[output_join - 1] * output_quat[output_join], axis=-1) > 0.0)
