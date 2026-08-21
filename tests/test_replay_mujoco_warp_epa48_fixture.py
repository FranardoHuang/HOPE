"""Host-only tests for the fixed EPA48 replay contract and classifier."""

import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/replay_mujoco_warp_epa48_fixture.py"
SPEC = importlib.util.spec_from_file_location("epa48_replay", SCRIPT)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def test_fixture_contract_names_the_exact_pose_geometry_and_probe():
    fixture, xml = M.load_fixture()
    expected = {
        "schema_version": 1,
        "fixture_id": "mujoco_warp_epa48_ellipsoid_cylinder_cross_v1",
        "mjcf_file": "mujoco_warp_epa48_ellipsoid_cylinder_cross_v1.xml",
        "mjcf_sha256": "f611bbf5189a5eb87b0b9da58261d7f6d1d31302757daea106cfbc22a4fc58ce",
        "geometry": {
            "fixed": {"name": "fixed_geom", "type": "ellipsoid", "size_m": [0.025, 0.075, 0.125]},
            "moving": {
                "name": "moving_geom", "type": "cylinder", "size_m": [0.04, 0.105], "density_kg_m3": 1000,
            },
        },
        "pose": {
            "translation_m": [0.0071089754719, -0.011792854753, 0.0173480845636],
            "quat_wxyz": [0.102804198523, 0.325362607212, 0.0882094053962, 0.935836295189],
        },
        "probe": {
            "nworld": 1, "ccd_iterations": 100, "ccd_tolerance": 1e-8, "disable_multiccd": True,
            "nconmax": 4, "nccdmax": 4, "njmax": 32, "gravity_m_s2": [0.0, 0.0, 0.0], "joint": "free",
        },
        "scientific_scope": {
            "diagnostic_unauthorized": True, "training_authorized": False,
            "stock_cpu_mujoco_is_oracle": False,
        },
    }
    assert fixture == expected
    assert '<option gravity="0 0 0"/>' in xml.read_text()
    assert '<freejoint name="moving_free"/>' in xml.read_text()


def test_fixture_loader_rejects_changed_bytes_and_symlink(monkeypatch, tmp_path):
    changed = tmp_path / "fixture.json"
    changed.write_bytes(M.FIXTURE_JSON.read_bytes() + b"\n")
    monkeypatch.setattr(M, "FIXTURE_JSON", changed)
    with pytest.raises(M.ReplayError, match="JSON SHA differs"):
        M.load_fixture()
    linked = tmp_path / "linked.json"
    linked.symlink_to(ROOT / "configs/fixtures/mujoco_warp_epa48_ellipsoid_cylinder_cross_v1.json")
    monkeypatch.setattr(M, "FIXTURE_JSON", linked)
    with pytest.raises(M.ReplayError, match="regular non-symlink"):
        M.load_fixture()


def test_classifier_accepts_only_exact_repeated_stock_overflow_and_finite_fork_contact():
    stock = {
        "schema_version": 1,
        "fixture_id": "mujoco_warp_epa48_ellipsoid_cylinder_cross_v1",
        "nworld": 1,
        "repeats": 3,
        "runtime": {
            "role": "stock24", "distribution_version": "3.10.0.3", "module_version": "3.10.0.3",
            "epa_horizon": 24, "epa_horizon_bit": 256,
            "types_sha256": "712e76f495d3dedcb45acc7c248e226f56144b5fef5d9841d5e04d279fa7fd4f",
            "package_file_count": 281,
            "package_manifest_sha256": "6fb7b2849955d952e69d67c534b96816f43db5914805ff2732692e70011ea3c2",
            "mujoco_version": "3.10.0", "warp_version": "1.16.0", "python_prefix": "/env/stock",
            "warp_cache_dir": "/cache/stock",
            "device": {"uuid": "GPU-11111111-2222-3333-4444-555555555555", "pci_bus_id": "00000001:BE:00"},
        },
        "observations": [
            {"repeat_index": 0, "overflow_mask": 256, "contact_count": 0, "contact_distances": [],
             "contact_positions": [], "contact_frames": [], "active_contact_finite": False},
            {"repeat_index": 1, "overflow_mask": 256, "contact_count": 0, "contact_distances": [],
             "contact_positions": [], "contact_frames": [], "active_contact_finite": False},
            {"repeat_index": 2, "overflow_mask": 256, "contact_count": 0, "contact_distances": [],
             "contact_positions": [], "contact_frames": [], "active_contact_finite": False},
        ],
        "scientific_scope": {
            "diagnostic_unauthorized": True, "training_authorized": False,
            "stock_cpu_mujoco_is_oracle": False,
        },
    }
    fork = {
        "schema_version": 1,
        "fixture_id": "mujoco_warp_epa48_ellipsoid_cylinder_cross_v1",
        "nworld": 1,
        "repeats": 3,
        "runtime": {
            "role": "fork48", "distribution_version": "3.10.0.3+hope.epa48.1",
            "module_version": "3.10.0.3+hope.epa48.1", "epa_horizon": 48, "epa_horizon_bit": 256,
            "types_sha256": "391e421eeede84389d6c7daeae39b19ce43132d29c11f7f3c328a50011c7a696",
            "package_file_count": 281,
            "package_manifest_sha256": "90ae9570e2e4e0dc45fd28315aae15b156802a5c2f92c0ba33dad3aac4385036",
            "mujoco_version": "3.10.0", "warp_version": "1.16.0", "python_prefix": "/env/fork",
            "warp_cache_dir": "/cache/fork",
            "device": {"uuid": "GPU-11111111-2222-3333-4444-555555555555", "pci_bus_id": "00000001:BE:00"},
        },
        "observations": [
            {"repeat_index": 0, "overflow_mask": 0, "contact_count": 1, "contact_distances": [-0.1],
             "contact_positions": [[1.0, 2.0, 3.0]], "contact_frames": [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]], "active_contact_finite": True},
            {"repeat_index": 1, "overflow_mask": 0, "contact_count": 1, "contact_distances": [-0.1],
             "contact_positions": [[1.0, 2.0, 3.0]], "contact_frames": [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]], "active_contact_finite": True},
            {"repeat_index": 2, "overflow_mask": 0, "contact_count": 1, "contact_distances": [-0.1],
             "contact_positions": [[1.0, 2.0, 3.0]], "contact_frames": [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]], "active_contact_finite": True},
        ],
        "scientific_scope": {
            "diagnostic_unauthorized": True, "training_authorized": False,
            "stock_cpu_mujoco_is_oracle": False,
        },
    }
    expected_uuid = "GPU-11111111-2222-3333-4444-555555555555"
    assert M.classify_results(stock, fork, 3, expected_uuid)["verdict"] == "PASS_EPA48_FIXED_FIXTURE_REPLAY"
    wrong_stock = copy.deepcopy(stock)
    wrong_stock["observations"][1]["overflow_mask"] = 0
    assert M.classify_results(wrong_stock, fork, 3, expected_uuid)["verdict"] == "FAIL_EPA48_FIXED_FIXTURE_REPLAY"
    unstable_fork = copy.deepcopy(fork)
    unstable_fork["observations"][2].update({
        "contact_count": 2, "contact_distances": [-0.1, -0.2],
        "contact_positions": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        "contact_frames": [[1.0] * 9, [2.0] * 9],
    })
    assert M.classify_results(stock, unstable_fork, 3, expected_uuid)["verdict"] == "FAIL_EPA48_FIXED_FIXTURE_REPLAY"
    nonfinite_fork = copy.deepcopy(fork)
    nonfinite_fork["observations"][0]["contact_frames"][0][0] = float("inf")
    nonfinite_fork["observations"][0]["active_contact_finite"] = False
    assert M.classify_results(stock, nonfinite_fork, 3, expected_uuid)["verdict"] == "FAIL_EPA48_FIXED_FIXTURE_REPLAY"


def test_json_writer_and_output_root_are_no_clobber(tmp_path):
    target = tmp_path / "result.json"
    M._write_json_x(target, {"first": True})
    with pytest.raises(M.ReplayError, match="refusing to overwrite"):
        M._write_json_x(target, {"second": True})
    output = tmp_path / "run"
    M._create_output_root(output)
    with pytest.raises(M.ReplayError, match="output root already exists"):
        M._create_output_root(output)
    assert json.loads(target.read_text()) == {"first": True}


def test_public_cli_does_not_disclose_private_mode():
    shown = subprocess.run([sys.executable, str(SCRIPT), "--help"], text=True, capture_output=True, check=False)
    assert shown.returncode == 0
    assert "replay" in shown.stdout
    assert "_worker" not in shown.stdout + shown.stderr
    refused = subprocess.run([sys.executable, str(SCRIPT), "not-a-mode"], text=True, capture_output=True, check=False)
    assert refused.returncode == 2
    assert "invalid choice" in refused.stderr
    assert "_worker" not in refused.stdout + refused.stderr


def test_actual_gpu_replay_is_explicit_opt_in(tmp_path):
    stock = os.environ.get("EPA48_REPLAY_STOCK_PYTHON")
    fork = os.environ.get("EPA48_REPLAY_FORK_PYTHON")
    uuid = os.environ.get("EPA48_REPLAY_GPU_UUID")
    if not all((stock, fork, uuid)):
        pytest.skip("set EPA48_REPLAY_STOCK_PYTHON/FORK_PYTHON/GPU_UUID for actual CUDA replay")
    completed = subprocess.run([
        stock, str(SCRIPT), "replay", "--stock-python", stock, "--fork-python", fork,
        "--expected-gpu-uuid", uuid, "--output-root", str(tmp_path / "actual"),
        "--device", os.environ.get("EPA48_REPLAY_DEVICE", "cuda:0"), "--repeats", "3",
    ], text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr
