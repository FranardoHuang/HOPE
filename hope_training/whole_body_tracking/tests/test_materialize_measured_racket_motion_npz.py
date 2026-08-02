from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import pickle

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/materialize_measured_racket_motion_npz.py"
SPEC = importlib.util.spec_from_file_location("measured_racket_materializer", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_materializer_aligns_contact_rows_and_emits_strict_contract(tmp_path, monkeypatch):
    motion = tmp_path / "motion.npz"
    np.savez(
        motion,
        fps=np.asarray([50]),
        joint_pos=np.zeros((5, 31), dtype=np.float32),
    )
    source_sha = "1" * 64
    motion_sha = MODULE._sha256(motion)
    joint_order = tmp_path / "joint-order.json"
    joint_order.write_text("{}")
    joint_order_sha = MODULE._sha256(joint_order)
    retarget = tmp_path / "retarget.pkl"
    retarget_report = tmp_path / "retarget.report.json"
    retarget_report.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "kind": "chingmu_canonical_racket_full_phase_retarget_v4",
                "action_id": "Take_x_unit00_FH",
                "admitted": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    receipt_sha = MODULE._sha256(retarget_report)
    position = np.column_stack(
        (np.arange(10, dtype=np.float32), np.zeros(10), np.zeros(10))
    )
    with retarget.open("wb") as stream:
        pickle.dump(
            {
                "wrist_mode": "canonical_right_racket_full_phase_v4",
                "measured_racket_retarget_admitted": True,
                "mount_normal_sign": -1.0,
                "measured_racket_input_pkl_sha256": "3" * 64,
                "measured_racket_site_pos_w_120": position,
                "measured_racket_normal_w_120": np.tile([0.0, 2.0, 0.0], (10, 1)),
                "measured_racket_long_axis_w_120": np.tile([2.0, 0.0, 0.0], (10, 1)),
                "measured_racket_robot_butt_to_blade_axis_local": (
                    MODULE.ROBOT_BUTT_TO_BLADE_AXIS_LOCAL.copy()
                ),
                "measured_racket_robot_rigid_visual_mesh_sha256": (
                    MODULE.ROBOT_RIGID_VISUAL_MESH_SHA256
                ),
                "measured_racket_source_sha256": source_sha,
                "measured_racket_retarget_receipt_sha256": receipt_sha,
                "joint_order_contract_id": "a3-gmr-dof-pos-to-runtime-articulation-v1",
                "joint_order_contract_sha256": joint_order_sha,
                "measured_racket_uid": "Take_x_unit00_FH",
                "measured_racket_selected_hit_frame_120": 5,
            },
            stream,
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "units": [
                    {
                        "uid": "Take_x_unit00_FH",
                        "T": 5,
                        "fps": 50,
                        "hit_frame_pkl_120": 5,
                        "hit_frame_50": 2,
                        "source_pkl_sha256": "3" * 64,
                        "npz_sha256": motion_sha,
                    }
                ]
            }
        )
    )
    manifest_sha = MODULE._sha256(manifest)
    catalog = tmp_path / "catalog.json"
    clips = [
        {
            "uid": "Take_x_unit00_FH",
            "sha256": motion_sha,
            "T": 5,
            "hit_frame_50": 2,
        }
    ] + [
        {
            "uid": f"dummy_{index:02d}",
            "sha256": str(index).zfill(64),
            "T": 1,
            "hit_frame_50": 0,
        }
        for index in range(72)
    ]
    catalog.write_text(
        json.dumps(
            {"n_clips": 73, "excluded": ["Take_085_unit00_FH"], "clips": clips}
        )
    )
    catalog_sha = MODULE._sha256(catalog)
    with retarget.open("rb") as stream:
        payload = pickle.load(stream)
    payload["measured_racket_manifest_sha256"] = manifest_sha
    payload["measured_racket_catalog_sha256"] = catalog_sha
    with retarget.open("wb") as stream:
        pickle.dump(payload, stream)
    monkeypatch.setattr(
        MODULE,
        "_rebuild_kinematics",
        lambda **kwargs: (kwargs["arrays"], 0.0, np.zeros(2)),
    )
    real_sha256 = MODULE._sha256
    monkeypatch.setattr(
        MODULE,
        "_sha256",
        lambda path: (
            MODULE.EXPECTED_MJCF_SHA256
            if Path(path).name == "model.xml"
            else real_sha256(path)
        ),
    )
    arrays = MODULE.build_arrays(
        motion_path=motion,
        retarget_path=retarget,
        retarget_report_path=retarget_report,
        manifest_path=manifest,
        catalog_path=catalog,
        uid="Take_x_unit00_FH",
        model_path=tmp_path / "model.xml",
        joint_order_contract_path=joint_order,
    )
    assert arrays["measured_racket_site_pos_w"].shape == (5, 3)
    assert arrays["measured_racket_site_pos_w"][2, 0] == 5.0
    np.testing.assert_allclose(
        arrays["measured_racket_normal_w"], np.tile([0.0, 1.0, 0.0], (5, 1))
    )
    assert arrays["measured_racket_retarget_admitted"].tolist() == [1]
    assert arrays["measured_racket_robot_mount_normal_sign"].tolist() == [-1]
    np.testing.assert_allclose(
        arrays["measured_racket_long_axis_w"], np.tile([1.0, 0.0, 0.0], (5, 1))
    )
    assert arrays["measured_racket_schema_version"].tolist() == [4]
    np.testing.assert_array_equal(
        arrays["measured_racket_robot_butt_to_blade_axis_local"],
        MODULE.ROBOT_BUTT_TO_BLADE_AXIS_LOCAL,
    )
    assert str(arrays["measured_racket_joint_order_contract_sha256"]) == joint_order_sha
    assert str(arrays["measured_racket_input_motion_sha256"]) == motion_sha


def test_atomic_npz_publish_is_complete_and_no_replace(tmp_path):
    output = tmp_path / "motion.npz"
    MODULE._atomic_savez_no_replace(output, {"x": np.arange(4)})
    with np.load(output, allow_pickle=False) as archive:
        assert archive["x"].tolist() == [0, 1, 2, 3]
    try:
        MODULE._atomic_savez_no_replace(output, {"x": np.arange(2)})
    except FileExistsError:
        pass
    else:
        raise AssertionError("atomic NPZ publication must not replace existing output")
    assert list(tmp_path.iterdir()) == [output]


def test_qpos_resampling_slerps_root_and_interpolates_joints():
    source = np.zeros((2, 38), dtype=np.float64)
    source[:, 3] = 1.0
    source[1, 3:7] = [0.0, 0.0, 0.0, 1.0]
    source[1, 7:] = 2.0

    result = MODULE._resample_qpos(source, np.asarray([0.5]))

    np.testing.assert_allclose(np.linalg.norm(result[0, 3:7]), 1.0)
    np.testing.assert_allclose(result[0, 7:], 1.0)


def test_world_heading_rotation_uses_shared_pivot_for_robot_and_racket():
    position = np.asarray([[2.0, 0.0, 0.0]])
    normal = np.asarray([[1.0, 0.0, 0.0]])
    quat = np.asarray([[1.0, 0.0, 0.0, 0.0]])

    pos, face, rotated_quat = MODULE._rotate_world(
        position,
        normal,
        quat,
        theta=np.pi / 2.0,
        pivot=np.asarray([1.0, 0.0]),
    )

    np.testing.assert_allclose(pos, [[1.0, 1.0, 0.0]], atol=1.0e-12)
    np.testing.assert_allclose(face, [[0.0, 1.0, 0.0]], atol=1.0e-12)
    np.testing.assert_allclose(
        rotated_quat,
        [[np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]],
        atol=1.0e-12,
    )
