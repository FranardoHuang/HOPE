"""Tests for the no-authority ChingMu N=73 source capsule."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import materialize_chingmu73_source_capsule as capsule  # noqa: E402


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json(path: Path, value) -> bytes:
    payload = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload


def _motion(path: Path, total: int = 6) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    body_quat = np.zeros((total, 32, 4), dtype=np.float32)
    body_quat[..., 0] = 1.0
    values = {
        "fps": np.asarray([50], dtype=np.int64),
        "joint_pos": np.zeros((total, 31), dtype=np.float32),
        "joint_vel": np.zeros((total, 31), dtype=np.float32),
        "body_pos_w": np.zeros((total, 32, 3), dtype=np.float32),
        "body_quat_w": body_quat,
        "body_lin_vel_w": np.zeros((total, 32, 3), dtype=np.float32),
        "body_ang_vel_w": np.zeros((total, 32, 3), dtype=np.float32),
        "kinematics_schema_version": np.asarray([2], dtype=np.int64),
        "body_pos_point": np.asarray("link_origin"),
        "body_lin_vel_point": np.asarray("center_of_mass"),
        "body_names": np.asarray(capsule.EXPECTED_BODY_NAMES),
    }
    np.savez(path, **values)
    return path.read_bytes()


def _ball(path: Path, ball_at_hit: list[float]) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    real = np.zeros((8, 3), dtype=np.float32)
    real[3] = np.asarray(ball_at_hit, dtype=np.float32)
    np.savez(
        path,
        ball_real_hope_m=real,
        ball_synth_hope_m=real.copy(),
        unit_offset=np.asarray(1, dtype=np.int64),
        n_unit=np.asarray(4, dtype=np.int64),
        src_range_120=np.asarray([0, 7], dtype=np.int64),
        hit_frames_ext=np.asarray([3], dtype=np.int64),
        fps=np.asarray(120.0, dtype=np.float64),
    )
    return path.read_bytes()


class Inputs:
    def __init__(self, tmp_path: Path):
        self.profile_root = tmp_path / "profile"
        self.batch_root = tmp_path / "batch"
        self.motion_root = self.profile_root / "motions/chingmu73_20260728"
        self.ball_root = tmp_path / "balls"
        self.output = tmp_path / "capsule"
        for directory in (
            self.profile_root,
            self.batch_root / "clips",
            self.motion_root,
            self.ball_root,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        prototype_path = (
            self.profile_root / "configs/stroke_prototypes_v1_20260727.json"
        )
        prototype_bytes = _json(
            prototype_path,
            {"schema_version": 1, "training_authorized": False},
        )
        actions = []
        units = []
        for index in range(capsule.ACTION_COUNT):
            uid = f"Take_{index:03d}_unit00_BH"
            action_id = uid.lower()
            motion_name = f"hope_{uid}.npz"
            motion_bytes = _motion(self.motion_root / motion_name)
            (self.batch_root / "clips" / motion_name).write_bytes(motion_bytes)
            station = [index / 1000.0, -0.2]
            ball_hit = [0.1 + index / 10000.0, -0.3, 0.25]
            ball_name = f"{uid}.ball.npz"
            _ball(self.ball_root / ball_name, ball_hit)
            unit = {
                "uid": uid,
                "family": "BH",
                "npz": f"clips/{motion_name}",
                "npz_sha256": _sha(motion_bytes),
                "source_pkl": f"/old/pod/source/{uid}.pkl",
                "source_pkl_sha256": _sha(uid.encode()),
                "ball_npz": f"/old/pod/ball_ext/{ball_name}",
                "T": 6,
                "fps": 50,
                "duration_s": 0.1,
                "hit_frame_ball_120": 2,
                "hit_frame_pkl_120": 2,
                "hit_frame_src120": 3,
                "hit_frame_50": 2,
                "strike_phase": 0.4,
                "n_hits_ext": 1,
                "station_xy_hope_m": station,
                "retime_factor": 1.0,
                "v_in_fit_hope_ms": [-1.0, 0.1, -0.2],
                "v_out_fit_hope_ms": [2.0, 0.2, 0.5],
                "ball_pos_hit_hope_m": ball_hit,
                "ball_coverage_unit": 1.0,
                "ball_coverage_pre30": 1.0,
                "warnings": [] if index else ["synthetic warning"],
            }
            units.append(unit)
            _json(
                self.batch_root / "clips" / f"hope_{uid}.meta.json",
                {
                    "clip": uid,
                    "source_fps": 120.0,
                    "retime_factor": 1.0,
                    "station_xy_hope_m": station,
                    "hits": [
                        {
                            "frame_src120": 3,
                            "frame_out": 2,
                            "side": "backhand",
                            "ball_hope_m": ball_hit,
                            "v_in_fit_hope_ms": [-1.0, 0.1, -0.2],
                            "v_out_fit_hope_ms": [2.0, 0.2, 0.5],
                        }
                    ],
                },
            )
            zero_travel = {
                key: [0.0, 0.0]
                for key in (
                    "base_travel_center_b_yaw_xy_m",
                    "base_travel_std_lower_initial_m",
                    "base_travel_std_lower_max_m",
                    "base_travel_std_upper_initial_m",
                    "base_travel_std_upper_max_m",
                    "base_travel_min_b_yaw_xy_m",
                    "base_travel_max_b_yaw_xy_m",
                )
            }
            actions.append(
                {
                    "action_id": action_id,
                    "action_uid": index + 1,
                    "motion_path": (
                        f"motions/chingmu73_20260728/{motion_name}"
                    ),
                    "motion_sha256": _sha(motion_bytes),
                    "strike_phase": 0.4,
                    "reference_t_hit_s": 0.04,
                    "reference_t_cycle_s": 0.1,
                    "reference_racket_site_speed_mps": 1.0,
                    "reaction_margin_s": 0.1,
                    "teacher_rate_min": 0.6,
                    "teacher_rate_max": 1.0,
                    "family": "backhand",
                    "mount_normal_sign": -1,
                    "ball_profile": {
                        "base_spawn_center_w_xy_m": [
                            station[0] + 0.5,
                            station[1] + 0.7625,
                        ],
                        **zero_travel,
                    },
                }
            )

        units.append({"uid": capsule.EXCLUDED_UID})
        self.batch_path = self.batch_root / "chingmu_manifest_v1.json"
        self.batch_bytes = _json(
            self.batch_path,
            {
                "schema": capsule.EXPECTED_BATCH_SCHEMA,
                "generated_utc": "synthetic",
                "units": units,
                "failures": [],
            },
        )
        self.action_path = (
            self.profile_root
            / "configs/action_ball_chingmu73_nomove_f10_20260728.json"
        )
        self.action_bytes = _json(
            self.action_path,
            {
                "schema_version": 3,
                "manifest_id": capsule.EXPECTED_MANIFEST_ID,
                "mobility_mode": "no_move",
                "action_order": [action["action_id"] for action in actions],
                "prototype": {
                    "path": "configs/stroke_prototypes_v1_20260727.json",
                    "sha256": _sha(prototype_bytes),
                    "scope": "full",
                },
                "solver_profile_sha256": _sha(b"solver-profile"),
                "physics_profile_sha256": _sha(b"physics-profile"),
                "actions": actions,
            },
        )
        self.report_path = (
            self.profile_root
            / "configs/action_ball_chingmu73_nomove_f10_20260728.buildreport.json"
        )
        self.report_bytes = _json(
            self.report_path,
            {
                "n_actions": capsule.ACTION_COUNT,
                "excluded_uids": [capsule.EXCLUDED_UID],
                "file_sha256": _sha(self.action_bytes),
                "batch_manifest_sha256": _sha(self.batch_bytes),
            },
        )

    def run(self):
        return capsule.materialize(
            action_manifest_path=self.action_path,
            expected_action_manifest_sha256=_sha(self.action_bytes),
            build_report_path=self.report_path,
            expected_build_report_sha256=_sha(self.report_bytes),
            batch_manifest_path=self.batch_path,
            expected_batch_manifest_sha256=_sha(self.batch_bytes),
            profile_root=self.profile_root,
            batch_root=self.batch_root,
            motion_root=self.motion_root,
            ball_root=self.ball_root,
            output_directory=self.output,
        )


def test_materializes_portable_content_bound_source_inventory(tmp_path: Path):
    inputs = Inputs(tmp_path)

    receipt = inputs.run()

    assert receipt["verdict"] == "PASS_SOURCE_INVENTORY_ONLY"
    assert receipt["authorization"] == {
        "compiler_candidate_authorized": False,
        "motion_admission_present": False,
        "training_authorized": False,
        "deployment_authorized": False,
        "hardware_authorized": False,
    }
    assert len(receipt["actions"]) == 73
    assert receipt["actions"][0]["warnings"] == ["synthetic warning"]
    persisted = json.loads(
        (inputs.output / capsule.RECEIPT_NAME).read_text()
    )
    assert persisted == receipt
    for row in receipt["actions"]:
        assert _sha((inputs.output / row["motion_path"]).read_bytes()) == row[
            "motion_sha256"
        ]
        assert _sha((inputs.output / row["metadata_path"]).read_bytes()) == row[
            "metadata_sha256"
        ]
        assert _sha((inputs.output / row["ball_path"]).read_bytes()) == row[
            "ball_sha256"
        ]
        assert not row["ball_path"].startswith("/")


def test_ball_semantic_drift_fails_before_publication(tmp_path: Path):
    inputs = Inputs(tmp_path)
    first = inputs.ball_root / "Take_000_unit00_BH.ball.npz"
    with np.load(first, allow_pickle=False) as archive:
        values = {key: np.asarray(archive[key]) for key in archive.files}
    values["hit_frames_ext"] = np.asarray([4], dtype=np.int64)
    np.savez(first, **values)

    with pytest.raises(
        capsule.ChingMu73CapsuleError,
        match="first hit minus unit_offset",
    ):
        inputs.run()

    assert not inputs.output.exists()


def test_symlink_and_no_clobber_fail_closed(tmp_path: Path):
    inputs = Inputs(tmp_path)
    first = inputs.motion_root / "hope_Take_000_unit00_BH.npz"
    external = tmp_path / "external.npz"
    external.write_bytes(first.read_bytes())
    first.unlink()
    first.symlink_to(external)

    with pytest.raises(
        capsule.ChingMu73CapsuleError,
        match="motion root may contain only regular",
    ):
        inputs.run()
    assert not inputs.output.exists()

    first.unlink()
    first.write_bytes(external.read_bytes())
    inputs.run()
    with pytest.raises(
        capsule.ChingMu73CapsuleError,
        match="no-clobber",
    ):
        inputs.run()
