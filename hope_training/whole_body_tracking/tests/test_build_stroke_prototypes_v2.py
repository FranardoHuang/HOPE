"""Synthetic proof that prototype v2 measures the selected face centre."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_stroke_prototypes.py"
SPEC = importlib.util.spec_from_file_location(
    "build_stroke_prototypes_v2_under_test",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
B = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(B)


def _yaw_quaternion(angle_rad: np.ndarray) -> np.ndarray:
    result = np.zeros((angle_rad.shape[0], 4), dtype=np.float64)
    result[:, 0] = np.cos(angle_rad / 2.0)
    result[:, 3] = np.sin(angle_rad / 2.0)
    return result


def test_derive_one_uses_face_center_not_site_velocity(tmp_path):
    frame_count = 9
    fps = 100.0
    wrist = 0
    pelvis = 1
    body_pos = np.zeros((frame_count, 2, 3), dtype=np.float64)
    body_pos[:, wrist, 0] = np.arange(frame_count) * 0.01
    body_pos[:, wrist, 2] = 1.0
    yaw = -0.30 + np.arange(frame_count) * 0.02
    body_quat = np.zeros((frame_count, 2, 4), dtype=np.float64)
    body_quat[:, wrist] = _yaw_quaternion(yaw)
    # Root turns during preparation; prototype direction must stay in the
    # frozen frame-0 ready yaw used by the runtime birth receipt.
    body_quat[:, pelvis] = _yaw_quaternion(
        np.arange(frame_count, dtype=np.float64) * 0.10
    )
    body_ang_vel = np.zeros_like(body_pos)
    body_ang_vel[:, wrist, 2] = 2.0
    joint_vel = np.linspace(0.1, 0.2, frame_count)[:, None]

    clip = tmp_path / "fh_loop.npz"
    np.savez(
        clip,
        fps=np.asarray(fps),
        body_names=np.asarray(
            [B.WRIST_BODY, B.BASE_BODY],
            dtype=str,
        ),
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        body_ang_vel_w=body_ang_vel,
        joint_vel=joint_vel,
    )
    args = SimpleNamespace(
        min_contact_z=0.5,
        min_contact_vz=-10.0,
        retime_s_min=0.6,
    )
    row = B.derive_one(
        clip,
        0.04,
        0.04,
        "chingmu_001",
        "upper",
        "source-sha",
        args,
        {"joint": 10.0},
        {"joint": 100.0},
        ["joint"],
    )

    rotation = B.quat_to_rot(body_quat[:, wrist])
    site = body_pos[:, wrist] + np.einsum(
        "tij,j->ti",
        rotation,
        B.RACKET_SITE_OFFSET_WRIST_M,
    )
    face_sign = row["face_sign"]
    contact = row["contact_frame"]
    assert contact == 4
    site_velocity_all = B.runtime_clean_site_diff(site, fps)
    site_velocity = site_velocity_all[contact]
    face_velocity = B.face_center_velocity_from_site_twist(
        site_velocity_all,
        body_ang_vel[:, wrist],
        rotation,
        face_sign,
    )[contact]
    expected_speed = float(np.linalg.norm(face_velocity))
    expected_direction = face_velocity / expected_speed

    assert np.allclose(
        row["racket_face_center_velocity_hat_b"],
        expected_direction,
        atol=1.0e-12,
        rtol=0.0,
    )
    assert np.isclose(
        row["racket_face_center_speed_nominal_mps"],
        expected_speed,
        atol=1.0e-12,
        rtol=0.0,
    )
    assert np.linalg.norm(face_velocity - site_velocity) > 1.0e-4
    assert "v_hat_b" not in row
    assert "speed_nominal_mps" not in row
    assert row["priority"] == 0
    assert row["enabled"] is True


def test_checked_in_native_centres_require_unit_teacher_rate():
    """The current four admitted fivebind motions expose the old >1 bug."""

    repo_root = ROOT.parents[1]
    manifest_path = (
        repo_root / "configs/action_ball_n5_nomove_f10_20260728.json"
    )
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    for action in document["actions"]:
        clip_path = repo_root / action["motion_path"]
        with np.load(clip_path) as clip:
            names = [str(value) for value in clip["body_names"]]
            wrist = names.index(B.WRIST_BODY)
            position = np.asarray(clip["body_pos_w"], dtype=np.float64)
            quaternion = np.asarray(
                clip["body_quat_w"],
                dtype=np.float64,
            )
            angular_velocity = np.asarray(
                clip["body_ang_vel_w"],
                dtype=np.float64,
            )
            fps = float(np.asarray(clip["fps"]).reshape(-1)[0])

        rotation = B.quat_to_rot(quaternion[:, wrist])
        site = position[:, wrist] + np.einsum(
            "tij,j->ti",
            rotation,
            B.RACKET_SITE_OFFSET_WRIST_M,
        )
        site_velocity = B.runtime_clean_site_diff(site, fps)
        strike = round(
            float(action["strike_phase"]) * (len(site) - 1)
        )
        face_velocity = B.face_center_velocity_from_site_twist(
            site_velocity,
            angular_velocity[:, wrist],
            rotation,
            action["mount_normal_sign"],
        )
        face_offset_w = (
            rotation[strike]
            @ B.face_center_from_site_local(
                action["mount_normal_sign"]
            )
        )
        recovered_site_velocity = (
            face_velocity[strike]
            - np.cross(
                angular_velocity[strike, wrist],
                face_offset_w,
            )
        )
        required_rate = float(
            np.linalg.norm(recovered_site_velocity)
            / action["reference_racket_site_speed_mps"]
        )
        assert np.isclose(
            required_rate,
            1.0,
            atol=1.0e-6,
            rtol=0.0,
        ), (action["action_id"], required_rate)
        assert required_rate <= float(
            action["teacher_rate_max"]
        ) + 1.0e-6


def _fresh_library():
    specs = [
        {
            "motion_id": motion_id,
            "source_sha256": f"{index + 1:064x}",
        }
        for index, motion_id in enumerate(B.FRESH_N5_BANK_MOTION_IDS)
    ]
    return {
        "training_authorized": False,
        "hardware_authorized": False,
        "canonical_ready": {"sha256": B.CANONICAL_READY_V1_SHA256},
        "motion_specs": specs,
        "required_output_matrix": {
            "motion_ids": list(B.FRESH_N5_BANK_MOTION_IDS),
            "scopes": ["upper", "full"],
            "candidate_count": 14,
        },
    }


def _fresh_manifest():
    outputs = []
    for index, motion_id in enumerate(B.FRESH_N5_BANK_MOTION_IDS):
        for scope_index, scope in enumerate(("upper", "full")):
            outputs.append(
                {
                    "motion_id": motion_id,
                    "scope": scope,
                    "filename": (
                        f"{motion_id}_{scope}_canonical_v2.npz"
                    ),
                    "output_npz_sha256": (
                        f"{100 + 2 * index + scope_index:064x}"
                    ),
                }
            )
    return {
        "output_matrix": {
            "motion_ids": list(B.FRESH_N5_BANK_MOTION_IDS),
            "scopes": ["upper", "full"],
            "candidate_count": 14,
        },
        "outputs": outputs,
    }


def test_fresh_n5_selects_exact_upper_view_in_training_order():
    source_sha = B._fresh_n5_library_source_sha(_fresh_library())
    assert tuple(source_sha) == B.FRESH_N5_BANK_MOTION_IDS

    selected = B._fresh_n5_upper_outputs(_fresh_manifest())
    assert tuple(row["motion_id"] for row in selected) == (
        B.FRESH_N5_UPPER_MOTION_IDS
    )
    assert {row["scope"] for row in selected} == {"upper"}
    assert not (
        {row["motion_id"] for row in selected}
        & B.FRESH_N5_FORBIDDEN_MOTION_IDS
    )


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda library: library["motion_specs"].__setitem__(
                5, library["motion_specs"][6]
            ),
            "immutable canonical five followed",
        ),
        (
            lambda library: library["canonical_ready"].__setitem__(
                "sha256", "0" * 64
            ),
                "exact registered legacy canonical ready v1",
        ),
        (
            lambda library: library.__setitem__(
                "training_authorized", True
            ),
            "may not authorize training",
        ),
    ],
)
def test_fresh_n5_library_rejects_identity_or_authorization_drift(
    mutation,
    match,
):
    library = _fresh_library()
    mutation(library)
    with pytest.raises(SystemExit, match=match):
        B._fresh_n5_library_source_sha(library)


def test_fresh_n5_manifest_rejects_partial_cycle_or_single_scope():
    manifest = _fresh_manifest()
    manifest["outputs"] = [
        row for row in manifest["outputs"] if row["scope"] == "upper"
    ]
    with pytest.raises(SystemExit, match="incomplete"):
        B._fresh_n5_upper_outputs(manifest)


def test_fresh_n5_manifest_rejects_stale_or_renamed_output():
    manifest = _fresh_manifest()
    row = next(
        row
        for row in manifest["outputs"]
        if row["motion_id"] == "v12_forehand_block"
        and row["scope"] == "upper"
    )
    row["filename"] = "fh_block_syn_upper_canonical_v2.npz"
    with pytest.raises(SystemExit, match="filename must be"):
        B._fresh_n5_upper_outputs(manifest)
