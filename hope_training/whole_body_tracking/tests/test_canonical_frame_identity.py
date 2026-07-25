"""Tests for content-addressed temporal-frame identity receipts."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest


_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS = _ROOT / "hope_training" / "whole_body_tracking" / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from canonical_frame_identity import (  # noqa: E402
    FrameIdentityError,
    build_frame_identity_receipt,
    canonical_json_bytes,
    verify_frame_identity_receipt,
    write_frame_identity_receipt,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _names_sha(names: list[str]) -> str:
    payload = json.dumps(names, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _qmul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    scalar = left[..., 0] * right[..., 0] - np.sum(
        left[..., 1:] * right[..., 1:], axis=-1
    )
    vector = (
        left[..., 0, None] * right[..., 1:]
        + right[..., 0, None] * left[..., 1:]
        + np.cross(left[..., 1:], right[..., 1:])
    )
    return np.concatenate((scalar[..., None], vector), axis=-1)


def _rz(yaw_deg: float) -> np.ndarray:
    angle = math.radians(yaw_deg)
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _fixture(
    tmp_path: Path,
    *,
    grip_bake: bool = True,
    duplicate_invariant_frame: bool = False,
) -> tuple[dict, dict[str, Path]]:
    frames, joints, bodies = 6, 31, 4
    joint_names = [f"joint_{index:02d}" for index in range(joints)]
    body_names = np.asarray(
        [
            "pelvis_link",
            "torso_Link",
            "right_wrist_roll_Link",
            "left_wrist_roll_Link",
        ]
    )
    invariant_indices = (
        [index for index in range(joints) if index not in {26, 28, 30}]
        if grip_bake
        else list(range(joints))
    )

    frame = np.arange(frames, dtype=np.float32)[:, None]
    column = np.arange(joints, dtype=np.float32)[None, :]
    event_q = (0.1 * frame + 0.003 * column).astype(np.float32)
    event_qd = (0.2 * frame - 0.005 * column).astype(np.float32)
    if duplicate_invariant_frame:
        event_q[3, invariant_indices] = event_q[2, invariant_indices]
        event_qd[3, invariant_indices] = event_qd[2, invariant_indices]
    bound_q = event_q.copy()
    bound_qd = event_qd.copy()
    if grip_bake:
        for offset, index in enumerate((26, 28, 30), start=1):
            bound_q[:, index] += (
                offset * 0.2 + np.arange(frames, dtype=np.float32) ** 2 * 0.01
            )
            bound_qd[:, index] -= (
                offset * 0.1 + np.arange(frames, dtype=np.float32) * 0.02
            )

    base_pos = np.arange(frames * bodies * 3, dtype=np.float64).reshape(
        frames, bodies, 3
    )
    base_pos = base_pos * 0.001 + np.asarray([0.2, -0.1, 0.8])
    base_lin = np.flip(base_pos, axis=0) * 0.4
    base_ang = base_pos * -0.2
    base_quat = np.zeros((frames, bodies, 4), dtype=np.float64)
    base_quat[..., 0] = 1.0

    event_yaw, event_translation = 0.0, np.asarray([-0.2, 0.0, 0.0])
    bound_yaw, bound_translation = -30.0, np.zeros(3)
    event_rotation, bound_rotation = _rz(event_yaw), _rz(bound_yaw)
    event_pos = np.einsum("ij,tbj->tbi", event_rotation, base_pos)
    event_pos += event_translation
    bound_pos = np.einsum("ij,tbj->tbi", bound_rotation, base_pos)
    bound_pos += bound_translation
    event_lin = np.einsum("ij,tbj->tbi", event_rotation, base_lin)
    bound_lin = np.einsum("ij,tbj->tbi", bound_rotation, base_lin)
    event_ang = np.einsum("ij,tbj->tbi", event_rotation, base_ang)
    bound_ang = np.einsum("ij,tbj->tbi", bound_rotation, base_ang)
    event_qyaw = np.asarray(
        [math.cos(math.radians(event_yaw) / 2), 0.0, 0.0, math.sin(math.radians(event_yaw) / 2)]
    )
    bound_qyaw = np.asarray(
        [math.cos(math.radians(bound_yaw) / 2), 0.0, 0.0, math.sin(math.radians(bound_yaw) / 2)]
    )
    event_body_quat = _qmul(
        np.broadcast_to(event_qyaw, base_quat.shape), base_quat
    )
    bound_body_quat = _qmul(
        np.broadcast_to(bound_qyaw, base_quat.shape), base_quat
    )
    excluded_bodies: list[str] = []
    if grip_bake:
        excluded_bodies = ["right_wrist_roll_Link"]
        bound_pos[:, 2, 0] += 0.3
        bound_lin[:, 2, 1] -= 0.4
        bound_ang[:, 2, 2] += 0.5
        bound_body_quat[:, 2] = np.asarray([0.0, 1.0, 0.0, 0.0])

    def arrays(
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        body_pos: np.ndarray,
        body_quat: np.ndarray,
        body_lin: np.ndarray,
        body_ang: np.ndarray,
    ) -> dict:
        return {
            "fps": np.asarray([50], dtype=np.int64),
            "joint_pos": joint_pos.astype(np.float32),
            "joint_vel": joint_vel.astype(np.float32),
            "body_pos_w": body_pos.astype(np.float32),
            "body_quat_w": body_quat.astype(np.float32),
            "body_lin_vel_w": body_lin.astype(np.float32),
            "body_ang_vel_w": body_ang.astype(np.float32),
            "kinematics_schema_version": np.asarray([2], dtype=np.int64),
            "body_pos_point": np.asarray("link_origin"),
            "body_lin_vel_point": np.asarray("link_origin"),
            "body_names": body_names,
        }

    event_npz = tmp_path / "event.npz"
    bound_npz = tmp_path / "bound.npz"
    np.savez(
        event_npz,
        **arrays(
            event_q,
            event_qd,
            event_pos,
            event_body_quat,
            event_lin,
            event_ang,
        ),
    )
    np.savez(
        bound_npz,
        **arrays(
            bound_q,
            bound_qd,
            bound_pos,
            bound_body_quat,
            bound_lin,
            bound_ang,
        ),
    )

    common = {
        "campaign": "test",
        "source_pkl": {
            "path": "/evidence/source.pkl",
            "sha256": "a" * 64,
            "frames": 4,
            "fps": 30,
        },
        "output": {
            "frames": frames,
            "fps": 50,
            "frame_formula": "round(((input_frames-1)/30)*50)+1",
        },
        "mjcf": {"path": "/model.xml", "sha256": "b" * 64},
        "joint_contract": "configs/a3_joint_order_bijection_v1.json",
    }
    event_provenance = tmp_path / "event.provenance.json"
    bound_provenance = tmp_path / "bound.provenance.json"
    _write_json(
        event_provenance,
        {
            **common,
            "grip_bake": {"method": "test"} if grip_bake else None,
            "se2_station_probe": {
                "translation_w_m": event_translation.tolist(),
                "yaw_deg": event_yaw,
                "semantics": "left action",
            },
        },
    )
    _write_json(
        bound_provenance,
        {
            **common,
            "grip_bake": None,
            "se2_station_probe": {
                "translation_w_m": bound_translation.tolist(),
                "yaw_deg": bound_yaw,
                "semantics": "left action",
            },
        },
    )

    joint_order = tmp_path / "joint_order.txt"
    joint_order.write_text(
        "# runtime order\n" + "\n".join(joint_names) + "\n",
        encoding="utf-8",
    )
    joint_bijection = tmp_path / "joint_bijection.json"
    _write_json(
        joint_bijection,
        {
            "schema_version": 1,
            "expected_joint_count": joints,
            "target_order": {
                "name": "runtime_articulation_joint_pos",
                "file_sha256": _sha(joint_order),
                "names_sha256": _names_sha(joint_names),
            },
        },
    )
    event_authority = tmp_path / "event_authority.json"
    _write_json(
        event_authority,
        {
            "schema_version": 1,
            "authority_id": "test_event_authority_v1",
            "review_status": "TEST_ONLY",
            "scope": "Test-only event frame and source binding.",
            "authorization": {
                "training_authorized": False,
                "behavior_authorized": False,
                "hardware_authorized": False,
                "artifact_promotion_authorized": False,
            },
            "semantic_contract": {
                "contact_truth_observed": False,
                "unresolved_or_mismatched_source_binding_policy": "fail_closed",
            },
            "motions": [
                {
                    "motion_id": "test_motion",
                    "bound_recipe_source": {
                        "path": "bound.npz",
                        "sha256": _sha(bound_npz),
                    },
                    "nominal_event": {
                        "contact_truth_observed": False,
                        "event_source_remote_path": "pod2:/evidence/event.npz",
                        "event_source_sha256": _sha(event_npz),
                        "evidence_json_pointer": "/test/event",
                        "frame": 4,
                        "mapping_to_bound_recipe_source": (
                            "UNRESOLVED_NO_CONTENT_ADDRESSED_FRAME_MAP_RECEIPT"
                        ),
                        "semantic_kind": "air_swing_nominal_strike_event",
                    },
                }
            ],
        },
    )

    paths = {
        "event_npz": event_npz,
        "bound_npz": bound_npz,
        "event_provenance": event_provenance,
        "bound_provenance": bound_provenance,
        "event_authority": event_authority,
        "joint_order": joint_order,
        "joint_bijection": joint_bijection,
    }
    spec = {
        "schema_version": 1,
        "motion_id": "test_motion",
        "mapping_role": "ordinary_nominal_event_to_bound_recipe_source",
        "inputs": {
            name: {"path": str(path), "sha256": _sha(path)}
            for name, path in paths.items()
        },
        "invariant_joints": [
            {"index": index, "name": joint_names[index]}
            for index in invariant_indices
        ],
        "excluded_world_bodies": excluded_bodies,
        "world_residual_tolerances": {
            "position_m": 2.0e-7,
            "linear_velocity_m_s": 1.0e-5,
            "angular_velocity_rad_s": 2.0e-5,
            "quaternion_l2_sign_invariant": 2.0e-7,
        },
        "semantic_contract": {
            "frame_identity_only": True,
            "observed_ball_contact": False,
            "behavior_authorized": False,
            "synthetic_behavior_nominal_claim": False,
        },
    }
    return spec, paths


def _refresh_test_authority_bindings(
    spec: dict, paths: dict[str, Path]
) -> None:
    authority = json.loads(
        paths["event_authority"].read_text(encoding="utf-8")
    )
    motion = authority["motions"][0]
    motion["bound_recipe_source"]["sha256"] = _sha(paths["bound_npz"])
    motion["nominal_event"]["event_source_sha256"] = _sha(paths["event_npz"])
    _write_json(paths["event_authority"], authority)
    spec["inputs"]["event_authority"]["sha256"] = _sha(
        paths["event_authority"]
    )


def test_grip_baked_pair_proves_temporal_identity_without_full_pose_claim(tmp_path):
    spec, _ = _fixture(tmp_path, grip_bake=True)
    receipt = build_frame_identity_receipt(spec)

    assert receipt["event_mapping"]["identity_map"] is True
    assert receipt["event_mapping"]["event_source_frame"] == 4
    assert receipt["event_mapping"]["bound_source_frame"] == 4
    assert receipt["integrity_contract"]["full_pose_identity_claimed"] is False
    invariant = receipt["invariant_joint_contract"]
    assert len(invariant["invariant_joints"]) == 28
    assert [row["index"] for row in invariant["excluded_joints"]] == [26, 28, 30]
    assert all(
        row["joint_pos_max_abs_rad"] > 0.0
        for row in invariant["excluded_joints"]
    )
    assert invariant["same_index_match_count"] == 6
    assert invariant["nonidentity_match_count"] == 0
    assert receipt["se2_world_contract"]["pass"] is True
    assert receipt["se2_world_contract"]["excluded_world_bodies"] == [
        "right_wrist_roll_Link"
    ]
    assert receipt["authorization"]["behavior_authorized"] is False
    assert (
        "complete_pose_identity_when_any_joint_or_world_body_is_excluded"
        in receipt["non_claims"]
    )


def test_full_joint_pair_may_claim_full_pose_identity_modulo_bound_se2(tmp_path):
    spec, _ = _fixture(tmp_path, grip_bake=False)
    receipt = build_frame_identity_receipt(spec)
    assert receipt["integrity_contract"]["full_pose_identity_claimed"] is True
    assert receipt["invariant_joint_contract"]["excluded_joints"] == []
    assert receipt["se2_world_contract"]["included_world_body_count"] == 4
    contract = receipt["npz_contract"]["array_contract"]
    assert contract["fps"] == {"shape": [1], "dtype": "<i8", "value": 50}
    assert contract["kinematics_schema_version"] == {
        "shape": [1],
        "dtype": "<i8",
        "value": 2,
    }
    assert contract["body_pos_point"]["value"] == "link_origin"
    assert contract["body_names"]["shape"] == [4]


def test_excluding_a_world_body_disables_full_pose_identity_claim(tmp_path):
    spec, _ = _fixture(tmp_path, grip_bake=False)
    spec["excluded_world_bodies"] = ["right_wrist_roll_Link"]
    receipt = build_frame_identity_receipt(spec)
    assert receipt["integrity_contract"]["full_pose_identity_claimed"] is False


def test_receipt_publication_is_no_clobber_and_verifies_by_exact_rebuild(tmp_path):
    spec, _ = _fixture(tmp_path, grip_bake=True)
    receipt = build_frame_identity_receipt(spec)
    output = tmp_path / "receipt.json"
    receipt_sha = write_frame_identity_receipt(receipt, output)
    assert receipt_sha == hashlib.sha256(canonical_json_bytes(receipt)).hexdigest()

    verified = verify_frame_identity_receipt(
        output, expected_receipt_sha256=receipt_sha
    )
    assert verified["receipt_id"] == "test_motion_frame_identity_v1"
    with pytest.raises(FrameIdentityError, match="already exists"):
        write_frame_identity_receipt(receipt, output)


def test_writer_rejects_nested_authorization_tamper_before_publication(tmp_path):
    spec, _ = _fixture(tmp_path, grip_bake=True)
    receipt = json.loads(canonical_json_bytes(build_frame_identity_receipt(spec)))
    receipt["authorization"]["behavior_authorized"] = True
    output = tmp_path / "tampered.json"
    with pytest.raises(FrameIdentityError, match="non-false authorization"):
        write_frame_identity_receipt(receipt, output)
    assert not output.exists()


def test_writer_rejects_derived_field_tamper_before_publication(tmp_path):
    spec, _ = _fixture(tmp_path, grip_bake=True)
    receipt = json.loads(canonical_json_bytes(build_frame_identity_receipt(spec)))
    receipt["event_mapping"]["bound_source_frame"] = 3
    output = tmp_path / "tampered.json"
    with pytest.raises(FrameIdentityError, match="does not exactly equal"):
        write_frame_identity_receipt(receipt, output)
    assert not output.exists()


def test_writer_rejects_broken_symlink_destination_without_creating_target(
    tmp_path,
):
    spec, _ = _fixture(tmp_path, grip_bake=True)
    receipt = build_frame_identity_receipt(spec)
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "created.json"
    output = tmp_path / "receipt.json"
    output.symlink_to(target)
    with pytest.raises(FrameIdentityError, match="already exists"):
        write_frame_identity_receipt(receipt, output)
    assert output.is_symlink()
    assert not target.exists()


def test_verifier_rejects_input_tamper_after_receipt_publication(tmp_path):
    spec, paths = _fixture(tmp_path, grip_bake=True)
    receipt = build_frame_identity_receipt(spec)
    output = tmp_path / "receipt.json"
    receipt_sha = write_frame_identity_receipt(receipt, output)
    paths["event_provenance"].write_text("{}\n", encoding="utf-8")
    with pytest.raises(FrameIdentityError, match="SHA-256 mismatch"):
        verify_frame_identity_receipt(
            output, expected_receipt_sha256=receipt_sha
        )


def test_invariant_channel_difference_fails_closed(tmp_path):
    spec, paths = _fixture(tmp_path, grip_bake=True)
    with np.load(paths["bound_npz"], allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    arrays["joint_pos"] = arrays["joint_pos"].copy()
    arrays["joint_pos"][4, 0] += np.float32(0.01)
    np.savez(paths["bound_npz"], **arrays)
    spec["inputs"]["bound_npz"]["sha256"] = _sha(paths["bound_npz"])
    _refresh_test_authority_bindings(spec, paths)
    with pytest.raises(FrameIdentityError, match="invariant joint_pos"):
        build_frame_identity_receipt(spec)


def test_npz_scalar_dtype_drift_fails_closed(tmp_path):
    spec, paths = _fixture(tmp_path, grip_bake=True)
    with np.load(paths["bound_npz"], allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    arrays["fps"] = arrays["fps"].astype(np.int32)
    np.savez(paths["bound_npz"], **arrays)
    spec["inputs"]["bound_npz"]["sha256"] = _sha(paths["bound_npz"])
    _refresh_test_authority_bindings(spec, paths)
    with pytest.raises(FrameIdentityError, match=r"shape \(1,\) and dtype <i8"):
        build_frame_identity_receipt(spec)


def test_npz_point_semantics_must_match_between_inputs(tmp_path):
    spec, paths = _fixture(tmp_path, grip_bake=True)
    with np.load(paths["bound_npz"], allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    arrays["body_lin_vel_point"] = np.asarray("center_of_mass")
    np.savez(paths["bound_npz"], **arrays)
    spec["inputs"]["bound_npz"]["sha256"] = _sha(paths["bound_npz"])
    _refresh_test_authority_bindings(spec, paths)
    with pytest.raises(FrameIdentityError, match="schema, fps, shape, or body order"):
        build_frame_identity_receipt(spec)


def test_declared_resampling_formula_is_validated_not_only_compared(tmp_path):
    spec, paths = _fixture(tmp_path, grip_bake=True)
    for name in ("event_provenance", "bound_provenance"):
        provenance = json.loads(paths[name].read_text(encoding="utf-8"))
        provenance["output"]["frame_formula"] = "same_but_wrong"
        _write_json(paths[name], provenance)
        spec["inputs"][name]["sha256"] = _sha(paths[name])
    with pytest.raises(FrameIdentityError, match="frame_formula must equal"):
        build_frame_identity_receipt(spec)


def test_world_residual_tolerances_cannot_exceed_fixed_v1_ceilings(tmp_path):
    spec, _ = _fixture(tmp_path, grip_bake=True)
    spec["world_residual_tolerances"]["position_m"] = 1.0e100
    with pytest.raises(FrameIdentityError, match="fixed v1 ceilings"):
        build_frame_identity_receipt(spec)


def test_invariant_joint_undercoverage_fails_closed(tmp_path):
    spec, _ = _fixture(tmp_path, grip_bake=True)
    spec["invariant_joints"] = spec["invariant_joints"][:1]
    with pytest.raises(FrameIdentityError, match="all 31 runtime joints"):
        build_frame_identity_receipt(spec)


def test_event_frame_cannot_be_injected_by_spec(tmp_path):
    spec, _ = _fixture(tmp_path, grip_bake=True)
    spec["event_source_frame"] = 0
    with pytest.raises(FrameIdentityError, match="extra=\\['event_source_frame'\\]"):
        build_frame_identity_receipt(spec)


def test_event_authority_must_bind_exact_event_source_sha(tmp_path):
    spec, paths = _fixture(tmp_path, grip_bake=True)
    authority = json.loads(
        paths["event_authority"].read_text(encoding="utf-8")
    )
    authority["motions"][0]["nominal_event"]["event_source_sha256"] = "c" * 64
    _write_json(paths["event_authority"], authority)
    spec["inputs"]["event_authority"]["sha256"] = _sha(
        paths["event_authority"]
    )
    with pytest.raises(FrameIdentityError, match="event source SHA disagrees"):
        build_frame_identity_receipt(spec)


def test_event_frame_is_derived_from_content_bound_authority(tmp_path):
    spec, paths = _fixture(tmp_path, grip_bake=True)
    authority = json.loads(
        paths["event_authority"].read_text(encoding="utf-8")
    )
    authority["motions"][0]["nominal_event"]["frame"] = 3
    _write_json(paths["event_authority"], authority)
    spec["inputs"]["event_authority"]["sha256"] = _sha(
        paths["event_authority"]
    )
    receipt = build_frame_identity_receipt(spec)
    assert receipt["event_mapping"]["event_source_frame"] == 3
    assert (
        receipt["event_authority_contract"][
            "event_frame_derived_from_authority_not_spec"
        ]
        is True
    )


def test_nonunique_invariant_frame_fingerprints_fail_closed(tmp_path):
    spec, _ = _fixture(
        tmp_path, grip_bake=True, duplicate_invariant_frame=True
    )
    with pytest.raises(FrameIdentityError, match="not all unique"):
        build_frame_identity_receipt(spec)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("observed_ball_contact", True, "observed_ball_contact"),
        ("behavior_authorized", True, "behavior_authorized"),
        (
            "synthetic_behavior_nominal_claim",
            True,
            "synthetic_behavior_nominal_claim",
        ),
    ],
)
def test_semantic_promotion_is_always_rejected(tmp_path, field, value, match):
    spec, _ = _fixture(tmp_path, grip_bake=True)
    spec["semantic_contract"][field] = value
    with pytest.raises(FrameIdentityError, match=match):
        build_frame_identity_receipt(spec)


def test_joint_name_and_index_are_both_bound(tmp_path):
    spec, _ = _fixture(tmp_path, grip_bake=True)
    spec["invariant_joints"][0]["name"] = "wrong_name"
    with pytest.raises(FrameIdentityError, match="disagrees with joint order"):
        build_frame_identity_receipt(spec)


def test_each_input_file_is_read_once_for_hash_and_parse(tmp_path, monkeypatch):
    spec, paths = _fixture(tmp_path, grip_bake=True)
    original = Path.read_bytes
    counts: dict[Path, int] = {}

    def counted(path: Path) -> bytes:
        resolved = path.resolve()
        counts[resolved] = counts.get(resolved, 0) + 1
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", counted)
    build_frame_identity_receipt(spec)
    assert counts == {path.resolve(): 1 for path in paths.values()}
