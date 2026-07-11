"""Dependency-light tests for motion body-velocity point provenance/migration."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import migrate_motion_kinematics as migrate  # noqa: E402
import motion_kinematics_contract as contract  # noqa: E402


def _base_motion(T=7, B=2):
    pos = np.zeros((T, B, 3), dtype=np.float32)
    quat = np.zeros((T, B, 4), dtype=np.float32)
    quat[..., 0] = 1.0
    ang = np.zeros((T, B, 3), dtype=np.float32)
    return {
        "fps": np.array([50], dtype=np.int64),
        "joint_pos": np.zeros((T, 1), dtype=np.float32),
        "joint_vel": np.zeros((T, 1), dtype=np.float32),
        "body_pos_w": pos,
        "body_quat_w": quat,
        "body_lin_vel_w": np.zeros_like(pos),
        "body_ang_vel_w": ang,
    }


def _body_names(B: int) -> list[str]:
    return [f"body_{index}" for index in range(B)]


def test_declared_metadata_round_trip_is_exact():
    data = _base_motion()
    data.update(contract.metadata_arrays(body_names=_body_names(2)))
    meta = contract.read_metadata(data)
    assert meta.schema_version == 2
    assert meta.body_pos_point == "link_origin"
    assert meta.body_lin_vel_point == "center_of_mass"
    assert meta.body_names == tuple(_body_names(2))
    assert meta.exact_motion_command_v2


def test_schema_one_or_missing_body_order_is_never_exact():
    data = _base_motion()
    data.update(
        kinematics_schema_version=np.array([1]),
        body_pos_point=np.array("link_origin"),
        body_lin_vel_point=np.array("center_of_mass"),
    )
    meta = contract.read_metadata(data)
    assert meta.schema_version == 1
    assert meta.body_names is None
    assert not meta.exact_motion_command_v2

    data["kinematics_schema_version"] = np.array([2])
    with pytest.raises(ValueError, match="requires body_names"):
        contract.read_metadata(data)


def test_explicit_and_embedded_body_orders_must_agree():
    data = _base_motion()
    data.update(contract.metadata_arrays(body_names=["pelvis", "torso"]))
    assert contract.resolve_body_names(
        data, explicit_body_names=["pelvis", "torso"], expected_count=2
    ) == ("pelvis", "torso")
    with pytest.raises(ValueError, match="disagrees"):
        contract.resolve_body_names(
            data, explicit_body_names=["torso", "pelvis"], expected_count=2
        )


def test_link_to_com_velocity_uses_omega_cross_rotated_com_offset():
    data = _base_motion(T=3, B=1)
    data["body_lin_vel_w"][:] = [1.0, 2.0, 3.0]
    data["body_ang_vel_w"][:] = [0.0, 0.0, 10.0]
    com = np.array([[0.2, 0.0, 0.0]])
    out = contract.link_velocity_to_com(
        data["body_lin_vel_w"], data["body_ang_vel_w"], data["body_quat_w"], com
    )
    np.testing.assert_allclose(out, np.array([[[1.0, 4.0, 3.0]]] * 3), atol=1.0e-12)


def test_content_signature_detects_old_mujoco_link_fd_but_not_static_clip():
    data = _base_motion(T=7, B=1)
    t = np.arange(7, dtype=np.float32) / 50.0
    data["body_pos_w"][:, 0, 0] = 2.0 * t
    data["body_lin_vel_w"] = np.gradient(data["body_pos_w"], 1.0 / 50.0, axis=0)
    data["body_ang_vel_w"][:, 0, 2] = 3.0
    sig = contract.link_fd_signature(data)
    assert sig["suspected_link_origin"]
    assert sig["max_abs_mps"] < 1.0e-6

    data["body_ang_vel_w"][:] = 0.0
    sig = contract.link_fd_signature(data)
    assert not sig["suspected_link_origin"]
    assert sig["ambiguous"]


def test_migration_tags_and_converts_without_mutating_source():
    data = _base_motion(T=3, B=1)
    data["body_lin_vel_w"][:] = [1.0, 2.0, 3.0]
    data["body_ang_vel_w"][:] = [0.0, 0.0, 10.0]
    source_copy = data["body_lin_vel_w"].copy()
    out, report = migrate.migrate_arrays(
        data,
        source_point="link_origin",
        com_pos_b=np.array([[0.2, 0.0, 0.0]]),
        body_names=["body_0"],
    )
    np.testing.assert_array_equal(data["body_lin_vel_w"], source_copy)
    np.testing.assert_allclose(out["body_lin_vel_w"], [[[1.0, 4.0, 3.0]]] * 3, atol=1.0e-6)
    assert contract.read_metadata(out).exact_motion_command_v2
    assert report["max_velocity_delta_mps"] == 2.0


def test_center_of_mass_assertion_is_metadata_only_and_bit_preserving():
    data = _base_motion(T=3, B=1)
    data["body_lin_vel_w"][:] = [1.25, -0.5, 0.75]
    out, report = migrate.migrate_arrays(
        data,
        source_point="center_of_mass",
        com_pos_b=None,
        body_names=["body_0"],
    )
    np.testing.assert_array_equal(out["body_lin_vel_w"], data["body_lin_vel_w"])
    assert report["max_velocity_delta_mps"] == 0.0
    assert contract.read_metadata(out).exact_motion_command_v2


def test_migration_reorders_every_body_array_from_source_to_runtime_order():
    data = _base_motion(T=3, B=2)
    for index, key in enumerate(migrate.BODY_ARRAY_KEYS):
        width = data[key].shape[-1]
        data[key][:, 0] = np.arange(width, dtype=np.float32) + 10 * index
        data[key][:, 1] = np.arange(width, dtype=np.float32) + 100 + 10 * index
    joint_pos = data["joint_pos"].copy()

    out, report = migrate.migrate_arrays(
        data,
        source_point="center_of_mass",
        com_pos_b=None,
        body_names=["old_first", "old_second"],
        target_body_names=["old_second", "old_first"],
    )

    for key in migrate.BODY_ARRAY_KEYS:
        np.testing.assert_array_equal(out[key][:, 0], data[key][:, 1])
        np.testing.assert_array_equal(out[key][:, 1], data[key][:, 0])
    np.testing.assert_array_equal(out["joint_pos"], joint_pos)
    assert contract.read_metadata(out).body_names == ("old_second", "old_first")
    assert report["body_order_reordered"] is True
    assert report["body_permutation"] == [1, 0]


def test_migration_rejects_target_body_order_that_is_not_a_permutation():
    with pytest.raises(ValueError, match="must be a permutation"):
        migrate.migrate_arrays(
            _base_motion(T=3, B=2),
            source_point="center_of_mass",
            com_pos_b=None,
            body_names=["a", "b"],
            target_body_names=["a", "c"],
        )


def test_static_bootstrap_declares_schema_two_body_order_semantics():
    source = (SCRIPTS / "make_static_motion.py").read_text(encoding="utf-8")
    assert "from motion_kinematics_contract import metadata_arrays" in source
    assert "log.update(metadata_arrays(body_names=robot.body_names))" in source


def test_legacy_link_velocity_runtime_escape_is_diagnostic_and_default_off():
    source = (
        Path(__file__).resolve().parents[1]
        / "source" / "whole_body_tracking" / "whole_body_tracking"
        / "tasks" / "tracking" / "mdp" / "commands.py"
    ).read_text(encoding="utf-8")
    assert "allow_legacy_link_origin_velocity: bool = False" in source
    assert '"status": "legacy_link_origin_velocity_diagnostic_only"' in source

    train = (Path(__file__).resolve().parents[1] / "scripts" / "train.py").read_text(
        encoding="utf-8"
    )
    assert '"allow_legacy_link_origin_velocity",' in train
    assert "task.motion.allow_legacy_link_origin_velocity" in train
    assert "motion_kinematics_exact=false and every descendant remains inexact" in train
