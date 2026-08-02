"""Measured-paddle motion-bank contract tests on the real MotionLoader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from test_reward_flags_mdp import (  # installs Isaac Lab stubs, then loads real commands.py
    _BODY_NAMES,
    _write_motion_npz,
    commands_mod,
)


def _add_measured_racket(path: Path, *, complete: bool = True) -> None:
    with np.load(path, allow_pickle=False) as source:
        arrays = {key: np.asarray(source[key]) for key in source.files}
    frames = int(arrays["joint_pos"].shape[0])
    arrays.update(
        {
            "measured_racket_site_pos_w": np.zeros((frames, 3), dtype=np.float32),
            "measured_racket_normal_w": np.tile(
                np.asarray([0.0, -1.0, 0.0], dtype=np.float32), (frames, 1)
            ),
            "measured_racket_long_axis_w": np.tile(
                np.asarray([1.0, 0.0, 0.0], dtype=np.float32), (frames, 1)
            ),
            "measured_racket_schema_version": np.asarray([4], dtype=np.int64),
            "measured_racket_position_semantics": np.asarray("physical_blade_center"),
            "measured_racket_normal_semantics": np.asarray(
                "signed_physical_hitting_face"
            ),
            "measured_racket_long_axis_semantics": np.asarray(
                "measured_paddle_butt_to_blade"
            ),
            "measured_racket_robot_mount_normal_sign": np.asarray([-1], dtype=np.int8),
            "measured_racket_robot_butt_to_blade_axis_local": np.asarray(
                [1.0 / np.sqrt(2.0), 0.0, 1.0 / np.sqrt(2.0)],
                dtype=np.float64,
            ),
            "measured_racket_robot_rigid_visual_mesh_sha256": np.asarray(
                "442ff2ecb82d3da481f1500d8a788192ba7d8bc2969f4d8c9d98266ea116b4dd"
            ),
            "measured_racket_source_sha256": np.asarray("a" * 64),
            "measured_racket_retarget_admitted": np.asarray([1], dtype=np.int64),
            "measured_racket_retarget_receipt_sha256": np.asarray("b" * 64),
            "measured_racket_joint_order_contract_id": np.asarray(
                "a3-gmr-dof-pos-to-runtime-articulation-v1"
            ),
            "measured_racket_joint_order_contract_sha256": np.asarray("c" * 64),
        }
    )
    if not complete:
        arrays.pop("measured_racket_retarget_receipt_sha256")
    np.savez(path, **arrays)


def _loader(paths: list[Path]):
    return commands_mod.MotionLoader(
        [str(path) for path in paths],
        [0],
        articulation_body_names=_BODY_NAMES,
        selected_body_names=[_BODY_NAMES[0]],
        device="cpu",
    )


def test_complete_measured_racket_bank_is_loaded(tmp_path: Path) -> None:
    first = Path(_write_motion_npz(tmp_path / "first.npz", 5))
    second = Path(_write_motion_npz(tmp_path / "second.npz", 7))
    _add_measured_racket(first)
    _add_measured_racket(second)

    loader = _loader([first, second])

    assert loader.measured_racket_available is True
    assert len(loader.measured_racket_contracts) == 2
    assert all(contract is not None for contract in loader.measured_racket_contracts)
    assert loader.measured_racket_mount_normal_sign_per_clip == (-1, -1)
    assert tuple(loader._measured_racket_site_pos_w.shape) == (12, 3)
    assert loader._measured_racket_normal_w[:, 1].tolist() == [-1.0] * 12
    assert loader._measured_racket_long_axis_w[:, 0].tolist() == [1.0] * 12


def test_partial_contract_and_mixed_bank_fail_closed(tmp_path: Path) -> None:
    partial = Path(_write_motion_npz(tmp_path / "partial.npz", 5))
    _add_measured_racket(partial, complete=False)
    with pytest.raises(ValueError, match="partial measured-racket contract"):
        _loader([partial])

    measured = Path(_write_motion_npz(tmp_path / "measured.npz", 5))
    legacy = Path(_write_motion_npz(tmp_path / "legacy.npz", 5))
    _add_measured_racket(measured)
    with pytest.raises(ValueError, match="mixed measured-racket availability"):
        _loader([measured, legacy])


def test_invalid_measured_mount_sign_fails_closed(tmp_path: Path) -> None:
    motion = Path(_write_motion_npz(tmp_path / "bad-sign.npz", 5))
    _add_measured_racket(motion)
    with np.load(motion, allow_pickle=False) as source:
        arrays = {key: np.asarray(source[key]) for key in source.files}
    arrays["measured_racket_robot_mount_normal_sign"] = np.asarray([0], dtype=np.int8)
    np.savez(motion, **arrays)

    with pytest.raises(ValueError, match="robot_mount_normal_sign must be scalar"):
        _loader([motion])


def test_nonorthogonal_measured_racket_axes_fail_closed(tmp_path: Path) -> None:
    motion = Path(_write_motion_npz(tmp_path / "nonorthogonal.npz", 5))
    _add_measured_racket(motion)
    with np.load(motion, allow_pickle=False) as source:
        arrays = {key: np.asarray(source[key]) for key in source.files}
    arrays["measured_racket_long_axis_w"] = np.tile(
        np.asarray([0.0, -1.0, 0.0], dtype=np.float32), (5, 1)
    )
    np.savez(motion, **arrays)

    with pytest.raises(ValueError, match="face/long axes are not orthogonal"):
        _loader([motion])
