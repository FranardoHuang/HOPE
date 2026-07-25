"""Contract tests for the §5.1 protected-window digest and receipt."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import canonical_protected_window as cpw  # noqa: E402


def _write_motion(path: Path, *, frames: int = 8, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    np.savez(
        path,
        fps=np.float64(50.0),
        joint_pos=rng.normal(size=(frames, 31)).astype("<f4"),
        joint_vel=rng.normal(size=(frames, 31)).astype("<f4"),
        body_pos_w=rng.normal(size=(frames, 32, 3)).astype("<f4"),
        body_quat_w=rng.normal(size=(frames, 32, 4)).astype("<f4"),
        body_lin_vel_w=rng.normal(size=(frames, 32, 3)).astype("<f4"),
        body_ang_vel_w=rng.normal(size=(frames, 32, 3)).astype("<f4"),
    )


def _bindings() -> dict[str, str]:
    return {
        "recipe_sha256": "1" * 64,
        "compiler_sha256": "2" * 64,
        "mjcf_sha256": "3" * 64,
        "urdf_sha256": "4" * 64,
        "body_order_sha256": "5" * 64,
    }


def _receipt(tmp_path: Path):
    source = tmp_path / "source.npz"
    output = tmp_path / "output.npz"
    _write_motion(source, seed=1)
    _write_motion(output, seed=2)
    source_digest = cpw.compute_protected_window_digest(
        source,
        role="source",
        motion_id="bh_block",
        scope="upper",
        frame_indices=cpw.source_window_indices((2, 5)),
    )
    indices, _, _ = cpw.output_window_indices(1.25, 4.75)
    output_digest = cpw.compute_protected_window_digest(
        output,
        role="output",
        motion_id="bh_block",
        scope="upper",
        frame_indices=indices,
    )
    receipt = cpw.build_transformation_receipt(
        motion_id="bh_block",
        scope="upper",
        source_digest=source_digest,
        output_digest=output_digest,
        source_span_inclusive=(2, 5),
        output_window_start_fractional_frame=1.25,
        output_window_end_fractional_frame=4.75,
        entry_frame=1,
        exit_frame=6,
        marker_time_map={
            "window_start": {"source_index": 2, "time_s": 0.04},
            "source_anchor": {"source_index": 4, "time_s": 0.08},
            "window_end": {"source_index": 5, "time_s": 0.10},
        },
        binding_sha256=_bindings(),
        allowed_transforms=["body_scope", "retime", "schema2_rebuild"],
    )
    return receipt, source, output


def test_source_indices_cover_every_inclusive_span_frame():
    assert cpw.source_window_indices((3, 6)) == (3, 4, 5, 6)
    with pytest.raises(cpw.ProtectedWindowError):
        cpw.source_window_indices((4, 3))


def test_output_indices_use_ceil_floor_and_exact_hex_not_rounding():
    indices, start_hex, end_hex = cpw.output_window_indices(1.9, 4.1)
    # Nearest-frame rounding would give (2, 3, 4) from 1.9 -> 2 and 4.1 -> 4,
    # which happens to agree, so pin a case where they differ:
    assert indices == (2, 3, 4)
    indices2, _, _ = cpw.output_window_indices(1.2, 4.9)
    assert indices2 == (2, 3, 4)  # ceil(1.2)=2, floor(4.9)=4 — never 1 or 5
    assert float.fromhex(start_hex) == 1.9
    assert float.fromhex(end_hex) == 4.1
    with pytest.raises(cpw.ProtectedWindowError):
        cpw.output_window_indices(2.4, 2.6)  # no integer frame inside


def test_digest_binds_bytes_role_and_indices(tmp_path):
    motion = tmp_path / "m.npz"
    _write_motion(motion, seed=3)
    base = cpw.compute_protected_window_digest(
        motion,
        role="source",
        motion_id="fh_loop",
        scope="upper",
        frame_indices=(1, 2, 3),
    )
    same = cpw.compute_protected_window_digest(
        motion,
        role="source",
        motion_id="fh_loop",
        scope="upper",
        frame_indices=(1, 2, 3),
    )
    assert base.digest_sha256 == same.digest_sha256

    other_indices = cpw.compute_protected_window_digest(
        motion,
        role="source",
        motion_id="fh_loop",
        scope="upper",
        frame_indices=(1, 2, 3, 4),
    )
    assert other_indices.digest_sha256 != base.digest_sha256

    other_role = cpw.compute_protected_window_digest(
        motion,
        role="output",
        motion_id="fh_loop",
        scope="upper",
        frame_indices=(1, 2, 3),
    )
    assert other_role.digest_sha256 != base.digest_sha256

    other_scope = cpw.compute_protected_window_digest(
        motion,
        role="source",
        motion_id="fh_loop",
        scope="full",
        frame_indices=(1, 2, 3),
    )
    assert other_scope.digest_sha256 != base.digest_sha256

    # One float in one channel inside the window flips the digest.
    with np.load(motion, allow_pickle=False) as payload:
        arrays = {name: payload[name].copy() for name in payload.files}
    arrays["body_ang_vel_w"][2, 7, 1] += np.float32(1e-3)
    np.savez(motion, **arrays)
    changed = cpw.compute_protected_window_digest(
        motion,
        role="source",
        motion_id="fh_loop",
        scope="upper",
        frame_indices=(1, 2, 3),
    )
    assert changed.digest_sha256 != base.digest_sha256


def test_digest_rejects_wrong_dtype_and_missing_channel(tmp_path):
    motion = tmp_path / "bad_dtype.npz"
    _write_motion(motion, seed=4)
    with np.load(motion, allow_pickle=False) as payload:
        arrays = {name: payload[name].copy() for name in payload.files}
    arrays["joint_pos"] = arrays["joint_pos"].astype("<f8")
    np.savez(motion, **arrays)
    with pytest.raises(cpw.ProtectedWindowError, match="stored as <f4"):
        cpw.compute_protected_window_digest(
            motion,
            role="source",
            motion_id="fh_loop",
            scope="upper",
            frame_indices=(0, 1),
        )

    missing = tmp_path / "missing.npz"
    _write_motion(missing, seed=5)
    with np.load(missing, allow_pickle=False) as payload:
        arrays = {
            name: payload[name].copy()
            for name in payload.files
            if name != "body_quat_w"
        }
    np.savez(missing, **arrays)
    with pytest.raises(cpw.ProtectedWindowError, match="missing timed channels"):
        cpw.compute_protected_window_digest(
            missing,
            role="source",
            motion_id="fh_loop",
            scope="upper",
            frame_indices=(0, 1),
        )


def test_receipt_round_trip_and_cross_splice_rejection(tmp_path):
    receipt, source, output = _receipt(tmp_path)
    verdict = cpw.verify_transformation_receipt(
        receipt, source_path=source, output_path=output
    )
    assert verdict["verdict"] == "WINDOW_DIGESTS_REPRODUCED"

    # Swapping the two files is a cross-splice and must fail.
    with pytest.raises(cpw.ProtectedWindowError):
        cpw.verify_transformation_receipt(
            receipt, source_path=output, output_path=source
        )

    # Any edit to the receipt body breaks its own seal.
    tampered = dict(receipt)
    tampered["entry_frame"] = 2
    with pytest.raises(cpw.ProtectedWindowError, match="drifted"):
        cpw.verify_transformation_receipt(
            tampered, source_path=source, output_path=output
        )

    # Rewriting output bytes (same shape) must be caught by the motion SHA.
    _write_motion(output, seed=9)
    with pytest.raises(cpw.ProtectedWindowError, match="drifted|reproduce"):
        cpw.verify_transformation_receipt(
            receipt, source_path=source, output_path=output
        )


def test_receipt_requires_distinct_digests_and_full_bindings(tmp_path):
    source = tmp_path / "s.npz"
    _write_motion(source, seed=6)
    digest = cpw.compute_protected_window_digest(
        source,
        role="source",
        motion_id="bh_block",
        scope="upper",
        frame_indices=cpw.source_window_indices((1, 3)),
    )
    output_digest = cpw.ProtectedWindowDigest(
        role="output",
        motion_id="bh_block",
        scope="upper",
        motion_sha256=digest.motion_sha256,
        frame_indices=digest.frame_indices,
        header=digest.header,
        digest_sha256=digest.digest_sha256,
    )
    with pytest.raises(cpw.ProtectedWindowError, match="identical"):
        cpw.build_transformation_receipt(
            motion_id="bh_block",
            scope="upper",
            source_digest=digest,
            output_digest=output_digest,
            source_span_inclusive=(1, 3),
            output_window_start_fractional_frame=1.0,
            output_window_end_fractional_frame=3.0,
            entry_frame=0,
            exit_frame=4,
            marker_time_map={"window_start": {"source_index": 1}},
            binding_sha256=_bindings(),
            allowed_transforms=["retime"],
        )

    bindings = _bindings()
    bindings.pop("mjcf_sha256")
    with pytest.raises(cpw.ProtectedWindowError, match="missing"):
        cpw.build_transformation_receipt(
            motion_id="bh_block",
            scope="upper",
            source_digest=digest,
            output_digest=cpw.ProtectedWindowDigest(
                role="output",
                motion_id="bh_block",
                scope="upper",
                motion_sha256="a" * 64,
                frame_indices=(1, 2, 3),
                header={},
                digest_sha256="b" * 64,
            ),
            source_span_inclusive=(1, 3),
            output_window_start_fractional_frame=1.0,
            output_window_end_fractional_frame=3.0,
            entry_frame=0,
            exit_frame=4,
            marker_time_map={"window_start": {"source_index": 1}},
            binding_sha256=bindings,
            allowed_transforms=["retime"],
        )
