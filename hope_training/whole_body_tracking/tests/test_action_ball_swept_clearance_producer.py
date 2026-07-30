"""Dedicated tests for the continuous ActionBall swept-clearance producer."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[3]
SCRIPTS = REPO / "hope_training/whole_body_tracking/scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import certify_action_ball_swept_clearance as swept  # noqa: E402
import canonical_motion_bank_gate as bank_gate  # noqa: E402
import test_canonical_motion_arbitrary_bank as arbitrary_fixtures  # noqa: E402
from mujoco_motion_player import RUNTIME_BODY_NAMES  # noqa: E402


def _snapshot(tmp_path: Path, name: str, payload: bytes) -> swept.FileSnapshot:
    path = tmp_path / name
    path.write_bytes(payload)
    return swept.read_snapshot(path, label=name)


def _envelopes(*, joint_reach: float = 0.0) -> tuple[swept.GeomEnvelope, ...]:
    reaches = (joint_reach,) + (0.0,) * 30
    return (
        swept.GeomEnvelope(
            name="torso_collision",
            body_name="torso_Link",
            root_rotation_reach_m=0.0,
            joint_rotation_reach_m=reaches,
            geom_rbound_m=0.1,
        ),
        swept.GeomEnvelope(
            name="right_racket_collision",
            body_name="right_wrist_yaw_Link",
            root_rotation_reach_m=0.0,
            joint_rotation_reach_m=reaches,
            geom_rbound_m=0.1,
        ),
        swept.GeomEnvelope(
            name="right_racket_handle_collision",
            body_name="right_wrist_yaw_Link",
            root_rotation_reach_m=0.0,
            joint_rotation_reach_m=reaches,
            geom_rbound_m=0.1,
        ),
    )


class FakeBackend:
    def __init__(
        self,
        distances: dict[tuple[str, str], float] | None = None,
        *,
        joint_reach: float = 0.0,
    ) -> None:
        self.robot_geometries = _envelopes(joint_reach=joint_reach)
        self.obstacle_roles = swept.ACTION_BALL_ROLES
        self.distances = distances or {}
        self.pose_count = 0

    def apply_pose(
        self,
        root_position: np.ndarray,
        root_quaternion_wxyz: np.ndarray,
        joint_position: np.ndarray,
    ) -> None:
        assert root_position.shape == (3,)
        assert root_quaternion_wxyz.shape == (4,)
        assert joint_position.shape == (31,)
        self.pose_count += 1

    def distance_saturation_query(
        self, robot_geom_name: str, obstacle_role: str, distmax_m: float
    ) -> tuple[float, bool]:
        actual = self.distances.get(
            (robot_geom_name, obstacle_role), 0.02
        )
        if not np.isfinite(actual):
            return float(actual), False
        if actual >= distmax_m:
            return float(distmax_m), True
        return float(actual), False


def _clip(
    tmp_path: Path,
    scope: str,
    *,
    frames: int = 2,
    first_joint_delta: float = 0.0,
) -> swept.MotionClip:
    snapshot = _snapshot(
        tmp_path,
        f"stroke_{scope}_canonical_v2.npz",
        f"exact-{scope}-motion-bytes".encode(),
    )
    joint_pos = np.zeros((frames, 31), dtype=np.float64)
    if first_joint_delta:
        joint_pos[-1, 0] = first_joint_delta
    root_pos = np.zeros((frames, 32, 3), dtype=np.float64)
    root_quat = np.zeros((frames, 32, 4), dtype=np.float64)
    root_quat[..., 0] = 1.0
    return swept.MotionClip(
        motion_id="stroke",
        scope=scope,
        snapshot=snapshot,
        fps=1.0,
        joint_pos=joint_pos,
        body_pos_w=root_pos,
        body_quat_w=root_quat,
        contact_window_start_s=0.25,
        contact_window_end_s=0.75,
    )


def _scene_contract() -> dict[str, Any]:
    raw = (
        ("top", (-2.0, -0.5, 0.70), (2.0, 0.5, 0.75)),
        ("keepout", (-2.0, -0.5, 0.0), (2.0, 0.5, 0.70)),
        ("net", (-0.005, -0.6, 0.75), (0.005, 0.6, 0.90)),
        ("post_left", (-0.01, 0.59, 0.75), (0.01, 0.61, 0.92)),
        ("post_right", (-0.01, -0.61, 0.75), (0.01, -0.59, 0.92)),
    )
    components = []
    for role, lo_raw, hi_raw in raw:
        lo = np.asarray(lo_raw, dtype=np.float64)
        hi = np.asarray(hi_raw, dtype=np.float64)
        center = (lo + hi) / 2.0
        extents = hi - lo
        components.append(
            {
                "role": role,
                "geom_name": swept.OBSTACLE_GEOM_NAMES[role],
                "center_m": center.tolist(),
                "full_extents_m": extents.tolist(),
                "aabb_lo_m": lo.tolist(),
                "aabb_hi_m": hi.tolist(),
            }
        )
    return {
        "scene_profile": swept.SCENE_PROFILE,
        "with_table": True,
        "near_x_m": 0.5,
        "surface_z_m": 0.75,
        "keepout_floor_z_m": 0.0,
        "action_ball_keepout_semantics": "robot_only_keepout_ball_excluded",
        "roles": list(swept.ACTION_BALL_ROLES),
        "components": components,
        "components_sha256": swept._canonical_json_sha256(components),
    }


def _source_pins(tmp_path: Path) -> dict[str, swept.FileSnapshot]:
    return {
        role: _snapshot(tmp_path, f"{role}.py", f"# {role}\n".encode())
        for role in (
            "geometry",
            "table_frame",
            "hope_commands",
            "scene_builder",
            "joint_order",
        )
    }


def _robot_geometry() -> dict[str, Any]:
    names = [envelope.name for envelope in _envelopes()]
    rows = [{"name": name, "fixture": True} for name in names]
    return {
        "all_enabled_collision_geoms": True,
        "collision_geom_count": len(names),
        "collision_geom_names": names,
        "collision_geometry_rows": rows,
        "collision_geometry_sha256": swept._canonical_json_sha256(rows),
        "racket_and_handle_geom_names": list(
            swept.RACKET_AND_HANDLE_GEOMS
        ),
        "reach_envelopes": [],
    }


def _receipt(tmp_path: Path) -> dict[str, Any]:
    clips = [_clip(tmp_path, scope) for scope in swept.REQUESTED_SCOPES]
    backend = FakeBackend()
    results = [
        swept.certify_motion_continuous(
            clip, backend, max_subdivision_depth=0
        )
        for clip in clips
    ]
    outputs = [
        {
            "motion_id": clip.motion_id,
            "scope": clip.scope,
            "filename": clip.snapshot.path.name,
            "path": str(clip.snapshot.path),
            "bytes": clip.snapshot.size,
            "sha256": clip.snapshot.sha256,
            "frames": clip.frames,
            "fps": clip.fps,
            "duration_s": clip.duration_s,
            "contact_window_start_s": clip.contact_window_start_s,
            "contact_window_end_s": clip.contact_window_end_s,
        }
        for clip in clips
    ]
    bank_binding = {
        "manifest": {
            "path": str(tmp_path / "manifest.json"),
            "bytes": 1,
            "sha256": "1" * 64,
        },
        "recipe": {
            "path": str(tmp_path / "recipe.json"),
            "bytes": 1,
            "sha256": "2" * 64,
        },
        "ready": {
            "path": str(tmp_path / "ready.npz"),
            "bytes": 1,
            "sha256": "3" * 64,
        },
        "mjcf": {
            "path": str(tmp_path / "robot.xml"),
            "bytes": 1,
            "sha256": "4" * 64,
        },
        "urdf": {
            "path": str(tmp_path / "robot.urdf"),
            "bytes": 1,
            "sha256": "5" * 64,
        },
        "body_order": {
            "path": str(tmp_path / "body_order.txt"),
            "bytes": 1,
            "sha256": "6" * 64,
        },
        "station_center_shift_xy_m": [0.0, 0.0],
        "output_matrix": {
            "motion_ids": ["stroke"],
            "scopes": list(swept.REQUESTED_SCOPES),
            "candidate_count": 2,
        },
        "outputs": outputs,
    }
    return swept.build_receipt(
        bank_binding=bank_binding,
        scene_contract=_scene_contract(),
        source_pins=_source_pins(tmp_path),
        dependency_pins={
            "python": {"version": "fixture"},
            "numpy": {"version": np.__version__},
            "mujoco": {"version": "fixture"},
        },
        robot_geometry=_robot_geometry(),
        results=results,
    )


def _write_public_roundtrip_clip(
    tmp_path: Path, scope: str
) -> swept.MotionClip:
    path = tmp_path / f"stroke_{scope}_canonical_v2.npz"
    joint_pos = np.zeros((2, 31), dtype=np.float64)
    joint_vel = np.zeros_like(joint_pos)
    body_pos = np.zeros((2, 32, 3), dtype=np.float64)
    body_quat = np.zeros((2, 32, 4), dtype=np.float64)
    body_quat[..., 0] = 1.0
    np.savez(
        path,
        fps=np.asarray([1.0], dtype=np.float64),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        body_lin_vel_w=np.zeros_like(body_pos),
        body_ang_vel_w=np.zeros_like(body_pos),
        kinematics_schema_version=np.asarray([2], dtype=np.int64),
        body_pos_point=np.asarray("link_origin"),
        body_lin_vel_point=np.asarray("center_of_mass"),
        body_names=np.asarray([f"body_{index}" for index in range(32)]),
    )
    return swept.MotionClip(
        motion_id="stroke",
        scope=scope,
        snapshot=swept.read_snapshot(path, label=f"{scope} output"),
        fps=1.0,
        joint_pos=joint_pos,
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        contact_window_start_s=0.25,
        contact_window_end_s=0.75,
    )


def _arbitrary_pinned_bank(
    tmp_path: Path,
    monkeypatch,
    *,
    count: int = 5,
) -> tuple[
    arbitrary_fixtures.BankFixture,
    Any,
    Path,
    dict[str, Any],
    Path,
]:
    fixture = arbitrary_fixtures.BankFixture(
        tmp_path / "arbitrary",
        count,
        monkeypatch,
    )
    loaded = fixture.load()
    # The compiler unit fixture intentionally uses a one-line body-order
    # placeholder because its fake backend does not consume names.  This
    # producer test exercises the real 32-name clearance input contract.
    body_order = loaded.canonical_recipe.model_paths["body_order"]
    body_order.write_text(
        "\n".join(RUNTIME_BODY_NAMES) + "\n",
        encoding="utf-8",
    )
    object.__setattr__(
        loaded.canonical_recipe,
        "model_hashes",
        {
            **loaded.canonical_recipe.model_hashes,
            "body_order": swept._sha256_bytes(body_order.read_bytes()),
        },
    )
    monkeypatch.setattr(
        arbitrary_fixtures.arbitrary,
        "load_canonical_motion_recipe",
        lambda *args, **kwargs: loaded.canonical_recipe,
    )
    bank = fixture.root / "bank"
    bank.mkdir()
    ready = loaded.canonical_recipe.ready
    frames = 6
    fps = 50.0
    duration_s = (frames - 1) / fps
    outputs = []
    for motion_id in fixture.motion_ids:
        for scope in swept.REQUESTED_SCOPES:
            filename = f"{motion_id}_{scope}_canonical_v2.npz"
            path = bank / filename
            joint_pos = np.repeat(
                np.asarray(ready.joint_pos, dtype=np.float64)[None, :],
                frames,
                axis=0,
            )
            body_pos = np.zeros((frames, 32, 3), dtype=np.float64)
            body_pos[:, 0] = np.asarray(
                ready.root_pos_w,
                dtype=np.float64,
            )
            body_quat = np.zeros((frames, 32, 4), dtype=np.float64)
            body_quat[..., 0] = 1.0
            body_quat[:, 0] = np.asarray(
                ready.root_quat_wxyz,
                dtype=np.float64,
            )
            np.savez(
                path,
                fps=np.asarray([fps], dtype=np.float64),
                joint_pos=joint_pos,
                joint_vel=np.zeros_like(joint_pos),
                body_pos_w=body_pos,
                body_quat_w=body_quat,
                body_lin_vel_w=np.zeros_like(body_pos),
                body_ang_vel_w=np.zeros_like(body_pos),
                kinematics_schema_version=np.asarray([2], dtype=np.int64),
                body_pos_point=np.asarray("link_origin"),
                body_lin_vel_point=np.asarray("center_of_mass"),
                body_names=np.asarray(RUNTIME_BODY_NAMES),
            )
            outputs.append(
                {
                    "motion_id": motion_id,
                    "scope": scope,
                    "filename": filename,
                    "output_npz_sha256": swept._sha256_bytes(
                        path.read_bytes()
                    ),
                    # These are source-clip indices in the arbitrary compiler,
                    # not indices into the six-frame compiled output.
                    "entry_frame": 1,
                    "exit_frame": 7,
                    "duration_s": duration_s,
                    "contact_window_start_s": 0.02,
                    "contact_window_end_s": 0.06,
                    "source_anchor_time_s": 0.04,
                    "search": {
                        "contact_opportunity": {
                            "marker_only": True,
                            "acceleration_allowed_through_window_end": True,
                        }
                    },
                }
            )
    manifest = {
        "schema_version": 1,
        "library_id": loaded.raw["bank_id"],
        "publication_class": "compiler_candidate",
        "build_verdict": "PASS_COMPILER_CANDIDATE_ONLY",
        "training_authorized": False,
        "hardware_authorized": False,
        "recipe": {
            "path": str(fixture.recipe_path.resolve()),
            "sha256": loaded.sha256,
        },
        "ready": {
            "path": str(ready.path.resolve()),
            "sha256": ready.sha256,
            "direct_endpoint_for_every_motion": True,
            "old_source_frame_zero_bridge_inserted": False,
        },
        "output_matrix": {
            "motion_ids": list(fixture.motion_ids),
            "scopes": list(swept.REQUESTED_SCOPES),
            "candidate_count": 2 * count,
        },
        "outputs": outputs,
    }
    manifest_path = fixture.root / "BUILD_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return fixture, loaded, bank, manifest, manifest_path


def _load_arbitrary_pinned_bank(
    fixture: arbitrary_fixtures.BankFixture,
    loaded: Any,
    bank: Path,
    manifest_path: Path,
):
    mjcf = loaded.canonical_recipe.model_paths["mjcf"]
    return swept.load_pinned_bank(
        manifest_path=manifest_path,
        expected_manifest_sha256=swept._sha256_bytes(
            manifest_path.read_bytes()
        ),
        recipe_path=fixture.recipe_path,
        expected_recipe_sha256=loaded.sha256,
        bank_dir=bank,
        mjcf_path=mjcf,
        expected_mjcf_sha256=loaded.canonical_recipe.model_hashes["mjcf"],
        recipe_repo_root=fixture.root,
    )


def _refresh_result_after_ledger_tamper(
    result: dict[str, Any],
) -> None:
    result["proof_ledger_sha256"] = swept._canonical_json_sha256(
        result["proof_ledger"]
    )
    result["summary"] = swept.validate_motion_proof_result(result)
    result["verdict"] = (
        "PASS"
        if result["summary"]["all_base_pair_intervals_certified"]
        else "FAIL_CLOSED"
    )


def _refresh_receipt_aggregate(receipt: dict[str, Any]) -> None:
    summaries = [result["summary"] for result in receipt["results"]]
    receipt["aggregate"] = {
        "output_count": len(summaries),
        "required_base_pair_interval_count": sum(
            row["required_base_pair_interval_count"] for row in summaries
        ),
        "certified_base_pair_interval_count": sum(
            row["certified_base_pair_interval_count"] for row in summaries
        ),
        "unknown_base_pair_interval_count": sum(
            row["unknown_base_pair_interval_count"] for row in summaries
        ),
        "unsafe_base_pair_interval_count": sum(
            row["unsafe_base_pair_interval_count"] for row in summaries
        ),
        "nonfinite_base_pair_interval_count": sum(
            row["nonfinite_base_pair_interval_count"] for row in summaries
        ),
        "all_outputs_complete": all(
            row["all_base_pair_intervals_certified"] for row in summaries
        ),
    }
    complete = receipt["aggregate"]["all_outputs_complete"]
    receipt["verdict"] = "PASS" if complete else "FAIL_CLOSED"
    receipt["authorization"]["swept_clearance_complete"] = complete


def test_exact_five_mm_passes_and_nextafter_below_fails_closed(
    tmp_path: Path,
) -> None:
    exact_backend = FakeBackend(
        {
            ("torso_collision", "top"): swept.HARD_CLEARANCE_M,
        }
    )
    exact = swept.certify_motion_continuous(
        _clip(tmp_path, "upper"),
        exact_backend,
        max_subdivision_depth=0,
    )
    assert exact["verdict"] == "PASS"
    assert (
        exact["summary"]["minimum_clearance_certified_lower_bound_m"]
        == swept.HARD_CLEARANCE_M
    )
    assert swept.clearance_threshold_passes(swept.HARD_CLEARANCE_M)

    below = float(np.nextafter(swept.HARD_CLEARANCE_M, -np.inf))
    assert not swept.clearance_threshold_passes(below)
    below_backend = FakeBackend({("torso_collision", "top"): below})
    failed = swept.certify_motion_continuous(
        _clip(tmp_path, "full"),
        below_backend,
        max_subdivision_depth=0,
    )
    assert failed["verdict"] == "FAIL_CLOSED"
    assert failed["summary"]["unsafe_base_pair_interval_count"] == 1
    assert failed["summary"]["unknown_base_pair_interval_count"] == 0


def test_recursive_interval_enclosure_is_not_a_sampled_only_boolean(
    tmp_path: Path,
) -> None:
    # At the root leaf, reach*half-angle = 5 mm, so 8 mm midpoint distance
    # cannot certify.  At the two depth-1 leaves the bound is 2.5 mm and both
    # continuous halves certify.
    backend = FakeBackend(
        {("torso_collision", role): 0.008 for role in swept.ACTION_BALL_ROLES},
        joint_reach=1.0,
    )
    result = swept.certify_motion_continuous(
        _clip(
            tmp_path,
            "upper",
            first_joint_delta=0.01,
        ),
        backend,
        max_subdivision_depth=1,
    )
    assert result["verdict"] == "PASS"
    torso_top = [
        row
        for row in result["proof_ledger"]
        if row["robot_geom"] == "torso_collision"
        and row["obstacle_role"] == "top"
    ]
    assert [(row["u_lo"], row["u_hi"]) for row in torso_top] == [
        (0.0, 0.5),
        (0.5, 1.0),
    ]
    assert all(row["status"] == "CERTIFIED" for row in torso_top)
    assert all(
        row["motion_displacement_upper_bound_m"] == pytest.approx(0.0025)
        for row in torso_top
    )


def test_skipped_source_interval_is_rejected_even_if_ledger_hash_is_refreshed(
    tmp_path: Path,
) -> None:
    result = swept.certify_motion_continuous(
        _clip(tmp_path, "upper", frames=3),
        FakeBackend(),
        max_subdivision_depth=0,
    )
    result["proof_ledger"] = [
        row
        for row in result["proof_ledger"]
        if not (
            row["source_interval"] == 1
            and row["robot_geom"] == "torso_collision"
            and row["obstacle_role"] == "top"
        )
    ]
    result["proof_ledger_sha256"] = swept._canonical_json_sha256(
        result["proof_ledger"]
    )
    with pytest.raises(swept.ClearanceError, match="coverage mismatch"):
        swept.validate_motion_proof_result(result)


def test_skipped_enabled_robot_geom_is_rejected(
    tmp_path: Path,
) -> None:
    receipt = _receipt(tmp_path)
    result = receipt["results"][0]
    result["proof_ledger"] = [
        row
        for row in result["proof_ledger"]
        if row["robot_geom"] != "torso_collision"
    ]
    result["proof_ledger_sha256"] = swept._canonical_json_sha256(
        result["proof_ledger"]
    )
    with pytest.raises(swept.ClearanceError, match="coverage mismatch"):
        swept.validate_receipt_self_consistency(receipt)


def test_skipped_obstacle_is_rejected(
    tmp_path: Path,
) -> None:
    receipt = _receipt(tmp_path)
    result = receipt["results"][0]
    result["obstacle_roles"] = list(swept.ACTION_BALL_ROLES[:-1])
    result["proof_ledger"] = [
        row
        for row in result["proof_ledger"]
        if row["obstacle_role"] != "post_right"
    ]
    result["proof_ledger_sha256"] = swept._canonical_json_sha256(
        result["proof_ledger"]
    )
    with pytest.raises(
        swept.ClearanceError, match="exact five-piece assembly|omits"
    ):
        swept.validate_receipt_self_consistency(receipt)


def test_keepout_geometry_tamper_is_rejected_after_self_consistent_rehash(
    tmp_path: Path,
) -> None:
    receipt = _receipt(tmp_path)
    keepout = receipt["scene_contract"]["components"][1]
    keepout["aabb_hi_m"][2] -= 0.01
    lo = np.asarray(keepout["aabb_lo_m"])
    hi = np.asarray(keepout["aabb_hi_m"])
    keepout["center_m"] = ((lo + hi) / 2.0).tolist()
    keepout["full_extents_m"] = (hi - lo).tolist()
    receipt["scene_contract"]["components_sha256"] = (
        swept._canonical_json_sha256(
            receipt["scene_contract"]["components"]
        )
    )
    with pytest.raises(
        swept.ClearanceError, match="floor-to-slab-underside"
    ):
        swept.validate_receipt_self_consistency(receipt)


def test_certified_leaf_below_exact_threshold_is_rejected_after_rehash(
    tmp_path: Path,
) -> None:
    receipt = _receipt(tmp_path)
    result = receipt["results"][0]
    result["proof_ledger"][0][
        "interval_clearance_certified_lower_bound_m"
    ] = float(np.nextafter(swept.HARD_CLEARANCE_M, -np.inf))
    result["proof_ledger_sha256"] = swept._canonical_json_sha256(
        result["proof_ledger"]
    )
    with pytest.raises(swept.ClearanceError, match="weakens"):
        swept.validate_receipt_self_consistency(receipt)


def test_output_bytes_tamper_is_rejected(
    tmp_path: Path,
) -> None:
    receipt = _receipt(tmp_path)
    receipt["results"][0]["sha256"] = "f" * 64
    with pytest.raises(swept.ClearanceError, match="exact output bytes"):
        swept.validate_receipt_self_consistency(receipt)


def test_arbitrary_recipe_first_middle_last_matrix_loads_through_strict_view(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fixture, loaded, bank, _manifest, manifest_path = _arbitrary_pinned_bank(
        tmp_path,
        monkeypatch,
        count=5,
    )
    assert "model_contract" not in fixture.recipe
    assert "canonical_ready" not in fixture.recipe

    bank_binding, clips, *_rest = _load_arbitrary_pinned_bank(
        fixture,
        loaded,
        bank,
        manifest_path,
    )

    observed = [(clip.motion_id, clip.scope) for clip in clips]
    expected = [
        (motion_id, scope)
        for motion_id in fixture.motion_ids
        for scope in swept.REQUESTED_SCOPES
    ]
    assert observed == expected
    assert observed[0] == (fixture.motion_ids[0], "upper")
    assert observed[4] == (fixture.motion_ids[2], "upper")
    assert observed[-1] == (fixture.motion_ids[-1], "full")
    assert bank_binding["output_matrix"] == {
        "motion_ids": list(fixture.motion_ids),
        "scopes": ["upper", "full"],
        "candidate_count": 10,
    }
    assert all(
        row["sha256"] == clip.snapshot.sha256
        for row, clip in zip(bank_binding["outputs"], clips)
    )
    for index in (0, 4, len(clips) - 1):
        proof = swept.certify_motion_continuous(
            clips[index],
            FakeBackend(),
            max_subdivision_depth=0,
        )
        assert proof["verdict"] == "PASS"
        assert proof["summary"]["all_base_pair_intervals_certified"] is True


def test_arbitrary_recipe_wrong_output_order_fails_closed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fixture, loaded, bank, manifest, manifest_path = _arbitrary_pinned_bank(
        tmp_path,
        monkeypatch,
        count=5,
    )
    manifest["outputs"][4], manifest["outputs"][5] = (
        manifest["outputs"][5],
        manifest["outputs"][4],
    )
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        swept.ClearanceError,
        match="output order changed arbitrary-N identity",
    ):
        _load_arbitrary_pinned_bank(
            fixture,
            loaded,
            bank,
            manifest_path,
        )


def test_arbitrary_recipe_middle_output_byte_drift_fails_closed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fixture, loaded, bank, manifest, manifest_path = _arbitrary_pinned_bank(
        tmp_path,
        monkeypatch,
        count=5,
    )
    middle = manifest["outputs"][4]
    middle_path = bank / middle["filename"]
    middle_path.write_bytes(middle_path.read_bytes() + b"drift")

    with pytest.raises(
        swept.ClearanceError,
        match="SHA-256 mismatch",
    ):
        _load_arbitrary_pinned_bank(
            fixture,
            loaded,
            bank,
            manifest_path,
        )


def test_unknown_subdivision_exhaustion_cannot_pass(
    tmp_path: Path,
) -> None:
    # Midpoint itself is safe, but 8 mm is insufficient to cover a 10 mm
    # continuous motion envelope when no subdivision is allowed.
    result = swept.certify_motion_continuous(
        _clip(
            tmp_path,
            "upper",
            first_joint_delta=0.02,
        ),
        FakeBackend(
            {
                ("torso_collision", role): 0.008
                for role in swept.ACTION_BALL_ROLES
            },
            joint_reach=1.0,
        ),
        max_subdivision_depth=0,
    )
    assert result["verdict"] == "FAIL_CLOSED"
    assert result["summary"]["unknown_base_pair_interval_count"] > 0
    assert result["summary"]["unsafe_base_pair_interval_count"] == 0


def test_backend_saturation_flag_cannot_forge_a_certificate(
    tmp_path: Path,
) -> None:
    class ContradictoryBackend(FakeBackend):
        def distance_saturation_query(
            self, robot_geom_name: str, obstacle_role: str, distmax_m: float
        ) -> tuple[float, bool]:
            del robot_geom_name, obstacle_role
            return float(np.nextafter(distmax_m, -np.inf)), True

    result = swept.certify_motion_continuous(
        _clip(tmp_path, "upper"),
        ContradictoryBackend(),
        max_subdivision_depth=0,
    )
    assert result["verdict"] == "FAIL_CLOSED"
    assert result["summary"]["nonfinite_base_pair_interval_count"] > 0
    assert result["summary"]["certified_base_pair_interval_count"] == 0


def test_runtime_sources_derive_exact_floor_to_underside_five_piece_assembly(
    ) -> None:
    geometry = swept.read_snapshot(
        swept.DEFAULT_GEOMETRY_SOURCE, label="geometry"
    )
    table_frame = swept.read_snapshot(
        swept.DEFAULT_TABLE_FRAME_SOURCE, label="table_frame"
    )
    commands = swept.read_snapshot(
        swept.DEFAULT_HOPE_COMMANDS_SOURCE, label="hope_commands"
    )
    scene_builder = swept.read_snapshot(
        swept.DEFAULT_SCENE_BUILDER_SOURCE, label="scene_builder"
    )
    scene = swept.derive_action_ball_assembly(
        geometry_source=geometry,
        table_frame_source=table_frame,
        hope_commands_source=commands,
        scene_builder_source=scene_builder,
    )
    assert scene["roles"] == list(swept.ACTION_BALL_ROLES)
    assert len(scene["components"]) == 5
    top, keepout, net, left, right = scene["components"]
    assert keepout["aabb_lo_m"][2] == 0.0
    assert keepout["aabb_hi_m"][2] == top["aabb_lo_m"][2]
    assert keepout["full_extents_m"][0:2] == top["full_extents_m"][0:2]
    assert net["role"] == "net"
    assert left["center_m"] != right["center_m"]
    for post in (left, right):
        overlap = np.minimum(net["aabb_hi_m"], post["aabb_hi_m"]) - np.maximum(
            net["aabb_lo_m"], post["aabb_lo_m"]
        )
        assert np.all(overlap > 0.0)


def test_unapproved_table_component_intrusion_is_rejected_after_rehash(
    tmp_path: Path,
) -> None:
    receipt = _receipt(tmp_path)
    net = receipt["scene_contract"]["components"][2]
    net["aabb_lo_m"][2] -= 0.001
    lo = np.asarray(net["aabb_lo_m"])
    hi = np.asarray(net["aabb_hi_m"])
    net["center_m"] = ((lo + hi) / 2.0).tolist()
    net["full_extents_m"] = (hi - lo).tolist()
    receipt["scene_contract"]["components_sha256"] = (
        swept._canonical_json_sha256(
            receipt["scene_contract"]["components"]
        )
    )
    with pytest.raises(
        swept.ClearanceError, match="net must meet|unapproved interior overlap"
    ):
        swept.validate_receipt_self_consistency(receipt)


def test_shifted_net_post_overlap_is_not_whitelisted_by_role_only(
    tmp_path: Path,
) -> None:
    receipt = _receipt(tmp_path)
    post = receipt["scene_contract"]["components"][3]
    post["aabb_lo_m"][1] -= 0.005
    post["aabb_hi_m"][1] -= 0.005
    lo = np.asarray(post["aabb_lo_m"])
    hi = np.asarray(post["aabb_hi_m"])
    post["center_m"] = ((lo + hi) / 2.0).tolist()
    post["full_extents_m"] = (hi - lo).tolist()
    receipt["scene_contract"]["components_sha256"] = (
        swept._canonical_json_sha256(
            receipt["scene_contract"]["components"]
        )
    )
    with pytest.raises(
        swept.ClearanceError, match="mirrored geometry|exact conservative net joint"
    ):
        swept.validate_receipt_self_consistency(receipt)


def test_wrong_source_pin_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "pinned.py"
    path.write_text("VALUE = 1\n", encoding="utf-8")
    with pytest.raises(swept.ClearanceError, match="SHA-256 mismatch"):
        swept.read_snapshot(
            path,
            label="pinned source",
            expected_sha256="0" * 64,
        )


def test_receipt_publication_is_external_and_no_clobber(
    tmp_path: Path,
) -> None:
    bank = tmp_path / "bank"
    bank.mkdir()
    external = tmp_path / "receipts" / "clearance.json"
    payload = {"verdict": "FAIL_CLOSED"}
    published = swept.write_json_no_clobber(
        payload, external, forbidden_tree=bank
    )
    assert published == external
    assert external.is_file()
    with pytest.raises(FileExistsError, match="overwrite"):
        swept.write_json_no_clobber(
            payload, external, forbidden_tree=bank
        )
    with pytest.raises(swept.ClearanceError, match="external"):
        swept.write_json_no_clobber(
            payload, bank / "receipt.json", forbidden_tree=bank
        )


def test_full_receipt_pass_has_zero_unknown_nonfinite_unsafe(
    tmp_path: Path,
) -> None:
    receipt = _receipt(tmp_path)
    swept.validate_receipt_self_consistency(receipt)
    assert receipt["verdict"] == "PASS"
    assert receipt["aggregate"]["unknown_base_pair_interval_count"] == 0
    assert receipt["aggregate"]["nonfinite_base_pair_interval_count"] == 0
    assert receipt["aggregate"]["unsafe_base_pair_interval_count"] == 0
    assert receipt["authorization"] == {
        "swept_clearance_complete": True,
        "training_authorized": False,
        "hardware_authorized": False,
    }


def test_real_producer_projection_roundtrips_through_bank_consumer(
    tmp_path: Path,
) -> None:
    """The producer, not a handwritten JSON fixture, feeds the bank gate."""

    files = {}
    for role, suffix in (
        ("manifest", "json"),
        ("recipe", "json"),
        ("ready", "npz"),
        ("mjcf", "xml"),
        ("urdf", "urdf"),
        ("body_order", "txt"),
    ):
        path = tmp_path / f"{role}.{suffix}"
        path.write_bytes(f"{role}-fixture\n".encode())
        files[role] = swept.read_snapshot(path, label=role)

    clips = [
        _write_public_roundtrip_clip(tmp_path, scope)
        for scope in swept.REQUESTED_SCOPES
    ]
    backend = FakeBackend()
    results = []
    for clip in clips:
        result = swept.certify_motion_continuous(
            clip, backend, max_subdivision_depth=0
        )
        result["endpoint_contract"] = {
            "start_frame": 0,
            "end_frame": clip.frames - 1,
            "shared_ready_joint_exact": True,
            "shared_ready_root_tolerance_m_rad": 2.0e-6,
            "start_root_position_error_m": 0.0,
            "end_root_position_error_m": 0.0,
            "start_root_orientation_error_rad": 0.0,
            "end_root_orientation_error_rad": 0.0,
            "endpoint_velocity_channels_exact_zero": True,
            "prepare_frame_count_minimum": 1,
            "recovery_frame_count_minimum": 1,
        }
        result["stored_frame_fk_contract"] = {
            "frame_count": clip.frames,
            "position_tolerance_m": swept.FK_POSITION_TOL_M,
            "orientation_tolerance_rad": swept.FK_ORIENTATION_TOL_RAD,
            "maximum_position_error_m": 0.0,
            "maximum_orientation_error_rad": 0.0,
            "pass": True,
        }
        results.append(result)

    source_pins = _source_pins(tmp_path)
    internal = swept.build_receipt(
        bank_binding={
            **{
                role: snapshot.binding()
                for role, snapshot in files.items()
            },
            "station_center_shift_xy_m": None,
            "output_matrix": {
                "motion_ids": ["stroke"],
                "scopes": list(swept.REQUESTED_SCOPES),
                "candidate_count": 2,
            },
            "outputs": [
                {
                    "motion_id": clip.motion_id,
                    "scope": clip.scope,
                    "filename": clip.snapshot.path.name,
                    "path": str(clip.snapshot.path),
                    "bytes": clip.snapshot.size,
                    "sha256": clip.snapshot.sha256,
                    "frames": clip.frames,
                    "fps": clip.fps,
                    "duration_s": clip.duration_s,
                    "contact_window_start_s": (
                        clip.contact_window_start_s
                    ),
                    "contact_window_end_s": clip.contact_window_end_s,
                }
                for clip in clips
            ],
        },
        scene_contract=_scene_contract(),
        source_pins=source_pins,
        dependency_pins={
            "python": {"fixture": True},
            "numpy": {"fixture": True},
            "mujoco": {"fixture": True},
        },
        robot_geometry=_robot_geometry(),
        results=results,
    )
    public = swept.project_bank_gate_receipt(internal)
    assert public["receipt_class"] == bank_gate._SWEPT_RECEIPT_CLASS
    receipt_path = tmp_path / "public_swept_receipt.json"
    receipt_path.write_text(
        json.dumps(public, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest_rows = {
        (clip.motion_id, clip.scope): (
            {
                "filename": clip.snapshot.path.name,
                "output_npz_sha256": clip.snapshot.sha256,
                "duration_s": clip.duration_s,
                "contact_window_start_s": (
                    clip.contact_window_start_s
                ),
                "contact_window_end_s": clip.contact_window_end_s,
            },
            clip.snapshot.path,
        )
        for clip in clips
    }
    bound = bank_gate._BoundFiles(
        manifest=files["manifest"].path,
        bank_dir=tmp_path,
        recipe=files["recipe"].path,
        compiler=files["recipe"].path,
        geometry_tool=files["recipe"].path,
        weighted_arc_tool=files["recipe"].path,
        ready=files["ready"].path,
        mjcf=files["mjcf"].path,
        urdf=files["urdf"].path,
        body_order=files["body_order"].path,
        hashes={
            "recipe": files["recipe"].sha256,
            "ready": files["ready"].sha256,
            "mjcf": files["mjcf"].sha256,
            "urdf": files["urdf"].sha256,
            "body_order": files["body_order"].sha256,
        },
        compiled_signature="0" * 64,
    )
    contract = bank_gate._VerificationContract(
        expected_motion_ids=("stroke",),
        expected_matrix=(
            ("stroke", "upper"),
            ("stroke", "full"),
        ),
        expected_filenames=tuple(
            clip.snapshot.path.name for clip in clips
        ),
        append_only_composition=None,
        station_center_shift_xy_m=None,
        base_recipe_path=None,
        base_manifest_path=None,
    )
    reopened = bank_gate._validate_swept_clearance_receipt(
        receipt_path,
        swept._sha256_bytes(receipt_path.read_bytes()),
        files=bound,
        contract=contract,
        matrix=manifest_rows,
    )
    assert reopened.sha256 == swept._sha256_bytes(
        receipt_path.read_bytes()
    )
    assert reopened.minimum_clearance_m >= swept.HARD_CLEARANCE_M
