"""Unit tests for the candidate-only canonical motion compiler."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import canonical_motion_compiler as cmc  # noqa: E402
import canonical_path_topp as cpt  # noqa: E402
from canonical_motion_geometry import CanonicalMotionGeometry  # noqa: E402
from canonical_motion_markers import (  # noqa: E402
    ConstructionMarker,
    MarkerSemantics,
    MarkerSemanticsRow,
)
from canonical_motion_recipe import (  # noqa: E402
    CanonicalMotionRecipe,
    MotionSource,
    ReadyState,
)
from canonical_path_topp import (  # noqa: E402
    MarkerMapping,
    RetimeResult,
    ScalarPathCollocationTrace,
)
from canonical_schema2_builder import Schema2Candidate  # noqa: E402
from mujoco_motion_player import (  # noqa: E402
    MotionClip,
    RUNTIME_BODY_NAMES,
    RUNTIME_JOINT_NAMES,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _fake_collocation_trace(
    q: np.ndarray,
    *,
    duration_s: float,
    weighted_arc_length_receipt=None,
) -> ScalarPathCollocationTrace:
    """Minimal immutable exact-state double for compiler-bound retimers."""

    q_node = np.asarray(q, dtype=np.float64)
    s_node = np.linspace(0.0, 1.0, len(q_node))
    s_mid = 0.5 * (s_node[:-1] + s_node[1:])
    q_mid = 0.5 * (q_node[:-1] + q_node[1:])
    zeros_node = np.zeros_like(q_node)
    zeros_mid = np.zeros_like(q_mid)
    tick_cell_index = np.clip(
        np.arange(len(q_node), dtype=np.int64), 0, len(q_node) - 2
    )
    tick_cell_side = np.full(len(q_node), "cell_interior", dtype="U32")
    tick_cell_side[0] = "right_cell_at_knot"
    tick_cell_side[-1] = "left_cell_at_path_end"
    tick_time = np.linspace(0.0, duration_s, len(q_node))
    return ScalarPathCollocationTrace(
        s_node=s_node,
        s_mid=s_mid,
        q_node=q_node,
        q_s_node=zeros_node,
        q_ss_node_left=zeros_node,
        q_ss_node_right=zeros_node,
        q_mid=q_mid,
        q_s_mid=zeros_mid,
        q_ss_mid=zeros_mid,
        x_node=np.zeros(len(q_node)),
        u_cell=np.zeros(len(q_node) - 1),
        time_node_s=tick_time,
        time_mid_s=0.5 * (tick_time[:-1] + tick_time[1:]),
        tick_time_s=tick_time,
        tick_s=s_node,
        tick_x=np.zeros(len(q_node)),
        tick_q=q_node,
        tick_q_s=zeros_node,
        tick_q_ss=zeros_node,
        tick_qdot=zeros_node,
        tick_qdd=zeros_node,
        tick_scalar_acceleration=np.zeros(len(q_node)),
        tick_cell_index=tick_cell_index,
        tick_cell_side=tick_cell_side,
        path_progress_contract=(
            "weighted_arc_length_v1"
            if weighted_arc_length_receipt is not None
            else "explicit_geometry_parameter_2jet_v1"
        ),
        path_evaluator_kind=(
            "weighted_arc_path_evaluate_l_exact_v1"
            if weighted_arc_length_receipt is not None
            else "quintic_hermite_endpoint_2jet_v1"
        ),
        path_evaluator_sha256_float64_le="0" * 64,
        geometry_continuity_contract=(
            "C2_endpoint_2jet_declared_and_one_sided_knot_verified"
        ),
        node_second_derivative_contract=(
            "cell_i_uses_right_i_at_start_and_left_i_plus_1_at_end"
        ),
        boundary_second_derivative_contract=(
            "start_left_duplicates_start_right;_end_right_duplicates_end_left"
        ),
        tick_second_derivative_contract="test_selected_cell_side",
        grid_subdivisions=4,
        time_scale=1.0,
        weighted_arc_length_receipt=weighted_arc_length_receipt,
    )


class FakePlantBackend:
    joint_names = tuple(RUNTIME_JOINT_NAMES)
    root_body_name = "pelvis_link"

    def __init__(self) -> None:
        self.position_lower = np.full(31, -5.0)
        self.position_upper = np.full(31, 5.0)
        self.velocity_limit = np.full(31, 10.0)
        self.effort_limit = np.full(31, 100.0)

    def site_pose(self, joint_pos, root_pos_w, root_quat_w):
        del joint_pos, root_pos_w, root_quat_w
        return np.zeros(3), np.eye(3)

    def diagonal_dynamics(self, joint_pos, root_pos_w, root_quat_w):
        del joint_pos, root_pos_w, root_quat_w
        return np.ones(31), np.zeros(31)


class FakeFaceResult:
    def __init__(self, joint_pos: np.ndarray, frame_indices) -> None:
        self.joint_pos = np.asarray(joint_pos, dtype=np.float64).copy()
        wrist = RUNTIME_JOINT_NAMES.index("right_wrist_yaw_joint")
        self.joint_pos[:, wrist] += 0.05
        self._frame_indices = tuple(frame_indices)

    def summary(self):
        return {
            "mode": "normal",
            "solver_strategy": "test_coordinated_window",
            "frames": len(self.joint_pos),
            "frame_indices": list(self._frame_indices),
        }


def _fake_schema2_builder(**kwargs) -> Schema2Candidate:
    frames = len(kwargs["joint_pos"])
    body_pos = np.zeros((frames, 32, 3), dtype=np.float32)
    body_quat = np.zeros((frames, 32, 4), dtype=np.float32)
    body_quat[..., 0] = 1.0
    body_lin = np.zeros((frames, 32, 3), dtype=np.float32)
    body_ang = np.zeros((frames, 32, 3), dtype=np.float32)
    arrays = {
        "fps": np.asarray([kwargs["fps"]], dtype=np.float32),
        "joint_pos": np.asarray(kwargs["joint_pos"], dtype=np.float32),
        "joint_vel": np.asarray(kwargs["joint_vel"], dtype=np.float32),
        "body_pos_w": body_pos,
        "body_quat_w": body_quat,
        "body_lin_vel_w": body_lin,
        "body_ang_vel_w": body_ang,
        "root_pos_w": np.asarray(kwargs["root_pos_w"], dtype=np.float32),
        "root_quat_wxyz": np.asarray(
            kwargs["root_quat_wxyz"], dtype=np.float32
        ),
        "root_lin_vel_w": np.asarray(
            kwargs["root_lin_vel_w"], dtype=np.float32
        ),
        "root_ang_vel_w": np.asarray(
            kwargs["root_ang_vel_w"], dtype=np.float32
        ),
    }
    payload = b"fake-schema2:" + kwargs["input_sha256"].encode("ascii")
    output_sha = hashlib.sha256(payload).hexdigest()
    hashes = {
        "output_npz_sha256": output_sha,
        "input_sha256": kwargs["input_sha256"],
    }
    return Schema2Candidate(
        arrays=MappingProxyType(arrays),
        npz_bytes=payload,
        manifest=MappingProxyType(
            {
                "publication_class": "compiler_candidate",
                "training_authorized": False,
                "hashes": hashes,
            }
        ),
        report=MappingProxyType(
            {
                "status": "PASS",
                "frames": len(arrays["joint_pos"]),
                "hashes": dict(hashes),
            }
        ),
    )


def _fast_marker_only_retime(q_path, *args, **kwargs) -> RetimeResult:
    """Cheap retimer double; the dedicated search test checks ranking."""

    del args
    q = np.asarray(q_path, dtype=np.float64)
    marker_rows = kwargs["markers"]
    progress = np.asarray(kwargs["path_progress"], dtype=np.float64)
    arc = kwargs.get("weighted_arc_path")
    weighted_receipt = (
        dict(cpt._weighted_arc_length_receipt(arc))
        if arc is not None
        else {
            "enabled": True,
            "contract": "weighted_arc_length_v1",
            "content_sha256": "a" * 64,
            "receipt_sha256": "b" * 64,
            "coordinate_scale_sha256_float64_le": "c" * 64,
        }
    )
    marker_times = {
        "window_start": 0.10,
        "source_anchor": 0.14,
        "window_end": 0.18,
    }
    markers = {
        name: MarkerMapping(
            source_index=float(source_index),
            time_s=marker_times[name],
            output_fractional_frame=marker_times[name] * 50.0,
            output_frame=int(round(marker_times[name] * 50.0)),
            path_position_at_frame=float(source_index),
        )
        for name, source_index in marker_rows.items()
    }
    uniform = (
        kwargs.get("uniform_scalar_path_acceleration_until_marker")
        is not None
    )
    return RetimeResult(
        q=q,
        qdot=np.zeros_like(q),
        path_position=np.arange(len(q), dtype=np.float64),
        path_speed=np.ones(len(q) - 1, dtype=np.float64),
        path_acceleration=np.zeros(len(q) - 1, dtype=np.float64),
        markers=markers,
        report={
            "duration_s": 0.30,
            "path_progress": {
                "contract": (
                    "weighted_arc_length_v1"
                    if arc is not None
                    else "explicit_geometry_parameter_2jet_v1"
                ),
                "explicit": True,
            },
            "path_evaluator": {
                "kind": (
                    "weighted_arc_path_evaluate_l_exact_v1"
                    if arc is not None
                    else "quintic_hermite_endpoint_2jet_v1"
                ),
                "derivative_inputs_consumed": True,
            },
            "weighted_arc_length": (
                weighted_receipt
            ),
            "markers": {
                name: {
                    "path_progress": float(progress[int(source_index)])
                }
                for name, source_index in marker_rows.items()
            },
            "marker_policy": "observe_only_no_window_lock",
            "nonnegative_acceleration_until_marker": (
                {"enabled": False, "marker": None}
                if uniform
                else {
                    "enabled": True,
                    "marker": "window_end",
                    "prefix_scalar_acceleration_min_continuous": 0.0,
                    "prefix_scalar_acceleration_min_50hz": 0.0,
                    "control_guard_enabled": True,
                    "output_interval_policy": (
                        "every_interval_starting_before_exact_marker_must_be_"
                        "nonnegative_including_the_straddling_interval"
                    ),
                }
            ),
            "uniform_scalar_path_acceleration_prefix_comparator": (
                {
                    "enabled": True,
                    "marker": "window_end",
                    "strictly_positive_on_every_guard_cell": True,
                    "cruise_cells_through_guard": 0,
                    "constant_scalar_path_acceleration": 1.0,
                    "constant_scalar_path_acceleration_units": (
                        "weighted_arc_length_per_second_squared"
                        if arc is not None
                        else "geometry_parameter_per_second_squared"
                    ),
                    "control_guard_path_progress": 1.0,
                    "control_guard_boundary_frame": 9,
                    "control_guard_iteration": 0,
                    "suffix_recovery_contract": "test_backward_recovery",
                    "suffix_speed_squared_scale_from_ordinary_profile": 1.0,
                }
                if uniform
                else {"enabled": False, "marker": None}
            ),
        },
        collocation_trace=_fake_collocation_trace(
            q,
            duration_s=0.30,
            weighted_arc_length_receipt=weighted_receipt,
        ),
    )


def _compiler_plumbing_geometry(
    source_q,
    ready_q,
    entry_frame,
    exit_frame,
    window_start,
    window_end,
    **kwargs,
) -> CanonicalMotionGeometry:
    """Geometry double for compiler plumbing tests that mock the retimer too."""

    del kwargs
    source = np.asarray(source_q, dtype=np.float64)
    ready = np.asarray(ready_q, dtype=np.float64)
    retained = source[entry_frame : exit_frame + 1]
    row_count = len(retained) + 2
    parameter = np.linspace(0.0, 2.0 * np.pi, row_count)
    q_path = np.broadcast_to(
        ready[None, :], (row_count, len(ready))
    ).copy()
    q_path[:, 0] += np.cos(parameter) - 1.0
    q_path[:, 1] += np.sin(parameter)
    q_path[0] = ready
    q_path[-1] = ready
    dq_ds = np.zeros_like(q_path)
    dq_ds[:, 0] = -np.sin(parameter)
    dq_ds[:, 1] = np.cos(parameter)
    d2q_ds2 = np.zeros_like(q_path)
    d2q_ds2[:, 0] = -np.cos(parameter)
    d2q_ds2[:, 1] = -np.sin(parameter)
    source_map = np.concatenate(
        (
            np.asarray([np.nan]),
            np.arange(entry_frame, exit_frame + 1, dtype=np.float64),
            np.asarray([np.nan]),
        )
    )
    finite = np.isfinite(source_map)
    window_mask = (
        finite
        & (source_map >= float(window_start))
        & (source_map <= float(window_end))
    )
    window_indices = np.flatnonzero(window_mask)
    waypoint_indices = np.arange(1, len(q_path) - 1, dtype=np.int64)
    return CanonicalMotionGeometry(
        q_path=q_path,
        path_parameter=parameter,
        dq_ds=dq_ds,
        d2q_ds2=d2q_ds2,
        source_frame_map=source_map,
        segment_labels=np.where(
            window_mask, "contact_opportunity", "test_plumbing"
        ),
        window_mask=window_mask,
        window_path_start=int(window_indices[0]),
        window_path_end=int(window_indices[-1]),
        source_waypoint_path_indices=waypoint_indices,
        entry_frame=int(entry_frame),
        exit_frame=int(exit_frame),
        window_start=int(window_start),
        window_end=int(window_end),
        continuity_report={
            "c2_continuous": True,
            "test_double_scope": "compiler_plumbing_with_mocked_retimer",
            "source_frame_map_receipt": {
                "encoding": "float64_le_canonical_qnan_v1"
            },
        },
        canonical_knot_path_indices=np.concatenate(
            (
                np.asarray([0], dtype=np.int64),
                waypoint_indices,
                np.asarray([len(q_path) - 1], dtype=np.int64),
            )
        ),
    )


def _make_clip(path: Path, motion_index: int, frames: int = 9) -> MotionClip:
    q = np.zeros((frames, 31), dtype=np.float64)
    progress = np.asarray(
        [0.0, 0.025, 0.060, 0.105, 0.160, 0.220, 0.275, 0.315, 0.340]
    )
    shoulder = RUNTIME_JOINT_NAMES.index("right_shoulder_pitch_joint")
    elbow = RUNTIME_JOINT_NAMES.index("right_elbow_joint")
    q[:, shoulder] = progress + 0.002 * motion_index
    q[:, elbow] = -0.35 * progress
    qd = np.zeros_like(q)

    body_pos = np.zeros((frames, len(RUNTIME_BODY_NAMES), 3), dtype=np.float64)
    body_quat = np.zeros((frames, len(RUNTIME_BODY_NAMES), 4), dtype=np.float64)
    body_quat[..., 0] = 1.0
    pelvis = RUNTIME_BODY_NAMES.index("pelvis_link")
    body_pos[:, pelvis, 0] = np.linspace(0.0, 0.008, frames)
    body_pos[:, pelvis, 2] = 1.0
    body_lin = np.zeros_like(body_pos)
    body_ang = np.zeros_like(body_pos)
    return MotionClip(
        path=path,
        fps=50.0,
        joint_pos=q,
        joint_vel=qd,
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        body_lin_vel_w=body_lin,
        body_ang_vel_w=body_ang,
        has_migration_provenance=False,
        body_lin_vel_point="body_com",
    )


def _make_recipe(tmp_path: Path) -> CanonicalMotionRecipe:
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text("{}\n", encoding="utf-8")
    ready_path = tmp_path / "ready.npz"
    ready_path.write_bytes(b"ready")
    ready_q = np.zeros(31, dtype=np.float64)
    ready = ReadyState(
        path=ready_path,
        sha256=_digest("ready"),
        joint_pos=ready_q,
        joint_vel=np.zeros(31),
        root_pos_w=np.asarray([0.0, 0.0, 1.0]),
        root_quat_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0]),
        source_segment="bh_loop_c",
        source_frame=0,
    )
    ids = ("fh_loop", "bh_loop_c", "fh_block_syn", "bh_block", "s0_highpress")
    sources = []
    marker_rows = []
    for index, motion_id in enumerate(ids):
        source_path = tmp_path / f"{motion_id}.npz"
        source_path.write_bytes(motion_id.encode("utf-8"))
        face = None
        scope_overrides = {}
        if motion_id == "fh_block_syn":
            face = {
                "mode": "signed_raw_plus_y_flip",
                "active_joints": "right_arm_7",
                "site_position": "preserve_per_source_frame",
                "orientation": "normal_hard_inplane_free",
                "single_axis_pi_overlay_forbidden": True,
            }
        if motion_id == "s0_highpress":
            scope_overrides = {
                "full": {
                    "grounding_policy": (
                        "minimum_constant_z_offset_for_1mm_clearance"
                    ),
                    "maximum_grounding_offset_m": 0.09,
                }
            }
        sources.append(
            MotionSource(
                motion_id=motion_id,
                human_role=motion_id,
                path=source_path,
                sha256=_digest(motion_id),
                clip=_make_clip(source_path, index),
                face_manifold=face,
                scope_overrides=scope_overrides,
            )
        )
        marker_rows.append(
            MarkerSemanticsRow(
                motion_id=motion_id,
                nominal_event=None if motion_id == "fh_block_syn" else 4,
                ge50_seed=(4, 5),
                ge80_seed=(4, 5),
                preferred_seed=None,
                construction_marker=(
                    ConstructionMarker(
                        annotation_frame=4,
                        donor_preferred_frame=4,
                        solve_span=(3, 6),
                    )
                    if motion_id == "fh_block_syn"
                    else None
                ),
                historical_adv2c3_start=2,
                bound_recipe_source_path=f"{motion_id}.npz",
                bound_recipe_source_sha256=_digest(motion_id),
                source_scan_remote_path=f"pod2:/scan/{motion_id}.json",
                source_scan_sha256=_digest(f"scan-{motion_id}"),
                frame_identity=None,
                post_retime_behavior_gate_status=(
                    "PENDING_POST_RETIME_BEHAVIOR_RESCAN"
                ),
            )
        )
    marker_authority_path = tmp_path / "marker_authority.json"
    marker_authority_path.write_text("{}\n", encoding="utf-8")
    marker_semantics = MarkerSemantics(
        path=marker_authority_path,
        repo_root=tmp_path,
        sha256=_digest("marker-authority"),
        authority_id="canonical_motion_marker_semantics_v2_20260724",
        review_status=(
            "REVIEWED_FOR_CONTENT_ADDRESSED_FRAME_MAPPING_AND_LEGACY_SEED_"
            "PROVENANCE"
        ),
        legacy_authority_sha256=_digest("marker-authority-v1"),
        rows=tuple(marker_rows),
    )
    model_paths = {}
    model_hashes = {}
    for name in ("mjcf", "urdf", "body_order"):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        model_paths[name] = path
        model_hashes[name] = _digest(name)
    raw = {
        "schema_version": 1,
        "library_id": "test_canonical_five_by_two",
        "publication_class": "compiler_candidate",
        "training_authorized": False,
        "hardware_authorized": False,
        "time_law": {
            "fps": 50.0,
            "joint_velocity_limit_fraction": 1.0,
            "post_retime_behavior_opportunity_minimum_s": 0.08,
            "legacy_seed_marker_policy": (
                "search_and_retime_marker_only_never_output_behavior_window"
            ),
            "kinematic_window_policy": (
                "nonnegative_scalar_acceleration_through_exact_window_end"
            ),
            "window_acceleration_allowed_through_end": True,
        },
        "entry_exit_search": {
            "mode": "enumerate_all_then_gate_and_rank",
            "legacy_ge80_halo_source_frames": 1,
            "retained_source_prefix_required": False,
            "retained_source_suffix_required": False,
            "historical_adv2c3_role": "comparator_only_not_default",
        },
        "post_build_gates": [
            "exact_vendor_mujoco_fk_playback",
            "torque_balance_contact",
        ],
    }
    return CanonicalMotionRecipe(
        path=recipe_path,
        repo_root=tmp_path,
        raw=raw,
        ready=ready,
        sources=tuple(sources),
        marker_semantics=marker_semantics,
        marker_authority_path=marker_authority_path,
        marker_authority_sha256=marker_semantics.sha256,
        model_paths=MappingProxyType(model_paths),
        model_hashes=MappingProxyType(model_hashes),
    )


def _options() -> cmc.CompilerOptions:
    return cmc.CompilerOptions(
        joint_acceleration_limits_rad_s2=np.full(31, 80.0),
        full_root_limits=cmc.FullRootPathLimits(
            position_lower=np.asarray([-2.0] * 6),
            position_upper=np.asarray([2.0] * 6),
            velocity=np.asarray([2.0] * 6),
            acceleration=np.asarray([20.0] * 6),
        ),
        s0_full_grounding_offset_m=0.01,
        samples_per_scaled_unit=6.0,
        min_connector_intervals=5,
        min_core_intervals=5,
        grid_subdivisions=4,
    )


def test_compile_builds_exact_five_by_two_direct_ready_candidates(
    tmp_path, monkeypatch
):
    recipe = _make_recipe(tmp_path)
    face_calls = []

    def fake_face_solver(
        source_joint_pos,
        root_pos_w,
        root_quat_w,
        canonical_ready_joint_pos,
        **kwargs,
    ):
        del root_pos_w, root_quat_w, canonical_ready_joint_pos
        face_calls.append(kwargs)
        return FakeFaceResult(source_joint_pos, kwargs["frame_indices"])

    monkeypatch.setattr(cmc, "solve_face_flipped_window", fake_face_solver)
    monkeypatch.setattr(cmc, "build_schema2_candidate", _fake_schema2_builder)
    monkeypatch.setattr(
        cmc, "build_canonical_geometry", _compiler_plumbing_geometry
    )
    monkeypatch.setattr(cmc, "retime_path", _fast_marker_only_retime)

    library = cmc.compile_loaded_canonical_motion_library(
        recipe,
        options=_options(),
        backend=FakePlantBackend(),
    )

    assert len(library.motions) == 10
    assert [(row.motion_id, row.scope) for row in library.motions] == [
        (source.motion_id, scope)
        for source in recipe.sources
        for scope in ("upper", "full")
    ]
    assert len(face_calls) == 2
    assert all(call["frame_indices"] == (3, 4, 5, 6) for call in face_calls)
    assert all(call["anchor_index"] == 1 for call in face_calls)

    for row in library.motions:
        arrays = row.schema2.arrays
        assert row.collocation_trace.path_progress_contract == (
            "weighted_arc_length_v1"
        )
        assert (
            row.collocation_trace.weighted_arc_length_receipt[
                "digest_verified"
            ]
            is True
        )
        assert row.collocation_trace.tick_q.flags.writeable is False
        np.testing.assert_array_equal(arrays["joint_pos"][0], recipe.ready.joint_pos)
        np.testing.assert_array_equal(arrays["joint_pos"][-1], recipe.ready.joint_pos)
        assert np.count_nonzero(arrays["joint_vel"][[0, -1]]) == 0
        assert np.count_nonzero(arrays["root_lin_vel_w"][[0, -1]]) == 0
        assert np.count_nonzero(arrays["root_ang_vel_w"][[0, -1]]) == 0
        assert row.search_report["selected"]["direct_path"] == (
            "canonical_ready_to_selected_core_to_canonical_ready"
        )
        assert row.search_report["selected"]["old_frame_zero_forced"] is False
        assert row.search_report["adv2c3_comparator"]["forced_or_seeded"] is False
        opportunity = row.search_report["contact_opportunity"]
        assert opportunity["marker_only"] is True
        assert opportunity["pose_locked"] is False
        assert opportunity["velocity_locked"] is False
        assert opportunity["acceleration_locked"] is False
        assert opportunity["acceleration_allowed_through_window_end"] is True
        assert (
            opportunity[
                "nonnegative_scalar_acceleration_through_window_end"
            ]
            is True
        )
        assert row.contact_window_end_s - row.contact_window_start_s >= (
            0.08 - 1.0e-12
        )
        brake = row.retime_report["scalar_no_early_brake_proxy"]
        assert brake["proxy_only"] is True
        assert brake["selection_criterion"] is True
        assert brake["no_negative_scalar_acceleration_before_window_end"] is True
        assert "uniform_actuator_torque" in brake["does_not_establish"]
        geometry = row.geometry_report
        assert geometry["source_frame_map_receipt"]["encoding"] == (
            "float64_le_canonical_qnan_v1"
        )
        assert set(geometry["recipe_marker_binding"]) == {
            "window_start",
            "source_anchor",
            "window_end",
        }
        assert row.schema2.manifest["hashes"]["input_sha256"] != "0" * 64
        assert (
            row.schema2.report["hashes"]["input_sha256"]
            == row.schema2.manifest["hashes"]["input_sha256"]
        )

    manifest = library.manifest
    assert manifest["publication_class"] == "compiler_candidate"
    assert manifest["training_authorized"] is False
    assert manifest["hardware_authorized"] is False
    assert manifest["output_matrix"]["candidate_count"] == 10
    assert manifest["ready"]["old_source_frame_zero_bridge_inserted"] is False
    # This plumbing test deliberately replaces both geometry and retiming; the
    # content-binding receipt must identify that exact test implementation.
    assert manifest["geometry_tool"]["path"].endswith(
        "test_canonical_motion_compiler.py"
    )
    assert manifest["search_contract"]["adv2c3"] == (
        "comparator_only_not_default"
    )
    assert manifest["search_contract"]["marker_authority"].startswith(
        "legacy_ranking_anchor_only"
    )
    assert manifest["search_contract"]["time_law_parameterization"].startswith(
        "weighted_arc_length_v1"
    )
    assert len(manifest["weighted_arc_tool"]["sha256"]) == 64
    assert manifest["contact_opportunity_contract"] == {
        "marker_only": True,
        "pose_speed_acceleration_freeze": False,
        "acceleration_allowed_through_window_end": True,
        "nonnegative_scalar_acceleration_through_window_end": True,
    }
    assert "uniform_actuator_torque" in manifest["non_claims"]
    options_receipt = manifest["compiler_options"]
    assert len(
        options_receipt["joint_acceleration_limits_rad_s2"]["values"]
    ) == 31
    assert options_receipt["full_root_limits"]["coordinate_order"] == list(
        cmc.ROOT_COORDINATE_NAMES
    )
    full_contract = options_receipt["effective_path_coordinate_contracts"]["full"]
    assert full_contract["dimension"] == 37
    assert full_contract["coordinate_order"][:31] == list(RUNTIME_JOINT_NAMES)
    assert full_contract["coordinate_order"][31:] == list(
        cmc.ROOT_COORDINATE_NAMES
    )
    assert full_contract["layout"].startswith("joint_31_then_root")
    assert options_receipt["s0_full_grounding_offset_m"] == pytest.approx(0.01)
    assert len(options_receipt["compiler_options_sha256"]) == 64
    assert len(
        options_receipt["face_manifold"]["active_candidate_seeds_sha256"]
    ) == 64
    assert all(
        gate["status"] == "pending" for gate in manifest["post_build_gates"]
    )


def test_search_score_prioritizes_quantized_hit_start_over_cycle_duration(
    tmp_path, monkeypatch
):
    recipe = _make_recipe(tmp_path)
    source = recipe.sources[0]
    source_coordinates = np.column_stack(
        (
            np.linspace(0.0, 0.08, 9),
            np.linspace(0.0, 0.04, 9),
        )
    )
    ready = np.zeros(2)
    contract = cmc._PathContract(
        position_lower=np.asarray([-20.0, -20.0]),
        position_upper=np.asarray([20.0, 20.0]),
        velocity=np.asarray([100.0, 100.0]),
        acceleration=np.asarray([100.0, 100.0]),
        coordinate_scale=np.asarray([1.0, 1.0]),
        coordinate_semantics=("joint_0", "joint_1"),
        coordinate_units=("rad", "rad"),
    )
    path_identity = {}

    def fake_geometry(
        source_q,
        ready_q,
        entry_frame,
        exit_frame,
        window_start,
        window_end,
        **kwargs,
    ):
        del source_q, ready_q, kwargs
        # Later entries have shorter cycles and lower path variation, but the
        # entry-0 candidate reaches the window one controller tick sooner.
        amplitude = float(10 - entry_frame)
        parameter = np.linspace(0.0, 2.0 * np.pi, 4)
        q_path = amplitude * np.column_stack(
            (np.cos(parameter) - 1.0, np.sin(parameter))
        )
        q_path[[0, -1]] = 0.0
        dq_ds = amplitude * np.column_stack(
            (-np.sin(parameter), np.cos(parameter))
        )
        d2q_ds2 = amplitude * np.column_stack(
            (-np.cos(parameter), -np.sin(parameter))
        )
        path_identity[q_path.tobytes(order="C")] = (
            entry_frame,
            exit_frame,
        )
        return CanonicalMotionGeometry(
            q_path=q_path,
            path_parameter=parameter,
            dq_ds=dq_ds,
            d2q_ds2=d2q_ds2,
            source_frame_map=np.asarray(
                [np.nan, float(window_start), float(window_end), np.nan]
            ),
            segment_labels=np.asarray(["ready", "window", "window", "ready"]),
            window_mask=np.asarray([False, True, True, False]),
            window_path_start=1,
            window_path_end=2,
            source_waypoint_path_indices=np.asarray([1, 2]),
            entry_frame=entry_frame,
            exit_frame=exit_frame,
            window_start=window_start,
            window_end=window_end,
            continuity_report={"c2_continuous": True},
            canonical_knot_path_indices=np.arange(4, dtype=np.int64),
        )

    def fake_retime(q_path, *args, **kwargs):
        entry, exit_ = path_identity[
            np.ascontiguousarray(q_path).tobytes(order="C")
        ]
        base = _fast_marker_only_retime(q_path, *args, **kwargs)
        duration = 1.0 if entry == 0 else 0.8
        window_start = 0.10 if entry == 0 else 0.12
        markers = {
            name: MarkerMapping(
                source_index=value,
                time_s={
                    "window_start": window_start,
                    "source_anchor": 0.14,
                    "window_end": 0.18,
                }[name],
                output_fractional_frame=0.0,
                output_frame=0,
                path_position_at_frame=0.0,
            )
            for name, value in kwargs["markers"].items()
        }
        report = dict(base.report)
        report.update(
            {
                "duration_s": duration,
                "candidate": [entry, exit_],
            }
        )
        return replace(
            base,
            markers=markers,
            report=report,
        )

    monkeypatch.setattr(cmc, "build_canonical_geometry", fake_geometry)
    monkeypatch.setattr(cmc, "retime_path", fake_retime)
    winner, report = cmc._search_core(
        recipe,
        source,
        source_coordinates,
        ready,
        contract,
        options=_options(),
    )

    assert winner.duration_s == pytest.approx(1.0)
    assert winner.candidate.entry_frame == 0
    assert report["ranking"] == [
        "kinematic_feasibility",
        "nonnegative_scalar_acceleration_through_window_end",
        "minimum_window_start_control_tick",
        "minimum_source_anchor_control_tick",
        "minimum_prewindow_zero_acceleration_plateau_ticks",
        "minimum_worst_recovery_control_ticks",
        "minimum_cycle_control_ticks",
        "minimum_scaled_l2_total_variation",
        "entry_frame",
        "exit_frame",
    ]
    # The cheap retimer double reports all-zero segment accelerations, so all
    # three pre-window segments count as plateau ticks for every candidate.
    assert winner.score[:5] == (5, 7, 3, 45, 50)
    assert report["selected"]["window_start_control_tick"] == 5
    assert report["selected"]["source_anchor_control_tick"] == 7
    assert report["selected"]["prewindow_zero_acceleration_plateau_ticks"] == 3
    assert report["selected"]["worst_recovery_control_ticks"] == 45
    assert report["selected"]["cycle_control_ticks"] == 50
    assert report["pruning"].startswith("disabled_")
    comparator = report["time_law_comparators"][
        "uniform_scalar_path_acceleration_prefix"
    ]
    assert comparator["dominance_status"] == (
        "PASS_NO_BRAKE_WEAKLY_DOMINATES"
    )
    assert comparator["uniform_prefix"]["cruise_cells_through_guard"] == 0
    assert "not_an_adopted_time_law" in comparator["role"]
    assert report["adv2c3_comparator"]["entry_frame"] == 2
    assert report["adv2c3_comparator"]["coincides_with_selected_entry"] is False
    assert report["enumerated_count"] == 12

    parallel_winner, parallel_report = cmc._search_core(
        recipe,
        source,
        source_coordinates,
        ready,
        contract,
        options=replace(_options(), search_workers=3),
    )
    assert parallel_winner.score == winner.score
    assert parallel_winner.candidate == winner.candidate
    assert parallel_report["selected"] == report["selected"]
    assert parallel_report["enumeration_sha256"] == report["enumeration_sha256"]
    assert parallel_report["search_workers"] == 3
    assert parallel_report["admissibly_pruned_count"] == 0
    assert parallel_report["algorithm"] == (
        "enumerate_all_then_parallel_exact_retime_then_control_tick_rank"
    )


def test_uniform_prefix_comparator_rejects_solver_dominance_gap():
    q = np.asarray([[0.0], [0.5], [0.0]])
    no_brake = _fast_marker_only_retime(
        q,
        markers={
            "window_start": 0.5,
            "source_anchor": 1.0,
            "window_end": 1.5,
        },
        path_progress=np.arange(3, dtype=np.float64),
    )
    earlier = dict(no_brake.markers)
    original = earlier["window_start"]
    earlier["window_start"] = replace(
        original,
        time_s=original.time_s - 0.04,
        output_fractional_frame=original.output_fractional_frame - 2.0,
        output_frame=original.output_frame - 2,
    )
    uniform_report = dict(no_brake.report)
    uniform_report["uniform_scalar_path_acceleration_prefix_comparator"] = {
        "enabled": True,
        "strictly_positive_on_every_guard_cell": True,
        "cruise_cells_through_guard": 0,
        "constant_scalar_path_acceleration": 1.0,
        "constant_scalar_path_acceleration_units": (
            "geometry_parameter_per_second_squared"
        ),
        "control_guard_path_progress": 1.0,
        "control_guard_boundary_frame": 9,
        "control_guard_iteration": 0,
        "suffix_recovery_contract": "test_backward_recovery",
        "suffix_speed_squared_scale_from_ordinary_profile": 1.0,
    }
    uniform = replace(
        no_brake,
        markers=earlier,
        report=uniform_report,
    )
    with pytest.raises(cmc.CanonicalMotionCompilerError, match="dominance gap"):
        cmc._uniform_prefix_comparator_receipt(
            no_brake, uniform, fps=50.0
        )


def test_process_exact_search_matches_serial_without_invalid_duration_pruning(tmp_path):
    recipe = _make_recipe(tmp_path)
    source = recipe.sources[0]
    # This is a process-transport determinism test, not another exhaustive
    # entry/exit-search test.  A construction solve span of (0, 6) around the
    # (4, 5) seed with halo 1 caps every candidate at exit 6, so both the
    # serial and process branches execute the real C2 geometry and fixed-grid
    # retimer on the same small candidate set.
    pinned_row = replace(
        recipe.marker_semantics.row(source.motion_id),
        construction_marker=ConstructionMarker(
            annotation_frame=4,
            donor_preferred_frame=4,
            solve_span=(0, 6),
        ),
    )
    recipe = replace(
        recipe,
        marker_semantics=replace(
            recipe.marker_semantics,
            rows=tuple(
                pinned_row if row.motion_id == source.motion_id else row
                for row in recipe.marker_semantics.rows
            ),
        ),
    )
    # A two-coordinate regular arc plus an off-arc ready state gives the
    # retained core a legitimate closed C2 loop.  One active coordinate would
    # have to reverse sign on the return and therefore contain a real cusp.
    angle = np.linspace(0.0, np.pi, 9)
    source_coordinates = np.column_stack(
        (np.cos(angle), np.sin(angle))
    )
    ready = np.asarray([0.0, -1.0])
    contract = cmc._PathContract(
        position_lower=np.asarray([-2.0, -2.0]),
        position_upper=np.asarray([2.0, 2.0]),
        # These are deliberately loose test-only limits.  They keep the
        # controller-tick convergence receipt away from a refinement boundary;
        # the production limits remain supplied by the compiler caller.
        velocity=np.asarray([10.0, 10.0]),
        acceleration=np.asarray([10_000.0, 10_000.0]),
        coordinate_scale=np.asarray([1.0, 1.0]),
        coordinate_semantics=("joint_0", "joint_1"),
        coordinate_units=("rad", "rad"),
    )
    serial_winner, serial_report = cmc._search_core(
        recipe,
        source,
        source_coordinates,
        ready,
        contract,
        options=replace(_options(), grid_subdivisions=16),
    )
    process_winner, process_report = cmc._search_core(
        recipe,
        source,
        source_coordinates,
        ready,
        contract,
        options=replace(
            _options(),
            grid_subdivisions=16,
            search_workers=2,
            search_parallel_backend="process",
        ),
    )
    assert process_winner.score == serial_winner.score
    assert process_winner.candidate == serial_winner.candidate
    np.testing.assert_array_equal(
        process_winner.geometry.q_path, serial_winner.geometry.q_path
    )
    np.testing.assert_array_equal(
        process_winner.geometry.source_frame_map,
        serial_winner.geometry.source_frame_map,
    )
    np.testing.assert_array_equal(
        process_winner.retimed.q, serial_winner.retimed.q
    )
    np.testing.assert_array_equal(
        process_winner.retimed.qdot, serial_winner.retimed.qdot
    )
    np.testing.assert_array_equal(
        process_winner.retimed.path_position,
        serial_winner.retimed.path_position,
    )
    np.testing.assert_array_equal(
        process_winner.retimed.path_speed,
        serial_winner.retimed.path_speed,
    )
    np.testing.assert_array_equal(
        process_winner.retimed.path_acceleration,
        serial_winner.retimed.path_acceleration,
    )
    assert process_winner.retimed.markers == serial_winner.retimed.markers
    assert process_winner.retimed.report == serial_winner.retimed.report
    weighted = cmc._weighted_arc_formal_path(
        serial_winner.geometry, contract.coordinate_scale
    )
    _, expected_marker_lengths = (
        cmc._weighted_arc_marker_indices_and_lengths(
            serial_winner.geometry,
            weighted,
            {
                "window_start": pinned_row.ge80_seed[0],
                "source_anchor": pinned_row.nominal_event,
                "window_end": pinned_row.ge80_seed[1],
            },
        )
    )
    arc_receipt = serial_winner.retimed.report["weighted_arc_length"]
    assert arc_receipt["content_sha256"] == weighted.arc_path.content_sha256
    assert arc_receipt["coordinate_scale_sha256_float64_le"] == hashlib.sha256(
        np.ascontiguousarray(
            contract.coordinate_scale, dtype="<f8"
        ).tobytes(order="C")
    ).hexdigest()
    for name, expected_length in expected_marker_lengths.items():
        assert (
            serial_winner.retimed.report["markers"][name]["path_progress"]
            == expected_length
        )
    assert process_report["selected"] == serial_report["selected"]
    assert process_report["enumeration_sha256"] == (
        serial_report["enumeration_sha256"]
    )
    assert process_report["search_parallel_backend"] == "process"
    assert process_report["process_batch_size"] == 2
    assert process_report["process_batch_count"] >= 1
    assert process_report["admissibly_pruned_count"] == 0
    assert (
        process_report["fully_retimed_feasible_count"]
        + process_report["retime_failure_count"]
        == process_report["eligible_count"]
    )
    assert process_report["algorithm"] == (
        "enumerate_all_then_process_parallel_exact_retime_then_"
        "control_tick_lexicographic_rank"
    )


def test_publish_is_candidate_only_and_no_clobber(
    tmp_path, monkeypatch
):
    recipe = _make_recipe(tmp_path)
    monkeypatch.setattr(
        cmc,
        "solve_face_flipped_window",
        lambda source_joint_pos, *args, **kwargs: FakeFaceResult(
            source_joint_pos, kwargs["frame_indices"]
        ),
    )
    monkeypatch.setattr(cmc, "build_schema2_candidate", _fake_schema2_builder)
    monkeypatch.setattr(
        cmc, "build_canonical_geometry", _compiler_plumbing_geometry
    )
    monkeypatch.setattr(cmc, "retime_path", _fast_marker_only_retime)
    library = cmc.compile_loaded_canonical_motion_library(
        recipe,
        options=_options(),
        backend=FakePlantBackend(),
    )
    output = tmp_path / "published"
    result = cmc.write_compiled_canonical_motion_library(library, output)
    assert result == output
    manifest_path = output / cmc.BUILD_MANIFEST_NAME
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    assert manifest["build_verdict"] == "PASS_COMPILER_CANDIDATE_ONLY"
    assert manifest["training_authorized"] is False
    assert len(list(output.glob("*.npz"))) == 10
    assert len(list(output.glob("*.npz.manifest.json"))) == 10
    assert len(list(output.glob("*.npz.report.json"))) == 10

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        cmc.write_compiled_canonical_motion_library(library, output)
    assert manifest_path.read_bytes() == manifest_bytes


def test_atomic_directory_publish_never_replaces_existing_target(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    (source / "payload").write_text("new", encoding="utf-8")
    target.mkdir()
    (target / "payload").write_text("old", encoding="utf-8")

    with pytest.raises(FileExistsError):
        cmc._rename_directory_noreplace(source, target)

    assert source.is_dir()
    assert (source / "payload").read_text(encoding="utf-8") == "new"
    assert (target / "payload").read_text(encoding="utf-8") == "old"

    fresh_target = tmp_path / "fresh-target"
    cmc._rename_directory_noreplace(source, fresh_target)
    assert not source.exists()
    assert (fresh_target / "payload").read_text(encoding="utf-8") == "new"


def test_path_loader_wrapper_is_not_bypassed(tmp_path, monkeypatch):
    recipe = _make_recipe(tmp_path)
    observed = {}

    def fake_loader(path, *, repo_root=None):
        observed["load"] = (Path(path), repo_root)
        return recipe

    sentinel = object()

    def fake_compile(loaded, *, options, backend=None):
        observed["compile"] = (loaded, options, backend)
        return sentinel

    monkeypatch.setattr(cmc, "load_canonical_motion_recipe", fake_loader)
    monkeypatch.setattr(cmc, "compile_loaded_canonical_motion_library", fake_compile)
    options = _options()
    backend = FakePlantBackend()
    result = cmc.compile_canonical_motion_library(
        tmp_path / "bound_recipe.json",
        repo_root=tmp_path,
        options=options,
        backend=backend,
    )
    assert result is sentinel
    assert observed["load"] == (tmp_path / "bound_recipe.json", tmp_path)
    assert observed["compile"] == (recipe, options, backend)


def test_scalar_no_early_brake_is_a_hard_gate_but_remains_proxy_only():
    markers = {
        "window_start": MarkerMapping(1.0, 0.02, 1.0, 1, 1.0),
        "source_anchor": MarkerMapping(2.0, 0.04, 2.0, 2, 2.0),
        "window_end": MarkerMapping(3.0, 0.06, 3.0, 3, 3.0),
    }
    retimed = RetimeResult(
        q=np.zeros((5, 1)),
        qdot=np.zeros((5, 1)),
        path_position=np.arange(5, dtype=float),
        path_speed=np.zeros(4),
        path_acceleration=np.asarray([0.2, 0.1, 0.05, -0.4]),
        markers=markers,
        report={
            "nonnegative_acceleration_until_marker": {
                "enabled": True,
                "marker": "window_end",
                "prefix_scalar_acceleration_min_continuous": 0.0,
                "prefix_scalar_acceleration_min_50hz": 0.05,
                "control_guard_enabled": True,
                "output_interval_policy": (
                    "every_interval_starting_before_exact_marker_must_be_"
                    "nonnegative_including_the_straddling_interval"
                ),
            }
        },
    )
    receipt = cmc._scalar_no_early_brake_diagnostic(retimed, fps=50.0)
    assert receipt["negative_segment_count_before_window_end"] == 0
    assert receipt["negative_segment_count_overlapping_window"] == 0
    assert receipt["first_negative_segment_before_window_end"] is None
    assert receipt["first_negative_time_s"] is None
    assert receipt["no_negative_scalar_acceleration_before_window_end"] is True
    assert receipt["selection_criterion"] is True
    assert receipt["proxy_only"] is True
    assert "useful_racket_speed" in receipt["does_not_establish"]
    # Braking after the window end is deliberately not called early braking.
    assert 3 not in [receipt["first_negative_segment_before_window_end"]]

    bad_output = replace(
        retimed,
        path_acceleration=np.asarray([0.2, -0.1, 0.05, -0.4]),
    )
    with pytest.raises(
        cmc.CanonicalMotionCompilerError,
        match="negative scalar acceleration",
    ):
        cmc._scalar_no_early_brake_diagnostic(bad_output, fps=50.0)

    bad_report = dict(retimed.report)
    bad_policy = dict(
        bad_report["nonnegative_acceleration_until_marker"]
    )
    bad_policy["prefix_scalar_acceleration_min_50hz"] = -0.1
    bad_report["nonnegative_acceleration_until_marker"] = bad_policy
    with pytest.raises(
        cmc.CanonicalMotionCompilerError,
        match="50 Hz window_end",
    ):
        cmc._scalar_no_early_brake_diagnostic(
            replace(retimed, report=bad_report), fps=50.0
        )


def _noisy_source_path(frames: int = 40) -> np.ndarray:
    """Deterministic smooth base plus an alternating high-frequency jitter."""

    t = np.linspace(0.0, 1.0, frames)
    base = np.column_stack((np.sin(2.0 * np.pi * t), 0.3 * t))
    spikes = np.zeros_like(base)
    spikes[1:-1:2, 0] = 0.05
    spikes[2:-1:2, 0] = -0.05
    return base + spikes


def test_probe_source_smoothing_respects_tolerance_and_pins_endpoints():
    raw = _noisy_source_path()
    tolerance = 0.02
    smoothed, passes, deviation = cmc._smooth_source_coordinates(raw, tolerance)

    assert passes >= 1
    assert smoothed.shape == raw.shape
    achieved = float(np.max(np.abs(smoothed - raw)))
    assert achieved <= tolerance
    assert deviation == pytest.approx(achieved)
    # Endpoints stay exactly the raw source; only interior frames move.
    assert np.array_equal(smoothed[0], raw[0])
    assert np.array_equal(smoothed[-1], raw[-1])
    # A stricter tolerance can never do more smoothing than a looser one.
    _, fewer_passes, _ = cmc._smooth_source_coordinates(raw, tolerance / 4.0)
    assert fewer_passes <= passes


def test_probe_source_smoothing_none_knob_is_byte_identical_noop():
    raw = _noisy_source_path()
    options = _options()
    assert options.probe_source_smoothing_tolerance_rad is None
    out, receipt = cmc._apply_probe_source_smoothing(raw, options)
    # The exact same array object flows into geometry construction untouched.
    assert out is raw
    assert out.tobytes(order="C") == raw.tobytes(order="C")
    assert receipt["active"] is False
    assert receipt["passes_applied"] == 0


def test_probe_source_smoothing_reduces_discrete_curvature():
    raw = _noisy_source_path()
    smoothed, passes, _ = cmc._smooth_source_coordinates(raw, 0.02)
    assert passes >= 1

    def second_difference_norm(values: np.ndarray) -> float:
        return float(np.sum(np.diff(values, n=2, axis=0) ** 2))

    assert second_difference_norm(smoothed) < second_difference_norm(raw)


def test_probe_source_smoothing_receipt_records_passes_and_deviation():
    raw = _noisy_source_path()
    options = replace(_options(), probe_source_smoothing_tolerance_rad=0.02)
    out, receipt = cmc._apply_probe_source_smoothing(raw, options)
    assert out is not raw
    assert receipt["active"] is True
    assert receipt["tolerance_rad"] == pytest.approx(0.02)
    assert receipt["passes_applied"] >= 1
    assert receipt["max_abs_deviation_reached"] <= 0.02
    assert receipt["endpoints_pinned_exact"] is True
    assert receipt["probe_grade_not_verifiable"] is True
    # The passes/deviation in the receipt match a direct recomputation.
    _, passes, deviation = cmc._smooth_source_coordinates(raw, 0.02)
    assert receipt["passes_applied"] == passes
    assert receipt["max_abs_deviation_reached"] == pytest.approx(deviation)


def test_probe_source_smoothing_records_in_manifest_and_scope_receipts(
    tmp_path, monkeypatch
):
    recipe = _make_recipe(tmp_path)
    monkeypatch.setattr(
        cmc,
        "solve_face_flipped_window",
        lambda source_joint_pos, *args, **kwargs: FakeFaceResult(
            source_joint_pos, kwargs["frame_indices"]
        ),
    )
    monkeypatch.setattr(cmc, "build_schema2_candidate", _fake_schema2_builder)
    monkeypatch.setattr(
        cmc, "build_canonical_geometry", _compiler_plumbing_geometry
    )
    monkeypatch.setattr(cmc, "retime_path", _fast_marker_only_retime)

    library = cmc.compile_loaded_canonical_motion_library(
        recipe,
        options=replace(
            _options(), probe_source_smoothing_tolerance_rad=0.5
        ),
        backend=FakePlantBackend(),
    )

    grid = library.manifest["compiler_options"]["geometry_and_grid"]
    assert grid["probe_source_smoothing_tolerance_rad"] == pytest.approx(0.5)
    assert grid["probe_source_smoothing_is_identity"] is False
    for row in library.motions:
        smoothing = row.scope_report["probe_source_smoothing"]
        assert smoothing["active"] is True
        assert smoothing["tolerance_rad"] == pytest.approx(0.5)
        assert smoothing["passes_applied"] >= 1
        assert smoothing["max_abs_deviation_reached"] <= 0.5

    identity_library = cmc.compile_loaded_canonical_motion_library(
        recipe,
        options=_options(),
        backend=FakePlantBackend(),
    )
    identity_grid = identity_library.manifest["compiler_options"][
        "geometry_and_grid"
    ]
    assert identity_grid["probe_source_smoothing_tolerance_rad"] is None
    assert identity_grid["probe_source_smoothing_is_identity"] is True
    for row in identity_library.motions:
        assert row.scope_report["probe_source_smoothing"]["active"] is False


def test_probe_source_smoothing_rejects_nonpositive_tolerance(tmp_path):
    recipe = _make_recipe(tmp_path)
    bad = replace(_options(), probe_source_smoothing_tolerance_rad=0.0)
    with pytest.raises(
        cmc.CanonicalMotionCompilerError,
        match="probe_source_smoothing_tolerance_rad",
    ):
        cmc.compile_loaded_canonical_motion_library(
            recipe, options=bad, backend=FakePlantBackend()
        )


def test_fails_closed_without_explicit_full_root_or_s0_grounding(tmp_path):
    recipe = _make_recipe(tmp_path)
    bad_root = cmc.CompilerOptions(
        joint_acceleration_limits_rad_s2=np.ones(31),
        full_root_limits=cmc.FullRootPathLimits(
            position_lower=np.zeros(6),
            position_upper=np.zeros(6),
            velocity=np.ones(6),
            acceleration=np.ones(6),
        ),
        s0_full_grounding_offset_m=None,
    )
    with pytest.raises(
        cmc.CanonicalMotionCompilerError,
        match="position lower limit",
    ):
        cmc.compile_loaded_canonical_motion_library(
            recipe, options=bad_root, backend=FakePlantBackend()
        )

    no_grounding = cmc.CompilerOptions(
        joint_acceleration_limits_rad_s2=np.ones(31),
        full_root_limits=cmc.FullRootPathLimits(
            position_lower=np.full(6, -2.0),
            position_upper=np.full(6, 2.0),
            velocity=np.ones(6),
            acceleration=np.ones(6),
        ),
        s0_full_grounding_offset_m=None,
        samples_per_scaled_unit=6.0,
        min_connector_intervals=5,
        min_core_intervals=5,
        grid_subdivisions=4,
    )
    # Call the narrow scope preprocessor so the test reaches the missing
    # measured offset without spending time compiling the preceding nine rows.
    with pytest.raises(
        cmc.CanonicalMotionCompilerError,
        match="explicitly measured grounding offset",
    ):
        cmc._preprocess_scope(
            recipe,
            recipe.source("s0_highpress"),
            "full",
            no_grounding,
        )
