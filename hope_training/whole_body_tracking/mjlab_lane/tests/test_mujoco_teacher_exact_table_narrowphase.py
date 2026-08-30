"""Focused tests for the opt-in teacher-replay table narrow phase."""

from __future__ import annotations

import inspect
import itertools
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch


LANE = Path(__file__).resolve().parents[1]
REPO = LANE.parents[2]
if str(LANE) not in sys.path:
    sys.path.insert(0, str(LANE))
if str(LANE.parent) not in sys.path:
    sys.path.insert(0, str(LANE.parent))

import mujoco_gpu_ac_full_mdp_initial_wait_env as wait_env
import mujoco_gpu_ac_full_mdp_wait_rsl3 as runner
import mujoco_gpu_ac_table_keepout as keepout
import mujoco_teacher_exact_table_narrowphase as exact
from mujoco_native import table_termination as authority


def _catalog() -> exact.SourceTriangleCatalog:
    return exact.SourceTriangleCatalog(
        repo_root=REPO,
        artifact_path=authority.COLLISION_PROXY_ARTIFACT,
        source_urdf=authority.PLANT_SOURCE_URDF,
    )


def _cube_triangles(half=1.0) -> np.ndarray:
    h = float(half)
    vertices = np.asarray(
        tuple(itertools.product((-h, h), repeat=3)), dtype=np.float64
    )
    index = {tuple(vertex): offset for offset, vertex in enumerate(vertices)}
    triangles = []
    for axis in range(3):
        other = tuple(value for value in range(3) if value != axis)
        for side in (-h, h):
            corners = []
            for first, second in ((-h, -h), (h, -h), (h, h), (-h, h)):
                point = [0.0, 0.0, 0.0]
                point[axis] = side
                point[other[0]] = first
                point[other[1]] = second
                corners.append(index[tuple(point)])
            triangles.extend(
                (vertices[[corners[0], corners[1], corners[2]]],
                 vertices[[corners[0], corners[2], corners[3]]])
            )
    return np.asarray(triangles, dtype=np.float64)


def test_triangle_aabb_is_inclusive_for_corner_edge_and_rotation():
    lo = np.asarray((-1.0, -1.0, -1.0))
    hi = np.asarray((1.0, 1.0, 1.0))
    corner_touch = np.asarray(
        ((1.0, 1.0, 1.0), (1.4, 1.0, 1.0), (1.0, 1.4, 1.0))
    )
    assert exact.triangle_aabb_overlap(corner_touch, lo, hi)

    theta = np.deg2rad(37.0)
    rotation = np.asarray(
        ((np.cos(theta), -np.sin(theta), 0.0),
         (np.sin(theta), np.cos(theta), 0.0),
         (0.0, 0.0, 1.0))
    )
    crossing = np.asarray(((-2.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 0.0, 2.0)))
    assert exact.triangle_aabb_overlap(crossing @ rotation.T, lo, hi)
    separated = corner_touch + np.asarray((0.001, 0.001, 0.001))
    assert not exact.triangle_aabb_overlap(separated, lo, hi)


def test_closed_source_solid_detects_table_box_containment():
    # No cube surface triangle intersects this inner AABB; solid parity must
    # still report overlap rather than clearing a real containment collision.
    hit, triangle = exact.triangles_aabb_overlap(
        _cube_triangles(),
        np.asarray((-0.1, -0.1, -0.1)),
        np.asarray((0.1, 0.1, 0.1)),
    )
    assert hit is True
    assert triangle is None

    vertices = np.asarray(
        tuple(itertools.product((-1.0, 1.0), repeat=3)), dtype=np.float64
    )
    assert exact.convex_hull_aabb_overlap(
        vertices,
        _cube_triangles(),
        np.asarray((-0.1, -0.1, -0.1)),
        np.asarray((0.1, 0.1, 0.1)),
    )


def test_20mm_expanded_table_is_the_positive_control_without_hidden_epsilon():
    triangle = np.asarray(
        ((1.01, -0.05, -0.05), (1.01, 0.05, -0.05), (1.01, 0.0, 0.05))
    )
    lo = np.asarray((-1.0, -1.0, -1.0))
    raw_hi = np.asarray((1.0, 1.0, 1.0))
    expanded_hi = np.asarray((1.02, 1.02, 1.02))
    assert not exact.triangle_aabb_overlap(triangle, lo, raw_hi)
    assert exact.triangle_aabb_overlap(triangle, lo - 0.02, expanded_hi)
    cube_triangles = _cube_triangles(0.001) + np.asarray((1.01, 0.0, 0.0))
    cube_vertices = np.unique(cube_triangles.reshape(-1, 3), axis=0)
    assert not exact.convex_hull_aabb_overlap(
        cube_vertices, cube_triangles, lo, raw_hi
    )
    assert exact.convex_hull_aabb_overlap(
        cube_vertices, cube_triangles, lo - 0.02, expanded_hi
    )


def test_component51_real_frame8_broad_positive_is_source_hull_clear():
    catalog = _catalog()
    component_index = 51
    component = catalog.load(component_index)
    row = catalog.rows[component_index]
    owner_position = np.asarray(
        (0.534522294998169, -0.3099048435688019, 0.845622718334198),
        dtype=np.float64,
    )
    owner_quaternion = np.asarray(
        (
            0.5015745759010315,
            -0.8510432243347168,
            -0.05899961292743683,
            0.14376181364059448,
        ),
        dtype=np.float64,
    )
    rotation = exact._quat_matrix_wxyz(owner_quaternion)
    center = owner_position + rotation @ np.asarray(
        row["local_center_owner_m"], dtype=np.float64
    )
    axes = np.asarray(row["local_half_axes_owner_m"], dtype=np.float64) @ rotation.T
    vertices = component.owner_vertices_m @ rotation.T + owner_position
    triangles = component.owner_hull_triangles_m @ rotation.T + owner_position
    scene = keepout._load_table_scene()
    table_rows = scene.action_ball_policy_obstacle_geometry()
    contract = scene.action_ball_policy_geometry_contract(table_rows)
    table_lo, table_hi = authority._validated_table_aabbs(contract)
    assert exact.obb_aabb_overlap(center, axes, table_lo[0], table_hi[0])
    assert not exact.convex_hull_aabb_overlap(
        vertices, triangles, table_lo[0], table_hi[0]
    )


@pytest.mark.parametrize("component_index", (51, 55, 57))
def test_component_heldouts_have_source_proven_empty_obb_regions(component_index):
    catalog = _catalog()
    component = catalog.load(component_index)
    row = catalog.rows[component_index]
    center = np.asarray(row["local_center_owner_m"], dtype=np.float64)
    axes = np.asarray(row["local_half_axes_owner_m"], dtype=np.float64)
    extent = float(np.min(np.linalg.norm(axes, axis=1)))
    found = None
    for coefficients in itertools.product((-0.98, -0.85, 0.85, 0.98), repeat=3):
        point = center + np.asarray(coefficients) @ axes
        half = max(1.0e-5, extent * 0.002)
        lo = point - half
        hi = point + half
        broad = exact.obb_aabb_overlap(center, axes, lo, hi)
        narrow = exact.convex_hull_aabb_overlap(
            component.owner_vertices_m, component.owner_hull_triangles_m, lo, hi
        )
        if broad and not narrow:
            found = (lo, hi)
            break
    assert found is not None, (
        f"component {component_index} has no deterministic empty OBB heldout"
    )


def _evaluator(
    *, table_lo, table_hi, geometry_dtype=np.float64
) -> exact.TeacherExactTableNarrowphase:
    catalog = _catalog()
    rows = catalog.rows
    owner_names = authority.TABLE_CONTACT_BODY_NAMES
    owner_index = {name: index for index, name in enumerate(owner_names)}
    return exact.TeacherExactTableNarrowphase(
        repo_root=REPO,
        artifact_path=authority.COLLISION_PROXY_ARTIFACT,
        source_urdf=authority.PLANT_SOURCE_URDF,
        runtime_mjb_sha256="a" * 64,
        mujoco_version="3.10.0",
        nativeccd_enabled=True,
        disableflags=0,
        component_ids=tuple(row["component_id"] for row in rows),
        owner_body_names=owner_names,
        component_owner_indices=np.asarray(
            [owner_index[row["owner_body_name"]] for row in rows]
        ),
        component_local_centers=np.asarray(
            [row["local_center_owner_m"] for row in rows], dtype=geometry_dtype
        ),
        component_local_half_axes=np.asarray(
            [row["local_half_axes_owner_m"] for row in rows], dtype=geometry_dtype
        ),
        table_lo=np.asarray(table_lo, dtype=np.float64),
        table_hi=np.asarray(table_hi, dtype=np.float64),
        blade_owner_index=owner_index["right_wrist_yaw_Link"],
        blade_center_offset=np.asarray((100.0, 100.0, 100.0)),
        blade_local_half_axes=np.eye(3) * 0.01,
        plant_identity={
            "root_mjcf_sha256": "b" * 64,
            "identity_manifest_sha256": "c" * 64,
            "portable_identity_sha256": "d" * 64,
            "verification_receipt_sha256": "e" * 64,
            "owner_local_frame_sha256": "f" * 64,
        },
        table_geometry_sha256="1" * 64,
        table_roles=authority.TABLE_ASSEMBLY_ROLES,
        table_expansion_m=authority.TABLE_GUARD_MARGIN_M,
    )


def test_all_source_hulls_close_eagerly_to_live_float32_broad_geometry():
    table_lo = np.zeros((5, 3), dtype=np.float32)
    table_hi = np.ones((5, 3), dtype=np.float32)
    receipt = _evaluator(
        table_lo=table_lo,
        table_hi=table_hi,
        geometry_dtype=np.float32,
    ).receipt()
    assert receipt["source_component_count_eagerly_closed"] == 62
    assert receipt["source_to_live_obb_max_escape_m"] <= 2.0e-8
    assert receipt["live_artifact_max_abs_delta_m"] <= 2.0e-8


def test_exact_positive_witness_binds_source_and_runtime_identity():
    catalog = _catalog()
    triangle = catalog.load(0).owner_hull_triangles_m[0]
    point = np.mean(triangle, axis=0)
    table_lo = np.full((5, 3), 200.0)
    table_hi = table_lo + 0.01
    table_lo[0] = point - 1.0e-3
    table_hi[0] = point + 1.0e-3
    evaluator = _evaluator(table_lo=table_lo, table_hi=table_hi)
    positions = np.zeros((32, 3), dtype=np.float64)
    quaternions = np.zeros((32, 4), dtype=np.float64)
    quaternions[:, 0] = 1.0
    result = evaluator.evaluate(
        body_position_env_m=positions,
        body_quaternion_wxyz=quaternions,
        capture_boundary="heldout",
        physics_substep_index=7,
    )
    assert result["exact_hit"] is True
    assert result["fail_closed"] is False
    evidence = evaluator.current_exact_evidence
    assert evidence["pair"]["exact_overlap"] is True
    assert evidence["narrowphase_identity_sha256"] == evaluator.receipt()[
        "identity_sha256"
    ]
    receipt = evaluator.receipt()
    assert receipt["runtime_mjb_sha256"] == "a" * 64
    assert receipt["nativeccd_enabled"] is True
    assert receipt["backend_contact_used_as_clearance_truth"] is False
    assert receipt["production_consumer_changed"] is False
    assert receipt["intersection_semantics"] == (
        "inclusive_closed_set_plus_20nm_roundoff_guard"
    )
    assert receipt["source_component_count_eagerly_closed"] == 62
    assert receipt["source_to_live_obb_max_escape_m"] <= 2.0e-8


def test_nonfinite_exact_input_fails_closed_to_the_broad_verdict():
    table_lo = np.zeros((5, 3), dtype=np.float64)
    table_hi = np.ones((5, 3), dtype=np.float64)
    evaluator = _evaluator(table_lo=table_lo, table_hi=table_hi)
    positions = np.zeros((32, 3), dtype=np.float64)
    positions[0, 0] = np.nan
    quaternions = np.zeros((32, 4), dtype=np.float64)
    quaternions[:, 0] = 1.0
    result = evaluator.evaluate(
        body_position_env_m=positions,
        body_quaternion_wxyz=quaternions,
        capture_boundary="heldout",
        physics_substep_index=None,
    )
    assert result["exact_hit"] is True
    assert result["fail_closed"] is True
    assert result["failure_type"] == "ExactTableNarrowphaseError"
    receipt = evaluator.receipt()
    assert receipt["failure_count"] == 1
    assert receipt["first_failure"]["capture_boundary"] == "heldout"


class _FakeEvaluator:
    def __init__(self, *, exact_hit=False, error=False):
        self.exact_hit = exact_hit
        self.error = error
        self.calls = 0
        self.current_exact_evidence = None
        self.failures = []

    def clear_current_exact_evidence(self):
        self.current_exact_evidence = None

    def record_failure(self, exc, **kwargs):
        self.failures.append((type(exc).__name__, kwargs))

    def evaluate(self, **_kwargs):
        self.calls += 1
        if self.error:
            raise RuntimeError("unknown exact result")
        return {"exact_hit": self.exact_hit}


def _refinement_rig(evaluator):
    return types.SimpleNamespace(
        _torch=torch,
        _diagnostic_exact_table_narrowphase=evaluator,
        _diagnostic_exact_table_host_pose=lambda _data: (
            np.zeros((32, 3)),
            np.tile(np.asarray((1.0, 0.0, 0.0, 0.0)), (32, 1)),
        ),
    )


def test_live_broad_phase_is_preserved_and_only_positive_rows_pay_exact_work():
    refine = wait_env.FullMdpInitialWaitVecEnv._diagnostic_refine_table_keepout
    evaluator = _FakeEvaluator(exact_hit=False)
    rig = _refinement_rig(evaluator)
    clear = torch.tensor([False])
    assert refine(
        rig,
        clear,
        data=object(),
        capture_boundary="heldout",
        substep_index=3,
    ) is clear
    assert evaluator.calls == 0
    broad = torch.tensor([True])
    result = refine(
        rig,
        broad,
        data=object(),
        capture_boundary="heldout",
        substep_index=3,
    )
    assert result.tolist() == [False]
    assert evaluator.calls == 1

    unknown = _FakeEvaluator(error=True)
    result = refine(
        _refinement_rig(unknown),
        broad,
        data=object(),
        capture_boundary="heldout",
        substep_index=3,
    )
    assert result.tolist() == [True]
    assert unknown.calls == 1
    assert unknown.failures == [
        (
            "RuntimeError",
            {"capture_boundary": "heldout", "physics_substep_index": 3},
        )
    ]


def test_attribution_preserves_real_broad_margin_and_appends_exact_evidence():
    plant = {
        "root_mjcf_sha256": "1" * 64,
        "identity_manifest_sha256": "2" * 64,
        "portable_identity_sha256": "3" * 64,
        "verification_receipt_sha256": "4" * 64,
        "owner_local_frame_sha256": "5" * 64,
    }
    broad_witness = {
        "schema": "action_ball_keepout_first_witness_v1",
        "selection": "first_production_order_overlap",
        "reason": "sat_overlap",
        "component_index": 51,
        "component_id": "component51",
        "component_kind": "body_proxy",
        "owner_body_index": 31,
        "owner_body_name": "right_wrist_yaw_Link",
        "table_index": 0,
        "table_role": "top",
        "sat_signed_margin_m": 0.00425,
        "root_position_env_m": [0.0, 0.0, 1.0],
        "root_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        "owner_position_env_m": [0.1, 0.2, 0.3],
        "owner_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        "plant_identity": plant,
        "collision_artifact_sha256": "6" * 64,
        "collision_content_sha256": "7" * 64,
    }
    evidence = {
        "schema": "action_ball_teacher_exact_overlap_evidence_v1",
        "selection": "first_exact_source_overlap_in_production_order",
        "pair": {"component_index": 55, "exact_overlap": True},
        "narrowphase_identity_sha256": "8" * 64,
    }
    rig = types.SimpleNamespace(
        _diagnostic_table_tick_first_positive=None,
        _diagnostic_first_table_terminal_source=None,
        _diagnostic_table_tick_keepout=False,
        _diagnostic_table_tick_final_resolved=False,
        _diagnostic_table_tick_resolved_any_substep=False,
        _diagnostic_first_resolved_substep=None,
        _diagnostic_contact_patch_transition_start_step=10,
        _diagnostic_exact_table_narrowphase=types.SimpleNamespace(
            current_exact_evidence=evidence
        ),
        _table_keepout=types.SimpleNamespace(
            diagnostic_first_positive_witness=lambda _data: dict(broad_witness)
        ),
        sim=types.SimpleNamespace(data=object()),
        decimation=4,
    )
    wait_env.FullMdpInitialWaitVecEnv._capture_diagnostic_table_attribution(
        rig,
        keepout=torch.tensor([True]),
        resolved=torch.tensor([False]),
        resolved_is_final=False,
        substep_index=2,
        capture_boundary="physics_substep_poststate",
    )
    witness = rig._diagnostic_table_tick_first_positive["keepout_witness"]
    assert witness["sat_signed_margin_m"] == 0.00425
    assert witness["component_index"] == 51
    assert witness["exact_narrowphase"] == evidence


def test_only_zero_ppo_teacher_replay_installs_the_exact_diagnostic():
    source = inspect.getsource(runner._run_teacher_replay)
    assert "enable_diagnostic_exact_table_narrowphase" in source
    assert "exact_table_narrowphase" in source
    refine = inspect.getsource(
        wait_env.FullMdpInitialWaitVecEnv._diagnostic_refine_table_keepout
    )
    assert "resolved_table_contact" not in refine
    assert "return keepout" in refine
    enable = inspect.getsource(
        wait_env.FullMdpInitialWaitVecEnv.enable_diagnostic_exact_table_narrowphase
    )
    assert "guard.sample = diagnostic_exact_sample" in enable
    assert "TABLE_CONTACT_BODY_NAMES" in enable
    assert "mujoco_teacher_exact_table_narrowphase" in enable
    after = inspect.getsource(
        wait_env.FullMdpInitialWaitVecEnv._after_physics_substep
    )
    final = inspect.getsource(
        wait_env.FullMdpInitialWaitVecEnv._latch_post_forward_table_keepout
    )
    assert "_diagnostic_refine_table_keepout" not in after
    assert "_diagnostic_refine_table_keepout" not in final
