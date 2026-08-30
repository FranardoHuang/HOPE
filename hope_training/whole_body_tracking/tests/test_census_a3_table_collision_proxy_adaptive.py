from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = REPO_ROOT / "hope_training" / "whole_body_tracking" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

SPEC = importlib.util.spec_from_file_location(
    "census_a3_table_collision_proxy_adaptive",
    SCRIPT_DIR / "census_a3_table_collision_proxy_adaptive.py",
)
assert SPEC is not None and SPEC.loader is not None
census = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(census)


def _component(rotation=None, translation=None):
    return {
        "owner_from_mesh_rotation": (
            rotation if rotation is not None else np.eye(3).tolist()
        ),
        "owner_from_mesh_translation": (
            translation if translation is not None else [0.0, 0.0, 0.0]
        ),
    }


def test_final_owner_float32_obb_preserves_rotated_vertex_coverage():
    angle = 0.713
    rotation = np.asarray(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    component = _component(rotation=rotation.tolist(), translation=[0.3, -0.2, 0.7])
    center, axes = census._final_owner_float32_obb(
        component,
        [0.25, -0.1, 0.05],
        [[0.4, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.1]],
    )
    mesh_corners = census._obb_corners(
        np.asarray([0.25, -0.1, 0.05]),
        np.asarray([[0.4, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.1]]),
    )
    owner_corners = census._owner_vertices(component, mesh_corners)
    maximum = census.proxy._float32_max_abs_obb_coefficient(
        center, axes, owner_corners
    )
    assert maximum <= 1.0
    assert center.dtype == np.float64
    assert axes.dtype == np.float64
    assert np.array_equal(center, center.astype(np.float32).astype(np.float64))
    assert np.array_equal(axes, axes.astype(np.float32).astype(np.float64))


def test_directed_hausdorff_uses_full_3d_empty_corner_not_one_axis_support():
    tetrahedron = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    result = census._directed_hausdorff_obb_to_convex_subset(
        np.asarray([0.5, 0.5, 0.5]),
        np.diag([0.5, 0.5, 0.5]),
        tetrahedron,
    )
    expected = 2.0 / math.sqrt(3.0)
    assert math.isclose(result["raw_distance_m"], expected, abs_tol=1.0e-12)
    assert result["certified_upper_bound_m"] > result["raw_distance_m"]
    assert np.allclose(result["obb_corner_owner_m"], [1.0, 1.0, 1.0])
    assert np.allclose(result["subset_witness_owner_m"], [1.0 / 3.0] * 3)


def test_closest_triangle_point_handles_face_edge_and_vertex_regions():
    triangle = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    assert np.allclose(
        census._closest_point_on_triangle([0.2, 0.3, 1.0], triangle),
        [0.2, 0.3, 0.0],
    )
    assert np.allclose(
        census._closest_point_on_triangle([0.8, 0.8, 0.0], triangle),
        [0.5, 0.5, 0.0],
    )
    assert np.allclose(
        census._closest_point_on_triangle([-1.0, -1.0, 0.0], triangle),
        [0.0, 0.0, 0.0],
    )


def test_candidate_is_deterministic_and_partitions_every_complete_tetrahedron():
    interior = (0.0, 0.0, 0.0)
    tetrahedra = [
        (interior, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        (interior, (-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)),
        (interior, (1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
        (interior, (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)),
    ]
    first = census._candidate(_component(), tetrahedra, 2)
    second = census._candidate(_component(), tetrahedra, 2)
    assert first == second
    assert first["whole_tetra_partition_proven"] is True
    assert sum(row["assigned_tetra_count"] for row in first["leaves"]) == 4
    assert first["owner_float32_max_abs_obb_coefficient"] <= 1.0


def test_source_enumerator_matches_reviewed_62_component_authority():
    components, source_sha = census._source_components(
        census.proxy.DEFAULT_SOURCE_URDF,
        census.proxy.DEFAULT_BODY_ORDER_SOURCE,
    )
    baseline, _file_sha = census._read_verified_baseline(
        census.DEFAULT_BASELINE_ARTIFACT
    )
    census._crosscheck_baseline_authority(components, source_sha, baseline)
    assert len(components) == 62
    assert len({row["source_component_id"] for row in components}) == 62
    assert source_sha == census.proxy.PINNED_SOURCE_URDF_SHA256
