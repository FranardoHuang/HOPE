"""Conservative full-convex-hull contracts for the A3 table proxy."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


WBT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WBT_ROOT.parents[1]
SCRIPT = WBT_ROOT / "scripts/materialize_a3_table_collision_proxy.py"
SPEC = importlib.util.spec_from_file_location(
    "_test_materialize_a3_table_collision_proxy", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


def test_complete_tetra_partition_is_deterministic_and_covered_by_leaf_obbs():
    tetrahedra = [
        (
            (offset, 0.0, 0.0),
            (offset + 0.2, 0.0, 0.0),
            (offset, 0.1, 0.0),
            (offset, 0.0, 0.1),
        )
        for offset in (0.0, 0.3, 1.0, 1.3)
    ]
    first = M._partition_simplices(tetrahedra, 2)
    assert first == M._partition_simplices(tetrahedra, 2)
    assert sorted(index for leaf in first for index in leaf) == list(range(4))
    assert not set(first[0]).intersection(first[1])
    obbs, receipts = M._partition_pca_obbs(tetrahedra, first)
    assert len(obbs) == len(receipts) == 2
    for leaf, (center, axes), receipt in zip(first, obbs, receipts):
        vertices = np.asarray(
            [vertex for index in leaf for vertex in tetrahedra[index]],
            dtype=np.float64,
        )
        coefficients = np.linalg.solve(
            np.asarray(axes, dtype=np.float64).T,
            (vertices - np.asarray(center, dtype=np.float64)).T,
        ).T
        assert float(np.max(np.abs(coefficients))) <= 1.0
        assert receipt["float32_max_abs_obb_coefficient"] <= 1.0


def test_pca_degenerate_eigenspace_has_canonical_right_handed_tie_rule():
    basis, eigenvalues, groups = M._canonical_pca_basis(
        np.eye(3, dtype=np.float64)
    )
    assert eigenvalues == (1.0, 1.0, 1.0)
    assert groups == (3,)
    assert np.array_equal(basis, np.eye(3, dtype=np.float64))
    assert np.linalg.det(basis) == 1.0


def test_tracked_multi_obb_artifact_seals_full_hull_tetra_cover():
    artifact_path = (
        REPO_ROOT
        / "configs/a3_table_collision_proxy_a3p0807_20260808"
        / "a3_table_collision_components.v2.json"
    )
    document = json.loads(artifact_path.read_text(encoding="ascii"))
    assert document["schema_version"] == 2
    assert document["artifact_type"] == M.ARTIFACT_TYPE
    assert document["source_component_count"] == 62
    assert document["component_count"] == 63
    assert document["plant_identity"]["kind"] == M.PLANT_IDENTITY_KIND
    decomposition = document["decomposition"]
    assert decomposition["algorithm"] == M.TRIANGLE_PARTITION_ALGORITHM
    assert decomposition["backend_collision_authority"] == (
        "Isaac split meshes under convexHull plus exact canonical MuJoCo "
        "mesh hulls and analytic wrist primitives"
    )
    assert decomposition["float32_coverage_validated"] is True
    assert decomposition["hull_toolchain"] == {
        "numpy_version": M.PINNED_HULL_NUMPY_VERSION,
        "pca_eigenvalue_tie_rtol": M.PCA_EIGENVALUE_TIE_RTOL,
        "qhull_options": M.PINNED_HULL_QHULL_OPTIONS,
        "scipy_version": M.PINNED_HULL_SCIPY_VERSION,
    }

    rows = [
        row
        for row in document["components"]
        if row["source_component_id"].endswith(
            ":right_hand_pingpang_link.stl"
        )
    ]
    assert [row["proxy_box_index"] for row in rows] == [0, 1]
    assert {row["proxy_box_count"] for row in rows} == {2}
    assert {row["coverage_basis"] for row in rows} == {
        "complete_convex_hull_tetra_fan_pca_obb_union_plus_complete_mujoco_actual_collision_cover"
    }
    assert {row["coverage_primitive_kind"] for row in rows} == {
        "convex_hull_tetrahedron"
    }
    assert [row["coverage_primitive_count"] for row in rows] == [305, 305]
    assert {row["source_triangle_count"] for row in rows} == {10258}

    receipts = [
        row
        for row in decomposition["partition_receipts"]
        if row["source_component_id"] == rows[0]["source_component_id"]
    ]
    assert len(receipts) == 1
    receipt = receipts[0]
    hull = receipt["convex_hull"]
    assert hull["hull_vertex_count"] == 307
    assert hull["facet_count"] == hull["tetra_count"] == 610
    assert hull["hull_geometry_sha256"] == (
        "96b05900b79150be2546423adf8a4f2e9db700ed9870e1c9148a57a72ee288a0"
    )
    assert receipt["leaf_primitive_counts"] == [305, 305]
    assert hull["tetra_fan_volume_abs_error_m3"] <= 1.0e-15
    mu_binding = document["mujoco_actual_collision_binding"]
    assert mu_binding["mjcf_sha256"] == M.PINNED_MUJOCO_MJCF_SHA256
    assert {
        row["name"] for row in mu_binding["target_colliders"]
    } == set(M.MUJOCO_COLLISION_SOURCE_GROUPS)
    mu_cover = receipt["mujoco_actual_collision_cover"]
    assert mu_cover["max_float32_projection_interval_excess_m"] <= 0.0
    assert mu_cover["max_float32_projection_interval_excess_m"] <= -1.0e-6
    assert {
        row["name"] for row in mu_cover["leaf_by_primitive"]
    } == {
        name
        for name, source in M.MUJOCO_COLLISION_SOURCE_GROUPS.items()
        if source == "right_hand_pingpang_link.stl"
    }
    assert all(
        row["coverage_primitive_indices_sha256"]
        == receipt["leaf_primitive_indices_sha256"][index]
        for index, row in enumerate(rows)
    )
