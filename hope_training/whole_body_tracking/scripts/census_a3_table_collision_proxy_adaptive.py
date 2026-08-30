#!/usr/bin/env python3
"""Census deterministic adaptive OBB counts for every A3 source collider.

This is deliberately an offline evidence producer, not a runtime collision
artifact.  For each tracked URDF collision component it partitions the whole
convex-hull tetrahedral fan, fits one conservative OBB per partition leaf, and
measures the full three-dimensional directed Hausdorff excess from each final
owner-frame float32 OBB to the convex subset assigned to that leaf.

The first leaf count in ``1..max_leaves`` below the provisional excess ruler is
recorded.  No component name, prior winner, or table normal participates in
selection.  A result can only recommend a later runtime implementation when
all source components pass and the total row budget is small; it does not
claim complete MuJoCo actual-collider authority, which remains a separate
promotion prerequisite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import materialize_a3_table_collision_proxy as proxy


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BASELINE_ARTIFACT = (
    REPO_ROOT
    / "configs"
    / "a3_table_collision_proxy_a3p0807_20260808"
    / "a3_table_collision_components.v2.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "configs"
    / "a3_table_collision_proxy_a3p0807_20260808"
    / "a3_table_collision_adaptive_obb_census.v1.json"
)
SCHEMA_VERSION = 1
ARTIFACT_TYPE = "a3_table_collision_adaptive_obb_census_v1"
DEFAULT_MAX_LEAVES = 8
DEFAULT_EXCESS_RULER_M = 0.005
DEFAULT_SOURCE_ROW_BUDGET = 128
FINAL_OWNER_OUTWARD_PAD_M = 1.0e-6
DISTANCE_CERTIFICATE_PAD_M = 1.0e-9
INSIDE_CLASSIFICATION_ULP_FACTOR = 256.0
FACET_CHUNK_SIZE = 4096


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-urdf", type=Path, default=proxy.DEFAULT_SOURCE_URDF)
    parser.add_argument(
        "--body-order-source", type=Path, default=proxy.DEFAULT_BODY_ORDER_SOURCE
    )
    parser.add_argument(
        "--baseline-artifact", type=Path, default=DEFAULT_BASELINE_ARTIFACT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-leaves", type=int, default=DEFAULT_MAX_LEAVES)
    parser.add_argument(
        "--excess-ruler-m", type=float, default=DEFAULT_EXCESS_RULER_M
    )
    parser.add_argument(
        "--source-row-budget", type=int, default=DEFAULT_SOURCE_ROW_BUDGET
    )
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def _read_verified_baseline(path: Path) -> tuple[dict[str, object], str]:
    payload = path.resolve().read_bytes()
    document = json.loads(payload)
    if not isinstance(document, dict):
        raise ValueError("baseline proxy artifact must be one JSON object")
    recorded = document.get("content_sha256")
    unsigned = dict(document)
    unsigned.pop("content_sha256", None)
    recomputed = _sha256_bytes(_canonical_json_bytes(unsigned))
    if recorded != recomputed:
        raise ValueError("baseline proxy artifact content SHA is invalid")
    if (
        document.get("schema_version") != proxy.SCHEMA_VERSION
        or document.get("artifact_type") != proxy.ARTIFACT_TYPE
        or document.get("source_component_count") != 62
    ):
        raise ValueError("baseline proxy artifact is not the reviewed v2 authority")
    return document, _sha256_bytes(payload)


def _runtime_owner_map(
    root: ET.Element, body_order: Sequence[str]
) -> dict[str, tuple[str, Any, Any]]:
    parent: dict[str, tuple[str, str, Any, Any]] = {}
    for joint in root.findall("joint"):
        parent_element = joint.find("parent")
        child_element = joint.find("child")
        if parent_element is None or child_element is None:
            raise ValueError("joint is missing parent/child")
        child_name = str(child_element.get("link"))
        if child_name in parent:
            raise ValueError(f"duplicate URDF joint child: {child_name}")
        rotation, translation = proxy._origin_transform(joint.find("origin"))
        parent[child_name] = (
            str(parent_element.get("link")),
            str(joint.get("type")),
            rotation,
            translation,
        )

    order_set = set(body_order)
    result: dict[str, tuple[str, Any, Any]] = {}
    for link in root.findall("link"):
        link_name = str(link.get("name"))
        if not link_name or link_name in result:
            raise ValueError(f"duplicate or empty URDF link name: {link_name!r}")
        owner = link_name
        rotation = proxy._identity_rotation()
        translation = (0.0, 0.0, 0.0)
        seen: set[str] = set()
        while owner in parent and parent[owner][1] == "fixed":
            if owner in seen:
                raise ValueError("fixed-joint cycle in A3 URDF")
            seen.add(owner)
            parent_name, _, joint_rotation, joint_translation = parent[owner]
            rotation, translation = proxy._compose(
                joint_rotation,
                joint_translation,
                rotation,
                translation,
            )
            owner = parent_name
        if owner not in order_set:
            raise ValueError(
                f"collision link {link_name!r} maps to unknown body {owner!r}"
            )
        result[link_name] = (owner, rotation, translation)
    return result


def _source_components(
    source_urdf: Path, body_order_source: Path
) -> tuple[list[dict[str, object]], str]:
    source_urdf = source_urdf.resolve()
    if REPO_ROOT.resolve() not in source_urdf.parents:
        raise ValueError("source URDF must remain inside the tracked repo")
    source_root = source_urdf.parents[1]
    mesh_root = (source_root / "meshes").resolve()
    body_order = proxy._body_order(body_order_source.resolve())
    root = ET.parse(source_urdf).getroot()
    owners = _runtime_owner_map(root, body_order)
    components: list[dict[str, object]] = []
    for link in root.findall("link"):
        link_name = str(link.get("name"))
        owner, owner_from_link_rotation, owner_from_link_translation = owners[
            link_name
        ]
        for collision_index, collision in enumerate(link.findall("collision")):
            mesh = collision.find("geometry/mesh")
            if mesh is None:
                raise ValueError(
                    f"non-mesh A3 collision is not supported: {link_name}"
                )
            filename = str(mesh.get("filename"))
            marker = "/meshes/"
            if marker not in filename:
                raise ValueError(f"unexpected A3 collision mesh URI: {filename}")
            relative_mesh = filename.rsplit(marker, 1)[1]
            mesh_path = (mesh_root / relative_mesh).resolve()
            if not mesh_path.is_file() or mesh_root not in mesh_path.parents:
                raise ValueError(f"collision mesh escapes or is missing: {relative_mesh}")
            scale = proxy._float_triplet(
                mesh.get("scale"), default=(1.0, 1.0, 1.0)
            )
            if not all(value > 0.0 for value in scale):
                raise ValueError("collision mesh scale must be positive")
            link_from_mesh_rotation, link_from_mesh_translation = (
                proxy._origin_transform(collision.find("origin"))
            )
            owner_from_mesh_rotation, owner_from_mesh_translation = proxy._compose(
                owner_from_link_rotation,
                owner_from_link_translation,
                link_from_mesh_rotation,
                link_from_mesh_translation,
            )
            repo_relative_mesh = mesh_path.relative_to(REPO_ROOT).as_posix()
            source_component_id = (
                f"{owner}:{link_name}:{collision_index}:{relative_mesh}"
            )
            components.append(
                {
                    "mesh_path": repo_relative_mesh,
                    "mesh_sha256": _sha256_bytes(mesh_path.read_bytes()),
                    "owner_body_name": owner,
                    "owner_from_mesh_rotation": owner_from_mesh_rotation,
                    "owner_from_mesh_translation": owner_from_mesh_translation,
                    "scale": scale,
                    "source_component_id": source_component_id,
                    "source_link_name": link_name,
                    "source_mesh_path": mesh_path,
                }
            )
    components.sort(key=lambda row: str(row["source_component_id"]))
    if len(components) != 62 or len(
        {str(row["source_component_id"]) for row in components}
    ) != 62:
        raise ValueError("source authority must contain 62 unique components")
    return components, _sha256_bytes(source_urdf.read_bytes())


def _crosscheck_baseline_authority(
    components: Sequence[Mapping[str, object]],
    source_urdf_sha256: str,
    baseline: Mapping[str, object],
) -> None:
    source_receipt = baseline.get("source_urdf")
    if not isinstance(source_receipt, dict) or source_receipt.get(
        "sha256"
    ) != source_urdf_sha256:
        raise ValueError("source URDF differs from baseline artifact authority")
    decomposition = baseline.get("decomposition")
    if not isinstance(decomposition, dict):
        raise ValueError("baseline decomposition receipt is missing")
    partition_rows = decomposition.get("partition_receipts")
    if not isinstance(partition_rows, list):
        raise ValueError("baseline partition receipts are missing")
    expected = {
        str(row["source_component_id"]): (
            str(row["mesh_path"]),
            str(row["mesh_sha256"]),
        )
        for row in partition_rows
        if isinstance(row, dict)
    }
    observed = {
        str(row["source_component_id"]): (
            str(row["mesh_path"]),
            str(row["mesh_sha256"]),
        )
        for row in components
    }
    if observed != expected:
        raise ValueError("source component closure differs from baseline artifact")


def _owner_vertices(component: Mapping[str, object], vertices: Any) -> Any:
    import numpy as np

    rotation = np.asarray(component["owner_from_mesh_rotation"], dtype=np.float64)
    translation = np.asarray(
        component["owner_from_mesh_translation"], dtype=np.float64
    )
    return np.asarray(vertices, dtype=np.float64) @ rotation.T + translation


def _final_owner_float32_obb(
    component: Mapping[str, object], center_mesh: Any, half_axes_mesh: Any
) -> tuple[Any, Any]:
    import numpy as np

    rotation = np.asarray(component["owner_from_mesh_rotation"], dtype=np.float64)
    translation = np.asarray(
        component["owner_from_mesh_translation"], dtype=np.float64
    )
    center_owner = rotation @ np.asarray(center_mesh, dtype=np.float64) + translation
    axes_owner = np.asarray(half_axes_mesh, dtype=np.float64) @ rotation.T
    lengths = np.linalg.norm(axes_owner, axis=1)
    if not bool(np.all(np.isfinite(lengths))) or bool(np.any(lengths <= 0.0)):
        raise ValueError("candidate OBB has a degenerate half axis")
    axes_owner *= ((lengths + FINAL_OWNER_OUTWARD_PAD_M) / lengths)[:, None]
    return (
        center_owner.astype(np.float32).astype(np.float64),
        axes_owner.astype(np.float32).astype(np.float64),
    )


def _obb_corners(center: Any, half_axes: Any) -> Any:
    import itertools
    import numpy as np

    center = np.asarray(center, dtype=np.float64)
    half_axes = np.asarray(half_axes, dtype=np.float64)
    return np.asarray(
        [
            center + sum(sign * half_axes[axis] for axis, sign in enumerate(signs))
            for signs in itertools.product((-1.0, 1.0), repeat=3)
        ],
        dtype=np.float64,
    )


def _closest_point_on_segment(point: Any, start: Any, end: Any) -> Any:
    import numpy as np

    point = np.asarray(point, dtype=np.float64)
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    direction = end - start
    denominator = float(direction @ direction)
    if denominator <= 0.0:
        return start.copy()
    amount = max(0.0, min(1.0, float((point - start) @ direction) / denominator))
    return start + amount * direction


def _closest_point_on_triangle(point: Any, triangle: Any) -> Any:
    """Return a valid closest-point candidate on one closed triangle."""

    import numpy as np

    point = np.asarray(point, dtype=np.float64)
    triangle = np.asarray(triangle, dtype=np.float64)
    a, b, c = triangle
    ab = b - a
    ac = c - a
    normal = np.cross(ab, ac)
    norm_sq = float(normal @ normal)
    candidates = [
        _closest_point_on_segment(point, a, b),
        _closest_point_on_segment(point, b, c),
        _closest_point_on_segment(point, c, a),
    ]
    if norm_sq > 0.0:
        projection = point - normal * float((point - a) @ normal) / norm_sq
        gram = np.asarray(
            [[float(ab @ ab), float(ab @ ac)], [float(ab @ ac), float(ac @ ac)]],
            dtype=np.float64,
        )
        determinant = float(np.linalg.det(gram))
        if determinant > 0.0:
            uv = np.linalg.solve(
                gram,
                np.asarray(
                    [float((projection - a) @ ab), float((projection - a) @ ac)]
                ),
            )
            weights = np.asarray(
                (1.0 - float(uv[0]) - float(uv[1]), float(uv[0]), float(uv[1])),
                dtype=np.float64,
            )
            if bool(np.all(weights >= 0.0)):
                weights = np.maximum(weights, 0.0)
                weights /= np.sum(weights)
                candidates.append(weights @ triangle)
    return min(candidates, key=lambda candidate: float(np.linalg.norm(point - candidate)))


def _convex_subset_boundary(vertices: Any) -> tuple[Any, Any]:
    import numpy as np
    from scipy.spatial import ConvexHull

    vertices = np.unique(np.asarray(vertices, dtype=np.float64), axis=0)
    hull = ConvexHull(vertices, qhull_options=proxy.PINNED_HULL_QHULL_OPTIONS)
    facets = np.sort(np.asarray(hull.simplices, dtype=np.int64), axis=1)
    order = np.lexsort((facets[:, 2], facets[:, 1], facets[:, 0]))
    triangles = vertices[facets[order]]
    return hull, triangles


def _point_to_convex_subset_distance(
    point: Any,
    hull: Any,
    boundary_triangles: Any,
    *,
    facet_chunk_size: int = FACET_CHUNK_SIZE,
) -> tuple[float, Any, float]:
    import numpy as np

    point = np.asarray(point, dtype=np.float64)
    signed = hull.equations[:, :3] @ point + hull.equations[:, 3]
    equation_normals = np.asarray(hull.equations[:, :3], dtype=np.float64)
    equation_offsets = np.asarray(hull.equations[:, 3], dtype=np.float64)
    inside_guards = (
        INSIDE_CLASSIFICATION_ULP_FACTOR
        * np.finfo(np.float64).eps
        * (
            np.sum(np.abs(equation_normals) * np.abs(point)[None, :], axis=1)
            + np.abs(equation_offsets)
            + 1.0
        )
    )
    max_inside_guard = float(np.max(inside_guards))
    # Only a point with a strict numerical reserve is classified inside.  A
    # grey-zone point is sent to the boundary path; that can overestimate a
    # true zero distance by a few ulps, but can never turn an outside point
    # into a false zero.
    if bool(np.all(signed + inside_guards < 0.0)):
        return 0.0, point.copy(), max_inside_guard

    # Batch every hull facet.  The former scalar loop performed one solve and
    # three segment projections per Python iteration and made the 62-source
    # census impractical.  All four candidates below are constructed as real
    # points of the closed triangle, so their distances remain sound upper
    # bounds even at floating-point region boundaries.
    if (
        isinstance(facet_chunk_size, bool)
        or not isinstance(facet_chunk_size, int)
        or facet_chunk_size <= 0
    ):
        raise ValueError("facet chunk size must be a positive integer")
    triangles = np.asarray(boundary_triangles, dtype=np.float64)
    best_squared = math.inf
    best_witness = None
    for start_index in range(0, triangles.shape[0], facet_chunk_size):
        chunk = triangles[start_index : start_index + facet_chunk_size]
        a = chunk[:, 0, :]
        b = chunk[:, 1, :]
        c = chunk[:, 2, :]

        def segment_candidates(start: Any, end: Any) -> Any:
            direction = end - start
            denominator = np.sum(direction * direction, axis=1)
            if bool(np.any(denominator <= 0.0)):
                raise ValueError("convex hull contains a degenerate triangle edge")
            amount = (
                np.sum((point[None, :] - start) * direction, axis=1) / denominator
            )
            amount = np.clip(amount, 0.0, 1.0)
            return start + amount[:, None] * direction

        edge_ab = segment_candidates(a, b)
        edge_bc = segment_candidates(b, c)
        edge_ca = segment_candidates(c, a)

        ab = b - a
        ac = c - a
        normal = np.cross(ab, ac)
        normal_sq = np.sum(normal * normal, axis=1)
        if bool(np.any(normal_sq <= 0.0)):
            raise ValueError("convex hull contains a degenerate triangle facet")
        plane_amount = (
            np.sum((point[None, :] - a) * normal, axis=1) / normal_sq
        )
        projection = point[None, :] - plane_amount[:, None] * normal
        # Cross-product barycentrics divide by the already-validated squared
        # facet area.  The algebraically equivalent Gram determinant suffers
        # catastrophic cancellation on A3's very thin but non-degenerate
        # facets and incorrectly classified them as singular.
        weight_a = np.sum(
            np.cross(b - projection, c - projection) * normal, axis=1
        ) / normal_sq
        weight_b = np.sum(
            np.cross(c - projection, a - projection) * normal, axis=1
        ) / normal_sq
        weight_c = np.sum(
            np.cross(a - projection, b - projection) * normal, axis=1
        ) / normal_sq
        raw_weights = np.stack((weight_a, weight_b, weight_c), axis=1)
        valid_face = np.all(raw_weights >= 0.0, axis=1)
        weights = np.maximum(raw_weights, 0.0)
        weight_sum = np.sum(weights, axis=1)
        if bool(np.any(weight_sum <= 0.0)):
            raise ValueError("triangle projection weights cannot be normalized")
        weights /= weight_sum[:, None]
        face = np.sum(weights[:, :, None] * chunk, axis=1)

        candidates = np.stack((edge_ab, edge_bc, edge_ca, face), axis=1)
        squared = np.sum((candidates - point[None, None, :]) ** 2, axis=2)
        squared[~valid_face, 3] = math.inf
        flat_index = int(np.argmin(squared))
        facet_index, candidate_index = np.unravel_index(flat_index, squared.shape)
        chunk_best = float(squared[facet_index, candidate_index])
        # Strict comparison preserves the first canonical facet/candidate on
        # exact ties, independent of the chosen chunk width.
        if chunk_best < best_squared:
            best_squared = chunk_best
            best_witness = candidates[facet_index, candidate_index].copy()
    if best_witness is None or not math.isfinite(best_squared):
        raise ValueError("convex-subset boundary produced no finite witness")
    return float(math.sqrt(best_squared)), best_witness, max_inside_guard


def _directed_hausdorff_obb_to_convex_subset(
    center: Any, half_axes: Any, subset_vertices: Any
) -> dict[str, object]:
    """Compute d_H(OBB -> assigned convex subset) over all OBB corners.

    Distance to a closed convex set is convex.  Every point of an OBB is a
    convex combination of its eight corners, so the maximum distance is
    attained at a corner.  Point-to-polytope distance is then the minimum over
    the triangulated closed hull boundary for outside points.
    """

    import numpy as np

    hull, boundary_triangles = _convex_subset_boundary(subset_vertices)
    maximum = -1.0
    maximum_inside_guard = 0.0
    corner_witness = None
    subset_witness = None
    for corner in _obb_corners(center, half_axes):
        distance, witness, inside_guard = _point_to_convex_subset_distance(
            corner, hull, boundary_triangles
        )
        maximum_inside_guard = max(maximum_inside_guard, inside_guard)
        if distance > maximum:
            maximum = distance
            corner_witness = corner.copy()
            subset_witness = witness.copy()
    if corner_witness is None or subset_witness is None or not math.isfinite(maximum):
        raise ValueError("directed Hausdorff evaluation produced no finite witness")
    certified = float(np.nextafter(maximum, math.inf)) + DISTANCE_CERTIFICATE_PAD_M
    return {
        "certified_upper_bound_m": certified,
        "max_inside_classification_guard_m": maximum_inside_guard,
        "obb_corner_owner_m": [float(value) for value in corner_witness],
        "raw_distance_m": maximum,
        "subset_witness_owner_m": [float(value) for value in subset_witness],
    }


def _candidate(
    component: Mapping[str, object], tetrahedra: Sequence[Any], leaf_count: int
) -> dict[str, object]:
    import numpy as np

    leaves = proxy._partition_simplices(tetrahedra, leaf_count)
    fitted, fit_receipts = proxy._partition_pca_obbs(tetrahedra, leaves)
    leaf_rows = []
    maximum_excess = -1.0
    maximum_coverage_coefficient = -1.0
    for leaf_index, (leaf, fit, fit_receipt) in enumerate(
        zip(leaves, fitted, fit_receipts)
    ):
        assigned_mesh = np.unique(
            np.asarray(
                [
                    tetrahedra[simplex_index][vertex_index]
                    for simplex_index in leaf
                    for vertex_index in range(4)
                ],
                dtype=np.float64,
            ),
            axis=0,
        )
        assigned_owner = _owner_vertices(component, assigned_mesh)
        center_owner, axes_owner = _final_owner_float32_obb(
            component, fit[0], fit[1]
        )
        coverage_coefficient = proxy._float32_max_abs_obb_coefficient(
            center_owner, axes_owner, assigned_owner
        )
        maximum_coverage_coefficient = max(
            maximum_coverage_coefficient, coverage_coefficient
        )
        directed = _directed_hausdorff_obb_to_convex_subset(
            center_owner, axes_owner, assigned_owner
        )
        maximum_excess = max(
            maximum_excess, float(directed["certified_upper_bound_m"])
        )
        leaf_rows.append(
            {
                "assigned_tetra_count": len(leaf),
                "assigned_tetra_indices_sha256": _sha256_bytes(
                    _canonical_json_bytes(list(leaf))
                ),
                "directed_hausdorff": directed,
                "fit_receipt": fit_receipt,
                "leaf_index": leaf_index,
                "local_center_owner_f32_m": [float(value) for value in center_owner],
                "local_half_axes_owner_f32_m": [
                    [float(value) for value in axis] for axis in axes_owner
                ],
                "owner_float32_max_abs_obb_coefficient": coverage_coefficient,
                "unique_assigned_vertex_count": int(assigned_owner.shape[0]),
            }
        )
    flattened = [index for leaf in leaves for index in leaf]
    if sorted(flattened) != list(range(len(tetrahedra))):
        raise ValueError("candidate does not cover every tetrahedron exactly once")
    return {
        "leaf_count": leaf_count,
        "leaves": leaf_rows,
        "max_directed_hausdorff_certified_m": maximum_excess,
        "owner_float32_max_abs_obb_coefficient": maximum_coverage_coefficient,
        "partition_sha256": _sha256_bytes(
            _canonical_json_bytes([list(leaf) for leaf in leaves])
        ),
        "whole_tetra_partition_proven": True,
    }


def census(
    *,
    source_urdf: Path,
    body_order_source: Path,
    baseline_artifact: Path,
    max_leaves: int,
    excess_ruler_m: float,
    source_row_budget: int,
) -> dict[str, object]:
    import numpy as np
    import scipy

    if (
        isinstance(max_leaves, bool)
        or max_leaves <= 0
        or max_leaves > 8
        or not math.isfinite(excess_ruler_m)
        or excess_ruler_m <= 0.0
        or isinstance(source_row_budget, bool)
        or source_row_budget <= 0
    ):
        raise ValueError("adaptive census bounds are invalid")
    if (
        np.__version__ != proxy.PINNED_HULL_NUMPY_VERSION
        or scipy.__version__ != proxy.PINNED_HULL_SCIPY_VERSION
    ):
        raise ValueError("adaptive census requires the pinned hull toolchain")
    baseline, baseline_file_sha256 = _read_verified_baseline(baseline_artifact)
    components, source_urdf_sha256 = _source_components(
        source_urdf, body_order_source
    )
    _crosscheck_baseline_authority(components, source_urdf_sha256, baseline)

    rows = []
    selected_counts: list[int] = []
    unresolved_ids: list[str] = []
    for component in components:
        scale = tuple(float(value) for value in component["scale"])
        triangles = tuple(
            tuple(
                tuple(float(vertex[axis]) * scale[axis] for axis in range(3))
                for vertex in triangle
            )
            for triangle in proxy._stl_triangles(component["source_mesh_path"])
        )
        tetrahedra, hull_receipt = proxy._convex_hull_tetrahedra(
            triangles
        )
        del triangles
        candidates = []
        selected_leaf_count = None
        evaluated_max_leaves = min(max_leaves, len(tetrahedra))
        for leaf_count in range(1, evaluated_max_leaves + 1):
            candidate = _candidate(component, tetrahedra, leaf_count)
            candidate["passes_provisional_excess_ruler"] = (
                float(candidate["max_directed_hausdorff_certified_m"])
                <= excess_ruler_m
            )
            candidates.append(candidate)
            if candidate["passes_provisional_excess_ruler"]:
                selected_leaf_count = leaf_count
                break
        component_id = str(component["source_component_id"])
        if selected_leaf_count is None:
            unresolved_ids.append(component_id)
        else:
            selected_counts.append(selected_leaf_count)
        rows.append(
            {
                "candidates": candidates,
                "convex_hull": hull_receipt,
                "evaluated_max_leaf_count": evaluated_max_leaves,
                "mesh_path": component["mesh_path"],
                "mesh_sha256": component["mesh_sha256"],
                "owner_body_name": component["owner_body_name"],
                "selected_leaf_count": selected_leaf_count,
                "source_component_id": component_id,
                "source_link_name": component["source_link_name"],
            }
        )

    baseline_proxy_rows = int(baseline["component_count"])
    selected_proxy_rows = sum(selected_counts) if not unresolved_ids else None
    within_budget = (
        selected_proxy_rows is not None
        and selected_proxy_rows <= source_row_budget
    )
    distribution = {
        str(count): selected_counts.count(count)
        for count in range(1, max_leaves + 1)
        if selected_counts.count(count)
    }
    summary = {
        "baseline_proxy_rows": baseline_proxy_rows,
        "baseline_runtime_rows_including_blade": baseline_proxy_rows + 1,
        "global_runtime_candidate_eligible": bool(within_budget),
        "max_tested_leaf_count": max_leaves,
        "provisional_excess_ruler_m": excess_ruler_m,
        "selected_leaf_count_distribution": distribution,
        "selected_proxy_rows": selected_proxy_rows,
        "selected_proxy_rows_linear_multiplier_vs_baseline": (
            None
            if selected_proxy_rows is None
            else selected_proxy_rows / float(baseline_proxy_rows)
        ),
        "selected_runtime_rows_including_blade": (
            None if selected_proxy_rows is None else selected_proxy_rows + 1
        ),
        "source_row_budget": source_row_budget,
        "unresolved_component_count": len(unresolved_ids),
        "unresolved_component_ids": unresolved_ids,
    }
    if unresolved_ids:
        recommendation = "stop_and_design_exact_convex_positive_pair_narrow_phase"
    elif not within_budget:
        recommendation = "stop_and_design_exact_convex_positive_pair_narrow_phase"
    else:
        recommendation = "eligible_for_separate_global_runtime_design_review"

    content: dict[str, object] = {
        "algorithm": {
            "distance_certificate_pad_m": DISTANCE_CERTIFICATE_PAD_M,
            "distance_definition": (
                "maximum over final owner-frame float32 OBB corners of exact "
                "point distance to the assigned closed convex tetra subset"
            ),
            "final_owner_outward_pad_m": FINAL_OWNER_OUTWARD_PAD_M,
            "inside_classification_ulp_factor": INSIDE_CLASSIFICATION_ULP_FACTOR,
            "facet_chunk_size": FACET_CHUNK_SIZE,
            "partition_algorithm": proxy.TRIANGLE_PARTITION_ALGORITHM,
            "selection_is_component_identity_blind": True,
        },
        "artifact_type": ARTIFACT_TYPE,
        "authority": {
            "backend_promotion_claimed": False,
            "baseline_artifact_content_sha256": baseline["content_sha256"],
            "baseline_artifact_file_sha256": baseline_file_sha256,
            "baseline_artifact_path": baseline_artifact.resolve().relative_to(
                REPO_ROOT
            ).as_posix(),
            "complete_source_convex_solid_coverage_proven": True,
            "mujoco_actual_collider_scope": (
                "not claimed by this source-only census; the reviewed v2 "
                "artifact has exact MuJoCo primitive binding only for its "
                "explicit target sources"
            ),
            "source_component_count": len(components),
            "source_urdf_path": source_urdf.resolve().relative_to(REPO_ROOT).as_posix(),
            "source_urdf_sha256": source_urdf_sha256,
        },
        "components": rows,
        "recommendation": recommendation,
        "schema_version": SCHEMA_VERSION,
        "summary": summary,
        "toolchain": {
            "numpy_version": np.__version__,
            "qhull_options": proxy.PINNED_HULL_QHULL_OPTIONS,
            "scipy_version": scipy.__version__,
        },
    }
    content["content_sha256"] = _sha256_bytes(_canonical_json_bytes(content))
    return content


def main() -> int:
    args = _parse_args()
    document = census(
        source_urdf=args.source_urdf,
        body_order_source=args.body_order_source,
        baseline_artifact=args.baseline_artifact,
        max_leaves=args.max_leaves,
        excess_ruler_m=args.excess_ruler_m,
        source_row_budget=args.source_row_budget,
    )
    encoded = _canonical_json_bytes(document) + b"\n"
    output = args.output.resolve()
    if args.check:
        if not output.is_file() or output.read_bytes() != encoded:
            print(f"[FAIL] adaptive OBB census differs: {output}")
            return 1
        print(
            "[census_a3_table_collision_proxy_adaptive] OK: "
            f"sha256={_sha256_bytes(encoded)}"
        )
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(encoded)
    print(
        "[census_a3_table_collision_proxy_adaptive] wrote "
        f"{output} sha256={_sha256_bytes(encoded)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
