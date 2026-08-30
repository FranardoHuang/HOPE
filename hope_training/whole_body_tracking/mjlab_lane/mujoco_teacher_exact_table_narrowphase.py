"""Source-convex narrow phase for the N=1 teacher-replay diagnostic.

The live FullMDP table guard deliberately uses one conservative OBB per source
collision component.  That is a sound broad phase, but an OBB corner is not a
claim that the source mesh occupies the corner.  This module is the rare-path
diagnostic which follows a broad-positive pair with the exact tracked source
STL vertex convex hull against the already 20 mm expanded table primitive.

It is intentionally host-only and N=1.  Production training does not import or
call it.  A malformed identity, pose, source mesh, transform, or geometric
result fails closed; backend ``contact == false`` is never accepted as proof of
clearance.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import numpy as np


SCHEMA = "action_ball_teacher_exact_table_narrowphase_v1"
ALGORITHM = (
    "broad_positive_component_table_pair__tracked_source_stl_triangle_vs_"
    "convex_hull_vs_20mm_expanded_axis_aligned_table_primitive_sat_v1"
)
AUTHORITY_KIND = (
    "tracked_source_urdf_collision_mesh_vertex_convex_hulls_in_compiled_"
    "owner_body_frames__"
    "not_backend_contact_and_not_compiled_actual_geom_distance"
)
TABLE_ROLES = ("top", "keepout", "net", "post_left", "post_right")
RACKET_BLADE_COMPONENT_INDEX = 62
RACKET_BLADE_COMPONENT_ID = "racket_blade"
_HEX = frozenset("0123456789abcdef")


class ExactTableNarrowphaseError(RuntimeError):
    """A diagnostic identity or geometric precondition is unknown."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")
    return _sha256(payload)


def _finite_array(value, shape, *, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ExactTableNarrowphaseError(f"{label} is not finite {shape}")
    return array


def _quat_matrix_wxyz(quaternion: np.ndarray) -> np.ndarray:
    q = _finite_array(quaternion, (4,), label="owner quaternion")
    norm = float(np.linalg.norm(q))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ExactTableNarrowphaseError("owner quaternion has zero norm")
    w, x, y, z = q / norm
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def _rpy_matrix(rpy: tuple[float, float, float]) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        ),
        dtype=np.float64,
    )


def _triplet(text: str | None, default=(0.0, 0.0, 0.0)) -> tuple[float, float, float]:
    values = tuple(float(value) for value in (text.split() if text else default))
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ExactTableNarrowphaseError("URDF transform is not one finite triplet")
    return values


def _origin(element: ET.Element | None) -> tuple[np.ndarray, np.ndarray]:
    if element is None:
        return np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
    return (
        _rpy_matrix(_triplet(element.get("rpy"))),
        np.asarray(_triplet(element.get("xyz")), dtype=np.float64),
    )


def _compose(
    parent_rotation: np.ndarray,
    parent_translation: np.ndarray,
    child_rotation: np.ndarray,
    child_translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        parent_rotation @ child_rotation,
        parent_rotation @ child_translation + parent_translation,
    )


def _stl_triangles(path: Path) -> np.ndarray:
    payload = path.read_bytes()
    vertices: list[tuple[float, float, float]] = []
    if len(payload) >= 84:
        count = struct.unpack_from("<I", payload, 80)[0]
        if 84 + count * 50 == len(payload):
            for triangle in range(count):
                base = 84 + triangle * 50 + 12
                for vertex in range(3):
                    vertices.append(
                        tuple(
                            float(value)
                            for value in struct.unpack_from(
                                "<fff", payload, base + vertex * 12
                            )
                        )
                    )
    if not vertices:
        try:
            text = payload.decode("ascii")
        except UnicodeDecodeError as exc:
            raise ExactTableNarrowphaseError("STL encoding is unknown") from exc
        for line in text.splitlines():
            fields = line.strip().split()
            if len(fields) == 4 and fields[0].lower() == "vertex":
                vertices.append(tuple(float(value) for value in fields[1:]))
    triangles = np.asarray(vertices, dtype=np.float64)
    if (
        triangles.ndim != 2
        or triangles.shape[1:] != (3,)
        or triangles.shape[0] == 0
        or triangles.shape[0] % 3 != 0
        or not np.isfinite(triangles).all()
    ):
        raise ExactTableNarrowphaseError("STL has no finite triangle soup")
    return triangles.reshape(-1, 3, 3)


def triangle_aabb_overlap(triangle, lo, hi) -> bool:
    """Exact closed-set triangle/AABB SAT, including edges and containment."""

    tri = _finite_array(triangle, (3, 3), label="triangle")
    lower = _finite_array(lo, (3,), label="table lower bound")
    upper = _finite_array(hi, (3,), label="table upper bound")
    if not np.all(upper > lower):
        raise ExactTableNarrowphaseError("table primitive is empty")
    center = 0.5 * (lower + upper)
    half = 0.5 * (upper - lower)
    local = tri - center
    edges = (local[1] - local[0], local[2] - local[1], local[0] - local[2])
    axes = [np.eye(3, dtype=np.float64)[axis] for axis in range(3)]
    axes.append(np.cross(edges[0], edges[1]))
    world = np.eye(3, dtype=np.float64)
    axes.extend(np.cross(edge, axis) for edge in edges for axis in world)
    for axis in axes:
        norm = float(np.linalg.norm(axis))
        if norm <= 1.0e-15:
            continue
        unit = axis / norm
        projection = local @ unit
        radius = float(np.dot(half, np.abs(unit)))
        if float(np.min(projection)) > radius or float(np.max(projection)) < -radius:
            return False
    return True


def _point_in_closed_triangle_mesh(point, triangles) -> bool:
    """Odd/even solid test; disagreement is unknown and therefore raises."""

    origin = _finite_array(point, (3,), label="solid-test point")
    value = np.asarray(triangles, dtype=np.float64)
    directions = np.asarray(
        ((1.0, 0.371, 0.127), (0.193, 1.0, 0.419), (0.257, 0.163, 1.0)),
        dtype=np.float64,
    )
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    votes = []
    for direction in directions:
        distances = []
        ambiguous = False
        for triangle in value:
            edge1 = triangle[1] - triangle[0]
            edge2 = triangle[2] - triangle[0]
            pvec = np.cross(direction, edge2)
            determinant = float(np.dot(edge1, pvec))
            if abs(determinant) <= 1.0e-14:
                continue
            inverse = 1.0 / determinant
            tvec = origin - triangle[0]
            u = float(np.dot(tvec, pvec)) * inverse
            qvec = np.cross(tvec, edge1)
            v = float(np.dot(direction, qvec)) * inverse
            distance = float(np.dot(edge2, qvec)) * inverse
            if distance <= 0.0 or u < 0.0 or v < 0.0 or u + v > 1.0:
                continue
            # A ray exactly through a shared edge/vertex has no stable parity.
            if min(u, v, 1.0 - u - v) <= 1.0e-12:
                ambiguous = True
                break
            distances.append(distance)
        if ambiguous:
            continue
        distances.sort()
        unique = []
        for distance in distances:
            if not unique or abs(distance - unique[-1]) > 1.0e-10:
                unique.append(distance)
        votes.append(bool(len(unique) % 2))
    if not votes or any(vote != votes[0] for vote in votes[1:]):
        raise ExactTableNarrowphaseError("source solid containment is ambiguous")
    return votes[0]


def triangles_aabb_overlap(triangles, lo, hi) -> tuple[bool, int | None]:
    """Closed source-mesh solid against AABB, including either containment."""

    value = np.asarray(triangles, dtype=np.float64)
    if value.ndim != 3 or value.shape[1:] != (3, 3) or not np.isfinite(value).all():
        raise ExactTableNarrowphaseError("triangle batch is malformed")
    for index, triangle in enumerate(value):
        if triangle_aabb_overlap(triangle, lo, hi):
            return True, index
    # If the table primitive lies wholly inside a closed source solid, no
    # surface triangle crosses the AABB.  Its center then proves containment.
    center = 0.5 * (
        _finite_array(lo, (3,), label="table lower bound")
        + _finite_array(hi, (3,), label="table upper bound")
    )
    if _point_in_closed_triangle_mesh(center, value):
        return True, None
    return False, None


def convex_hull_aabb_overlap(hull_vertices, hull_triangles, lo, hi) -> bool:
    """Complete closed-set SAT for one convex source hull and one AABB."""

    vertices = np.asarray(hull_vertices, dtype=np.float64)
    triangles = np.asarray(hull_triangles, dtype=np.float64)
    lower = _finite_array(lo, (3,), label="table lower bound")
    upper = _finite_array(hi, (3,), label="table upper bound")
    if (
        vertices.ndim != 2
        or vertices.shape[1:] != (3,)
        or vertices.shape[0] < 4
        or triangles.ndim != 3
        or triangles.shape[1:] != (3, 3)
        or triangles.shape[0] < 4
        or not np.isfinite(vertices).all()
        or not np.isfinite(triangles).all()
        or not np.all(upper > lower)
    ):
        raise ExactTableNarrowphaseError("source convex hull/AABB is malformed")
    center = 0.5 * (lower + upper)
    half = 0.5 * (upper - lower)
    world = np.eye(3, dtype=np.float64)
    axes = [world[index] for index in range(3)]
    for triangle in triangles:
        edges = (
            triangle[1] - triangle[0],
            triangle[2] - triangle[1],
            triangle[0] - triangle[2],
        )
        axes.append(np.cross(edges[0], edges[1]))
        axes.extend(np.cross(edge, basis) for edge in edges for basis in world)
    for axis in axes:
        norm = float(np.linalg.norm(axis))
        if norm <= 1.0e-15:
            continue
        unit = axis / norm
        projection = vertices @ unit
        box_center = float(np.dot(center, unit))
        box_radius = float(np.dot(half, np.abs(unit)))
        if (
            float(np.min(projection)) > box_center + box_radius
            or float(np.max(projection)) < box_center - box_radius
        ):
            return False
    return True


def obb_aabb_overlap(center, half_axes, lo, hi) -> bool:
    """Closed-set 15-axis OBB/AABB SAT matching the live broad phase."""

    obb_center = _finite_array(center, (3,), label="OBB center")
    axes = _finite_array(half_axes, (3, 3), label="OBB half axes")
    lower = _finite_array(lo, (3,), label="table lower bound")
    upper = _finite_array(hi, (3,), label="table upper bound")
    if not np.all(upper > lower) or np.any(np.linalg.norm(axes, axis=1) <= 0.0):
        raise ExactTableNarrowphaseError("OBB/table primitive is degenerate")
    box_center = 0.5 * (lower + upper)
    box_half = 0.5 * (upper - lower)
    delta = box_center - obb_center
    world = np.eye(3, dtype=np.float64)
    candidates = [world[index] for index in range(3)]
    unit_axes = axes / np.linalg.norm(axes, axis=1)[:, None]
    candidates.extend(unit_axes)
    candidates.extend(np.cross(axis, basis) for axis in unit_axes for basis in world)
    for axis in candidates:
        norm = float(np.linalg.norm(axis))
        if norm <= 1.0e-15:
            continue
        unit = axis / norm
        separation = abs(float(np.dot(delta, unit)))
        obb_radius = float(np.sum(np.abs(axes @ unit)))
        box_radius = float(np.dot(box_half, np.abs(unit)))
        if separation > obb_radius + box_radius:
            return False
    return True


@dataclass(frozen=True)
class _SourceComponent:
    component_id: str
    owner_body_name: str
    mesh_path: str
    mesh_sha256: str
    owner_vertices_m: np.ndarray
    owner_hull_triangles_m: np.ndarray


class SourceTriangleCatalog:
    """Lazy, identity-bound reconstruction of tracked URDF collision meshes."""

    def __init__(self, *, repo_root: Path, artifact_path: Path, source_urdf: Path):
        try:
            import scipy
            from scipy.spatial import ConvexHull
        except Exception as exc:
            raise ExactTableNarrowphaseError(
                "diagnostic source convex-hull dependency is unavailable"
            ) from exc
        self.scipy_version = str(scipy.__version__)
        self._convex_hull = ConvexHull
        self.repo_root = repo_root.resolve()
        self.artifact_path = artifact_path.resolve(strict=True)
        self.source_urdf = source_urdf.resolve(strict=True)
        raw = self.artifact_path.read_bytes()
        self.artifact_sha256 = _sha256(raw)
        try:
            document = json.loads(raw.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ExactTableNarrowphaseError("collision artifact is not strict JSON") from exc
        rows = document.get("components") if isinstance(document, dict) else None
        if not isinstance(rows, list) or len(rows) != 62:
            raise ExactTableNarrowphaseError("collision artifact component count differs")
        self.content_sha256 = str(document.get("content_sha256"))
        unsigned = dict(document)
        unsigned.pop("content_sha256", None)
        if _canonical_sha(unsigned) != self.content_sha256:
            raise ExactTableNarrowphaseError("collision artifact content seal differs")
        self.rows = tuple(rows)
        self._cache: dict[int, _SourceComponent] = {}
        root = ET.parse(self.source_urdf).getroot()
        self._links = {str(link.get("name")): link for link in root.findall("link")}
        self._parent = {}
        for joint in root.findall("joint"):
            parent = joint.find("parent")
            child = joint.find("child")
            if parent is None or child is None:
                raise ExactTableNarrowphaseError("URDF joint is incomplete")
            name = str(child.get("link"))
            if name in self._parent:
                raise ExactTableNarrowphaseError("URDF child has multiple parents")
            rotation, translation = _origin(joint.find("origin"))
            self._parent[name] = (
                str(parent.get("link")), str(joint.get("type")), rotation, translation
            )

    def _owner_transform(self, link_name: str) -> tuple[str, np.ndarray, np.ndarray]:
        owner = link_name
        rotation = np.eye(3, dtype=np.float64)
        translation = np.zeros(3, dtype=np.float64)
        seen = set()
        while owner in self._parent and self._parent[owner][1] == "fixed":
            if owner in seen:
                raise ExactTableNarrowphaseError("fixed-joint cycle")
            seen.add(owner)
            parent, _kind, joint_r, joint_t = self._parent[owner]
            rotation, translation = _compose(
                joint_r, joint_t, rotation, translation
            )
            owner = parent
        return owner, rotation, translation

    def load(self, index: int) -> _SourceComponent:
        if index in self._cache:
            return self._cache[index]
        if type(index) is not int or not 0 <= index < len(self.rows):
            raise ExactTableNarrowphaseError("component index is out of range")
        row = self.rows[index]
        if not isinstance(row, dict):
            raise ExactTableNarrowphaseError("component row is malformed")
        component_id = str(row.get("component_id"))
        owner_name = str(row.get("owner_body_name"))
        source_link = str(row.get("source_link_name"))
        try:
            declared_owner, declared_link, collision_text, relative_mesh = (
                component_id.split(":", 3)
            )
            collision_index = int(collision_text)
            link = self._links[source_link]
            collision = link.findall("collision")[collision_index]
            mesh = collision.find("geometry/mesh")
        except (ValueError, KeyError, IndexError) as exc:
            raise ExactTableNarrowphaseError(
                "component does not map to one URDF collision"
            ) from exc
        if (
            declared_owner != owner_name
            or declared_link != source_link
            or mesh is None
            or not str(mesh.get("filename")).endswith("/meshes/" + relative_mesh)
        ):
            raise ExactTableNarrowphaseError("component/URDF identity differs")
        path = (self.repo_root / str(row.get("mesh_path"))).resolve(strict=True)
        if (
            self.repo_root not in path.parents
            or _sha256(path.read_bytes()) != row.get("mesh_sha256")
        ):
            raise ExactTableNarrowphaseError("component mesh bytes differ")
        triangles = _stl_triangles(path)
        scale = np.asarray(_triplet(mesh.get("scale"), (1.0, 1.0, 1.0)), dtype=np.float64)
        if np.any(scale <= 0.0):
            raise ExactTableNarrowphaseError("component mesh scale is invalid")
        triangles = triangles * scale[None, None, :]
        link_r, link_t = _origin(collision.find("origin"))
        runtime_owner, owner_r, owner_t = self._owner_transform(source_link)
        if runtime_owner != owner_name:
            raise ExactTableNarrowphaseError("component runtime owner differs")
        owner_from_mesh_r, owner_from_mesh_t = _compose(
            owner_r, owner_t, link_r, link_t
        )
        owner_triangles = triangles @ owner_from_mesh_r.T + owner_from_mesh_t
        try:
            source_vertices = np.unique(owner_triangles.reshape(-1, 3), axis=0)
            hull = self._convex_hull(source_vertices)
        except Exception as exc:
            raise ExactTableNarrowphaseError(
                "source collision convex hull cannot be reconstructed"
            ) from exc
        owner_vertices = source_vertices[hull.vertices]
        owner_hull_triangles = source_vertices[hull.simplices]
        # Bind the reconstructed source soup back to the OBB artifact it refines.
        center = _finite_array(row.get("local_center_owner_m"), (3,), label="artifact OBB center")
        half_axes = _finite_array(
            row.get("local_half_axes_owner_m"),
            (3, 3),
            label="artifact OBB axes",
        )
        directions = half_axes / np.linalg.norm(half_axes, axis=1)[:, None]
        extents = np.linalg.norm(half_axes, axis=1)
        projection = (owner_vertices - center) @ directions.T
        if np.any(np.abs(projection) > extents[None] + 2.0e-6):
            raise ExactTableNarrowphaseError("source triangles escape their broad OBB")
        owner_vertices.setflags(write=False)
        owner_hull_triangles.setflags(write=False)
        result = _SourceComponent(
            component_id=component_id,
            owner_body_name=owner_name,
            mesh_path=str(row.get("mesh_path")),
            mesh_sha256=str(row.get("mesh_sha256")),
            owner_vertices_m=owner_vertices,
            owner_hull_triangles_m=owner_hull_triangles,
        )
        self._cache[index] = result
        return result


class TeacherExactTableNarrowphase:
    """Rare-path evaluator and compact receipt accumulator for one world."""

    def __init__(
        self,
        *,
        repo_root: Path,
        artifact_path: Path,
        source_urdf: Path,
        runtime_mjb_sha256: str,
        mujoco_version: str,
        nativeccd_enabled: bool,
        disableflags: int,
        component_ids,
        owner_body_names,
        component_owner_indices,
        component_local_centers,
        component_local_half_axes,
        table_lo,
        table_hi,
        blade_owner_index: int,
        blade_center_offset,
        blade_local_half_axes,
        plant_identity: dict,
        table_geometry_sha256: str,
    ) -> None:
        if (
            type(runtime_mjb_sha256) is not str
            or len(runtime_mjb_sha256) != 64
            or any(char not in _HEX for char in runtime_mjb_sha256)
        ):
            raise ExactTableNarrowphaseError("runtime MJB SHA-256 is malformed")
        self.catalog = SourceTriangleCatalog(
            repo_root=repo_root,
            artifact_path=artifact_path,
            source_urdf=source_urdf,
        )
        self.component_ids = tuple(str(value) for value in component_ids)
        if len(self.component_ids) != 62 or self.component_ids != tuple(
            str(row["component_id"]) for row in self.catalog.rows
        ):
            raise ExactTableNarrowphaseError("live/source component order differs")
        self.owner_body_names = tuple(str(value) for value in owner_body_names)
        if len(self.owner_body_names) != 32 or len(set(self.owner_body_names)) != 32:
            raise ExactTableNarrowphaseError("compiled owner-body order differs")
        self.owner = np.asarray(component_owner_indices, dtype=np.int64)
        self.centers = _finite_array(component_local_centers, (62, 3), label="component centers")
        self.axes = _finite_array(component_local_half_axes, (62, 3, 3), label="component axes")
        self.table_lo = _finite_array(table_lo, (5, 3), label="table lower bounds")
        self.table_hi = _finite_array(table_hi, (5, 3), label="table upper bounds")
        if self.owner.shape != (62,) or np.any(self.owner < 0) or np.any(self.owner >= 32):
            raise ExactTableNarrowphaseError("component owner map differs")
        artifact_centers = np.asarray(
            [row["local_center_owner_m"] for row in self.catalog.rows],
            dtype=np.float64,
        )
        artifact_axes = np.asarray(
            [row["local_half_axes_owner_m"] for row in self.catalog.rows],
            dtype=np.float64,
        )
        if not np.array_equal(self.centers, artifact_centers) or not np.array_equal(
            self.axes, artifact_axes
        ):
            raise ExactTableNarrowphaseError("live/source broad geometry differs")
        for component, row in enumerate(self.catalog.rows):
            if self.owner_body_names[int(self.owner[component])] != str(
                row["owner_body_name"]
            ):
                raise ExactTableNarrowphaseError(
                    "source component does not close to its compiled owner body"
                )
        self.blade_owner_index = int(blade_owner_index)
        if not 0 <= self.blade_owner_index < 32:
            raise ExactTableNarrowphaseError("blade owner differs")
        if self.owner_body_names[self.blade_owner_index] != "right_wrist_yaw_Link":
            raise ExactTableNarrowphaseError("blade compiled owner body differs")
        self.blade_center = _finite_array(blade_center_offset, (3,), label="blade center")
        self.blade_axes = _finite_array(blade_local_half_axes, (3, 3), label="blade axes")
        if not isinstance(plant_identity, dict) or not plant_identity:
            raise ExactTableNarrowphaseError("plant identity is absent")
        self.identity = {
            "schema": SCHEMA,
            "algorithm": ALGORITHM,
            "authority_kind": AUTHORITY_KIND,
            "backend_contact_used_as_clearance_truth": False,
            "compiled_actual_geom_distance_used": False,
            "runtime_mjb_sha256": runtime_mjb_sha256,
            "mujoco_version": str(mujoco_version),
            "scipy_version": self.catalog.scipy_version,
            "nativeccd_enabled": bool(nativeccd_enabled),
            "disableflags": int(disableflags),
            "collision_artifact_sha256": self.catalog.artifact_sha256,
            "collision_content_sha256": self.catalog.content_sha256,
            "source_urdf_sha256": _sha256(self.catalog.source_urdf.read_bytes()),
            "plant_identity": dict(plant_identity),
            "table_geometry_sha256": str(table_geometry_sha256),
            "expanded_table_aabbs_sha256": _canonical_sha(
                {
                    "lo": self.table_lo.tolist(),
                    "hi": self.table_hi.tolist(),
                    "inclusive_closed_set": True,
                }
            ),
            "table_roles": list(TABLE_ROLES),
            "table_expansion_m": 0.02,
            "intersection_semantics": "inclusive_closed_set_no_extra_tolerance",
            "owner_body_names": list(self.owner_body_names),
        }
        self.identity["identity_sha256"] = _canonical_sha(self.identity)
        self.evaluation_count = 0
        self.broad_positive_pair_count = 0
        self.exact_positive_pair_count = 0
        self.false_positive_pair_count = 0
        self.first_false_positive = None
        self.first_exact_positive = None
        self.current_exact_witness = None

    def evaluate(
        self,
        *,
        body_position_env_m,
        body_quaternion_wxyz,
        capture_boundary: str,
        physics_substep_index: int | None,
    ) -> dict:
        self.evaluation_count += 1
        self.current_exact_witness = None
        try:
            positions = _finite_array(body_position_env_m, (32, 3), label="body positions")
            quaternions = _finite_array(body_quaternion_wxyz, (32, 4), label="body quaternions")
            rotations = np.stack(tuple(_quat_matrix_wxyz(value) for value in quaternions))
            broad_pairs = []
            exact_pairs = []
            false_pairs = []
            for component in range(63):
                if component < 62:
                    owner = int(self.owner[component])
                    center_local = self.centers[component]
                    axes_local = self.axes[component]
                    component_id = self.component_ids[component]
                else:
                    owner = self.blade_owner_index
                    center_local = self.blade_center
                    axes_local = self.blade_axes
                    component_id = RACKET_BLADE_COMPONENT_ID
                rotation = rotations[owner]
                center = positions[owner] + rotation @ center_local
                axes = axes_local @ rotation.T
                for table_index, (lo, hi) in enumerate(zip(self.table_lo, self.table_hi)):
                    if not obb_aabb_overlap(center, axes, lo, hi):
                        continue
                    pair = {
                        "component_index": component,
                        "component_id": component_id,
                        "owner_body_index": owner,
                        "owner_body_name": (
                            self.owner_body_names[owner]
                            if component == RACKET_BLADE_COMPONENT_INDEX
                            else self.catalog.rows[component]["owner_body_name"]
                        ),
                        "table_index": table_index,
                        "table_role": TABLE_ROLES[table_index],
                    }
                    broad_pairs.append(pair)
                    if component == RACKET_BLADE_COMPONENT_INDEX:
                        hit, triangle_index = True, None
                        source = {
                            "kind": "exact_blade_obb_primitive",
                            "mesh_path": None,
                            "mesh_sha256": None,
                        }
                    else:
                        source_component = self.catalog.load(component)
                        vertices = (
                            source_component.owner_vertices_m @ rotation.T
                            + positions[owner]
                        )
                        hull_triangles = (
                            source_component.owner_hull_triangles_m @ rotation.T
                            + positions[owner]
                        )
                        hit = convex_hull_aabb_overlap(
                            vertices, hull_triangles, lo, hi
                        )
                        triangle_index = None
                        source = {
                            "kind": "tracked_source_stl_vertex_convex_hull",
                            "mesh_path": source_component.mesh_path,
                            "mesh_sha256": source_component.mesh_sha256,
                        }
                    evaluated = {
                        **pair,
                        **source,
                        "triangle_index": triangle_index,
                        "capture_boundary": str(capture_boundary),
                        "physics_substep_index": physics_substep_index,
                        "exact_overlap": bool(hit),
                    }
                    (exact_pairs if hit else false_pairs).append(evaluated)
            if not broad_pairs:
                raise ExactTableNarrowphaseError(
                    "GPU broad-positive has no CPU broad-positive pair"
                )
            self.broad_positive_pair_count += len(broad_pairs)
            self.exact_positive_pair_count += len(exact_pairs)
            self.false_positive_pair_count += len(false_pairs)
            if false_pairs and self.first_false_positive is None:
                self.first_false_positive = dict(false_pairs[0])
            if exact_pairs and self.first_exact_positive is None:
                self.first_exact_positive = dict(exact_pairs[0])
            exact_hit = bool(exact_pairs)
            if exact_hit:
                chosen = dict(exact_pairs[0])
                owner = int(chosen["owner_body_index"])
                self.current_exact_witness = {
                    "schema": "action_ball_keepout_first_witness_v1",
                    "selection": "first_exact_source_overlap_in_production_order",
                    "reason": "sat_overlap",
                    **{key: chosen[key] for key in (
                        "component_index", "component_id", "owner_body_index",
                        "owner_body_name", "table_index", "table_role",
                    )},
                    "component_kind": (
                        "blade" if chosen["component_index"] == 62 else "body_proxy"
                    ),
                    "sat_signed_margin_m": 0.0,
                    "root_position_env_m": positions[0].tolist(),
                    "root_quaternion_wxyz": quaternions[0].tolist(),
                    "owner_position_env_m": positions[owner].tolist(),
                    "owner_quaternion_wxyz": quaternions[owner].tolist(),
                    "plant_identity": dict(self.identity["plant_identity"]),
                    "collision_artifact_sha256": self.catalog.artifact_sha256,
                    "collision_content_sha256": self.catalog.content_sha256,
                    "exact_narrowphase": chosen,
                    "narrowphase_identity_sha256": self.identity["identity_sha256"],
                }
            return {
                "exact_hit": exact_hit,
                "fail_closed": False,
                "broad_pairs": broad_pairs,
                "exact_pairs": exact_pairs,
                "false_positive_pairs": false_pairs,
            }
        except Exception as exc:
            # The broad verdict is retained on every unknown input or identity.
            return {
                "exact_hit": True,
                "fail_closed": True,
                "failure_type": type(exc).__name__,
                "failure_message": str(exc),
                "broad_pairs": [],
                "exact_pairs": [],
                "false_positive_pairs": [],
            }

    def receipt(self) -> dict:
        return {
            **self.identity,
            "diagnostic_unauthorized": True,
            "training_authorized": False,
            "production_consumer_changed": False,
            "evaluation_count": self.evaluation_count,
            "broad_positive_pair_count": self.broad_positive_pair_count,
            "exact_positive_pair_count": self.exact_positive_pair_count,
            "false_positive_pair_count": self.false_positive_pair_count,
            "first_false_positive": self.first_false_positive,
            "first_exact_positive": self.first_exact_positive,
        }


__all__ = (
    "ALGORITHM",
    "ExactTableNarrowphaseError",
    "SCHEMA",
    "SourceTriangleCatalog",
    "TeacherExactTableNarrowphase",
    "convex_hull_aabb_overlap",
    "obb_aabb_overlap",
    "triangle_aabb_overlap",
    "triangles_aabb_overlap",
)
