#!/usr/bin/env python3
"""Derive a no-clobber A3P-0807 USD with named wrist collider shapes.

The reviewed 0807 URDF contains five distinct fixed collision components below
``right_wrist_yaw_Link``.  Isaac Lab's production conversion merges them into
one PhysX collision shape, so a contact header cannot identify the selected
rubber independently.

This offline producer keeps the reviewed USD bundle as the articulation/base
reference, but builds the wrist, red rubber, black rubber, and handle collider
meshes directly from the reviewed URDF and STL bytes.  The published directory
also contains those exact sources, so ``--check`` can independently rederive
the converter cache identity, fixed transforms, mesh topology, and complete
overlay instead of believing a producer-written receipt.  SHA-256 values here
are asset-integrity pins only; live PhysX evidence remains a separate consumer
boundary and this asset never authorizes training or deployment.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile
from typing import Mapping
import xml.etree.ElementTree as ET

import yaml


SCHEMA = "action_ball_a3p0807_split_rubber_usd_v2"
ROOT_PRIM = "A3P_P1_0807_OP3_pingpang_31action_normalized_v1"
WRIST_BODY = "right_wrist_yaw_Link"

BUNDLE_PINS: Mapping[str, str] = {
    ".asset_hash": "a78a2f8fb207cbf479cc1b308cf9d3c58e1a55eb7da9dbc2caf34be697e9c993",
    "config.yaml": "f349c3f4d80a915f5ca3ce53d49785dfd7e6eeca2645dcd7b402d4d8a2288eb9",
    "configuration/model_base.usd": "108a4b45b96a8db8396d3a8feb995481c5db87efcde80066e6347ed494e658fc",
    "configuration/model_physics.usd": "390cf66cc052ea697e88e9ef0131bf7e2eee96e70c35c0861e1ce33d363747f5",
    "configuration/model_sensor.usd": "4e16201f146db3240b8a0082ae14e3aca41255a75812c5331bf8f4e39701355c",
    "model.usd": "13e5ecfe02238fbf1d20c13ed7177e18ed93d84bca8e0a592b6605f7fb85f351",
}

URDF_PIN = "15c83f5f3beea71350583143aef4d622d5219df65a0bed9a660a0edb7d388d09"
ISAACLAB_ASSET_HASH = "676efde5febed3c0fde0f2ad59650cdf"
# Updated only after deterministic source rederivation and independent tests.
DERIVED_MODEL_SHA256 = "a3cd382943ff9f70beecf88c729a6cc1c052a3c0a0cbffe91003ec319ab78140"


@dataclass(frozen=True)
class ComponentPin:
    source_name: str
    output_name: str
    mesh_sha256: str


REFERENCED_COMPONENTS = (
    ComponentPin(
        "right_wrist_yaw_link",
        "wrist_shell_collider",
        "1b96ef7f1618fd7565c7e0f9beeef2e56efe30d078837f89aae68dbb94c24bf6",
    ),
    ComponentPin(
        "pingpang_black_link",
        "black_rubber_collider",
        "5f0e772ea9ed81e5b70f5dfb4ded49f9d269c54c893249857209f85168361b1b",
    ),
    ComponentPin(
        "pingpang_red_link",
        "red_rubber_collider",
        "94182ec1c7c64db8c5ec7ce5f9aad44d427f433a6aae5cf23aa655e077633842",
    ),
)

HAND_SOURCE = ComponentPin(
    "right_hand_pingpang_link",
    "racket_handle_collider",
    "442ff2ecb82d3da481f1500d8a788192ba7d8bc2969f4d8c9d98266ea116b4dd",
)
MARKER_SOURCE = ComponentPin(
    "pingbang_ball_link",
    "excluded_nonphysical_racket_center_marker",
    "21c39c9f6112304776f4eadf7439193163a814b59391790df027ff5aa8249c93",
)
SOURCE_MESHES = REFERENCED_COMPONENTS + (HAND_SOURCE, MARKER_SOURCE)

URDF_LINK_BY_SOURCE = {
    "right_wrist_yaw_link": "right_wrist_yaw_Link",
    "right_hand_pingpang_link": "right_hand_pingpang_Link",
    "pingpang_red_link": "pingpang_red_Link",
    "pingpang_black_link": "pingpang_black_Link",
    "pingbang_ball_link": "pingbang_ball_Link",
}
EXPECTED_FIXED_MOUNTS = {
    "right_hand_pingpang_Link": (
        "right_hand_pingpang_joint",
        "right_wrist_yaw_Link",
    ),
    "pingpang_red_Link": ("pingpang_red_joint", "right_hand_pingpang_Link"),
    "pingpang_black_Link": ("pingpang_black_joint", "right_hand_pingpang_Link"),
    "pingbang_ball_Link": ("pingbang_ball_joint", "right_hand_pingpang_Link"),
}


class DerivationError(RuntimeError):
    """The requested output is not the exact reviewed diagnostic derivation."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, label: str) -> Path:
    try:
        path.lstat()
    except FileNotFoundError as exc:
        raise DerivationError(f"missing {label}: {path}") from exc
    if not path.is_file() or path.is_symlink():
        raise DerivationError(f"{label} must be one regular non-symlink file: {path}")
    return path


def _real_directory(path: Path, label: str) -> Path:
    try:
        path.lstat()
    except FileNotFoundError as exc:
        raise DerivationError(f"missing {label}: {path}") from exc
    if not path.is_dir() or path.is_symlink():
        raise DerivationError(f"{label} must be one real non-symlink directory: {path}")
    return path


def _reject_symlink_ancestors(path: Path, label: str) -> None:
    """Reject indirection in every existing component of an absolute path."""

    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if not current.exists() and not current.is_symlink():
            break
        if current.is_symlink():
            raise DerivationError(f"{label} has a symlink path component: {current}")


def _verify_exact_file(path: Path, expected: str, label: str) -> dict[str, object]:
    path = _regular_file(path, label)
    actual = _sha256(path)
    if actual != expected:
        raise DerivationError(
            f"{label} SHA-256 differs: expected {expected}, observed {actual}"
        )
    return {"bytes": path.stat().st_size, "sha256": actual}


def _inventory(root: Path) -> list[str]:
    return sorted(
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    )


def _contains_symlink_entry(
    root: Path, *, ignored_relative: frozenset[str] = frozenset()
) -> bool:
    """Walk without following links and detect any symlink, including dirs."""

    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in directory_names + file_names:
            path = base / name
            relative = str(path.relative_to(root))
            if path.is_symlink() and relative not in ignored_relative:
                return True
    return False


def _bundle_inventory(
    bundle: Path, pins: Mapping[str, str]
) -> dict[str, dict[str, object]]:
    _real_directory(bundle, "source bundle")
    actual = _inventory(bundle)
    expected = sorted(pins)
    if actual != expected:
        raise DerivationError(
            f"source bundle inventory differs: expected {expected}, observed {actual}"
        )
    return {
        relative: _verify_exact_file(
            bundle / relative, pins[relative], f"source bundle {relative}"
        )
        for relative in expected
    }


def _component_mesh_path(mesh_root: Path, component: ComponentPin) -> Path:
    return mesh_root / f"{component.source_name}.stl"


def _binary_stl_triangles(path: Path) -> tuple[tuple[tuple[float, float, float], ...], ...]:
    data = _regular_file(path, "source STL").read_bytes()
    if len(data) < 84:
        raise DerivationError(f"source STL is shorter than a binary header: {path}")
    triangle_count = struct.unpack_from("<I", data, 80)[0]
    if len(data) != 84 + triangle_count * 50:
        raise DerivationError(f"source STL is not exact binary triangle encoding: {path}")
    triangles = []
    for index in range(triangle_count):
        values = struct.unpack_from("<12fH", data, 84 + index * 50)
        triangles.append(
            tuple(
                tuple(float(value) for value in values[offset : offset + 3])
                for offset in (3, 6, 9)
            )
        )
    return tuple(triangles)


def _connected_triangle_components(
    triangles: tuple[tuple[tuple[float, float, float], ...], ...]
) -> tuple[tuple[int, ...], ...]:
    parent = list(range(len(triangles)))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    by_vertex: dict[tuple[float, float, float], list[int]] = {}
    for triangle_index, triangle in enumerate(triangles):
        for vertex in triangle:
            by_vertex.setdefault(vertex, []).append(triangle_index)
    for members in by_vertex.values():
        for other in members[1:]:
            union(members[0], other)
    groups: dict[int, list[int]] = {}
    for index in range(len(triangles)):
        groups.setdefault(find(index), []).append(index)
    return tuple(tuple(value) for value in groups.values())


def _vector(origin: ET.Element, attribute: str, label: str) -> tuple[float, float, float]:
    try:
        values = tuple(float(value) for value in origin.get(attribute, "").split())
    except ValueError as exc:
        raise DerivationError(f"{label} {attribute} is malformed") from exc
    if len(values) != 3:
        raise DerivationError(f"{label} {attribute} is not one 3-vector")
    return values


def _is_zero(vector: tuple[float, float, float]) -> bool:
    return vector == (0.0, 0.0, 0.0)


def _urdf_source_facts(urdf: Path) -> dict[str, object]:
    try:
        root = ET.parse(_regular_file(urdf, "reviewed source URDF")).getroot()
    except ET.ParseError as exc:
        raise DerivationError("reviewed source URDF is malformed") from exc
    links = root.findall("link")
    joints = root.findall("joint")
    link_by_name = {link.get("name"): link for link in links}
    if len(link_by_name) != len(links):
        raise DerivationError("reviewed source URDF has duplicate link names")
    joint_by_child: dict[str, ET.Element] = {}
    for joint in joints:
        child = joint.find("child")
        child_name = None if child is None else child.get("link")
        if not child_name or child_name in joint_by_child:
            raise DerivationError("reviewed source URDF joint tree is ambiguous")
        joint_by_child[child_name] = joint

    nonfixed = sum(joint.get("type") != "fixed" for joint in joints)
    if len(links) != 63 or len(joints) != 62 or nonfixed != 31:
        raise DerivationError(
            "reviewed source articulation topology differs from 63 links / 62 joints / 31 movable joints"
        )

    translations: dict[str, tuple[float, float, float]] = {}
    collision_meshes: dict[str, str] = {}
    for component in SOURCE_MESHES:
        link_name = URDF_LINK_BY_SOURCE[component.source_name]
        link = link_by_name.get(link_name)
        if link is None:
            raise DerivationError(f"reviewed source URDF lacks link {link_name}")
        collisions = link.findall("collision")
        if len(collisions) != 1:
            raise DerivationError(f"{link_name} must have one exact collision")
        collision = collisions[0]
        origin = collision.find("origin")
        mesh = collision.find("geometry/mesh")
        if origin is None or mesh is None:
            raise DerivationError(f"{link_name} collision lacks origin or mesh")
        if not _is_zero(_vector(origin, "rpy", f"{link_name} collision")):
            raise DerivationError(f"{link_name} collision rotation differs")
        translation = _vector(origin, "xyz", f"{link_name} collision")
        expected_filename = f"../meshes/{component.source_name}.stl"
        if mesh.get("filename") != expected_filename:
            raise DerivationError(f"{link_name} collision mesh reference differs")
        collision_meshes[component.source_name] = expected_filename

        current = link_name
        visited: set[str] = set()
        while current != WRIST_BODY:
            if current in visited:
                raise DerivationError("reviewed source fixed mount chain contains a cycle")
            visited.add(current)
            joint = joint_by_child.get(current)
            expected_mount = EXPECTED_FIXED_MOUNTS.get(current)
            if joint is None or expected_mount is None:
                raise DerivationError(f"{link_name} is not fixed below {WRIST_BODY}")
            parent = joint.find("parent")
            joint_origin = joint.find("origin")
            if (
                joint.get("name") != expected_mount[0]
                or joint.get("type") != "fixed"
                or parent is None
                or parent.get("link") != expected_mount[1]
                or joint_origin is None
                or not _is_zero(_vector(joint_origin, "rpy", joint.get("name", "joint")))
            ):
                raise DerivationError(f"{current} fixed mount differs")
            delta = _vector(joint_origin, "xyz", joint.get("name", "joint"))
            translation = tuple(translation[axis] + delta[axis] for axis in range(3))
            current = expected_mount[1]
        translations[component.source_name] = translation

    if translations["pingpang_red_link"] != translations["pingpang_black_link"]:
        raise DerivationError("red and black rubber mounts differ")
    return {
        "link_count": len(links),
        "source_joint_count": len(joints),
        "joint_count": nonfixed,
        "rigid_body_count": nonfixed + 1,
        "translations": translations,
        "collision_meshes": collision_meshes,
    }


def _bounds(
    triangles: tuple[tuple[tuple[float, float, float], ...], ...],
    indices: tuple[int, ...] | None = None,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    selected = range(len(triangles)) if indices is None else indices
    vertices = [vertex for index in selected for vertex in triangles[index]]
    return (
        tuple(min(vertex[axis] for vertex in vertices) for axis in range(3)),
        tuple(max(vertex[axis] for vertex in vertices) for axis in range(3)),
    )


def _handle_only_triangles(
    *, hand_mesh: Path, racket_site: tuple[float, float, float]
) -> tuple[tuple[tuple[tuple[float, float, float], ...], ...], dict[str, object]]:
    triangles = _binary_stl_triangles(hand_mesh)
    components = _connected_triangle_components(triangles)
    containing = []
    component_bounds = []
    for index, members in enumerate(components):
        lower, upper = _bounds(triangles, members)
        component_bounds.append((lower, upper))
        if all(lower[axis] <= racket_site[axis] <= upper[axis] for axis in range(3)):
            containing.append(index)
    if len(containing) != 1:
        raise DerivationError(
            "right-hand STL must have exactly one connected component containing the URDF racket site"
        )
    blade_index = containing[0]
    blade_members = components[blade_index]
    handle_indices = tuple(
        index
        for component_index, members in enumerate(components)
        if component_index != blade_index
        for index in members
    )
    if len(components) != 6 or len(blade_members) != 604 or len(handle_indices) != 9654:
        raise DerivationError(
            "right-hand STL connected-component topology differs from reviewed source"
        )
    handle = tuple(triangles[index] for index in handle_indices)
    lower, upper = _bounds(triangles, handle_indices)
    blade_lower, blade_upper = component_bounds[blade_index]
    return handle, {
        "source_triangle_count": len(triangles),
        "source_connected_component_count": len(components),
        "excluded_blade_component_triangle_count": len(blade_members),
        "excluded_blade_component_bounds_m": [list(blade_lower), list(blade_upper)],
        "handle_triangle_count": len(handle),
        "handle_bounds_m": [list(lower), list(upper)],
        "classification": "unique_source_component_aabb_containing_urdf_red_rubber_mount",
    }


def _number(value: float) -> str:
    return format(value, ".9g")


def _mesh_lines(
    *,
    output_name: str,
    triangles: tuple[tuple[tuple[float, float, float], ...], ...],
    translation: tuple[float, float, float],
) -> list[str]:
    vertices = [vertex for triangle in triangles for vertex in triangle]
    lower, upper = _bounds(triangles)
    point_text = ", ".join(
        f"({_number(vertex[0])}, {_number(vertex[1])}, {_number(vertex[2])})"
        for vertex in vertices
    )
    count_text = ", ".join("3" for _ in triangles)
    index_text = ", ".join(str(index) for index in range(len(vertices)))
    translate = ", ".join(_number(value) for value in translation)
    return [
        f'            def Mesh "{output_name}" (',
        '                prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]',
        "            )",
        "            {",
        "                bool physics:collisionEnabled = 1",
        '                uniform token physics:approximation = "convexHull"',
        f"                double3 xformOp:translate = ({translate})",
        '                uniform token[] xformOpOrder = ["xformOp:translate"]',
        "                float3[] extent = "
        f"[({_number(lower[0])}, {_number(lower[1])}, {_number(lower[2])}), "
        f"({_number(upper[0])}, {_number(upper[1])}, {_number(upper[2])})]",
        f"                int[] faceVertexCounts = [{count_text}]",
        f"                int[] faceVertexIndices = [{index_text}]",
        f"                point3f[] points = [{point_text}]",
        "            }",
        "",
    ]


def _source_geometry(
    *, source_urdf: Path, source_mesh_root: Path
) -> tuple[
    dict[str, object],
    dict[str, tuple[tuple[tuple[float, float, float], ...], ...]],
    dict[str, object],
]:
    urdf_facts = _urdf_source_facts(source_urdf)
    translations = urdf_facts["translations"]
    assert isinstance(translations, dict)
    raw = {
        component.source_name: _binary_stl_triangles(
            _component_mesh_path(source_mesh_root, component)
        )
        for component in SOURCE_MESHES
    }
    racket_site = translations["pingpang_red_link"]
    handle, handle_evidence = _handle_only_triangles(
        hand_mesh=_component_mesh_path(source_mesh_root, HAND_SOURCE),
        racket_site=racket_site,
    )
    output_meshes = {
        "wrist_shell_collider": raw["right_wrist_yaw_link"],
        "black_rubber_collider": raw["pingpang_black_link"],
        "red_rubber_collider": raw["pingpang_red_link"],
        "racket_handle_collider": handle,
    }
    evidence = {
        "source_triangle_counts": {name: len(value) for name, value in raw.items()},
        "output_triangle_counts": {
            name: len(value) for name, value in output_meshes.items()
        },
        "derived_handle": handle_evidence,
    }
    return urdf_facts, output_meshes, evidence


def _build_model_usda(
    *,
    output_meshes: Mapping[
        str, tuple[tuple[tuple[float, float, float], ...], ...]
    ],
    translations: Mapping[str, tuple[float, float, float]],
) -> str:
    lines = [
        "#usda 1.0",
        "(",
        f'    defaultPrim = "{ROOT_PRIM}"',
        '    metersPerUnit = 1',
        '    upAxis = "Z"',
        ")",
        "",
        f'def Xform "{ROOT_PRIM}" (',
        "    prepend references = @source_bundle/model.usd@",
        ")",
        "{",
        f'    over "{WRIST_BODY}"',
        "    {",
        '        over "collisions"',
        "        {",
        "            bool physics:collisionEnabled = 0",
        "        }",
        "",
        '        def Xform "action_ball_named_colliders"',
        "        {",
    ]
    sources_by_output = {
        "wrist_shell_collider": "right_wrist_yaw_link",
        "black_rubber_collider": "pingpang_black_link",
        "red_rubber_collider": "pingpang_red_link",
        "racket_handle_collider": "right_hand_pingpang_link",
    }
    for output_name in (
        "wrist_shell_collider",
        "black_rubber_collider",
        "red_rubber_collider",
        "racket_handle_collider",
    ):
        lines.extend(
            _mesh_lines(
                output_name=output_name,
                triangles=output_meshes[output_name],
                translation=translations[sources_by_output[output_name]],
            )
        )
    lines.extend(["        }", "    }", "}", ""])
    return "\n".join(lines)


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _rederive_isaaclab_asset_hash(config_path: Path, urdf_path: Path) -> str:
    try:
        loaded = yaml.safe_load(_regular_file(config_path, "converter config").read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise DerivationError("converter config is unreadable") from exc
    if not isinstance(loaded, dict):
        raise DerivationError("converter config is not one mapping")
    payload = dict(loaded)
    for key in ("asset_path", "usd_dir", "usd_file_name"):
        payload.pop(key, None)
    digest = hashlib.md5()
    digest.update(json.dumps(payload).encode())
    with _regular_file(urdf_path, "reviewed source URDF").open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_bundle_derivation(
    bundle: Path, urdf: Path, *, expected_asset_hash: str = ISAACLAB_ASSET_HASH
) -> dict[str, str]:
    stored = _regular_file(bundle / ".asset_hash", "converter asset hash").read_text().strip()
    rederived = _rederive_isaaclab_asset_hash(bundle / "config.yaml", urdf)
    if stored != rederived:
        raise DerivationError(
            "source bundle is not a converter cache of the enclosed reviewed URDF"
        )
    if rederived != expected_asset_hash:
        raise DerivationError("source bundle converter derivation differs from reviewed identity")
    return {"stored": stored, "rederived_from_enclosed_urdf": rederived}


def _receipt(
    *,
    bundle_receipt: dict[str, dict[str, object]],
    urdf_receipt: dict[str, object],
    mesh_receipts: dict[str, dict[str, object]],
    bundle_derivation: dict[str, str],
    urdf_facts: dict[str, object],
    geometry_evidence: dict[str, object],
    model_path: Path,
) -> dict[str, object]:
    translations = urdf_facts["translations"]
    assert isinstance(translations, dict)
    return {
        "schema": SCHEMA,
        "diagnostic_unauthorized": True,
        "launch_authorized": False,
        "source_bundle": bundle_receipt,
        "source_urdf": urdf_receipt,
        "source_meshes": mesh_receipts,
        "bundle_derivation": bundle_derivation,
        "articulation_invariants": {
            "rigid_actor": WRIST_BODY,
            "joint_count": urdf_facts["joint_count"],
            "rigid_body_count": urdf_facts["rigid_body_count"],
            "source_link_count": urdf_facts["link_count"],
            "source_joint_count": urdf_facts["source_joint_count"],
            "mass_and_inertia_source": "unchanged_source_bundle",
            "joint_topology_source": "unchanged_source_bundle",
        },
        "collision_change": {
            "disabled": f"{WRIST_BODY}/collisions",
            "replacement_parent": f"{WRIST_BODY}/action_ball_named_colliders",
            "components": [
                {
                    "source_name": component.source_name,
                    "output_name": component.output_name,
                    "approximation": "convexHull",
                    "translation_in_wrist_m": list(translations[component.source_name]),
                }
                for component in REFERENCED_COMPONENTS + (HAND_SOURCE,)
            ],
            "geometry_evidence": geometry_evidence,
            "excluded_nonphysical_marker": MARKER_SOURCE.source_name,
            "excluded_blade_core_from_handle": True,
        },
        "model_usd": {"bytes": model_path.stat().st_size, "sha256": _sha256(model_path)},
        "required_live_evidence": [
            "selected_rubber_CONTACT_FOUND_header",
            "opposite_rubber_CONTACT_FOUND_header",
            "handle_or_wrist_CONTACT_FOUND_header",
            "no_contact_negative_control",
            "teardown_and_stale_subscription_epoch",
        ],
    }


def _expected_output_inventory(bundle_pins: Mapping[str, str]) -> list[str]:
    return sorted(
        [
            "model.usd",
            "source/urdf/model.urdf",
            *(f"source/meshes/{component.source_name}.stl" for component in SOURCE_MESHES),
            *(f"source_bundle/{relative}" for relative in bundle_pins),
        ]
    )


def _verify_model_semantics(
    model_text: str,
    *,
    geometry_evidence: Mapping[str, object],
    translations: Mapping[str, tuple[float, float, float]],
) -> None:
    if model_text.count("physics:collisionEnabled = 0") != 1:
        raise DerivationError("derived model does not disable exactly one merged collider")
    if model_text.count('prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]') != 4:
        raise DerivationError("derived model does not contain exactly four named collision meshes")
    for component in REFERENCED_COMPONENTS + (HAND_SOURCE,):
        if model_text.count(f'def Mesh "{component.output_name}"') != 1:
            raise DerivationError(f"derived model lacks exact collider {component.output_name}")
        translation = ", ".join(
            _number(value) for value in translations[component.source_name]
        )
        if model_text.count(f"double3 xformOp:translate = ({translation})") < 1:
            raise DerivationError(f"derived model transform differs for {component.output_name}")
    if "PhysicsRigidBodyAPI" in model_text or "PhysicsJoint" in model_text:
        raise DerivationError("derived model illegally changes actor or joint topology")
    if "configuration/model_base.usd" in model_text or "/colliders/" in model_text:
        raise DerivationError("derived colliders still reference converted source geometry")
    counts = geometry_evidence.get("output_triangle_counts")
    expected_counts = {
        "wrist_shell_collider": 2806,
        "black_rubber_collider": 178,
        "red_rubber_collider": 178,
        "racket_handle_collider": 9654,
    }
    if counts != expected_counts:
        raise DerivationError("derived model geometry evidence differs")


def _validate_output(
    output_root: Path,
    *,
    bundle_pins: Mapping[str, str],
    urdf_pin: str,
    components: tuple[ComponentPin, ...],
    expected_asset_hash: str = ISAACLAB_ASSET_HASH,
) -> dict[str, object]:
    _reject_symlink_ancestors(output_root, "derived output root")
    _real_directory(output_root, "derived output root")
    # Reject top-level indirection before recursive inventory can merely report
    # it as an unexplained extra/partial entry.
    bundle = _real_directory(output_root / "source_bundle", "enclosed source bundle")
    source = _real_directory(output_root / "source", "enclosed source root")
    observed_inventory = _inventory(output_root)
    core_inventory = _expected_output_inventory(bundle_pins)
    permitted_inventories = (
        core_inventory,
        sorted(core_inventory + ["DERIVATION_RECEIPT.json"]),
    )
    if _contains_symlink_entry(
        output_root, ignored_relative=frozenset({"DERIVATION_RECEIPT.json"})
    ) or observed_inventory not in permitted_inventories:
        raise DerivationError("derived output inventory differs or is partial")
    mesh_root = _real_directory(source / "meshes", "enclosed source mesh root")
    urdf_root = _real_directory(source / "urdf", "enclosed source URDF root")
    urdf = urdf_root / "model.urdf"
    bundle_receipt = _bundle_inventory(bundle, bundle_pins)
    urdf_receipt = _verify_exact_file(urdf, urdf_pin, "enclosed reviewed source URDF")
    if tuple((value.source_name, value.output_name) for value in components) != tuple(
        (value.source_name, value.output_name) for value in SOURCE_MESHES
    ):
        raise DerivationError("component set/order must remain reviewed five-component tuple")
    mesh_receipts = {
        component.source_name: _verify_exact_file(
            _component_mesh_path(mesh_root, component),
            component.mesh_sha256,
            f"enclosed source mesh {component.source_name}",
        )
        for component in components
    }
    bundle_derivation = _verify_bundle_derivation(
        bundle, urdf, expected_asset_hash=expected_asset_hash
    )
    urdf_facts, output_meshes, geometry_evidence = _source_geometry(
        source_urdf=urdf, source_mesh_root=mesh_root
    )
    translations = urdf_facts["translations"]
    assert isinstance(translations, dict)
    canonical = _build_model_usda(
        output_meshes=output_meshes, translations=translations
    ).encode()
    model = _regular_file(output_root / "model.usd", "derived model.usd")
    if model.read_bytes() != canonical:
        raise DerivationError("derived model.usd differs from enclosed source rederivation")
    if _sha256(model) != DERIVED_MODEL_SHA256:
        raise DerivationError("derived model.usd differs from reviewed output integrity pin")
    _verify_model_semantics(
        canonical.decode(),
        geometry_evidence=geometry_evidence,
        translations=translations,
    )
    recomputed_receipt = _receipt(
        bundle_receipt=bundle_receipt,
        urdf_receipt=urdf_receipt,
        mesh_receipts=mesh_receipts,
        bundle_derivation=bundle_derivation,
        urdf_facts=urdf_facts,
        geometry_evidence=geometry_evidence,
        model_path=model,
    )
    receipt_path = output_root / "DERIVATION_RECEIPT.json"
    receipt_present = receipt_path.exists() or receipt_path.is_symlink()
    receipt_parseable = False
    receipt_matches = False
    if receipt_present and receipt_path.is_file() and not receipt_path.is_symlink():
        try:
            observed_receipt = json.loads(receipt_path.read_text())
        except (OSError, json.JSONDecodeError):
            observed_receipt = None
        receipt_parseable = isinstance(observed_receipt, dict)
        receipt_matches = observed_receipt == recomputed_receipt

    # Only independently recomputed facts are returned as the verification
    # result.  The producer receipt is optional telemetry and cannot authorize,
    # block, or modify this verdict.
    return {
        "schema": SCHEMA,
        "actual_sources_verified": True,
        "source_bundle": bundle_receipt,
        "source_urdf": urdf_receipt,
        "source_meshes": mesh_receipts,
        "bundle_derivation": bundle_derivation,
        "articulation_facts": {
            "joint_count": urdf_facts["joint_count"],
            "rigid_body_count": urdf_facts["rigid_body_count"],
            "source_link_count": urdf_facts["link_count"],
            "source_joint_count": urdf_facts["source_joint_count"],
        },
        "geometry_evidence": geometry_evidence,
        "model_usd": {"bytes": model.stat().st_size, "sha256": _sha256(model)},
        "live_stage_evidence": "not_evaluated_by_offline_asset_check",
        "receipt_audit": {
            "present": receipt_present,
            "parseable": receipt_parseable,
            "matches_recomputed": receipt_matches,
        },
    }


def _rename_no_replace(source: Path, destination: Path) -> None:
    """Atomically publish one directory without an existence-check race."""

    if source.parent.stat().st_dev != destination.parent.stat().st_dev:
        raise DerivationError("staging and output parent are not on one filesystem")

    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        function = libc.renameat2
        function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        function.restype = ctypes.c_int
        result = function(-100, source_bytes, -100, destination_bytes, 1)
    elif sys.platform == "darwin" and hasattr(libc, "renamex_np"):
        function = libc.renamex_np
        function.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        function.restype = ctypes.c_int
        result = function(source_bytes, destination_bytes, 0x00000004)
    else:
        raise DerivationError("platform lacks atomic no-replace directory publication")
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in (errno.EEXIST, errno.ENOTEMPTY):
        raise DerivationError(f"output already exists; no-clobber refused: {destination}")
    raise DerivationError(
        f"atomic no-clobber publication failed: {os.strerror(error)}"
    )


def _materialize_with_policy(
    *,
    source_bundle: Path,
    source_urdf: Path,
    source_mesh_root: Path,
    output_root: Path,
    bundle_pins: Mapping[str, str],
    urdf_pin: str,
    components: tuple[ComponentPin, ...],
    expected_asset_hash: str,
) -> dict[str, object]:
    """Build privately, validate completely, then atomically publish once."""

    _reject_symlink_ancestors(source_bundle, "source bundle")
    _reject_symlink_ancestors(source_urdf, "reviewed source URDF")
    _reject_symlink_ancestors(source_mesh_root, "reviewed source mesh root")
    _reject_symlink_ancestors(output_root.parent, "output parent")
    source_bundle = _real_directory(source_bundle.absolute(), "source bundle")
    source_urdf = _regular_file(source_urdf.absolute(), "reviewed source URDF")
    source_mesh_root = _real_directory(
        source_mesh_root.absolute(), "reviewed source mesh root"
    )
    output_root = output_root.absolute()
    parent = _real_directory(output_root.parent, "output parent")
    if output_root.exists() or output_root.is_symlink():
        raise DerivationError(f"output already exists; no-clobber refused: {output_root}")

    # Verify before copying, then verify only the enclosed copy during staging.
    _bundle_inventory(source_bundle, bundle_pins)
    _verify_exact_file(source_urdf, urdf_pin, "reviewed source URDF")
    for component in components:
        _verify_exact_file(
            _component_mesh_path(source_mesh_root, component),
            component.mesh_sha256,
            f"source mesh {component.source_name}",
        )
    _verify_bundle_derivation(
        source_bundle, source_urdf, expected_asset_hash=expected_asset_hash
    )

    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.staging-", dir=parent))
    published = False
    try:
        shutil.copytree(source_bundle, staging / "source_bundle", symlinks=False)
        (staging / "source" / "urdf").mkdir(parents=True)
        (staging / "source" / "meshes").mkdir()
        shutil.copy2(source_urdf, staging / "source" / "urdf" / "model.urdf")
        for component in SOURCE_MESHES:
            shutil.copy2(
                _component_mesh_path(source_mesh_root, component),
                staging / "source" / "meshes" / f"{component.source_name}.stl",
            )
        urdf_facts, output_meshes, _ = _source_geometry(
            source_urdf=staging / "source" / "urdf" / "model.urdf",
            source_mesh_root=staging / "source" / "meshes",
        )
        translations = urdf_facts["translations"]
        assert isinstance(translations, dict)
        (staging / "model.usd").write_text(
            _build_model_usda(output_meshes=output_meshes, translations=translations)
        )
        # Build the optional telemetry sidecar from independently re-read
        # enclosed inputs.  Its contents never participate in check verdicts.
        bundle_receipt = _bundle_inventory(staging / "source_bundle", bundle_pins)
        urdf_receipt = _verify_exact_file(
            staging / "source" / "urdf" / "model.urdf",
            urdf_pin,
            "enclosed reviewed source URDF",
        )
        mesh_receipts = {
            component.source_name: _verify_exact_file(
                staging / "source" / "meshes" / f"{component.source_name}.stl",
                component.mesh_sha256,
                f"enclosed source mesh {component.source_name}",
            )
            for component in SOURCE_MESHES
        }
        bundle_derivation = _verify_bundle_derivation(
            staging / "source_bundle",
            staging / "source" / "urdf" / "model.urdf",
            expected_asset_hash=expected_asset_hash,
        )
        _, _, geometry_evidence = _source_geometry(
            source_urdf=staging / "source" / "urdf" / "model.urdf",
            source_mesh_root=staging / "source" / "meshes",
        )
        receipt = _receipt(
            bundle_receipt=bundle_receipt,
            urdf_receipt=urdf_receipt,
            mesh_receipts=mesh_receipts,
            bundle_derivation=bundle_derivation,
            urdf_facts=urdf_facts,
            geometry_evidence=geometry_evidence,
            model_path=staging / "model.usd",
        )
        (staging / "DERIVATION_RECEIPT.json").write_bytes(_canonical_json(receipt))
        checked = _validate_output(
            staging,
            bundle_pins=bundle_pins,
            urdf_pin=urdf_pin,
            components=components,
            expected_asset_hash=expected_asset_hash,
        )
        _rename_no_replace(staging, output_root)
        published = True
        return checked
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


def materialize(
    *,
    source_bundle: Path,
    source_urdf: Path,
    source_mesh_root: Path,
    output_root: Path,
) -> dict[str, object]:
    """Production entrypoint with lexical reviewed pins and component policy."""

    return _materialize_with_policy(
        source_bundle=source_bundle,
        source_urdf=source_urdf,
        source_mesh_root=source_mesh_root,
        output_root=output_root,
        bundle_pins=BUNDLE_PINS,
        urdf_pin=URDF_PIN,
        components=SOURCE_MESHES,
        expected_asset_hash=ISAACLAB_ASSET_HASH,
    )


def _materialize_for_test(
    *,
    source_bundle: Path,
    source_urdf: Path,
    source_mesh_root: Path,
    output_root: Path,
    bundle_pins: Mapping[str, str],
    isaaclab_asset_hash: str,
) -> dict[str, object]:
    """Private fixture seam; neither CLI nor production callers can select it."""

    return _materialize_with_policy(
        source_bundle=source_bundle,
        source_urdf=source_urdf,
        source_mesh_root=source_mesh_root,
        output_root=output_root,
        bundle_pins=bundle_pins,
        urdf_pin=URDF_PIN,
        components=SOURCE_MESHES,
        expected_asset_hash=isaaclab_asset_hash,
    )


def _check_for_test(
    output_root: Path,
    *,
    bundle_pins: Mapping[str, str],
    isaaclab_asset_hash: str,
    components: tuple[ComponentPin, ...] = SOURCE_MESHES,
) -> dict[str, object]:
    """Private fixture verifier using explicit non-production expectations."""

    return _validate_output(
        output_root.absolute(),
        bundle_pins=bundle_pins,
        urdf_pin=URDF_PIN,
        components=components,
        expected_asset_hash=isaaclab_asset_hash,
    )


def check(output_root: Path) -> dict[str, object]:
    """Recompute deterministic source/model facts; ignore receipt authority."""

    return _validate_output(
        output_root.absolute(),
        bundle_pins=BUNDLE_PINS,
        urdf_pin=URDF_PIN,
        components=SOURCE_MESHES,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", type=Path)
    parser.add_argument("--source-urdf", type=Path)
    parser.add_argument("--source-mesh-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        if args.check:
            result = check(args.output_root)
        else:
            missing = [
                name
                for name in ("source_bundle", "source_urdf", "source_mesh_root")
                if getattr(args, name) is None
            ]
            if missing:
                raise DerivationError(
                    "materialization requires --source-bundle, --source-urdf, "
                    "and --source-mesh-root"
                )
            result = materialize(
                source_bundle=args.source_bundle,
                source_urdf=args.source_urdf,
                source_mesh_root=args.source_mesh_root,
                output_root=args.output_root,
            )
    except DerivationError as exc:
        print(f"[REFUSED] {exc}", file=sys.stderr)
        return 2
    print(
        "[OK] diagnostic split-rubber derivation "
        f"model_sha256={result['model_usd']['sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
