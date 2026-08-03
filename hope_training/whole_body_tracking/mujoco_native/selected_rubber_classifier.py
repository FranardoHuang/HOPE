"""Versioned, fail-closed selected-rubber authority for native MuJoCo.

The vendor MJCF intentionally keeps one generic racket collision proxy.  A
generic collision is useful physics evidence, but by itself it cannot say
which rubber face was hit.  This module adds that missing identity without
changing the contact mechanics:

* the official URDF red/black meshes and the current MJCF/site are verified;
* an immutable ActionBall manifest uniquely binds ``action_id``,
  ``action_uid`` and ``mount_normal_sign``;
* the compiled scene/backend identity is part of the classifier binding; and
* only a conservative, mesh-derived interior disk is classifiable.  Edge,
  rim, and between-plane contacts remain explicitly ambiguous.

The classifier consumes an already-observed contact edge.  It never invents a
contact from distance, a target, or a strike window.  The original generic
blade contact ledger remains authoritative for whether MuJoCo reported a
collision.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import struct
import sys
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MJCF = (
    REPO_ROOT
    / "agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/"
    "a3_pingpong/a3_pingpong.xml"
)
OFFICIAL_URDF = (
    REPO_ROOT
    / "agi/URDF/A3T2.5-URDF-std-pingpang/urdf/URDF-JOINT-LINK.urdf"
)
OFFICIAL_URDF_MESH_DIR = OFFICIAL_URDF.parent.parent / "meshes"
IDENTITY_MANIFEST = REPO_ROOT / "configs/a3_mujoco_identity_v2_20260803.json"
GEOMETRY_SOURCE_PY = (
    REPO_ROOT
    / "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp/racket_contact_geometry.py"
)

CLASSIFIER_BINDING_KIND = "a3_mujoco_selected_rubber_classifier_binding_v1"
ACTION_LINEAGE_KIND = "a3_mujoco_selected_rubber_action_lineage_v1"
CLASSIFICATION_KIND = "a3_mujoco_selected_rubber_contact_classification_v1"
RACKET_SITE_NAME = "right_racket"
GENERIC_BLADE_GEOM_NAME = "right_racket_collision"
RAW_A_AXIS_LOCAL = (0.0, 1.0, 0.0)

STATUS_SELECTED = "selected_rubber_face"
STATUS_OPPOSITE = "opposite_rubber_face"
STATUS_EDGE_RIM_AMBIGUOUS = "edge_or_rim_ambiguous"
STATUS_BETWEEN_PLANES_AMBIGUOUS = "between_outer_planes_ambiguous"
CLASSIFICATION_STATUSES = (
    STATUS_SELECTED,
    STATUS_OPPOSITE,
    STATUS_EDGE_RIM_AMBIGUOUS,
    STATUS_BETWEEN_PLANES_AMBIGUOUS,
)
AMBIGUITY_EDGE_RIM = "ball_footprint_not_strictly_inside_mesh_derived_safe_disk"
AMBIGUITY_BETWEEN_PLANES = "ball_center_not_strictly_outside_one_urdf_outer_plane"


class SelectedRubberClassifierError(ValueError):
    """Asset, action-lineage, or runtime classification is not exact."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise SelectedRubberClassifierError(
            "selected-rubber payload is not finite canonical JSON"
        ) from exc


def _plain_sha256(value: Any, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SelectedRubberClassifierError(f"{name} must be lowercase SHA-256")
    return value


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise SelectedRubberClassifierError(
            f"selected-rubber authority must live inside repository: {path}"
        ) from exc


def _unique_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise SelectedRubberClassifierError(
                f"duplicate JSON key in selected-rubber authority: {key}"
            )
        out[key] = value
    return out


def _strict_json(path: Path) -> tuple[bytes, dict[str, Any]]:
    try:
        raw = path.read_bytes()
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                SelectedRubberClassifierError(
                    f"non-finite JSON constant is forbidden: {value}"
                )
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SelectedRubberClassifierError(
            f"cannot read selected-rubber authority {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise SelectedRubberClassifierError(
            f"selected-rubber authority root must be an object: {path}"
        )
    return raw, value


@lru_cache(maxsize=1)
def _geometry_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_mujoco_selected_rubber_geometry_authority", GEOMETRY_SOURCE_PY
    )
    if spec is None or spec.loader is None:
        raise SelectedRubberClassifierError(
            f"cannot import geometry authority from {GEOMETRY_SOURCE_PY}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _xyz(text: str, name: str) -> np.ndarray:
    values = np.fromstring(text, sep=" ", dtype=np.float64)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise SelectedRubberClassifierError(f"{name} must be three finite scalars")
    return values


def _binary_stl_triangles(path: Path) -> tuple[bytes, np.ndarray]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise SelectedRubberClassifierError(f"cannot read binary STL {path}") from exc
    if len(raw) < 84:
        raise SelectedRubberClassifierError(f"binary STL is truncated: {path}")
    count = struct.unpack_from("<I", raw, 80)[0]
    if len(raw) != 84 + 50 * count:
        raise SelectedRubberClassifierError(f"STL is not exact binary STL: {path}")
    triangles = np.stack(
        [
            np.frombuffer(
                raw, dtype="<f4", count=9, offset=84 + 50 * index + 12
            ).reshape(3, 3)
            for index in range(count)
        ]
    ).astype(np.float64)
    if not np.isfinite(triangles).all():
        raise SelectedRubberClassifierError(f"STL contains non-finite vertices: {path}")
    return raw, triangles


def _one_named(root: ET.Element, tag: str, name: str) -> ET.Element:
    rows = [
        element
        for element in root.iter(tag)
        if element.attrib.get("name") == name
    ]
    if len(rows) != 1:
        raise SelectedRubberClassifierError(
            f"expected exactly one {tag} named {name!r}, found {len(rows)}"
        )
    return rows[0]


def _joint_origin(urdf: ET.Element, name: str) -> np.ndarray:
    joint = _one_named(urdf, "joint", name)
    origin = joint.find("origin")
    if origin is None:
        raise SelectedRubberClassifierError(f"URDF joint {name!r} has no origin")
    if _xyz(origin.attrib.get("rpy", "0 0 0"), f"{name}.rpy").tolist() != [
        0.0,
        0.0,
        0.0,
    ]:
        raise SelectedRubberClassifierError(f"URDF joint {name!r} is rotated")
    return _xyz(origin.attrib.get("xyz", ""), f"{name}.xyz")


def _closure_member(
    closure: Mapping[str, Any], suffix: str, expected_sha256: str
) -> str:
    matches = [
        key
        for key, value in closure.items()
        if str(key).endswith(suffix) and value == expected_sha256
    ]
    if len(matches) != 1:
        raise SelectedRubberClassifierError(
            f"compiled mesh closure does not uniquely bind {suffix}"
        )
    return str(matches[0])


def verify_urdf_mjcf_geometry(
    mjcf_path: Path | str = DEFAULT_MJCF,
) -> dict[str, Any]:
    """Verify the exact URDF/MJCF facts used by the classifier."""

    geometry = _geometry_module()
    mjcf_source = Path(mjcf_path).expanduser().resolve()
    if mjcf_source != DEFAULT_MJCF.resolve():
        raise SelectedRubberClassifierError(
            "selected-rubber authority requires the canonical root MJCF path"
        )
    try:
        mjcf_raw = mjcf_source.read_bytes()
        urdf_raw = OFFICIAL_URDF.read_bytes()
        mjcf = ET.fromstring(mjcf_raw)
        urdf = ET.fromstring(urdf_raw)
    except (OSError, ET.ParseError) as exc:
        raise SelectedRubberClassifierError(
            f"cannot read canonical racket URDF/MJCF: {exc}"
        ) from exc

    identity_raw, identity = _strict_json(IDENTITY_MANIFEST)
    expected_root_sha = _plain_sha256(
        identity.get("expected", {}).get("root_mjcf_sha256"),
        "identity expected root_mjcf_sha256",
    )
    if _sha256(mjcf_raw) != expected_root_sha:
        raise SelectedRubberClassifierError(
            "canonical MJCF differs from the v2 identity manifest"
        )

    site = _one_named(mjcf, "site", RACKET_SITE_NAME)
    site_pos = _xyz(site.attrib.get("pos", ""), "right_racket.pos")
    expected_site = np.asarray(geometry.RACKET_SITE_OFFSET_WRIST_M, np.float64)
    if not np.array_equal(site_pos, expected_site):
        raise SelectedRubberClassifierError("MJCF official racket site differs")
    if site.attrib.get("quat", "1 0 0 0") != "1 0 0 0":
        raise SelectedRubberClassifierError("MJCF official racket site is rotated")

    red_joint = _joint_origin(urdf, "pingpang_red_joint")
    black_joint = _joint_origin(urdf, "pingpang_black_joint")
    if not np.array_equal(red_joint, expected_site) or not np.array_equal(
        black_joint, expected_site
    ):
        raise SelectedRubberClassifierError(
            "URDF red/black fixed joints differ from the official racket site"
        )

    red_path = OFFICIAL_URDF_MESH_DIR / "pingpang_red_Link.STL"
    black_path = OFFICIAL_URDF_MESH_DIR / "pingpang_black_Link.STL"
    proxy_path = (
        mjcf_source.parent / "meshes/collision_optimized/right_racket_face_collision.STL"
    )
    red_raw, red = _binary_stl_triangles(red_path)
    black_raw, black = _binary_stl_triangles(black_path)
    proxy_raw, proxy = _binary_stl_triangles(proxy_path)
    if _sha256(red_raw) != geometry.RED_SELECTED_FACE_MESH_SHA256:
        raise SelectedRubberClassifierError("official red rubber STL SHA differs")
    if _sha256(black_raw) != geometry.BLACK_SELECTED_FACE_MESH_SHA256:
        raise SelectedRubberClassifierError("official black rubber STL SHA differs")

    red_outer_y = float(red[:, :, 1].max())
    black_outer_y = float(black[:, :, 1].min())
    if not math.isclose(
        red_outer_y,
        float(geometry.RED_OUTER_Y_FROM_SITE_M),
        rel_tol=0.0,
        abs_tol=2.0e-9,
    ) or not math.isclose(
        black_outer_y,
        float(geometry.BLACK_OUTER_Y_FROM_SITE_M),
        rel_tol=0.0,
        abs_tol=2.0e-9,
    ):
        raise SelectedRubberClassifierError(
            "official rubber STL outer planes differ from geometry authority"
        )

    proxy_mesh = _one_named(mjcf, "mesh", "collision_right_racket_face")
    proxy_scale = _xyz(proxy_mesh.attrib.get("scale", "1 1 1"), "proxy.scale")
    proxy = proxy * proxy_scale
    proxy_geom = _one_named(mjcf, "geom", GENERIC_BLADE_GEOM_NAME)
    proxy_pos = _xyz(proxy_geom.attrib.get("pos", ""), "proxy.pos")
    proxy_outer = np.asarray(
        [
            proxy_pos[1] + float(proxy[:, :, 1].max()),
            proxy_pos[1] + float(proxy[:, :, 1].min()),
        ],
        np.float64,
    )
    urdf_outer = np.asarray(
        [site_pos[1] + red_outer_y, site_pos[1] + black_outer_y], np.float64
    )
    if not np.allclose(proxy_outer, urdf_outer, rtol=0.0, atol=2.0e-9):
        raise SelectedRubberClassifierError(
            "generic MuJoCo blade outer planes differ from URDF rubber faces"
        )

    return {
        "geometry_source_sha256": _plain_sha256(
            geometry.GEOMETRY_SOURCE_SHA256, "geometry source SHA"
        ),
        "geometry_source_path": _repo_relative(GEOMETRY_SOURCE_PY),
        "official_urdf_path": _repo_relative(OFFICIAL_URDF),
        "official_urdf_sha256": _sha256(urdf_raw),
        "identity_manifest_path": _repo_relative(IDENTITY_MANIFEST),
        "identity_manifest_sha256": _sha256(identity_raw),
        "canonical_mjcf_path": _repo_relative(mjcf_source),
        "canonical_mjcf_sha256": _sha256(mjcf_raw),
        "red_rubber_mesh_sha256": _sha256(red_raw),
        "black_rubber_mesh_sha256": _sha256(black_raw),
        "generic_blade_mesh_sha256": _sha256(proxy_raw),
        "racket_site_offset_wrist_m": site_pos.tolist(),
        "raw_a_axis_local": list(RAW_A_AXIS_LOCAL),
        "red_outer_y_from_site_m": red_outer_y,
        "black_outer_y_from_site_m": black_outer_y,
        "face_area_center_xz_from_site_m": list(
            geometry.FACE_AREA_CENTER_XZ_FROM_SITE_M
        ),
        "ball_radius_m": float(geometry.BALL_RADIUS_M),
        "formal_face_edge_guard_m": float(geometry.FORMAL_FACE_EDGE_GUARD_M),
        "safe_ball_center_tangential_radius_m": float(
            geometry.SAFE_BALL_CENTER_TANGENTIAL_RADIUS_M
        ),
    }


def build_classifier_binding(
    *, scene_binding: Mapping[str, Any], mjcf_path: Path | str = DEFAULT_MJCF
) -> dict[str, Any]:
    """Bind the classifier to exact geometry, compiled scene, and backend."""

    if not isinstance(scene_binding, Mapping):
        raise SelectedRubberClassifierError("scene binding must be a mapping")
    scene_sha = _plain_sha256(
        scene_binding.get("binding_sha256"), "scene binding SHA"
    )
    geometry = verify_urdf_mjcf_geometry(mjcf_path)
    if scene_binding.get("canonical_mjcf_sha256") != geometry[
        "canonical_mjcf_sha256"
    ]:
        raise SelectedRubberClassifierError(
            "compiled scene does not bind the verified canonical MJCF"
        )
    compiled = scene_binding.get("compiled_runtime")
    if not isinstance(compiled, Mapping):
        raise SelectedRubberClassifierError(
            "scene binding has no compiled runtime/backend identity"
        )
    backend_version = compiled.get("mujoco_version")
    closure = compiled.get("mesh_source_closure_sha256")
    if type(backend_version) is not str or not backend_version.strip():
        raise SelectedRubberClassifierError("MuJoCo backend version is absent")
    if not isinstance(closure, Mapping):
        raise SelectedRubberClassifierError("compiled mesh closure is absent")
    closure_members = {
        "red": _closure_member(
            closure,
            "pingpang_red_Link.STL",
            geometry["red_rubber_mesh_sha256"],
        ),
        "black": _closure_member(
            closure,
            "pingpang_black_Link.STL",
            geometry["black_rubber_mesh_sha256"],
        ),
        "generic_blade": _closure_member(
            closure,
            "right_racket_face_collision.STL",
            geometry["generic_blade_mesh_sha256"],
        ),
    }
    payload = {
        "schema_version": 1,
        "kind": CLASSIFIER_BINDING_KIND,
        "classifier_source_sha256": _sha256(Path(__file__).read_bytes()),
        "classifier_source_path": _repo_relative(Path(__file__)),
        "scene_binding_sha256": scene_sha,
        "assembled_xml_sha256": _plain_sha256(
            scene_binding.get("assembled_xml_sha256"), "assembled XML SHA"
        ),
        "mujoco_backend_version": backend_version,
        "generic_blade_geom_name": GENERIC_BLADE_GEOM_NAME,
        "official_racket_site_name": RACKET_SITE_NAME,
        "generic_blade_contact_preserved": True,
        "classification_semantics": (
            "after_observed_generic_blade_contact;ball_center_in_official_site_frame;"
            "strictly_outside_one_urdf_outer_plane;strict_mesh_derived_inscribed_"
            "safe_disk;edge_rim_or_between_planes_fail_ambiguous"
        ),
        "classification_statuses": list(CLASSIFICATION_STATUSES),
        "compiled_mesh_closure_members": closure_members,
        "geometry": geometry,
    }
    payload["content_sha256"] = _sha256(_canonical_json_bytes(payload))
    return payload


def validate_classifier_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SelectedRubberClassifierError("classifier binding must be a mapping")
    payload = dict(value)
    declared = _plain_sha256(
        payload.pop("content_sha256", None), "classifier binding content SHA"
    )
    if payload.get("schema_version") != 1 or payload.get("kind") != (
        CLASSIFIER_BINDING_KIND
    ):
        raise SelectedRubberClassifierError("classifier binding kind/schema differs")
    if _sha256(_canonical_json_bytes(payload)) != declared:
        raise SelectedRubberClassifierError("classifier binding content seal differs")
    payload["content_sha256"] = declared
    return payload


def bind_action_manifest(
    *,
    manifest_path: Path | str,
    expected_manifest_sha256: str,
    action_uid: int,
    motion_sha256: str,
    mount_normal_sign: int,
    geometry_source_sha256: str,
    physics_sha256: str,
    classifier_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve one stable action ID from the exact manifest without guessing."""

    source = Path(manifest_path).expanduser().resolve()
    raw, manifest = _strict_json(source)
    expected_sha = _plain_sha256(expected_manifest_sha256, "action manifest SHA")
    if _sha256(raw) != expected_sha:
        raise SelectedRubberClassifierError(
            "action manifest bytes differ from immutable-tape authority"
        )
    if type(action_uid) is not int or action_uid < 0:
        raise SelectedRubberClassifierError("action_uid must be non-negative plain int")
    if type(mount_normal_sign) is not int or mount_normal_sign not in (-1, 1):
        raise SelectedRubberClassifierError("mount_normal_sign must be exact -1 or +1")
    motion_sha = _plain_sha256(motion_sha256, "action motion SHA")
    geometry_sha = _plain_sha256(geometry_source_sha256, "geometry source SHA")
    physics_sha = _plain_sha256(physics_sha256, "physics profile SHA")
    if manifest.get("schema_version") != 3:
        raise SelectedRubberClassifierError("action manifest must be exact schema 3")
    if manifest.get("physics_profile_sha256") != physics_sha:
        raise SelectedRubberClassifierError(
            "action manifest physics profile differs from immutable tape"
        )
    actions = manifest.get("actions")
    if not isinstance(actions, list):
        raise SelectedRubberClassifierError("action manifest actions must be a list")
    matches = [
        row
        for row in actions
        if isinstance(row, Mapping) and row.get("action_uid") == action_uid
    ]
    if len(matches) != 1:
        raise SelectedRubberClassifierError(
            "action_uid does not resolve uniquely in exact action manifest"
        )
    action = matches[0]
    action_id = action.get("action_id")
    if type(action_id) is not str or not action_id.strip():
        raise SelectedRubberClassifierError("resolved action_id is absent")
    if action.get("motion_sha256") != motion_sha:
        raise SelectedRubberClassifierError(
            "resolved action motion differs from immutable tape"
        )
    raw_sign = action.get("mount_normal_sign")
    if (
        isinstance(raw_sign, bool)
        or not isinstance(raw_sign, (int, float))
        or not math.isfinite(float(raw_sign))
    ):
        raise SelectedRubberClassifierError(
            "resolved action mount_normal_sign is invalid"
        )
    if float(raw_sign) != float(mount_normal_sign):
        raise SelectedRubberClassifierError(
            "resolved action mount_normal_sign differs from measured receipt"
        )

    binding = validate_classifier_binding(classifier_binding)
    binding_geometry = binding.get("geometry")
    if not isinstance(binding_geometry, Mapping) or binding_geometry.get(
        "geometry_source_sha256"
    ) != geometry_sha:
        raise SelectedRubberClassifierError(
            "action geometry source differs from classifier geometry authority"
        )
    payload = {
        "schema_version": 1,
        "kind": ACTION_LINEAGE_KIND,
        "action_id": action_id,
        "action_uid": action_uid,
        "mount_normal_sign": mount_normal_sign,
        "motion_sha256": motion_sha,
        "physics_sha256": physics_sha,
        "geometry_source_sha256": geometry_sha,
        "action_manifest_repo_relative_path": _repo_relative(source),
        "action_manifest_sha256": expected_sha,
        "scene_binding_sha256": binding["scene_binding_sha256"],
        "mujoco_backend_version": binding["mujoco_backend_version"],
        "classifier_binding_sha256": binding["content_sha256"],
    }
    payload["content_sha256"] = _sha256(_canonical_json_bytes(payload))
    return payload


def validate_action_lineage(
    value: Mapping[str, Any], *, classifier_binding: Mapping[str, Any]
) -> dict[str, Any]:
    payload = validate_action_lineage_seal(value)
    unsigned = dict(payload)
    unsigned.pop("content_sha256")
    binding = validate_classifier_binding(classifier_binding)
    if (
        unsigned.get("scene_binding_sha256") != binding["scene_binding_sha256"]
        or unsigned.get("mujoco_backend_version")
        != binding["mujoco_backend_version"]
        or unsigned.get("classifier_binding_sha256") != binding["content_sha256"]
    ):
        raise SelectedRubberClassifierError(
            "action lineage differs from current scene/backend classifier"
        )
    source = REPO_ROOT / unsigned["action_manifest_repo_relative_path"]
    rebound = bind_action_manifest(
        manifest_path=source,
        expected_manifest_sha256=unsigned["action_manifest_sha256"],
        action_uid=unsigned["action_uid"],
        motion_sha256=unsigned["motion_sha256"],
        mount_normal_sign=unsigned["mount_normal_sign"],
        geometry_source_sha256=unsigned["geometry_source_sha256"],
        physics_sha256=unsigned["physics_sha256"],
        classifier_binding=binding,
    )
    if rebound != payload:
        raise SelectedRubberClassifierError(
            "selected-rubber action lineage cannot be independently rebuilt"
        )
    return rebound


def validate_action_lineage_seal(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the portable lineage seal without opening its source files."""

    if not isinstance(value, Mapping):
        raise SelectedRubberClassifierError("selected-rubber action lineage is absent")
    payload = dict(value)
    expected_keys = {
        "schema_version",
        "kind",
        "action_id",
        "action_uid",
        "mount_normal_sign",
        "motion_sha256",
        "physics_sha256",
        "geometry_source_sha256",
        "action_manifest_repo_relative_path",
        "action_manifest_sha256",
        "scene_binding_sha256",
        "mujoco_backend_version",
        "classifier_binding_sha256",
        "content_sha256",
    }
    if set(payload) != expected_keys:
        raise SelectedRubberClassifierError("action lineage keys differ")
    declared = _plain_sha256(
        payload.pop("content_sha256", None), "action lineage content SHA"
    )
    if payload.get("schema_version") != 1 or payload.get("kind") != (
        ACTION_LINEAGE_KIND
    ):
        raise SelectedRubberClassifierError("action lineage kind/schema differs")
    if _sha256(_canonical_json_bytes(payload)) != declared:
        raise SelectedRubberClassifierError("action lineage content seal differs")
    if type(payload["action_id"]) is not str or not payload["action_id"].strip():
        raise SelectedRubberClassifierError("action lineage action_id is absent")
    if type(payload["action_uid"]) is not int or payload["action_uid"] < 0:
        raise SelectedRubberClassifierError("action lineage action_uid is invalid")
    if type(payload["mount_normal_sign"]) is not int or payload[
        "mount_normal_sign"
    ] not in (-1, 1):
        raise SelectedRubberClassifierError(
            "action lineage mount_normal_sign is invalid"
        )
    for key in (
        "motion_sha256",
        "physics_sha256",
        "geometry_source_sha256",
        "action_manifest_sha256",
        "scene_binding_sha256",
        "classifier_binding_sha256",
    ):
        _plain_sha256(payload[key], f"action lineage {key}")
    relative = Path(payload["action_manifest_repo_relative_path"])
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise SelectedRubberClassifierError(
            "action lineage manifest path is not repository-relative"
        )
    if (
        type(payload["mujoco_backend_version"]) is not str
        or not payload["mujoco_backend_version"].strip()
    ):
        raise SelectedRubberClassifierError("action lineage backend is absent")
    payload["content_sha256"] = declared
    return payload


def validate_classification_seal(
    value: Mapping[str, Any], *, action_lineage: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate a portable contact classification and its action binding."""

    lineage = validate_action_lineage_seal(action_lineage)
    if not isinstance(value, Mapping):
        raise SelectedRubberClassifierError("contact classification must be a mapping")
    payload = dict(value)
    expected_keys = {
        "schema_version",
        "kind",
        "policy_tick",
        "physics_substep",
        "status",
        "ambiguity_reason",
        "observed_face_sign",
        "selected_rubber",
        "ball_center_local_m",
        "tangential_distance_from_face_center_m",
        "safe_ball_center_tangential_radius_m",
        "action_lineage_sha256",
        "classifier_binding_sha256",
        "content_sha256",
    }
    if set(payload) != expected_keys:
        raise SelectedRubberClassifierError("contact classification keys differ")
    declared = _plain_sha256(
        payload.pop("content_sha256", None), "classification content SHA"
    )
    if payload.get("schema_version") != 1 or payload.get("kind") != (
        CLASSIFICATION_KIND
    ):
        raise SelectedRubberClassifierError("classification kind/schema differs")
    if _sha256(_canonical_json_bytes(payload)) != declared:
        raise SelectedRubberClassifierError("classification content seal differs")
    for key in ("policy_tick", "physics_substep"):
        if type(payload[key]) is not int or payload[key] < 0:
            raise SelectedRubberClassifierError(
                f"classification {key} must be non-negative int"
            )
    status = payload["status"]
    if status not in CLASSIFICATION_STATUSES:
        raise SelectedRubberClassifierError("classification status differs")
    if status in (STATUS_SELECTED, STATUS_OPPOSITE):
        if payload["ambiguity_reason"] is not None:
            raise SelectedRubberClassifierError(
                "classified face cannot carry ambiguity reason"
            )
        if payload["observed_face_sign"] not in (-1, 1):
            raise SelectedRubberClassifierError(
                "classified face must carry observed sign"
            )
        expected_selected = status == STATUS_SELECTED
        if payload["selected_rubber"] is not expected_selected:
            raise SelectedRubberClassifierError(
                "classification selected-rubber boolean disagrees with status"
            )
        if (
            payload["observed_face_sign"] == lineage["mount_normal_sign"]
        ) is not expected_selected:
            raise SelectedRubberClassifierError(
                "classification face sign disagrees with selected mount sign"
            )
    else:
        expected_reason = (
            AMBIGUITY_EDGE_RIM
            if status == STATUS_EDGE_RIM_AMBIGUOUS
            else AMBIGUITY_BETWEEN_PLANES
        )
        if (
            payload["ambiguity_reason"] != expected_reason
            or payload["observed_face_sign"] is not None
            or payload["selected_rubber"] is not None
        ):
            raise SelectedRubberClassifierError(
                "ambiguous classification carries face identity"
            )
    local = np.asarray(payload["ball_center_local_m"], dtype=np.float64)
    if local.shape != (3,) or not np.isfinite(local).all():
        raise SelectedRubberClassifierError(
            "classification local ball center must be finite length 3"
        )
    for key in (
        "tangential_distance_from_face_center_m",
        "safe_ball_center_tangential_radius_m",
    ):
        scalar = payload[key]
        if (
            isinstance(scalar, bool)
            or not isinstance(scalar, (int, float))
            or not math.isfinite(float(scalar))
            or float(scalar) < 0.0
        ):
            raise SelectedRubberClassifierError(
                f"classification {key} must be non-negative finite"
            )
    if payload["action_lineage_sha256"] != lineage["content_sha256"]:
        raise SelectedRubberClassifierError(
            "classification action lineage SHA differs"
        )
    if payload["classifier_binding_sha256"] != lineage[
        "classifier_binding_sha256"
    ]:
        raise SelectedRubberClassifierError(
            "classification classifier binding SHA differs"
        )
    payload["content_sha256"] = declared
    return payload


def classify_observed_generic_blade_contact(
    *,
    ball_center_w_m: Sequence[float],
    racket_site_position_w_m: Sequence[float],
    racket_rotation_w_from_local: Sequence[Sequence[float]],
    action_lineage: Mapping[str, Any],
    classifier_binding: Mapping[str, Any],
    policy_tick: int,
    physics_substep: int,
) -> dict[str, Any]:
    """Classify one already-observed generic blade contact conservatively."""

    binding = validate_classifier_binding(classifier_binding)
    lineage = validate_action_lineage(action_lineage, classifier_binding=binding)
    if type(policy_tick) is not int or policy_tick < 0:
        raise SelectedRubberClassifierError("policy_tick must be non-negative int")
    if type(physics_substep) is not int or physics_substep < 0:
        raise SelectedRubberClassifierError(
            "physics_substep must be non-negative int"
        )
    ball = np.asarray(ball_center_w_m, dtype=np.float64)
    site = np.asarray(racket_site_position_w_m, dtype=np.float64)
    rotation = np.asarray(racket_rotation_w_from_local, dtype=np.float64)
    if (
        ball.shape != (3,)
        or site.shape != (3,)
        or rotation.shape != (3, 3)
        or not np.isfinite(ball).all()
        or not np.isfinite(site).all()
        or not np.isfinite(rotation).all()
    ):
        raise SelectedRubberClassifierError(
            "contact classifier pose must be finite 3-D/3x3"
        )
    if not np.allclose(
        rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1.0e-9
    ) or not math.isclose(
        float(np.linalg.det(rotation)), 1.0, rel_tol=0.0, abs_tol=1.0e-9
    ):
        raise SelectedRubberClassifierError(
            "contact classifier racket rotation is not a proper rotation"
        )

    local = rotation.T @ (ball - site)
    geometry = binding["geometry"]
    center_xz = np.asarray(
        geometry["face_area_center_xz_from_site_m"], dtype=np.float64
    )
    tangential = float(np.linalg.norm(local[[0, 2]] - center_xz))
    safe_radius = float(geometry["safe_ball_center_tangential_radius_m"])
    red_outer = float(geometry["red_outer_y_from_site_m"])
    black_outer = float(geometry["black_outer_y_from_site_m"])
    observed_sign = None
    selected = None
    ambiguity = None
    if not tangential < safe_radius:
        status = STATUS_EDGE_RIM_AMBIGUOUS
        ambiguity = AMBIGUITY_EDGE_RIM
    elif float(local[1]) > red_outer:
        observed_sign = 1
        selected = lineage["mount_normal_sign"] == observed_sign
        status = STATUS_SELECTED if selected else STATUS_OPPOSITE
    elif float(local[1]) < black_outer:
        observed_sign = -1
        selected = lineage["mount_normal_sign"] == observed_sign
        status = STATUS_SELECTED if selected else STATUS_OPPOSITE
    else:
        status = STATUS_BETWEEN_PLANES_AMBIGUOUS
        ambiguity = AMBIGUITY_BETWEEN_PLANES

    payload = {
        "schema_version": 1,
        "kind": CLASSIFICATION_KIND,
        "policy_tick": policy_tick,
        "physics_substep": physics_substep,
        "status": status,
        "ambiguity_reason": ambiguity,
        "observed_face_sign": observed_sign,
        "selected_rubber": selected,
        "ball_center_local_m": local.tolist(),
        "tangential_distance_from_face_center_m": tangential,
        "safe_ball_center_tangential_radius_m": safe_radius,
        "action_lineage_sha256": lineage["content_sha256"],
        "classifier_binding_sha256": binding["content_sha256"],
    }
    payload["content_sha256"] = _sha256(_canonical_json_bytes(payload))
    return payload
