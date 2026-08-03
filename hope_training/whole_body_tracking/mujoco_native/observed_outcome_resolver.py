"""Source-bound native net-crossing and first-landing evidence.

This module resolves only events that MuJoCo has actually observed.  It does
not predict a trajectory, infer a hit from a target, or assign reward.  The
resolver is bound to the compiled physical-ball scene, its table geometry,
the ball radius, the exact question, and this source file.

The first contact-free ball state after a racket hit arms the resolver.  Each
later physics-substep sample may then establish:

* a directed crossing of the scene's net centre plane;
* a native net/post collision;
* the first native table contact; or
* a native floor contact.

Legal landing is deliberately conservative.  The observed ball centre must
be strictly inside the ball-radius-eroded table footprint and strictly on the
scene-authoritative opponent side of the net.  A crossing and a terminal
contact first seen in the same physics substep are temporally ambiguous and
stay fail-closed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import physical_ball_scene


REPO_ROOT = Path(__file__).resolve().parents[3]

RESOLVER_BINDING_KIND = "a3_mujoco_observed_outcome_resolver_binding_v2"
QUESTION_BINDING_KIND = "a3_mujoco_observed_outcome_question_binding_v2"
SNAPSHOT_KIND = "a3_mujoco_observed_outcome_snapshot_v2"
SUMMARY_KIND = "a3_mujoco_observed_outcome_summary_v2"

STATUS_UNARMED = "unarmed_no_outgoing_flight"
STATUS_TRACKING = "tracking_post_contact_flight"
STATUS_NET_COLLISION = "resolved_native_net_or_post_collision"
STATUS_FIRST_TABLE_LANDING = "resolved_native_first_table_contact"
STATUS_FLOOR_CONTACT = "resolved_native_floor_contact"
STATUS_SAME_SUBSTEP_AMBIGUOUS = (
    "fail_closed_same_substep_net_crossing_and_terminal_contact"
)
STATUS_OUTGOING_OVERLAP_AMBIGUOUS = (
    "fail_closed_outgoing_state_overlaps_terminal_contact"
)
STATUSES = (
    STATUS_UNARMED,
    STATUS_TRACKING,
    STATUS_NET_COLLISION,
    STATUS_FIRST_TABLE_LANDING,
    STATUS_FLOOR_CONTACT,
    STATUS_SAME_SUBSTEP_AMBIGUOUS,
    STATUS_OUTGOING_OVERLAP_AMBIGUOUS,
)
RESOLVED_STATUSES = (
    STATUS_NET_COLLISION,
    STATUS_FIRST_TABLE_LANDING,
    STATUS_FLOOR_CONTACT,
)
AMBIGUOUS_STATUSES = (
    STATUS_SAME_SUBSTEP_AMBIGUOUS,
    STATUS_OUTGOING_OVERLAP_AMBIGUOUS,
)
CONTACT_LABELS = ("racket", "table", "net", "floor")
TIME_DELTA_ABS_TOLERANCE_S = 1.0e-12
SEMANTICS = (
    "observed_substep_ball_centers_and_native_contact_labels_only;"
    "directed_net_plane_interpolation;strict_ball_radius_eroded_"
    "table_footprint;same_substep_crossing_terminal_contact_ambiguous"
)


class ObservedOutcomeResolverError(ValueError):
    """Observed outcome authority or chronological evidence is invalid."""


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
        raise ObservedOutcomeResolverError(
            "observed-outcome payload is not finite canonical JSON"
        ) from exc


def _plain_sha256(value: Any, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ObservedOutcomeResolverError(f"{name} must be lowercase SHA-256")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ObservedOutcomeResolverError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ObservedOutcomeResolverError(f"{name} must be finite")
    return result


def _vector(value: Any, width: int, name: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != width:
        raise ObservedOutcomeResolverError(
            f"{name} must contain exactly {width} values"
        )
    return [_finite(item, f"{name}[{index}]") for index, item in enumerate(value)]


def _stamp(policy_tick: Any, physics_substep: Any, name: str) -> dict[str, int]:
    if type(policy_tick) is not int or policy_tick < 0:
        raise ObservedOutcomeResolverError(
            f"{name}.policy_tick must be non-negative plain int"
        )
    if type(physics_substep) is not int or physics_substep < 0:
        raise ObservedOutcomeResolverError(
            f"{name}.physics_substep must be non-negative plain int"
        )
    return {
        "policy_tick": policy_tick,
        "physics_substep": physics_substep,
    }


def _stamp_tuple(value: Mapping[str, Any]) -> tuple[int, int]:
    canonical = _stamp(
        value.get("policy_tick"), value.get("physics_substep"), "stamp"
    )
    return canonical["policy_tick"], canonical["physics_substep"]


def _seal(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(payload))
    result["content_sha256"] = _sha256(_canonical_json_bytes(result))
    return result


def _validate_seal(
    value: Mapping[str, Any],
    *,
    expected_kind: str,
    expected_schema_version: int,
    expected_keys: set[str],
    name: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ObservedOutcomeResolverError(f"{name} must be a mapping")
    payload = copy.deepcopy(dict(value))
    if set(payload) != expected_keys:
        raise ObservedOutcomeResolverError(f"{name} keys differ")
    declared = _plain_sha256(
        payload.pop("content_sha256", None), f"{name} content SHA"
    )
    if (
        payload.get("schema_version") != expected_schema_version
        or payload.get("kind") != expected_kind
    ):
        raise ObservedOutcomeResolverError(f"{name} kind/schema differs")
    if _sha256(_canonical_json_bytes(payload)) != declared:
        raise ObservedOutcomeResolverError(f"{name} content seal differs")
    payload["content_sha256"] = declared
    return payload


def _box(row: Mapping[str, Any], name: str) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(row, Mapping) or row.get("name") != name:
        raise ObservedOutcomeResolverError(f"scene obstacle {name!r} is absent")
    center = np.asarray(_vector(row.get("center_mjcf_world_m"), 3, f"{name}.center"))
    extents = np.asarray(_vector(row.get("full_extents_m"), 3, f"{name}.extents"))
    if bool(np.any(extents <= 0.0)):
        raise ObservedOutcomeResolverError(f"scene obstacle {name!r} is empty")
    return center - 0.5 * extents, center + 0.5 * extents


def build_resolver_binding(
    *,
    scene_binding: Mapping[str, Any],
    obstacle_rows: Mapping[str, Any],
    plant_binding_sha256: str,
    policy_step_dt_s: float,
    control_decimation: int,
) -> dict[str, Any]:
    """Bind observed outcome semantics to the exact compiled physical scene."""

    if not isinstance(scene_binding, Mapping):
        raise ObservedOutcomeResolverError("scene binding must be a mapping")
    scene_sha = _plain_sha256(
        scene_binding.get("binding_sha256"), "scene binding SHA"
    )
    if (
        scene_binding.get("kind")
        != "a3_mujoco_physical_ball_scene_binding_v1"
        or scene_binding.get("with_ball") is not True
        or scene_binding.get("strict_pair_filter") is not True
    ):
        raise ObservedOutcomeResolverError(
            "resolver requires the strict native physical-ball scene"
        )
    compiled = scene_binding.get("compiled_runtime")
    if not isinstance(compiled, Mapping):
        raise ObservedOutcomeResolverError(
            "scene binding has no compiled runtime identity"
        )
    backend = compiled.get("mujoco_version")
    if type(backend) is not str or not backend.strip():
        raise ObservedOutcomeResolverError("MuJoCo backend identity is absent")

    table_scene = physical_ball_scene._load_table_scene_module()
    try:
        geometry_contract = table_scene.action_ball_policy_geometry_contract(
            obstacle_rows
        )
    except Exception as exc:  # noqa: BLE001 - source module is an authority boundary
        raise ObservedOutcomeResolverError(
            f"cannot validate ActionBall table geometry: {exc}"
        ) from exc
    if scene_binding.get("table_geometry_contract_sha256") != geometry_contract[
        "sha256"
    ]:
        raise ObservedOutcomeResolverError(
            "scene table geometry differs from compiled scene binding"
        )

    compiled_obstacles = compiled.get("obstacle_geometry")
    compiled_obstacle_ids = compiled.get("obstacle_geom_ids")
    if not isinstance(compiled_obstacles, Mapping) or not isinstance(
        compiled_obstacle_ids, Mapping
    ):
        raise ObservedOutcomeResolverError(
            "compiled obstacle geometry authority is absent"
        )
    source_obstacle_rows = (
        obstacle_rows.get("table_top"),
        obstacle_rows.get("robot_keepout"),
        obstacle_rows.get("net"),
        *(obstacle_rows.get("net_posts") or ()),
    )
    if any(not isinstance(row, Mapping) for row in source_obstacle_rows):
        raise ObservedOutcomeResolverError(
            "source table assembly rows are incomplete"
        )
    source_by_name = {row["name"]: row for row in source_obstacle_rows}
    required_names = set(physical_ball_scene.TABLE_ASSEMBLY_GEOM_NAMES)
    if (
        set(source_by_name) != required_names
        or set(compiled_obstacles) != required_names
        or set(compiled_obstacle_ids) != required_names
    ):
        raise ObservedOutcomeResolverError(
            "compiled/source table assembly does not contain exact five geoms"
        )
    for name in sorted(required_names):
        source_row = source_by_name[name]
        compiled_row = compiled_obstacles.get(name)
        if (
            not isinstance(compiled_row, Mapping)
            or compiled_row.get("name") != source_row.get("name")
            or compiled_row.get("geom_id") != compiled_obstacle_ids.get(name)
            or compiled_row.get("body_id") != 0
            or compiled_row.get("primitive")
            != "axis_aligned_box_full_extents_m"
            or _vector(
                compiled_row.get("center_mjcf_world_m"),
                3,
                f"compiled {name} center",
            )
            != _vector(
                source_row.get("center_mjcf_world_m"),
                3,
                f"source {name} center",
            )
            or _vector(
                compiled_row.get("full_extents_m"),
                3,
                f"compiled {name} extents",
            )
            != _vector(
                source_row.get("full_extents_m"),
                3,
                f"source {name} extents",
            )
        ):
            raise ObservedOutcomeResolverError(
                f"compiled obstacle {name!r} differs from table source authority"
            )
    table_lo, table_hi = _box(
        compiled_obstacles.get(physical_ball_scene.TABLE_GEOM_NAME),
        physical_ball_scene.TABLE_GEOM_NAME,
    )
    net_lo, net_hi = _box(
        compiled_obstacles.get(physical_ball_scene.NET_GEOM_NAMES[0]),
        physical_ball_scene.NET_GEOM_NAMES[0],
    )
    ball = scene_binding.get("ball")
    if not isinstance(ball, Mapping):
        raise ObservedOutcomeResolverError("scene ball contract is absent")
    radius = _finite(compiled.get("ball_radius_m"), "compiled ball radius")
    if radius <= 0.0:
        raise ObservedOutcomeResolverError("ball radius must be positive")
    if radius != _finite(ball.get("radius_m"), "source ball radius"):
        raise ObservedOutcomeResolverError(
            "compiled ball radius differs from source ball contract"
        )
    physics_dt = _finite(compiled.get("model_timestep_s"), "model timestep")
    if physics_dt <= 0.0:
        raise ObservedOutcomeResolverError("model timestep must be positive")
    if type(control_decimation) is not int or control_decimation < 1:
        raise ObservedOutcomeResolverError(
            "control decimation must be a positive plain int"
        )
    plant_sha = _plain_sha256(plant_binding_sha256, "plant binding SHA")
    policy_dt = _finite(policy_step_dt_s, "policy step dt")
    if not math.isclose(
        policy_dt,
        physics_dt * control_decimation,
        rel_tol=0.0,
        abs_tol=TIME_DELTA_ABS_TOLERANCE_S,
    ):
        raise ObservedOutcomeResolverError(
            "policy step dt differs from physics dt times control decimation"
        )
    if not (
        table_lo[0] + radius < table_hi[0] - radius
        and table_lo[1] + radius < table_hi[1] - radius
    ):
        raise ObservedOutcomeResolverError(
            "ball-radius-eroded table footprint is empty"
        )
    near_x, _surface_z = table_scene.virtual_table_pose()
    if float(near_x) != float(table_lo[0]):
        raise ObservedOutcomeResolverError(
            "table near-side source differs from scene table bounds"
        )

    geometry = {
        "table_x_bounds_w_m": [float(table_lo[0]), float(table_hi[0])],
        "table_y_bounds_w_m": [float(table_lo[1]), float(table_hi[1])],
        "table_surface_z_w_m": float(table_hi[2]),
        "net_plane_x_w_m": float(0.5 * (net_lo[0] + net_hi[0])),
        "net_top_z_w_m": float(net_hi[2]),
        "ball_radius_m": radius,
        "required_ball_center_net_clear_z_w_m": float(net_hi[2] + radius),
        "legal_ball_center_x_bounds_w_m": [
            float(table_lo[0] + radius),
            float(table_hi[0] - radius),
        ],
        "legal_ball_center_y_bounds_w_m": [
            float(table_lo[1] + radius),
            float(table_hi[1] - radius),
        ],
        "robot_near_table_x_w_m": float(near_x),
        "opponent_direction_x": 1,
    }
    payload = {
        "schema_version": 2,
        "kind": RESOLVER_BINDING_KIND,
        "resolver_source_path": (
            Path(__file__).resolve().relative_to(REPO_ROOT).as_posix()
        ),
        "resolver_source_sha256": _sha256(Path(__file__).read_bytes()),
        "physical_ball_scene_source_path": (
            Path(physical_ball_scene.__file__)
            .resolve()
            .relative_to(REPO_ROOT)
            .as_posix()
        ),
        "physical_ball_scene_source_sha256": _sha256(
            Path(physical_ball_scene.__file__).read_bytes()
        ),
        "table_scene_source_path": physical_ball_scene.TABLE_SCENE_PY.resolve()
        .relative_to(REPO_ROOT)
        .as_posix(),
        "table_scene_source_sha256": _sha256(
            physical_ball_scene.TABLE_SCENE_PY.read_bytes()
        ),
        "scene_binding_sha256": scene_sha,
        "plant_binding_sha256": plant_sha,
        "assembled_xml_sha256": _plain_sha256(
            scene_binding.get("assembled_xml_sha256"), "assembled XML SHA"
        ),
        "canonical_mjcf_sha256": _plain_sha256(
            scene_binding.get("canonical_mjcf_sha256"), "canonical MJCF SHA"
        ),
        "table_geometry_contract_sha256": geometry_contract["sha256"],
        "ball_contract_source_sha256": _plain_sha256(
            scene_binding.get("ball_contract_source", {}).get("sha256"),
            "ball contract source SHA",
        ),
        "mujoco_backend_version": backend,
        "physics_step_dt_s": physics_dt,
        "policy_step_dt_s": policy_dt,
        "control_decimation": control_decimation,
        "time_delta_abs_tolerance_s": TIME_DELTA_ABS_TOLERANCE_S,
        "geometry": geometry,
        "semantics": SEMANTICS,
    }
    return _seal(payload)


def validate_resolver_binding_seal(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "schema_version",
        "kind",
        "resolver_source_path",
        "resolver_source_sha256",
        "physical_ball_scene_source_path",
        "physical_ball_scene_source_sha256",
        "table_scene_source_path",
        "table_scene_source_sha256",
        "scene_binding_sha256",
        "plant_binding_sha256",
        "assembled_xml_sha256",
        "canonical_mjcf_sha256",
        "table_geometry_contract_sha256",
        "ball_contract_source_sha256",
        "mujoco_backend_version",
        "physics_step_dt_s",
        "policy_step_dt_s",
        "control_decimation",
        "time_delta_abs_tolerance_s",
        "geometry",
        "semantics",
        "content_sha256",
    }
    payload = _validate_seal(
        value,
        expected_kind=RESOLVER_BINDING_KIND,
        expected_schema_version=2,
        expected_keys=expected,
        name="observed-outcome resolver binding",
    )
    for key in (
        "resolver_source_sha256",
        "physical_ball_scene_source_sha256",
        "table_scene_source_sha256",
        "scene_binding_sha256",
        "plant_binding_sha256",
        "assembled_xml_sha256",
        "canonical_mjcf_sha256",
        "table_geometry_contract_sha256",
        "ball_contract_source_sha256",
    ):
        _plain_sha256(payload[key], f"resolver binding {key}")
    expected_resolver_path = (
        Path(__file__).resolve().relative_to(REPO_ROOT).as_posix()
    )
    expected_table_scene_path = (
        physical_ball_scene.TABLE_SCENE_PY.resolve()
        .relative_to(REPO_ROOT)
        .as_posix()
    )
    expected_physical_scene_path = (
        Path(physical_ball_scene.__file__)
        .resolve()
        .relative_to(REPO_ROOT)
        .as_posix()
    )
    if (
        payload["resolver_source_path"] != expected_resolver_path
        or payload["resolver_source_sha256"]
        != _sha256(Path(__file__).read_bytes())
        or payload["physical_ball_scene_source_path"]
        != expected_physical_scene_path
        or payload["physical_ball_scene_source_sha256"]
        != _sha256(Path(physical_ball_scene.__file__).read_bytes())
        or payload["table_scene_source_path"] != expected_table_scene_path
        or payload["table_scene_source_sha256"]
        != _sha256(physical_ball_scene.TABLE_SCENE_PY.read_bytes())
    ):
        raise ObservedOutcomeResolverError(
            "resolver binding differs from current source authority"
        )
    if (
        type(payload["mujoco_backend_version"]) is not str
        or not payload["mujoco_backend_version"].strip()
        or payload["semantics"] != SEMANTICS
    ):
        raise ObservedOutcomeResolverError(
            "resolver binding backend/semantics differ"
        )
    geometry = payload.get("geometry")
    if not isinstance(geometry, Mapping) or set(geometry) != {
        "table_x_bounds_w_m",
        "table_y_bounds_w_m",
        "table_surface_z_w_m",
        "net_plane_x_w_m",
        "net_top_z_w_m",
        "ball_radius_m",
        "required_ball_center_net_clear_z_w_m",
        "legal_ball_center_x_bounds_w_m",
        "legal_ball_center_y_bounds_w_m",
        "robot_near_table_x_w_m",
        "opponent_direction_x",
    }:
        raise ObservedOutcomeResolverError("resolver binding geometry keys differ")
    bounds: dict[str, list[float]] = {}
    for key in (
        "table_x_bounds_w_m",
        "table_y_bounds_w_m",
        "legal_ball_center_x_bounds_w_m",
        "legal_ball_center_y_bounds_w_m",
    ):
        lo, hi = _vector(geometry[key], 2, f"geometry.{key}")
        if not lo < hi:
            raise ObservedOutcomeResolverError(f"geometry.{key} is empty")
        bounds[key] = [lo, hi]
    radius = _finite(geometry["ball_radius_m"], "geometry.ball_radius_m")
    if radius <= 0.0:
        raise ObservedOutcomeResolverError("geometry ball radius must be positive")
    for key in (
        "table_surface_z_w_m",
        "net_plane_x_w_m",
        "net_top_z_w_m",
        "required_ball_center_net_clear_z_w_m",
        "robot_near_table_x_w_m",
    ):
        _finite(geometry[key], f"geometry.{key}")
    if geometry["opponent_direction_x"] != 1:
        raise ObservedOutcomeResolverError(
            "resolver opponent direction differs from near-side table authority"
        )
    if geometry["robot_near_table_x_w_m"] != geometry[
        "table_x_bounds_w_m"
    ][0]:
        raise ObservedOutcomeResolverError(
            "resolver near-side table derivation differs"
        )
    if bounds["legal_ball_center_x_bounds_w_m"] != [
        bounds["table_x_bounds_w_m"][0] + radius,
        bounds["table_x_bounds_w_m"][1] - radius,
    ] or bounds["legal_ball_center_y_bounds_w_m"] != [
        bounds["table_y_bounds_w_m"][0] + radius,
        bounds["table_y_bounds_w_m"][1] - radius,
    ]:
        raise ObservedOutcomeResolverError(
            "resolver legal footprint is not derived from table and ball radius"
        )
    if not (
        bounds["table_x_bounds_w_m"][0]
        < geometry["net_plane_x_w_m"]
        < bounds["table_x_bounds_w_m"][1]
        and geometry["table_surface_z_w_m"] < geometry["net_top_z_w_m"]
    ):
        raise ObservedOutcomeResolverError(
            "resolver net plane/top differ from table geometry"
        )
    expected_required_z = _finite(
        geometry["net_top_z_w_m"], "geometry.net_top_z_w_m"
    ) + radius
    if geometry["required_ball_center_net_clear_z_w_m"] != expected_required_z:
        raise ObservedOutcomeResolverError("resolver net-clear height derivation differs")
    if _finite(payload["physics_step_dt_s"], "physics_step_dt_s") <= 0.0:
        raise ObservedOutcomeResolverError("physics step must be positive")
    if (
        type(payload["control_decimation"]) is not int
        or payload["control_decimation"] < 1
    ):
        raise ObservedOutcomeResolverError(
            "resolver control decimation must be a positive plain int"
        )
    if not math.isclose(
        _finite(payload["policy_step_dt_s"], "policy_step_dt_s"),
        payload["physics_step_dt_s"] * payload["control_decimation"],
        rel_tol=0.0,
        abs_tol=TIME_DELTA_ABS_TOLERANCE_S,
    ):
        raise ObservedOutcomeResolverError(
            "resolver policy/physics timing derivation differs"
        )
    if (
        payload["time_delta_abs_tolerance_s"]
        != TIME_DELTA_ABS_TOLERANCE_S
    ):
        raise ObservedOutcomeResolverError(
            "resolver time-delta tolerance differs"
        )
    return payload


def validate_resolver_binding(
    value: Mapping[str, Any],
    *,
    expected_scene_binding: Mapping[str, Any],
    expected_obstacle_rows: Mapping[str, Any],
    expected_plant_binding_sha256: str,
    expected_policy_step_dt_s: float,
    expected_control_decimation: int,
    expected_resolver_source_sha256: str,
) -> dict[str, Any]:
    """Rebuild a binding from external compiled-scene/table authority."""

    canonical = validate_resolver_binding_seal(value)
    if canonical["resolver_source_sha256"] != _plain_sha256(
        expected_resolver_source_sha256,
        "expected resolver source SHA",
    ):
        raise ObservedOutcomeResolverError(
            "resolver binding differs from external resolver source authority"
        )
    expected = build_resolver_binding(
        scene_binding=expected_scene_binding,
        obstacle_rows=expected_obstacle_rows,
        plant_binding_sha256=expected_plant_binding_sha256,
        policy_step_dt_s=expected_policy_step_dt_s,
        control_decimation=expected_control_decimation,
    )
    if canonical != expected:
        raise ObservedOutcomeResolverError(
            "resolver binding differs from external scene/table authority"
        )
    return expected


def bind_question(
    *,
    resolver_binding: Mapping[str, Any],
    question_source_sha256: str,
    landing_aim_xy_w_m: Sequence[float],
    action_lineage_sha256: str | None,
) -> dict[str, Any]:
    """Bind one exact question without using its desired aim as outcome truth."""

    binding = validate_resolver_binding_seal(resolver_binding)
    question_sha = _plain_sha256(question_source_sha256, "question source SHA")
    aim = _vector(landing_aim_xy_w_m, 2, "landing aim")
    geometry = binding["geometry"]
    direction = geometry["opponent_direction_x"]
    lineage_sha = (
        None
        if action_lineage_sha256 is None
        else _plain_sha256(action_lineage_sha256, "action lineage SHA")
    )
    payload = {
        "schema_version": 2,
        "kind": QUESTION_BINDING_KIND,
        "resolver_binding_sha256": binding["content_sha256"],
        "scene_binding_sha256": binding["scene_binding_sha256"],
        "question_source_sha256": question_sha,
        "action_lineage_sha256": lineage_sha,
        "landing_aim_xy_w_m": aim,
        "opponent_direction_x": direction,
    }
    return _seal(payload)


def validate_question_binding_seal(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "schema_version",
        "kind",
        "resolver_binding_sha256",
        "scene_binding_sha256",
        "question_source_sha256",
        "action_lineage_sha256",
        "landing_aim_xy_w_m",
        "opponent_direction_x",
        "content_sha256",
    }
    payload = _validate_seal(
        value,
        expected_kind=QUESTION_BINDING_KIND,
        expected_schema_version=2,
        expected_keys=expected,
        name="observed-outcome question binding",
    )
    for key in (
        "resolver_binding_sha256",
        "scene_binding_sha256",
        "question_source_sha256",
    ):
        _plain_sha256(payload[key], f"question binding {key}")
    lineage = payload["action_lineage_sha256"]
    if lineage is not None:
        _plain_sha256(lineage, "question binding action lineage SHA")
    _vector(payload["landing_aim_xy_w_m"], 2, "question binding landing aim")
    if payload["opponent_direction_x"] != 1:
        raise ObservedOutcomeResolverError("question opponent direction differs")
    return payload


def validate_question_binding(
    value: Mapping[str, Any],
    *,
    resolver_binding: Mapping[str, Any],
    expected_question_source_sha256: str,
    expected_landing_aim_xy_w_m: Sequence[float],
    expected_action_lineage_sha256: str | None,
) -> dict[str, Any]:
    question = validate_question_binding_seal(value)
    binding = validate_resolver_binding_seal(resolver_binding)
    if (
        question["resolver_binding_sha256"] != binding["content_sha256"]
        or question["scene_binding_sha256"] != binding["scene_binding_sha256"]
    ):
        raise ObservedOutcomeResolverError(
            "question binding differs from current resolver/scene"
        )
    rebound = bind_question(
        resolver_binding=binding,
        question_source_sha256=expected_question_source_sha256,
        landing_aim_xy_w_m=expected_landing_aim_xy_w_m,
        action_lineage_sha256=expected_action_lineage_sha256,
    )
    if rebound != question:
        raise ObservedOutcomeResolverError(
            "observed-outcome question binding cannot be independently rebuilt"
        )
    return rebound


def _canonical_labels(value: Sequence[str]) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple, set)):
        raise ObservedOutcomeResolverError("active contact labels must be a sequence")
    labels = tuple(sorted(value))
    if len(set(labels)) != len(labels) or any(label not in CONTACT_LABELS for label in labels):
        raise ObservedOutcomeResolverError("active contact labels differ")
    return labels


def _canonical_sample(
    *,
    policy_tick: Any,
    physics_substep: Any,
    time_s: Any,
    ball_center_w_m: Any,
    active_contact_labels: Sequence[str],
) -> dict[str, Any]:
    return {
        "stamp": _stamp(policy_tick, physics_substep, "sample stamp"),
        "time_s": _finite(time_s, "sample time_s"),
        "ball_center_w_m": _vector(ball_center_w_m, 3, "sample ball center"),
        "active_contact_labels": list(_canonical_labels(active_contact_labels)),
    }


def _trace_sha256(samples: Sequence[Mapping[str, Any]]) -> str:
    return _sha256(_canonical_json_bytes({"samples": list(samples)}))


def _require_continuous_sample(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    resolver_binding: Mapping[str, Any],
) -> None:
    decimation = resolver_binding["control_decimation"]
    previous_tick, previous_substep = _stamp_tuple(previous["stamp"])
    current_tick, current_substep = _stamp_tuple(current["stamp"])
    if previous_substep >= decimation or current_substep >= decimation:
        raise ObservedOutcomeResolverError(
            "observed-outcome substep exceeds bound control decimation"
        )
    expected_stamp = (
        (previous_tick, previous_substep + 1)
        if previous_substep + 1 < decimation
        else (previous_tick + 1, 0)
    )
    if (current_tick, current_substep) != expected_stamp:
        raise ObservedOutcomeResolverError(
            "observed-outcome transcript is not substep-continuous"
        )
    delta = float(current["time_s"]) - float(previous["time_s"])
    if not math.isclose(
        delta,
        float(resolver_binding["physics_step_dt_s"]),
        rel_tol=0.0,
        abs_tol=float(resolver_binding["time_delta_abs_tolerance_s"]),
    ):
        raise ObservedOutcomeResolverError(
            "observed-outcome transcript time delta differs from physics step"
        )


def _crossing(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    question_binding: Mapping[str, Any],
    resolver_binding: Mapping[str, Any],
) -> dict[str, Any] | None:
    geometry = resolver_binding["geometry"]
    direction = question_binding["opponent_direction_x"]
    net_x = float(geometry["net_plane_x_w_m"])
    p0 = np.asarray(previous["ball_center_w_m"], dtype=np.float64)
    p1 = np.asarray(current["ball_center_w_m"], dtype=np.float64)
    before = direction * (float(p0[0]) - net_x)
    after = direction * (float(p1[0]) - net_x)
    if not (before <= 0.0 < after):
        return None
    denominator = float(p1[0] - p0[0])
    if denominator == 0.0:
        raise ObservedOutcomeResolverError("directed net crossing has zero x delta")
    alpha = float((net_x - float(p0[0])) / denominator)
    if not 0.0 <= alpha <= 1.0:
        raise ObservedOutcomeResolverError("net crossing interpolation left segment")
    point = p0 + alpha * (p1 - p0)
    time_s = float(
        previous["time_s"]
        + alpha * (current["time_s"] - previous["time_s"])
    )
    y_lo, y_hi = geometry["legal_ball_center_y_bounds_w_m"]
    required_z = float(geometry["required_ball_center_net_clear_z_w_m"])
    inside_width = bool(y_lo < float(point[1]) < y_hi)
    endpoint_envelope_certified = bool(
        y_lo < float(p0[1]) < y_hi
        and y_lo < float(p1[1]) < y_hi
        and float(p0[2]) > required_z
        and float(p1[2]) > required_z
    )
    cleared = bool(
        inside_width
        and float(point[2]) > required_z
        and endpoint_envelope_certified
    )
    return {
        "from_stamp": dict(previous["stamp"]),
        "to_stamp": dict(current["stamp"]),
        "from_ball_center_w_m": p0.tolist(),
        "to_ball_center_w_m": p1.tolist(),
        "from_time_s": float(previous["time_s"]),
        "to_time_s": float(current["time_s"]),
        "alpha": alpha,
        "time_s": time_s,
        "ball_center_w_m": point.tolist(),
        "required_center_z_w_m": required_z,
        "inside_strict_eroded_table_width": inside_width,
        "strict_segment_endpoint_envelope_certified": (
            endpoint_envelope_certified
        ),
        "cleared": cleared,
    }


def _landing(
    sample: Mapping[str, Any],
    *,
    question_binding: Mapping[str, Any],
    resolver_binding: Mapping[str, Any],
) -> dict[str, Any]:
    geometry = resolver_binding["geometry"]
    point = sample["ball_center_w_m"]
    x, y, z = (float(value) for value in point)
    x_lo, x_hi = geometry["legal_ball_center_x_bounds_w_m"]
    y_lo, y_hi = geometry["legal_ball_center_y_bounds_w_m"]
    net_x = float(geometry["net_plane_x_w_m"])
    radius = float(geometry["ball_radius_m"])
    direction = question_binding["opponent_direction_x"]
    strict_footprint = bool(x_lo < x < x_hi and y_lo < y < y_hi)
    target_half = bool(direction * (x - net_x) > radius)
    above_table_surface = bool(z > float(geometry["table_surface_z_w_m"]))
    return {
        "stamp": dict(sample["stamp"]),
        "time_s": float(sample["time_s"]),
        "ball_center_w_m": [x, y, z],
        "strict_ball_radius_eroded_footprint": strict_footprint,
        "strict_opponent_side_of_net": target_half,
        "ball_center_above_table_surface": above_table_surface,
        "native_top_landing_candidate": bool(
            strict_footprint and target_half and above_table_surface
        ),
    }


class ObservedOutcomeResolver:
    """Incremental deterministic resolver for one question/episode."""

    def __init__(
        self,
        *,
        resolver_binding: Mapping[str, Any],
        question_binding: Mapping[str, Any],
    ) -> None:
        self.resolver_binding = validate_resolver_binding_seal(resolver_binding)
        self.question_binding = validate_question_binding_seal(question_binding)
        if (
            self.question_binding["resolver_binding_sha256"]
            != self.resolver_binding["content_sha256"]
            or self.question_binding["scene_binding_sha256"]
            != self.resolver_binding["scene_binding_sha256"]
        ):
            raise ObservedOutcomeResolverError(
                "question binding differs from resolver/scene parent"
            )
        self._outgoing: dict[str, Any] | None = None
        self._previous: dict[str, Any] | None = None
        self._samples: list[dict[str, Any]] = []
        self._net_crossing: dict[str, Any] | None = None
        self._first_landing: dict[str, Any] | None = None
        self._status = STATUS_UNARMED
        self._outcome_stamp: dict[str, int] | None = None
        self._observed_net_clear: bool | None = None
        self._observed_legal_landing: bool | None = None
        self._fail_closed_reason: str | None = None
        self._terminal = False

    @property
    def armed(self) -> bool:
        return self._outgoing is not None

    def arm(
        self,
        outgoing_flight: Mapping[str, Any],
        *,
        active_contact_labels: Sequence[str] = (),
    ) -> None:
        if self._outgoing is not None:
            raise ObservedOutcomeResolverError("resolver is already armed")
        if not isinstance(outgoing_flight, Mapping):
            raise ObservedOutcomeResolverError("outgoing flight must be a mapping")
        sample = _canonical_sample(
            policy_tick=outgoing_flight.get("policy_tick"),
            physics_substep=outgoing_flight.get("physics_substep"),
            time_s=outgoing_flight.get("time_s"),
            ball_center_w_m=outgoing_flight.get("position_w_m"),
            active_contact_labels=active_contact_labels,
        )
        if sample["stamp"]["physics_substep"] >= self.resolver_binding[
            "control_decimation"
        ]:
            raise ObservedOutcomeResolverError(
                "outgoing substep exceeds bound control decimation"
            )
        self._outgoing = copy.deepcopy(sample)
        self._previous = copy.deepcopy(sample)
        self._samples.append(copy.deepcopy(sample))
        self._status = STATUS_TRACKING
        terminal = set(sample["active_contact_labels"]) & {"net", "table", "floor"}
        if terminal:
            self._status = STATUS_OUTGOING_OVERLAP_AMBIGUOUS
            self._fail_closed_reason = (
                "first_contact_free_outgoing_sample_overlaps_"
                + "_".join(sorted(terminal))
            )
            self._terminal = True

    def observe_substep(
        self,
        *,
        policy_tick: int,
        physics_substep: int,
        time_s: float,
        ball_center_w_m: Sequence[float],
        active_contact_labels: Sequence[str],
    ) -> None:
        if self._outgoing is None or self._previous is None:
            raise ObservedOutcomeResolverError(
                "resolver must be armed from achieved outgoing flight"
            )
        current = _canonical_sample(
            policy_tick=policy_tick,
            physics_substep=physics_substep,
            time_s=time_s,
            ball_center_w_m=ball_center_w_m,
            active_contact_labels=active_contact_labels,
        )
        _require_continuous_sample(
            self._previous,
            current,
            resolver_binding=self.resolver_binding,
        )
        self._samples.append(copy.deepcopy(current))
        if self._terminal:
            self._previous = current
            return
        crossing = None
        if self._net_crossing is None:
            crossing = _crossing(
                self._previous,
                current,
                question_binding=self.question_binding,
                resolver_binding=self.resolver_binding,
            )
        labels = set(current["active_contact_labels"])
        terminal_labels = labels & {"net", "table", "floor"}
        if len(terminal_labels) > 1:
            self._status = STATUS_SAME_SUBSTEP_AMBIGUOUS
            self._fail_closed_reason = (
                "multiple_terminal_contacts_"
                + "_".join(sorted(terminal_labels))
                + "_first_observed_in_same_physics_substep"
            )
            self._terminal = True
        elif crossing is not None and terminal_labels:
            self._status = STATUS_SAME_SUBSTEP_AMBIGUOUS
            self._fail_closed_reason = (
                "net_plane_crossing_and_"
                + "_".join(sorted(terminal_labels))
                + "_first_observed_in_same_physics_substep"
            )
            self._terminal = True
        else:
            if crossing is not None:
                self._net_crossing = crossing
            if "net" in labels:
                self._status = STATUS_NET_COLLISION
                self._outcome_stamp = dict(current["stamp"])
                self._observed_net_clear = False
                self._observed_legal_landing = False
                self._terminal = True
            elif "table" in labels:
                self._first_landing = _landing(
                    current,
                    question_binding=self.question_binding,
                    resolver_binding=self.resolver_binding,
                )
                net_clear = bool(
                    self._net_crossing is not None
                    and self._net_crossing["cleared"]
                )
                legal = bool(
                    net_clear
                    and self._first_landing["native_top_landing_candidate"]
                )
                self._status = STATUS_FIRST_TABLE_LANDING
                self._outcome_stamp = dict(current["stamp"])
                self._observed_net_clear = net_clear
                self._observed_legal_landing = legal
                self._terminal = True
            elif "floor" in labels:
                self._status = STATUS_FLOOR_CONTACT
                self._outcome_stamp = dict(current["stamp"])
                self._observed_net_clear = bool(
                    self._net_crossing is not None
                    and self._net_crossing["cleared"]
                )
                self._observed_legal_landing = False
                self._terminal = True
        self._previous = current

    def snapshot(self) -> dict[str, Any]:
        payload = {
            "schema_version": 2,
            "kind": SNAPSHOT_KIND,
            "resolver_binding_sha256": self.resolver_binding["content_sha256"],
            "question_binding_sha256": self.question_binding["content_sha256"],
            "status": self._status,
            "armed": self.armed,
            "outcome_resolved": self._status in RESOLVED_STATUSES,
            "outgoing_sample": copy.deepcopy(self._outgoing),
            "last_sample_stamp": (
                None if self._previous is None else dict(self._previous["stamp"])
            ),
            "net_crossing": copy.deepcopy(self._net_crossing),
            "first_table_landing": copy.deepcopy(self._first_landing),
            "outcome_stamp": copy.deepcopy(self._outcome_stamp),
            "observed_net_clear": self._observed_net_clear,
            "observed_legal_landing": self._observed_legal_landing,
            "fail_closed_reason": self._fail_closed_reason,
            "sample_count": len(self._samples),
            "trace_sha256": _trace_sha256(self._samples),
            "last_sample": (
                None if self._previous is None else copy.deepcopy(self._previous)
            ),
            "transcript_samples": copy.deepcopy(self._samples),
        }
        return _seal(payload)


def validate_snapshot(
    value: Mapping[str, Any],
    *,
    question_binding: Mapping[str, Any],
    resolver_binding: Mapping[str, Any],
    expected_question_binding_sha256: str,
    expected_resolver_binding_sha256: str,
) -> dict[str, Any]:
    binding = validate_resolver_binding_seal(resolver_binding)
    question = validate_question_binding_seal(question_binding)
    if (
        binding["content_sha256"]
        != _plain_sha256(
            expected_resolver_binding_sha256,
            "expected resolver binding SHA",
        )
        or question["content_sha256"]
        != _plain_sha256(
            expected_question_binding_sha256,
            "expected question binding SHA",
        )
    ):
        raise ObservedOutcomeResolverError(
            "snapshot binding differs from external parent authority"
        )
    if (
        question["resolver_binding_sha256"] != binding["content_sha256"]
        or question["scene_binding_sha256"]
        != binding["scene_binding_sha256"]
    ):
        raise ObservedOutcomeResolverError(
            "snapshot question differs from resolver/scene parent"
        )
    expected = {
        "schema_version",
        "kind",
        "resolver_binding_sha256",
        "question_binding_sha256",
        "status",
        "armed",
        "outcome_resolved",
        "outgoing_sample",
        "last_sample_stamp",
        "net_crossing",
        "first_table_landing",
        "outcome_stamp",
        "observed_net_clear",
        "observed_legal_landing",
        "fail_closed_reason",
        "sample_count",
        "trace_sha256",
        "last_sample",
        "transcript_samples",
        "content_sha256",
    }
    payload = _validate_seal(
        value,
        expected_kind=SNAPSHOT_KIND,
        expected_schema_version=2,
        expected_keys=expected,
        name="observed-outcome snapshot",
    )
    if (
        payload["question_binding_sha256"] != question["content_sha256"]
        or payload["resolver_binding_sha256"]
        != question["resolver_binding_sha256"]
    ):
        raise ObservedOutcomeResolverError(
            "observed-outcome snapshot binding differs"
        )
    status = payload["status"]
    if status not in STATUSES:
        raise ObservedOutcomeResolverError("observed-outcome status differs")
    if (
        type(payload["armed"]) is not bool
        or type(payload["outcome_resolved"]) is not bool
    ):
        raise ObservedOutcomeResolverError("snapshot availability flags must be bool")
    if payload["outcome_resolved"] is not (status in RESOLVED_STATUSES):
        raise ObservedOutcomeResolverError("snapshot resolved flag disagrees with status")
    samples = payload["transcript_samples"]
    if type(samples) is not list:
        raise ObservedOutcomeResolverError(
            "snapshot transcript samples must be a JSON list"
        )
    if (
        type(payload["sample_count"]) is not int
        or payload["sample_count"] != len(samples)
        or payload["trace_sha256"] != _trace_sha256(samples)
        or payload["last_sample"] != (None if not samples else samples[-1])
    ):
        raise ObservedOutcomeResolverError(
            "snapshot transcript count/digest/last sample differs"
        )
    canonical_samples = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping) or set(sample) != {
            "stamp",
            "time_s",
            "ball_center_w_m",
            "active_contact_labels",
        }:
            raise ObservedOutcomeResolverError(
                f"snapshot transcript sample {index} keys differ"
            )
        canonical = _canonical_sample(
            policy_tick=sample["stamp"].get("policy_tick"),
            physics_substep=sample["stamp"].get("physics_substep"),
            time_s=sample["time_s"],
            ball_center_w_m=sample["ball_center_w_m"],
            active_contact_labels=sample["active_contact_labels"],
        )
        if canonical != sample:
            raise ObservedOutcomeResolverError(
                f"snapshot transcript sample {index} is not canonical"
            )
        if canonical_samples:
            _require_continuous_sample(
                canonical_samples[-1],
                canonical,
                resolver_binding=binding,
            )
        canonical_samples.append(canonical)
    if status == STATUS_UNARMED:
        if payload["armed"] or samples or any(
            payload[key] is not None
            for key in (
                "outgoing_sample",
                "last_sample_stamp",
                "net_crossing",
                "first_table_landing",
                "outcome_stamp",
                "observed_net_clear",
                "observed_legal_landing",
                "fail_closed_reason",
                "last_sample",
            )
        ):
            raise ObservedOutcomeResolverError("unarmed snapshot carries evidence")
        if (
            payload["sample_count"] != 0
            or payload["trace_sha256"] != _trace_sha256(())
        ):
            raise ObservedOutcomeResolverError(
                "unarmed snapshot transcript closure differs"
            )
        return payload
    if not payload["armed"] or not isinstance(payload["outgoing_sample"], Mapping):
        raise ObservedOutcomeResolverError("armed snapshot lacks outgoing sample")
    outgoing = payload["outgoing_sample"]
    if set(outgoing) != {
        "stamp",
        "time_s",
        "ball_center_w_m",
        "active_contact_labels",
    }:
        raise ObservedOutcomeResolverError("outgoing sample keys differ")
    _stamp_tuple(outgoing["stamp"])
    _finite(outgoing["time_s"], "outgoing sample time")
    _vector(outgoing["ball_center_w_m"], 3, "outgoing sample ball center")
    _canonical_labels(outgoing["active_contact_labels"])
    if payload["last_sample_stamp"] is None:
        raise ObservedOutcomeResolverError("armed snapshot lacks last stamp")
    if _stamp_tuple(payload["last_sample_stamp"]) < _stamp_tuple(outgoing["stamp"]):
        raise ObservedOutcomeResolverError("snapshot last stamp precedes outgoing")
    if not samples or samples[0] != outgoing:
        raise ObservedOutcomeResolverError(
            "snapshot transcript does not start at outgoing sample"
        )
    if payload["last_sample_stamp"] != payload["last_sample"]["stamp"]:
        raise ObservedOutcomeResolverError(
            "snapshot last stamp differs from sealed last sample"
        )
    if status in RESOLVED_STATUSES:
        if payload["outcome_stamp"] is None:
            raise ObservedOutcomeResolverError("resolved outcome lacks stamp")
        if _stamp_tuple(payload["outcome_stamp"]) <= _stamp_tuple(outgoing["stamp"]):
            raise ObservedOutcomeResolverError(
                "resolved outcome is not strictly after outgoing flight"
            )
        if type(payload["observed_net_clear"]) is not bool or type(
            payload["observed_legal_landing"]
        ) is not bool:
            raise ObservedOutcomeResolverError("resolved outcome lacks booleans")
        if payload["observed_legal_landing"] and not payload["observed_net_clear"]:
            raise ObservedOutcomeResolverError(
                "observed legal landing requires observed net clear"
            )
        if payload["fail_closed_reason"] is not None:
            raise ObservedOutcomeResolverError("resolved outcome carries ambiguity")
    elif status in AMBIGUOUS_STATUSES:
        if (
            type(payload["fail_closed_reason"]) is not str
            or not payload["fail_closed_reason"]
            or payload["outcome_stamp"] is not None
            or payload["observed_net_clear"] is not None
            or payload["observed_legal_landing"] is not None
        ):
            raise ObservedOutcomeResolverError(
                "ambiguous outcome carries resolved evidence"
            )
    elif status == STATUS_TRACKING and any(
        payload[key] is not None
        for key in (
            "outcome_stamp",
            "observed_net_clear",
            "observed_legal_landing",
            "fail_closed_reason",
        )
    ):
        raise ObservedOutcomeResolverError("tracking outcome carries terminal facts")
    if status == STATUS_FIRST_TABLE_LANDING:
        if not isinstance(payload["first_table_landing"], Mapping):
            raise ObservedOutcomeResolverError("table outcome lacks first landing")
        landing = payload["first_table_landing"]
        if set(landing) != {
            "stamp",
            "time_s",
            "ball_center_w_m",
            "strict_ball_radius_eroded_footprint",
            "strict_opponent_side_of_net",
            "ball_center_above_table_surface",
            "native_top_landing_candidate",
        }:
            raise ObservedOutcomeResolverError("first landing keys differ")
        landing_sample = {
            "stamp": _stamp(
                landing["stamp"].get("policy_tick"),
                landing["stamp"].get("physics_substep"),
                "first landing stamp",
            ),
            "time_s": _finite(landing["time_s"], "first landing time"),
            "ball_center_w_m": _vector(
                landing["ball_center_w_m"], 3, "first landing ball center"
            ),
        }
        if _landing(
            landing_sample,
            question_binding=question,
            resolver_binding=binding,
        ) != landing:
            raise ObservedOutcomeResolverError(
                "first landing cannot be independently rebuilt"
            )
        if payload["outcome_stamp"] != landing["stamp"]:
            raise ObservedOutcomeResolverError(
                "table outcome stamp differs from first landing"
            )
        if payload["observed_legal_landing"] is not bool(
            payload["observed_net_clear"]
            and payload["first_table_landing"].get(
                "native_top_landing_candidate"
            )
        ):
            raise ObservedOutcomeResolverError(
                "observed legal landing derivation differs"
            )
    elif payload["first_table_landing"] is not None:
        raise ObservedOutcomeResolverError("non-table outcome carries landing")
    if status == STATUS_NET_COLLISION and (
        payload["observed_net_clear"] is not False
        or payload["observed_legal_landing"] is not False
    ):
        raise ObservedOutcomeResolverError("net collision cannot be a clear return")
    if (
        status == STATUS_FLOOR_CONTACT
        and payload["observed_legal_landing"] is not False
    ):
        raise ObservedOutcomeResolverError("floor contact cannot be legal landing")
    crossing = payload["net_crossing"]
    if crossing is not None:
        if not isinstance(crossing, Mapping) or set(crossing) != {
            "from_stamp",
            "to_stamp",
            "from_ball_center_w_m",
            "to_ball_center_w_m",
            "from_time_s",
            "to_time_s",
            "alpha",
            "time_s",
            "ball_center_w_m",
            "required_center_z_w_m",
            "inside_strict_eroded_table_width",
            "strict_segment_endpoint_envelope_certified",
            "cleared",
        }:
            raise ObservedOutcomeResolverError("net crossing keys differ")
        previous = {
            "stamp": _stamp(
                crossing["from_stamp"].get("policy_tick"),
                crossing["from_stamp"].get("physics_substep"),
                "net crossing from stamp",
            ),
            "time_s": _finite(crossing["from_time_s"], "net crossing from time"),
            "ball_center_w_m": _vector(
                crossing["from_ball_center_w_m"],
                3,
                "net crossing from ball center",
            ),
        }
        current = {
            "stamp": _stamp(
                crossing["to_stamp"].get("policy_tick"),
                crossing["to_stamp"].get("physics_substep"),
                "net crossing to stamp",
            ),
            "time_s": _finite(crossing["to_time_s"], "net crossing to time"),
            "ball_center_w_m": _vector(
                crossing["to_ball_center_w_m"],
                3,
                "net crossing to ball center",
            ),
        }
        if (
            _stamp_tuple(current["stamp"]) <= _stamp_tuple(previous["stamp"])
            or current["time_s"] <= previous["time_s"]
            or _crossing(
                previous,
                current,
                question_binding=question,
                resolver_binding=binding,
            )
            != crossing
        ):
            raise ObservedOutcomeResolverError(
                "net crossing cannot be independently rebuilt"
            )
    expected_net_clear = bool(crossing is not None and crossing["cleared"])
    if status in (STATUS_FIRST_TABLE_LANDING, STATUS_FLOOR_CONTACT) and (
        payload["observed_net_clear"] is not expected_net_clear
    ):
        raise ObservedOutcomeResolverError(
            "resolved outcome net-clear boolean differs from crossing"
        )
    if status in AMBIGUOUS_STATUSES and crossing is not None:
        raise ObservedOutcomeResolverError(
            "ambiguous same-substep outcome cannot claim ordered net crossing"
        )
    replayed = ObservedOutcomeResolver(
        resolver_binding=binding,
        question_binding=question,
    )
    replayed.arm(
        {
            "policy_tick": samples[0]["stamp"]["policy_tick"],
            "physics_substep": samples[0]["stamp"]["physics_substep"],
            "time_s": samples[0]["time_s"],
            "position_w_m": samples[0]["ball_center_w_m"],
        },
        active_contact_labels=samples[0]["active_contact_labels"],
    )
    for sample in samples[1:]:
        replayed.observe_substep(
            policy_tick=sample["stamp"]["policy_tick"],
            physics_substep=sample["stamp"]["physics_substep"],
            time_s=sample["time_s"],
            ball_center_w_m=sample["ball_center_w_m"],
            active_contact_labels=sample["active_contact_labels"],
        )
    if replayed.snapshot() != payload:
        raise ObservedOutcomeResolverError(
            "snapshot cannot be rebuilt from its complete transcript"
        )
    return payload


def replay_trace(
    *,
    resolver_binding: Mapping[str, Any],
    question_binding: Mapping[str, Any],
    expected_scene_binding: Mapping[str, Any],
    expected_obstacle_rows: Mapping[str, Any],
    expected_plant_binding_sha256: str,
    expected_policy_step_dt_s: float,
    expected_control_decimation: int,
    expected_resolver_source_sha256: str,
    expected_question_source_sha256: str,
    expected_landing_aim_xy_w_m: Sequence[float],
    expected_action_lineage_sha256: str | None,
    outgoing_flight: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Replay a finite ordered transcript and return its sealed snapshot."""

    binding = validate_resolver_binding(
        resolver_binding,
        expected_scene_binding=expected_scene_binding,
        expected_obstacle_rows=expected_obstacle_rows,
        expected_plant_binding_sha256=expected_plant_binding_sha256,
        expected_policy_step_dt_s=expected_policy_step_dt_s,
        expected_control_decimation=expected_control_decimation,
        expected_resolver_source_sha256=expected_resolver_source_sha256,
    )
    question = validate_question_binding(
        question_binding,
        resolver_binding=binding,
        expected_question_source_sha256=expected_question_source_sha256,
        expected_landing_aim_xy_w_m=expected_landing_aim_xy_w_m,
        expected_action_lineage_sha256=expected_action_lineage_sha256,
    )
    resolver = ObservedOutcomeResolver(
        resolver_binding=binding,
        question_binding=question,
    )
    resolver.arm(outgoing_flight)
    for sample in samples:
        if not isinstance(sample, Mapping) or set(sample) != {
            "policy_tick",
            "physics_substep",
            "time_s",
            "ball_center_w_m",
            "active_contact_labels",
        }:
            raise ObservedOutcomeResolverError("replay sample keys differ")
        resolver.observe_substep(**sample)
    snapshot = resolver.snapshot()
    validate_snapshot(
        snapshot,
        question_binding=question,
        resolver_binding=binding,
        expected_question_binding_sha256=question["content_sha256"],
        expected_resolver_binding_sha256=binding["content_sha256"],
    )
    return snapshot


def summarize_snapshots(
    rows: Sequence[Mapping[str, Any]],
    *,
    question_binding_by_sha256: Mapping[str, Mapping[str, Any]],
    resolver_binding_by_sha256: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Produce exact status/count conservation over resolver snapshots."""

    counts = {status: 0 for status in STATUSES}
    armed = resolved = net_clear = legal = 0
    row_shas = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ObservedOutcomeResolverError(f"summary row {index} is not a mapping")
        question_sha = row.get("question_binding_sha256")
        question = question_binding_by_sha256.get(question_sha)
        if question is None:
            raise ObservedOutcomeResolverError(
                f"summary row {index} has no question binding authority"
            )
        binding = resolver_binding_by_sha256.get(
            question["resolver_binding_sha256"]
        )
        if binding is None:
            raise ObservedOutcomeResolverError(
                f"summary row {index} has no resolver binding authority"
            )
        canonical = validate_snapshot(
            row,
            question_binding=question,
            resolver_binding=binding,
            expected_question_binding_sha256=question_sha,
            expected_resolver_binding_sha256=(
                question["resolver_binding_sha256"]
            ),
        )
        counts[canonical["status"]] += 1
        armed += int(canonical["armed"])
        resolved += int(canonical["outcome_resolved"])
        net_clear += int(canonical["observed_net_clear"] is True)
        legal += int(canonical["observed_legal_landing"] is True)
        row_shas.append(canonical["content_sha256"])
    if sum(counts.values()) != len(rows):
        raise ObservedOutcomeResolverError("outcome status count does not close")
    if resolved != sum(counts[status] for status in RESOLVED_STATUSES):
        raise ObservedOutcomeResolverError("resolved outcome count does not close")
    if not 0 <= legal <= net_clear <= resolved <= armed <= len(rows):
        raise ObservedOutcomeResolverError("observed outcome numerator chain differs")
    payload = {
        "schema_version": 2,
        "kind": SUMMARY_KIND,
        "rows": len(rows),
        "armed": armed,
        "resolved": resolved,
        "unresolved": len(rows) - resolved,
        "observed_net_clear": net_clear,
        "observed_legal_landing": legal,
        "status_counts": counts,
        "row_content_sha256": row_shas,
        "sum_closure": {
            "status_counts_equal_rows": True,
            "resolved_status_counts_equal_resolved": True,
            "legal_le_net_clear_le_resolved_le_armed_le_rows": True,
        },
    }
    return _seal(payload)


__all__ = [
    "AMBIGUOUS_STATUSES",
    "ObservedOutcomeResolver",
    "ObservedOutcomeResolverError",
    "QUESTION_BINDING_KIND",
    "RESOLVED_STATUSES",
    "RESOLVER_BINDING_KIND",
    "SNAPSHOT_KIND",
    "STATUSES",
    "STATUS_FIRST_TABLE_LANDING",
    "STATUS_FLOOR_CONTACT",
    "STATUS_NET_COLLISION",
    "STATUS_OUTGOING_OVERLAP_AMBIGUOUS",
    "STATUS_SAME_SUBSTEP_AMBIGUOUS",
    "STATUS_TRACKING",
    "STATUS_UNARMED",
    "bind_question",
    "build_resolver_binding",
    "replay_trace",
    "summarize_snapshots",
    "validate_question_binding",
    "validate_question_binding_seal",
    "validate_resolver_binding",
    "validate_resolver_binding_seal",
    "validate_snapshot",
]
