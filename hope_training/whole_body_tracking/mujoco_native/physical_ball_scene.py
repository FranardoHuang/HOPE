"""Shared native-MuJoCo physical-ball scene assembly.

The original ball-tax benchmark owned the first implementation of this scene.
This module makes the exact same default assembly reusable by a diagnostic N1
environment while preserving the benchmark's byte and contact semantics.

The N1 core opts into ``strict_pair_filter=True``.  In that mode the ball has
no mask-driven contacts: only the explicit racket/table/net/floor pairs can
touch it.  This is important because the fifth ActionBall obstacle is a
robot-only under-table keep-out and must never become a ball surface.

No aerodynamic or calibrated racket-contact model is installed here.  The
receipt says so explicitly; native MuJoCo contact is only an engineering
bring-up path, not a physics-fidelity or training authorization.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MJCF = (
    REPO_ROOT
    / "agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/a3_pingpong/a3_pingpong.xml"
)
TABLE_SCENE_PY = REPO_ROOT / "scripts/mujoco_table_scene.py"

# These names pre-date the production module.  They remain byte-stable so the
# ad1499df ball-tax benchmark keeps the same assembled XML semantics.
BALL_BODY_NAME = "benchmark_physical_ball_body"
BALL_JOINT_NAME = "benchmark_physical_ball_freejoint"
BALL_GEOM_NAME = "benchmark_physical_ball_geom"
RACKET_GEOM_NAME = "right_racket_collision"
TABLE_GEOM_NAME = "motion_table_top"
NET_GEOM_NAMES = (
    "motion_net",
    "motion_net_post_left",
    "motion_net_post_right",
)
FLOOR_GEOM_NAME = "floor"


class PhysicalBallSceneError(RuntimeError):
    """The physical-ball source or compiled scene is invalid."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _portable_binding_bytes(receipt: Mapping[str, Any]) -> bytes:
    """Hash semantic bytes/identities, never checkout-specific display paths."""

    payload = json.loads(_canonical_json_bytes(receipt).decode("utf-8"))
    payload.pop("binding_sha256", None)
    for key in ("ball_contract_source", "table_scene_source"):
        source = payload.get(key)
        if isinstance(source, dict):
            source.pop("path", None)
    return _canonical_json_bytes(payload)


def _reject_constant(value: str) -> None:
    raise PhysicalBallSceneError(f"non-finite JSON constant is forbidden: {value}")


def _unique_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise PhysicalBallSceneError(f"duplicate JSON key is forbidden: {key}")
        out[key] = value
    return out


def _load_table_scene_module(path: Path | str = TABLE_SCENE_PY) -> Any:
    source = Path(path).expanduser().resolve()
    spec = importlib.util.spec_from_file_location(
        "_mujoco_native_physical_ball_table_scene", source
    )
    if spec is None or spec.loader is None:
        raise PhysicalBallSceneError(f"cannot import table scene from {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class BallContract:
    source_path: str
    source_sha256: str
    radius_m: float
    mass_kg: float
    inertia_coeff: float
    physics_step_dt_s: float

    def as_mapping(self) -> dict[str, Any]:
        return {
            "path": self.source_path,
            "sha256": self.source_sha256,
            "radius_m": self.radius_m,
            "mass_kg": self.mass_kg,
            "inertia_coeff": self.inertia_coeff,
            "physics_step_dt_s": self.physics_step_dt_s,
        }


def load_ball_contract(path: Path | str) -> BallContract:
    source = Path(path).expanduser().resolve()
    try:
        raw = source.read_bytes()
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_pairs,
            parse_constant=_reject_constant,
        )
        venue = payload["action_ball_training"]["runtime"]["counter_rally"][
            "venue_physics"
        ]
        values = (
            float(venue["ball_radius_m"]),
            float(venue["ball_mass_kg"]),
            float(venue["ball_inertia_coeff"]),
            float(payload["physics_step_dt_s"]),
        )
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise PhysicalBallSceneError(
            f"cannot read ball/step contract {source}: {exc}"
        ) from exc
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise PhysicalBallSceneError("ball/step contract values must be finite and positive")
    return BallContract(
        source_path=str(source),
        source_sha256=_sha256(raw),
        radius_m=values[0],
        mass_kg=values[1],
        inertia_coeff=values[2],
        physics_step_dt_s=values[3],
    )


def _pair_targets(*, include_floor: bool) -> tuple[tuple[str, str], ...]:
    rows = (
        ("benchmark_ball_racket", RACKET_GEOM_NAME),
        ("benchmark_ball_table", TABLE_GEOM_NAME),
        *((f"benchmark_ball_{name}", name) for name in NET_GEOM_NAMES),
    )
    if include_floor:
        rows += (("benchmark_ball_floor", FLOOR_GEOM_NAME),)
    return rows


def assemble_scene_xml(
    canonical_xml: bytes,
    *,
    table_scene: Any,
    ball_contract: BallContract | Mapping[str, Any],
    with_ball: bool,
    strict_pair_filter: bool = False,
    include_floor_pair: bool = False,
) -> tuple[bytes, dict[str, Any]]:
    """Return the five-solid A3 scene, optionally with one free-joint ball.

    Defaults intentionally reproduce the original timing benchmark.  The N1
    core uses strict explicit pairs and includes the floor terminal surface.
    """

    contract = (
        ball_contract.as_mapping()
        if isinstance(ball_contract, BallContract)
        else dict(ball_contract)
    )
    required = ("radius_m", "mass_kg", "inertia_coeff", "physics_step_dt_s")
    try:
        numeric = {key: float(contract[key]) for key in required}
    except (KeyError, TypeError, ValueError) as exc:
        raise PhysicalBallSceneError(f"invalid ball contract mapping: {exc}") from exc
    if any(not math.isfinite(value) or value <= 0.0 for value in numeric.values()):
        raise PhysicalBallSceneError("ball contract mapping must be finite and positive")

    rows = table_scene.action_ball_policy_obstacle_geometry()
    geometry_contract = table_scene.action_ball_policy_geometry_contract(rows)
    table_scene_source = Path(table_scene.__file__).expanduser().resolve()
    augmented = table_scene.augment_mjcf_xml(canonical_xml, rows, collidable=True)
    augmented = table_scene.append_action_ball_policy_keepout_xml(
        augmented, rows, collidable=True
    )
    try:
        root = ET.fromstring(augmented)
    except ET.ParseError as exc:
        raise PhysicalBallSceneError(f"cannot parse augmented vendor MJCF: {exc}") from exc
    option = root.find("./option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", format(numeric["physics_step_dt_s"], ".17g"))

    pair_rows: tuple[tuple[str, str], ...] = ()
    if with_ball:
        worldbody = root.find("./worldbody")
        if worldbody is None:
            raise PhysicalBallSceneError("vendor MJCF has no worldbody")
        if root.find(f".//body[@name='{BALL_BODY_NAME}']") is not None:
            raise PhysicalBallSceneError("physical ball name already exists in vendor MJCF")
        radius = numeric["radius_m"]
        mass = numeric["mass_kg"]
        inertia = numeric["inertia_coeff"] * mass * radius * radius
        body = ET.SubElement(
            worldbody, "body", {"name": BALL_BODY_NAME, "pos": "0 0 100"}
        )
        ET.SubElement(
            body,
            "inertial",
            {
                "pos": "0 0 0",
                "mass": format(mass, ".17g"),
                "diaginertia": " ".join([format(inertia, ".17g")] * 3),
            },
        )
        ET.SubElement(body, "freejoint", {"name": BALL_JOINT_NAME})
        ET.SubElement(
            body,
            "geom",
            {
                "name": BALL_GEOM_NAME,
                "type": "sphere",
                "size": format(radius, ".17g"),
                "rgba": "1 0.5 0 1",
                "contype": "0" if strict_pair_filter else "1",
                "conaffinity": "0" if strict_pair_filter else "7",
                "condim": "3",
            },
        )
        contact = root.find("./contact")
        if contact is None:
            contact = ET.SubElement(root, "contact")
        pair_rows = _pair_targets(include_floor=include_floor_pair)
        for pair_name, other_geom in pair_rows:
            if root.find(f".//geom[@name='{other_geom}']") is None:
                raise PhysicalBallSceneError(
                    f"native ball contact pair references missing geom {other_geom!r}"
                )
            ET.SubElement(
                contact,
                "pair",
                {
                    "name": pair_name,
                    "geom1": BALL_GEOM_NAME,
                    "geom2": other_geom,
                    "condim": "3",
                },
            )
    final = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    receipt = {
        "schema_version": 1,
        "kind": "a3_mujoco_physical_ball_scene_binding_v1",
        "canonical_mjcf_sha256": _sha256(canonical_xml),
        "assembled_xml_sha256": _sha256(final),
        "ball_contract_source": {
            "path": contract.get("path"),
            "sha256": contract.get("sha256"),
        },
        "table_scene_source": {
            "path": str(table_scene_source),
            "sha256": _sha256(table_scene_source.read_bytes()),
        },
        "table_geometry_contract_sha256": geometry_contract["sha256"],
        "with_ball": bool(with_ball),
        "strict_pair_filter": bool(strict_pair_filter),
        "explicit_pair_targets": [value for _name, value in pair_rows],
        "robot_only_keepout_is_ball_surface": False if strict_pair_filter else None,
        "ball": {
            "radius_m": numeric["radius_m"],
            "mass_kg": numeric["mass_kg"],
            "inertia_coeff": numeric["inertia_coeff"],
            "native_contact": bool(with_ball),
            "aerodynamics": "not_implemented",
            "magnus_and_spin_flight": "not_implemented",
            "contact_material_status": "mujoco_pair_defaults_engineering_only",
        },
    }
    receipt["binding_sha256"] = _sha256(_portable_binding_bytes(receipt))
    return final, receipt


@dataclass(frozen=True)
class PhysicalBallScene:
    model: Any
    obstacle_geom_ids: dict[str, int]
    obstacle_names: tuple[str, ...]
    obstacle_rows: dict[str, Any]
    collidable: bool
    augmented_xml_sha256: str
    canonical_xml_sha256: str
    near_x: float
    surface_z: float
    geom_index_shift: int
    ball_body_id: int
    ball_joint_id: int
    ball_geom_id: int
    ball_qpos_adr: int
    ball_dof_adr: int
    binding: dict[str, Any]
    compile_wall_time_s: float

    def obstacle_of(self, geom_id: int) -> str | None:
        for name, value in self.obstacle_geom_ids.items():
            if int(value) == int(geom_id):
                return name
        return None


def compile_physical_ball_scene(
    mujoco: Any,
    *,
    mjcf_path: Path | str,
    ball_contract: BallContract,
    table_scene_path: Path | str = TABLE_SCENE_PY,
    strict_pair_filter: bool = True,
    include_floor_pair: bool = True,
) -> PhysicalBallScene:
    """Compile one exact A3 + five-solid table + native physical ball."""

    source = Path(mjcf_path).expanduser().resolve()
    if not source.is_file():
        raise PhysicalBallSceneError(f"MJCF not found: {source}")
    canonical = source.read_bytes()
    table_scene = _load_table_scene_module(table_scene_path)
    xml, receipt = assemble_scene_xml(
        canonical,
        table_scene=table_scene,
        ball_contract=ball_contract,
        with_ball=True,
        strict_pair_filter=strict_pair_filter,
        include_floor_pair=include_floor_pair,
    )
    assets = table_scene._mesh_assets(canonical, source.parent)
    started = time.perf_counter_ns()
    try:
        model = mujoco.MjModel.from_xml_string(xml.decode("utf-8"), assets=assets)
    except Exception as exc:  # noqa: BLE001 - MuJoCo uses extension exceptions
        raise PhysicalBallSceneError(f"cannot compile physical-ball scene: {exc}") from exc
    elapsed = (time.perf_counter_ns() - started) * 1.0e-9

    obstacle_names = tuple(table_scene.ACTION_BALL_POLICY_OBSTACLE_NAMES)
    obstacle_ids: dict[str, int] = {}
    for name in obstacle_names:
        geom_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name))
        if geom_id < 0:
            raise PhysicalBallSceneError(f"compiled scene is missing obstacle {name!r}")
        obstacle_ids[name] = geom_id

    def named_id(kind: Any, name: str) -> int:
        value = int(mujoco.mj_name2id(model, kind, name))
        if value < 0:
            raise PhysicalBallSceneError(f"compiled scene is missing {name!r}")
        return value

    body_id = named_id(mujoco.mjtObj.mjOBJ_BODY, BALL_BODY_NAME)
    joint_id = named_id(mujoco.mjtObj.mjOBJ_JOINT, BALL_JOINT_NAME)
    geom_id = named_id(mujoco.mjtObj.mjOBJ_GEOM, BALL_GEOM_NAME)
    if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise PhysicalBallSceneError("physical ball joint is not FREE")
    if int(model.jnt_bodyid[joint_id]) != body_id or int(model.geom_bodyid[geom_id]) != body_id:
        raise PhysicalBallSceneError("physical ball joint/geom do not belong to ball body")
    if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_SPHERE):
        raise PhysicalBallSceneError("physical ball geom is not a sphere")
    if not math.isclose(
        float(model.geom_size[geom_id, 0]), ball_contract.radius_m,
        rel_tol=0.0, abs_tol=1.0e-12,
    ) or not math.isclose(
        float(model.body_mass[body_id]), ball_contract.mass_kg,
        rel_tol=0.0, abs_tol=1.0e-12,
    ):
        raise PhysicalBallSceneError("compiled ball radius/mass differ from contract")
    expected_inertia = (
        ball_contract.inertia_coeff
        * ball_contract.mass_kg
        * ball_contract.radius_m
        * ball_contract.radius_m
    )
    if not all(
        math.isclose(float(value), expected_inertia, rel_tol=0.0, abs_tol=1.0e-12)
        for value in model.body_inertia[body_id]
    ):
        raise PhysicalBallSceneError("compiled ball inertia differs from contract")
    if strict_pair_filter and (
        int(model.geom_contype[geom_id]) != 0
        or int(model.geom_conaffinity[geom_id]) != 0
    ):
        raise PhysicalBallSceneError("strict physical ball must disable mask contacts")
    expected_pair_ids = {
        named_id(mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in receipt["explicit_pair_targets"]
    }
    compiled_pair_ids: set[int] = set()
    for pair_index in range(int(model.npair)):
        g1 = int(model.pair_geom1[pair_index])
        g2 = int(model.pair_geom2[pair_index])
        if geom_id in (g1, g2):
            compiled_pair_ids.add(g2 if g1 == geom_id else g1)
    if compiled_pair_ids != expected_pair_ids:
        raise PhysicalBallSceneError("compiled physical-ball explicit pair set differs")
    compiled_binding = dict(receipt)
    compiled_binding.pop("binding_sha256")
    compiled_binding["compiled_runtime"] = {
        "mujoco_version": str(getattr(mujoco, "__version__", "unknown")),
        "model_nq": int(model.nq),
        "model_nv": int(model.nv),
        "model_nbody": int(model.nbody),
        "model_ngeom": int(model.ngeom),
        "model_timestep_s": float(model.opt.timestep),
        "ball_body_id": body_id,
        "ball_joint_id": joint_id,
        "ball_geom_id": geom_id,
        "ball_qpos_adr": int(model.jnt_qposadr[joint_id]),
        "ball_dof_adr": int(model.jnt_dofadr[joint_id]),
        "obstacle_geom_ids": obstacle_ids,
        "mesh_source_closure_sha256": {
            name: _sha256(raw) for name, raw in sorted(assets.items())
        },
    }
    compiled_binding["binding_sha256"] = _sha256(
        _portable_binding_bytes(compiled_binding)
    )
    return PhysicalBallScene(
        model=model,
        obstacle_geom_ids=obstacle_ids,
        obstacle_names=obstacle_names,
        obstacle_rows=table_scene.action_ball_policy_obstacle_geometry(),
        collidable=True,
        augmented_xml_sha256=receipt["assembled_xml_sha256"],
        canonical_xml_sha256=receipt["canonical_mjcf_sha256"],
        near_x=float(table_scene.virtual_table_pose()[0]),
        surface_z=float(table_scene.virtual_table_pose()[1]),
        geom_index_shift=len(obstacle_names),
        ball_body_id=body_id,
        ball_joint_id=joint_id,
        ball_geom_id=geom_id,
        ball_qpos_adr=int(model.jnt_qposadr[joint_id]),
        ball_dof_adr=int(model.jnt_dofadr[joint_id]),
        binding=compiled_binding,
        compile_wall_time_s=elapsed,
    )
