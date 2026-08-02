#!/usr/bin/env python3
"""Matched CPU timing for the incremental cost of one native MuJoCo ball.

This is a diagnostic throughput benchmark, not a physics-validity gate.  It
compiles the same vendor A3 plus ActionBall five-solid table scene in two
forms: without a ball and with one 40 mm free-joint ball.  The ball model uses
the radius, mass and inertia coefficient from the supplied schema-3 training
contract and the native-contact collision semantics already used by
``mujoco_teacher_motion_native_ball_diagnostic.py``.

Three runtime arms separate the costs:

``ball_off``
    No ball body or free joint exists.
``ball_flight``
    The ball exists and moves in collision-free air.
``ball_contact``
    The same ball is launched downward over the table so native contacts are
    actually solved.

The N environments are independent ``MjData`` instances sharing one compiled
``MjModel``.  They are stepped sequentially by Python, so N=64 is a CPU
fallback measurement and MUST NOT be linearly extrapolated to a 4096-env GPU
trainer.  Scene compilation, reset, physics stepping and contact-active
substeps are reported separately.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import statistics
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


_SCRIPT_PATH = Path(__file__).resolve()
# A copied diagnostic script may be executed outside the repository when every
# source path is supplied explicitly.  Keep defaults convenient in-tree without
# making import itself depend on an in-tree parent depth.
REPO_ROOT = _SCRIPT_PATH.parents[3] if len(_SCRIPT_PATH.parents) > 3 else Path.cwd()
DEFAULT_MJCF = (
    REPO_ROOT
    / "agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/a3_pingpong/a3_pingpong.xml"
)
DEFAULT_CONTRACT = (
    REPO_ROOT
    / "configs/a3_vendor_runtime_authority_20260802_r9/"
    "bh_loop_c.shared_ready.training_contract.json"
)
TABLE_SCENE_PY = REPO_ROOT / "scripts/mujoco_table_scene.py"

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
ARMS = ("ball_off", "ball_flight", "ball_contact")


class BenchmarkError(RuntimeError):
    """The matched benchmark contract or scene is invalid."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _load_table_scene_module(path: Path | str = TABLE_SCENE_PY) -> Any:
    source = Path(path).expanduser().resolve()
    spec = importlib.util.spec_from_file_location(
        "_mujoco_ball_tax_table_scene", source
    )
    if spec is None or spec.loader is None:
        raise BenchmarkError(f"cannot import table scene from {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_ball_and_step_contract(path: Path | str) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        raw = source.read_bytes()
        payload = json.loads(raw)
        venue = payload["action_ball_training"]["runtime"]["counter_rally"][
            "venue_physics"
        ]
        dt = float(payload["physics_step_dt_s"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"cannot read ball/step contract {source}: {exc}") from exc
    values = {
        "radius_m": float(venue["ball_radius_m"]),
        "mass_kg": float(venue["ball_mass_kg"]),
        "inertia_coeff": float(venue["ball_inertia_coeff"]),
        "physics_step_dt_s": dt,
    }
    if any(not math.isfinite(value) or value <= 0.0 for value in values.values()):
        raise BenchmarkError("ball/step contract values must be finite and positive")
    values.update({"path": str(source), "sha256": _sha256(raw)})
    return values


def assemble_scene_xml(
    canonical_xml: bytes,
    *,
    table_scene: Any,
    ball_contract: Mapping[str, Any],
    with_ball: bool,
) -> tuple[bytes, dict[str, Any]]:
    """Return one five-solid A3 scene, optionally with a native-contact ball."""

    rows = table_scene.action_ball_policy_obstacle_geometry()
    augmented = table_scene.augment_mjcf_xml(canonical_xml, rows, collidable=True)
    augmented = table_scene.append_action_ball_policy_keepout_xml(
        augmented, rows, collidable=True
    )
    try:
        root = ET.fromstring(augmented)
    except ET.ParseError as exc:
        raise BenchmarkError(f"cannot parse augmented vendor MJCF: {exc}") from exc
    option = root.find("./option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", format(float(ball_contract["physics_step_dt_s"]), ".17g"))

    if with_ball:
        worldbody = root.find("./worldbody")
        if worldbody is None:
            raise BenchmarkError("vendor MJCF has no worldbody")
        if root.find(f".//body[@name='{BALL_BODY_NAME}']") is not None:
            raise BenchmarkError("benchmark ball name already exists in vendor MJCF")
        radius = float(ball_contract["radius_m"])
        mass = float(ball_contract["mass_kg"])
        inertia = float(ball_contract["inertia_coeff"]) * mass * radius * radius
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
                "contype": "1",
                "conaffinity": "7",
                "condim": "3",
            },
        )
        contact = root.find("./contact")
        if contact is None:
            contact = ET.SubElement(root, "contact")
        pair_names = (
            ("benchmark_ball_racket", RACKET_GEOM_NAME),
            ("benchmark_ball_table", TABLE_GEOM_NAME),
            *((f"benchmark_ball_{name}", name) for name in NET_GEOM_NAMES),
        )
        for pair_name, other_geom in pair_names:
            if root.find(f".//geom[@name='{other_geom}']") is None:
                raise BenchmarkError(
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
    return final, {
        "canonical_mjcf_sha256": _sha256(canonical_xml),
        "assembled_xml_sha256": _sha256(final),
        "with_ball": bool(with_ball),
        "ball": {
            "radius_m": float(ball_contract["radius_m"]),
            "mass_kg": float(ball_contract["mass_kg"]),
            "inertia_coeff": float(ball_contract["inertia_coeff"]),
            "native_contact": bool(with_ball),
            "contact_material_status": "mujoco_pair_defaults_timing_only_not_fidelity",
        },
    }


def _model_assets(table_scene: Any, canonical_xml: bytes, mjcf_path: Path) -> dict[str, bytes]:
    # The table-scene module owns the exact vendor mesh path resolution used by
    # the native core.  Keeping this call here avoids a second, drifting loader.
    return table_scene._mesh_assets(canonical_xml, mjcf_path.parent)


def compile_model(
    mujoco: Any,
    *,
    mjcf_path: Path,
    table_scene: Any,
    ball_contract: Mapping[str, Any],
    with_ball: bool,
) -> tuple[Any, dict[str, Any], float]:
    canonical_xml = mjcf_path.read_bytes()
    xml, receipt = assemble_scene_xml(
        canonical_xml,
        table_scene=table_scene,
        ball_contract=ball_contract,
        with_ball=with_ball,
    )
    assets = _model_assets(table_scene, canonical_xml, mjcf_path)
    started = time.perf_counter_ns()
    model = mujoco.MjModel.from_xml_string(xml.decode("utf-8"), assets=assets)
    elapsed_s = (time.perf_counter_ns() - started) * 1.0e-9
    return model, receipt, elapsed_s


def _name_id(mujoco: Any, model: Any, kind: Any, name: str) -> int:
    value = int(mujoco.mj_name2id(model, kind, name))
    if value < 0:
        raise BenchmarkError(f"compiled model is missing {name!r}")
    return value


def _reset_one(
    mujoco: Any,
    model: Any,
    data: Any,
    *,
    arm: str,
    table_center: np.ndarray,
    table_top_z: float,
    ball_radius: float,
    ball_qpos_adr: int | None,
    ball_dof_adr: int | None,
) -> None:
    key_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand"))
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
        mujoco.mj_resetData(model, data)
    data.ctrl[:] = 0.0
    if arm != "ball_off":
        assert ball_qpos_adr is not None and ball_dof_adr is not None
        if arm == "ball_flight":
            # Above and laterally outside the table/robot.  The deliberately
            # high parking altitude keeps the full default/qualified timing
            # horizon collision-free under gravity; the receipt still proves
            # this with an independently counted zero-contact pass.
            position = np.asarray((table_center[0], 3.0, 100.0), np.float64)
            velocity = np.asarray((2.0, 0.0, 0.0), np.float64)
        elif arm == "ball_contact":
            position = np.asarray(
                (table_center[0] + 0.45, table_center[1], table_top_z + ball_radius + 0.002),
                np.float64,
            )
            velocity = np.asarray((0.0, 0.0, -2.0), np.float64)
        else:
            raise BenchmarkError(f"unknown arm {arm!r}")
        data.qpos[ball_qpos_adr : ball_qpos_adr + 3] = position
        data.qpos[ball_qpos_adr + 3 : ball_qpos_adr + 7] = (1.0, 0.0, 0.0, 0.0)
        data.qvel[ball_dof_adr : ball_dof_adr + 3] = velocity
        data.qvel[ball_dof_adr + 3 : ball_dof_adr + 6] = 0.0
    data.time = 0.0
    data.qacc_warmstart[:] = 0.0
    mujoco.mj_forward(model, data)


def _summary(samples: Sequence[float]) -> dict[str, float]:
    values = sorted(float(value) for value in samples)
    if not values:
        raise BenchmarkError("cannot summarize an empty sample set")
    return {
        "min_s": values[0],
        "median_s": statistics.median(values),
        "max_s": values[-1],
        "mean_s": statistics.fmean(values),
    }


def benchmark_arm(
    mujoco: Any,
    model: Any,
    *,
    arm: str,
    num_envs: int,
    physics_steps_per_reset: int,
    timed_resets: int,
    warmup_resets: int,
    warmup_env_steps: int,
    table_center: np.ndarray,
    table_top_z: float,
    ball_radius: float,
) -> dict[str, Any]:
    if arm not in ARMS or num_envs <= 0 or min(
        physics_steps_per_reset, timed_resets
    ) <= 0 or warmup_resets < 0 or warmup_env_steps < 0:
        raise BenchmarkError("invalid benchmark arm/counts")
    ball_qpos_adr = None
    ball_dof_adr = None
    ball_geom_id = None
    if arm != "ball_off":
        joint_id = _name_id(
            mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, BALL_JOINT_NAME
        )
        ball_qpos_adr = int(model.jnt_qposadr[joint_id])
        ball_dof_adr = int(model.jnt_dofadr[joint_id])
        ball_geom_id = _name_id(
            mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, BALL_GEOM_NAME
        )
    data_rows = [mujoco.MjData(model) for _ in range(num_envs)]

    def reset_all() -> None:
        for data in data_rows:
            _reset_one(
                mujoco,
                model,
                data,
                arm=arm,
                table_center=table_center,
                table_top_z=table_top_z,
                ball_radius=ball_radius,
                ball_qpos_adr=ball_qpos_adr,
                ball_dof_adr=ball_dof_adr,
            )

    effective_warmup_resets = max(
        warmup_resets,
        int(math.ceil(warmup_env_steps / float(num_envs * physics_steps_per_reset))),
    )
    for _ in range(effective_warmup_resets):
        reset_all()
        for _step in range(physics_steps_per_reset):
            for data in data_rows:
                mujoco.mj_step(model, data)

    reset_samples = []
    step_samples = []
    total_env_steps = num_envs * physics_steps_per_reset
    for _ in range(timed_resets):
        started = time.perf_counter_ns()
        reset_all()
        reset_samples.append((time.perf_counter_ns() - started) * 1.0e-9)
        started = time.perf_counter_ns()
        for _step in range(physics_steps_per_reset):
            for data in data_rows:
                mujoco.mj_step(model, data)
        step_samples.append((time.perf_counter_ns() - started) * 1.0e-9)

    # Contact validation is deliberately outside the timed samples.  Scanning
    # ``data.contact`` in only the ball arms would otherwise turn Python
    # bookkeeping into an apparent physics tax.
    reset_all()
    active_ball_contact_substeps = 0
    ball_contact_points = 0
    for _step in range(physics_steps_per_reset):
        for data in data_rows:
            mujoco.mj_step(model, data)
            if ball_geom_id is not None:
                ball_contacts = 0
                for contact_index in range(int(data.ncon)):
                    contact = data.contact[contact_index]
                    if int(contact.geom1) == ball_geom_id or int(contact.geom2) == ball_geom_id:
                        ball_contacts += 1
                active_ball_contact_substeps += int(ball_contacts > 0)
                ball_contact_points += ball_contacts
    step_summary = _summary(step_samples)
    reset_summary = _summary(reset_samples)
    median_step_s = step_summary["median_s"]
    return {
        "arm": arm,
        "num_envs": num_envs,
        "physics_steps_per_reset": physics_steps_per_reset,
        "timed_resets": timed_resets,
        "requested_minimum_warmup_resets": warmup_resets,
        "requested_minimum_warmup_env_steps": warmup_env_steps,
        "effective_warmup_resets": effective_warmup_resets,
        "effective_warmup_env_steps": (
            effective_warmup_resets * num_envs * physics_steps_per_reset
        ),
        "env_physics_steps_per_timed_sample": total_env_steps,
        "reset_wall_time": reset_summary,
        "physics_wall_time": step_summary,
        "median_us_per_env_physics_step": median_step_s * 1.0e6 / total_env_steps,
        "median_env_physics_steps_per_second": total_env_steps / median_step_s,
        "ball_contact_active_env_substeps": active_ball_contact_substeps,
        "ball_contact_points": ball_contact_points,
        "ball_contact_active_fraction": (
            active_ball_contact_substeps
            / float(total_env_steps)
        ),
        "contact_counter_passes": 1,
        "contact_counter_inside_timed_region": False,
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import mujoco
    except ImportError as exc:
        raise BenchmarkError("mujoco Python package is required") from exc
    mjcf = Path(args.mjcf).expanduser().resolve()
    if not mjcf.is_file():
        raise BenchmarkError(f"MJCF not found: {mjcf}")
    contract = load_ball_and_step_contract(args.contract)
    table_scene = _load_table_scene_module(args.table_scene)

    compile_samples: dict[str, list[float]] = {"ball_off": [], "ball_on": []}
    compiled: dict[str, tuple[Any, dict[str, Any]]] = {}
    for label, with_ball in (("ball_off", False), ("ball_on", True)):
        for _ in range(args.compile_repeats):
            model, receipt, elapsed = compile_model(
                mujoco,
                mjcf_path=mjcf,
                table_scene=table_scene,
                ball_contract=contract,
                with_ball=with_ball,
            )
            compile_samples[label].append(elapsed)
            compiled[label] = (model, receipt)

    off_model, off_receipt = compiled["ball_off"]
    on_model, on_receipt = compiled["ball_on"]
    table_id = _name_id(
        mujoco, on_model, mujoco.mjtObj.mjOBJ_GEOM, TABLE_GEOM_NAME
    )
    probe = mujoco.MjData(on_model)
    mujoco.mj_forward(on_model, probe)
    table_center = np.asarray(probe.geom_xpos[table_id], dtype=np.float64).copy()
    table_top_z = float(table_center[2] + on_model.geom_size[table_id, 2])

    rows = []
    for num_envs in args.num_envs:
        for arm in ARMS:
            model = off_model if arm == "ball_off" else on_model
            rows.append(
                benchmark_arm(
                    mujoco,
                    model,
                    arm=arm,
                    num_envs=num_envs,
                    physics_steps_per_reset=args.physics_steps_per_reset,
                    timed_resets=args.timed_resets,
                    warmup_resets=args.warmup_resets,
                    warmup_env_steps=args.warmup_env_steps,
                    table_center=table_center,
                    table_top_z=table_top_z,
                    ball_radius=float(contract["radius_m"]),
                )
            )
    by_n = {}
    for num_envs in args.num_envs:
        arm_rows = {row["arm"]: row for row in rows if row["num_envs"] == num_envs}
        off = arm_rows["ball_off"]["median_us_per_env_physics_step"]
        flight = arm_rows["ball_flight"]["median_us_per_env_physics_step"]
        contact = arm_rows["ball_contact"]["median_us_per_env_physics_step"]
        by_n[str(num_envs)] = {
            "ball_dof_tax_percent": 100.0 * (flight / off - 1.0),
            "contact_increment_over_flight_percent": 100.0 * (contact / flight - 1.0),
            "total_ball_contact_tax_percent": 100.0 * (contact / off - 1.0),
        }

    payload = {
        "schema_version": 1,
        "kind": "a3_mujoco_cpu_physical_ball_tax_benchmark_v1",
        "status": "DIAGNOSTIC_COMPLETE",
        "scope": "MuJoCo_CPU_sequential_MjData_only",
        "environment": {
            "mujoco_version": str(getattr(mujoco, "__version__", "unknown")),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "pid": os.getpid(),
        },
        "sources": {
            "mjcf": {"path": str(mjcf), "sha256": _sha256(mjcf.read_bytes())},
            "table_scene": {
                "path": str(Path(args.table_scene).expanduser().resolve()),
                "sha256": _sha256(Path(args.table_scene).expanduser().resolve().read_bytes()),
            },
            "training_contract": contract,
            "ball_off_scene": off_receipt,
            "ball_on_scene": on_receipt,
        },
        "fixed_protocol": {
            "num_envs": list(args.num_envs),
            "physics_step_dt_s": float(contract["physics_step_dt_s"]),
            "physics_steps_per_reset": args.physics_steps_per_reset,
            "timed_resets": args.timed_resets,
            "warmup_resets": args.warmup_resets,
            "minimum_warmup_env_steps_per_arm_and_stratum": args.warmup_env_steps,
            "compile_repeats": args.compile_repeats,
            "model_sharing": "one_MjModel_many_independent_MjData",
            "stepping": "sequential_python_CPU",
        },
        "scene_compile_wall_time": {
            label: _summary(samples) for label, samples in compile_samples.items()
        },
        "rows": rows,
        "matched_tax_by_num_envs": by_n,
        "interpretation_limits": {
            "isaac_or_physx_claim": False,
            "gpu_vectorized_claim": False,
            "ppo_update_claim": False,
            "4096_env_measurement": False,
            "linear_extrapolation_to_4096_allowed": False,
            "contact_material_fidelity_claim": False,
            "why": (
                "N<=64 sequential CPU MjData isolates native MuJoCo rigid-body/contact tax; "
                "it does not include observation, reward, PPO, GPU batching or Isaac/PhysX."
            ),
        },
        "authorization": {
            "training": False,
            "promotion": False,
            "deployment": False,
            "hardware": False,
        },
    }
    payload["content_sha256"] = _sha256(_canonical_json_bytes(payload))
    return payload


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _parse_num_envs(value: str) -> tuple[int, ...]:
    try:
        rows = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("num-envs must be comma-separated integers") from exc
    if not rows or any(item <= 0 for item in rows) or len(set(rows)) != len(rows):
        raise argparse.ArgumentTypeError("num-envs must be unique positive integers")
    return rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--table-scene", type=Path, default=TABLE_SCENE_PY)
    parser.add_argument("--num-envs", type=_parse_num_envs, default=(1, 8, 32, 64))
    parser.add_argument("--physics-steps-per-reset", type=_positive_int, default=100)
    parser.add_argument("--timed-resets", type=_positive_int, default=7)
    parser.add_argument("--warmup-resets", type=_nonnegative_int, default=0)
    parser.add_argument("--warmup-env-steps", type=_nonnegative_int, default=10_000)
    parser.add_argument("--compile-repeats", type=_positive_int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        payload = run_benchmark(args)
        encoded = _canonical_json_bytes(payload)
        output = args.out.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except (BenchmarkError, OSError, ValueError) as exc:
        print(f"[mujoco-ball-tax][ERROR] {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
