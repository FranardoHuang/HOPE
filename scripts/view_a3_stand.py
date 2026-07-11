#!/usr/bin/env python3
"""Inspect the vendor A3 standing under the tracked production PD_STAND contract.

This is a plain-MuJoCo diagnostic.  It does not start AimRT, a planner, a policy,
Gate3/Gate3B, or hardware.  The controller parses ``a3_default_angles`` and the
``a3_pd_stand_{kps,kds}`` arrays from the production header at runtime; the two
neck joints remain passive exactly as that 29-DOF contract specifies.

Examples:
  python scripts/view_a3_stand.py --identity-only
  python scripts/view_a3_stand.py --check --duration 10 --report-json /tmp/a3-stand.json
  python scripts/view_a3_stand.py --snapshot /tmp/a3-stand.png
  python scripts/view_a3_stand.py  # interactive viewer
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile


REPO = Path(__file__).resolve().parents[1]
DEFAULT_MJCF = (
    REPO
    / "agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/a3_pingpong/a3_pingpong.xml"
)
DEFAULT_GAIN_HEADER = (
    REPO
    / "agi/a3_deploy_example/src/a3/a3_deploy_onnx_ref/include/a3_policy_parameters.hpp"
)

# Exact 29-DOF MuJoCo policy-view order documented by a3_policy_parameters.hpp.
POLICY_JOINT_ORDER = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)
PASSIVE_JOINTS = ("head_yaw_joint", "head_pitch_joint")
FOOT_GEOMS = ("left_ankle_roll_collision", "right_ankle_roll_collision")

_ARRAY_RE = re.compile(
    r"constexpr\s+std::array\s*<\s*double\s*,\s*(\d+)\s*>\s*"
    r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\{(.*?)\};",
    re.DOTALL,
)
_NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


@dataclass(frozen=True)
class ProductionStandParameters:
    default_angles: tuple[float, ...]
    kps: tuple[float, ...]
    kds: tuple[float, ...]
    source_path: str
    source_sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_identity(path: Path) -> dict[str, str]:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise FileNotFoundError(f"source is not a regular file: {resolved}")
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def parse_cpp_double_arrays(text: str) -> dict[str, tuple[float, ...]]:
    """Parse literal ``constexpr std::array<double, N>`` initializers strictly."""

    arrays: dict[str, tuple[float, ...]] = {}
    for match in _ARRAY_RE.finditer(text):
        declared, name, body = int(match.group(1)), match.group(2), match.group(3)
        if name in arrays:
            raise ValueError(f"duplicate C++ array {name}")
        body = re.sub(r"/\*.*?\*/", "", body, flags=re.DOTALL)
        body = re.sub(r"//[^\n]*", "", body)
        residue = _NUMBER_RE.sub("", body)
        if re.sub(r"[\s,]", "", residue):
            raise ValueError(f"{name} contains a non-literal initializer token: {residue!r}")
        values = tuple(float(token) for token in _NUMBER_RE.findall(body))
        if len(values) != declared:
            raise ValueError(f"{name} declares {declared} values but parsed {len(values)}")
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{name} contains a non-finite value")
        arrays[name] = values
    return arrays


def load_production_parameters(path: Path = DEFAULT_GAIN_HEADER) -> ProductionStandParameters:
    identity = source_identity(path)
    text = Path(identity["path"]).read_text(encoding="utf-8")
    arrays = parse_cpp_double_arrays(text)
    required = ("a3_default_angles", "a3_pd_stand_kps", "a3_pd_stand_kds")
    missing = [name for name in required if name not in arrays]
    if missing:
        raise ValueError(f"production gain header is missing arrays: {missing}")
    for name in required:
        if len(arrays[name]) != len(POLICY_JOINT_ORDER):
            raise ValueError(
                f"{name} has {len(arrays[name])} entries, expected {len(POLICY_JOINT_ORDER)}"
            )
    return ProductionStandParameters(
        default_angles=arrays["a3_default_angles"],
        kps=arrays["a3_pd_stand_kps"],
        kds=arrays["a3_pd_stand_kds"],
        source_path=identity["path"],
        source_sha256=identity["sha256"],
    )


def contract_report(mjcf: Path, gain_header: Path) -> tuple[dict[str, object], ProductionStandParameters]:
    mjcf_id = source_identity(mjcf)
    params = load_production_parameters(gain_header)
    report: dict[str, object] = {
        "schema_version": 1,
        "claim_scope": "plain_mujoco_pd_stand_diagnostic_only",
        "gate3": "not_run",
        "hardware": "not_run",
        "sources": {
            "vendor_mjcf": mjcf_id,
            "production_parameter_header": {
                "path": params.source_path,
                "sha256": params.source_sha256,
            },
        },
        "controller_contract": {
            "pose_array": "a3_default_angles",
            "kp_array": "a3_pd_stand_kps",
            "kd_array": "a3_pd_stand_kds",
            "policy_joint_order": list(POLICY_JOINT_ORDER),
            "passive_joint_names": list(PASSIVE_JOINTS),
        },
    }
    return report, params


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True, allow_nan=False)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp_name, destination)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


class StandPD:
    """Production 29-DOF PD_STAND controller with passive neck slots."""

    def __init__(self, mujoco_module, model, params: ProductionStandParameters):
        self._mj = mujoco_module
        index = {name: i for i, name in enumerate(POLICY_JOINT_ORDER)}
        seen: set[str] = set()
        passive_seen: set[str] = set()
        rows = []
        for actuator_id in range(model.nu):
            joint_id = int(model.actuator_trnid[actuator_id, 0])
            if joint_id < 0:
                raise ValueError(f"actuator {actuator_id} is not bound to a joint")
            joint_name = mujoco_module.mj_id2name(
                model, mujoco_module.mjtObj.mjOBJ_JOINT, joint_id
            )
            if int(model.jnt_type[joint_id]) != int(mujoco_module.mjtJoint.mjJNT_HINGE):
                raise ValueError(f"PD_STAND joint must be a hinge: {joint_name}")
            if joint_name in PASSIVE_JOINTS:
                if joint_name in passive_seen:
                    raise ValueError(f"multiple actuators target passive neck joint {joint_name}")
                passive_seen.add(joint_name)
                rows.append((actuator_id, None, None, 0.0, 0.0, 0.0, None))
                continue
            if joint_name not in index:
                raise ValueError(f"actuator joint is outside production PD_STAND contract: {joint_name}")
            if joint_name in seen:
                raise ValueError(f"multiple actuators target production joint {joint_name}")
            seen.add(joint_name)
            policy_id = index[joint_name]
            qpos_address = int(model.jnt_qposadr[joint_id])
            dof_address = int(model.jnt_dofadr[joint_id])
            ctrl_range = None
            if bool(model.actuator_ctrllimited[actuator_id]):
                ctrl_range = (
                    float(model.actuator_ctrlrange[actuator_id, 0]),
                    float(model.actuator_ctrlrange[actuator_id, 1]),
                )
            rows.append(
                (
                    actuator_id,
                    qpos_address,
                    dof_address,
                    params.default_angles[policy_id],
                    params.kps[policy_id],
                    params.kds[policy_id],
                    ctrl_range,
                )
            )
        missing = sorted(set(POLICY_JOINT_ORDER) - seen)
        if missing:
            raise ValueError(f"vendor MJCF is missing production PD_STAND joints: {missing}")
        if passive_seen != set(PASSIVE_JOINTS):
            raise ValueError(
                f"vendor MJCF passive-neck set mismatch: got {sorted(passive_seen)}, "
                f"expected {list(PASSIVE_JOINTS)}"
            )
        self.rows = tuple(rows)

    def __call__(self, model, data) -> None:
        for actuator_id, qadr, vadr, q_ref, kp, kd, ctrl_range in self.rows:
            if qadr is None:
                effort = 0.0
            else:
                effort = kp * (q_ref - float(data.qpos[qadr])) - kd * float(data.qvel[vadr])
                if ctrl_range is not None:
                    effort = min(max(effort, ctrl_range[0]), ctrl_range[1])
            data.ctrl[actuator_id] = effort


def _named_id(mujoco_module, model, object_type, name: str) -> int:
    object_id = int(mujoco_module.mj_name2id(model, object_type, name))
    if object_id < 0:
        raise ValueError(f"vendor MJCF is missing required name {name!r}")
    return object_id


def _foot_contact_flags(data, floor_id: int, left_id: int, right_id: int) -> tuple[bool, bool]:
    left = False
    right = False
    for contact_id in range(int(data.ncon)):
        contact = data.contact[contact_id]
        pair = {int(contact.geom1), int(contact.geom2)}
        if floor_id not in pair:
            continue
        left = left or left_id in pair
        right = right or right_id in pair
    return left, right


def run_headless(mujoco_module, model, data, controller: StandPD, args) -> dict[str, object]:
    import numpy as np

    pelvis_id = _named_id(mujoco_module, model, mujoco_module.mjtObj.mjOBJ_BODY, "pelvis_link")
    floor_id = _named_id(mujoco_module, model, mujoco_module.mjtObj.mjOBJ_GEOM, "floor")
    left_id = _named_id(mujoco_module, model, mujoco_module.mjtObj.mjOBJ_GEOM, FOOT_GEOMS[0])
    right_id = _named_id(mujoco_module, model, mujoco_module.mjtObj.mjOBJ_GEOM, FOOT_GEOMS[1])

    timestep = float(model.opt.timestep)
    steps = max(1, int(math.ceil(args.duration / timestep)))
    print_stride = max(1, int(round(args.print_every / timestep)))
    initial_z = float(data.xpos[pelvis_id, 2])
    min_z = initial_z
    max_z = initial_z
    max_tilt_deg = 0.0
    left_samples = right_samples = both_samples = 0
    failures: list[str] = []

    for step in range(steps):
        controller(model, data)
        mujoco_module.mj_step(model, data)
        finite_arrays = (data.qpos, data.qvel, data.qacc, data.ctrl, data.actuator_force)
        if not all(bool(np.all(np.isfinite(values))) for values in finite_arrays):
            failures.append(f"non_finite_state_at_step_{step + 1}")
            break

        pelvis_z = float(data.xpos[pelvis_id, 2])
        min_z = min(min_z, pelvis_z)
        max_z = max(max_z, pelvis_z)
        upright_cos = float(np.clip(data.xmat[pelvis_id].reshape(3, 3)[2, 2], -1.0, 1.0))
        tilt_deg = math.degrees(math.acos(upright_cos))
        max_tilt_deg = max(max_tilt_deg, tilt_deg)
        left, right = _foot_contact_flags(data, floor_id, left_id, right_id)
        left_samples += int(left)
        right_samples += int(right)
        both_samples += int(left and right)

        if step == 0 or (step + 1) % print_stride == 0 or step + 1 == steps:
            print(
                f"t={float(data.time):7.3f}s pelvis_z={pelvis_z:.4f}m "
                f"tilt={tilt_deg:.3f}deg contacts=L{int(left)}R{int(right)}"
            )

    completed = step + 1
    denominator = max(1, completed)
    max_z_drift = max(abs(min_z - initial_z), abs(max_z - initial_z))
    both_fraction = both_samples / denominator
    if max_z_drift > args.max_z_drift:
        failures.append(f"pelvis_z_drift>{args.max_z_drift}")
    if max_tilt_deg > args.max_tilt_deg:
        failures.append(f"pelvis_tilt>{args.max_tilt_deg}")
    if both_fraction < args.min_both_contact_fraction:
        failures.append(f"both_foot_contact_fraction<{args.min_both_contact_fraction}")

    return {
        "duration_requested_s": args.duration,
        "duration_completed_s": float(data.time),
        "steps_requested": steps,
        "steps_completed": completed,
        "finite": not any(item.startswith("non_finite") for item in failures),
        "pelvis_z_initial_m": initial_z,
        "pelvis_z_min_m": min_z,
        "pelvis_z_max_m": max_z,
        "pelvis_z_max_drift_m": max_z_drift,
        "pelvis_tilt_max_deg": max_tilt_deg,
        "left_foot_contact_fraction": left_samples / denominator,
        "right_foot_contact_fraction": right_samples / denominator,
        "both_foot_contact_fraction": both_fraction,
        "thresholds": {
            "max_z_drift_m": args.max_z_drift,
            "max_tilt_deg": args.max_tilt_deg,
            "min_both_foot_contact_fraction": args.min_both_contact_fraction,
        },
        "diagnostic_pass": not failures,
        "failures": failures,
    }


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF)
    ap.add_argument("--gain-header", type=Path, default=DEFAULT_GAIN_HEADER)
    ap.add_argument("--keyframe", default="stand")
    ap.add_argument(
        "--identity-only",
        action="store_true",
        help="print source SHA/PD contract without importing MuJoCo or loading the model",
    )
    ap.add_argument("--check", action="store_true", help="run a headless stability diagnostic")
    ap.add_argument("--snapshot", type=Path, help="render a PNG after the headless diagnostic")
    ap.add_argument("--duration", type=float, default=10.0, help="headless simulated seconds")
    ap.add_argument("--print-every", type=float, default=1.0, help="headless progress cadence in seconds")
    ap.add_argument("--max-z-drift", type=float, default=0.15, help="diagnostic failure threshold, metres")
    ap.add_argument("--max-tilt-deg", type=float, default=20.0, help="diagnostic failure threshold")
    ap.add_argument(
        "--min-both-contact-fraction",
        type=float,
        default=0.90,
        help="minimum fraction of sampled steps with both foot hulls contacting floor",
    )
    ap.add_argument("--report-json", type=Path, help="optional root-source-bound diagnostic report")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    numeric_args = (
        args.duration,
        args.print_every,
        args.max_z_drift,
        args.max_tilt_deg,
        args.min_both_contact_fraction,
    )
    if not all(math.isfinite(value) for value in numeric_args):
        raise SystemExit("diagnostic duration, cadence and thresholds must all be finite")
    if args.duration <= 0.0 or args.print_every <= 0.0:
        raise SystemExit("--duration and --print-every must be positive")
    if args.max_z_drift < 0.0 or args.max_tilt_deg < 0.0:
        raise SystemExit("diagnostic maxima must be non-negative")
    if not 0.0 <= args.min_both_contact_fraction <= 1.0:
        raise SystemExit("--min-both-contact-fraction must be in [0, 1]")

    report, params = contract_report(args.mjcf, args.gain_header)
    print(json.dumps(report, sort_keys=True))
    if args.identity_only:
        report["mode"] = "identity_only"
        if args.report_json:
            write_json_atomic(args.report_json, report)
        return 0

    try:
        import mujoco
    except ImportError as exc:
        raise SystemExit(
            "MuJoCo Python bindings are required for simulation; --identity-only and --help "
            "remain dependency-light"
        ) from exc

    model = mujoco.MjModel.from_xml_path(str(Path(args.mjcf).expanduser().resolve(strict=True)))
    data = mujoco.MjData(model)
    key_id = _named_id(mujoco, model, mujoco.mjtObj.mjOBJ_KEY, args.keyframe)
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    controller = StandPD(mujoco, model, params)
    try:
        integrator = mujoco.mjtIntegrator(int(model.opt.integrator)).name
    except (TypeError, ValueError):
        integrator = str(int(model.opt.integrator))
    report["model"] = {
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "timestep_s": float(model.opt.timestep),
        "integrator": integrator,
        "keyframe": args.keyframe,
    }

    if args.check or args.snapshot:
        report["mode"] = "headless"
        report["diagnostic"] = run_headless(mujoco, model, data, controller, args)
        if args.snapshot:
            camera_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "torso_follow"))
            model.vis.global_.offwidth = 1280
            model.vis.global_.offheight = 720
            with mujoco.Renderer(model, height=720, width=1280) as renderer:
                renderer.update_scene(data, camera=camera_id if camera_id >= 0 else -1)
                try:
                    from PIL import Image
                except ImportError as exc:
                    raise SystemExit("Pillow is required for --snapshot") from exc
                snapshot = args.snapshot.expanduser().resolve()
                snapshot.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(renderer.render()).save(snapshot)
                report["snapshot"] = str(snapshot)
        if args.report_json:
            write_json_atomic(args.report_json, report)
        print(json.dumps(report["diagnostic"], sort_keys=True))
        return 0 if report["diagnostic"]["diagnostic_pass"] else 1

    report["mode"] = "interactive"
    if args.report_json:
        write_json_atomic(args.report_json, report)
    print("[diagnostic-only] interactive viewer; this is not a Gate3/Gate3B result")
    import mujoco.viewer as mj_viewer

    mujoco.set_mjcb_control(controller)
    try:
        mj_viewer.launch(model, data)
    finally:
        mujoco.set_mjcb_control(None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
