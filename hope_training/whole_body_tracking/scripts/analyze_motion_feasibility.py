#!/usr/bin/env python3
"""Say what a captured motion would need changed before an A3 could track it.

This runs downstream of the retarget: give it a materialised motion NPZ (or a whole bank) and it
reports, per joint, what makes the motion hard and what to do about it. It is meant to be pointed
at a fresh capture session so the answer arrives before the next one is recorded.

Every finding carries a remedy, because the same number means different things:

* a joint past its position limit is a POSE problem -- stance, grip or table height,
* a joint past its speed limit is a TEMPO problem in that segment,
* a joint past its motor torque envelope is two different problems wearing one number.
  If the joint's own acceleration dominates the torque, the motor genuinely cannot drive that
  segment and the segment has to slow down. If gravity, centrifugal or cross-coupling dominates,
  the joint is being flung or dragged and the torque is a *reaction*: it is the tracking stiffness
  that is expensive, not the motion, and the fix is compliance or a slower PARENT segment.

That distinction was learned the hard way. On the 2026-08 bank the wrists looked badly overloaded
until the torque was decomposed: own-acceleration was 5-19% of it and centrifugal was up to 98%.
The wrists were absorbing the swing, not producing it, and telling a performer to "swing the wrist
less" would have been the wrong instruction.

Three traps this encodes so they are not re-discovered:

* Contacts must be disabled. The bank stores no root state, so the feet get driven through the
  floor and contact forces swamp the torque being measured -- once reading 41,934 N*m on a 6 N*m
  wrist.
* ``mj_forward`` recomputes ``qacc``, so the target acceleration is written AFTER it and before
  ``mj_inverse``. Getting that order wrong reads as good news: every peak collapses to a few
  percent of capacity.
* A joint frozen by the retarget still costs torque to hold rigid while the body swings, and will
  masquerade as the worst offender. Frozen joints are detected and reported as an artifact.

The six waist and ankle joints are parallel 2x2 mechanisms whose joint-space torque maps to motor
space through a pose-dependent Jacobian we do not hold. They are reported UNKNOWN and never PASS.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
ENVELOPE_CSV = REPO_ROOT / "configs" / "a3_motor_tn" / "current_conservative_tn_envelope.csv"
JOINT_MAP_CSV = REPO_ROOT / "configs" / "a3_motor_tn" / "a3_joint_motor_mapping.csv"
RUNTIME_JOINT_ORDER = REPO_ROOT / "configs" / "a3_runtime_articulation_joint_order.txt"

SOFT_LIMIT_FACTOR = 0.9  # matches ArticulationCfg.soft_joint_pos_limit_factor
FROZEN_RANGE_RAD = 1.0e-6
# A term counts as dominant when it carries at least this share of the peak torque.
DOMINANCE_SHARE = 0.5


class FeasibilityError(RuntimeError):
    """Raised when an input violates the recorded contract."""


# ----------------------------------------------------------------------------------------
# authorities
# ----------------------------------------------------------------------------------------


def load_joint_order() -> list[str]:
    return [
        line.strip()
        for line in RUNTIME_JOINT_ORDER.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


def load_envelope() -> dict[str, dict[str, float]]:
    families: dict[str, list[tuple[float, float]]] = {}
    with ENVELOPE_CSV.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            families.setdefault(row["family"], []).append(
                (float(row["speed_rad_s"]), float(row["torque_nm"]))
            )
    out = {}
    for family, points in families.items():
        points.sort()
        if len(points) != 3 or points[2][1] != 0.0:
            raise FeasibilityError(f"{family} is not the documented three-point envelope")
        out[family] = {
            "plateau_torque_nm": points[0][1],
            "plateau_end_rad_s": points[1][0],
            "zero_torque_rad_s": points[2][0],
        }
    return out


def load_joint_map() -> dict[str, dict[str, str]]:
    with JOINT_MAP_CSV.open(encoding="utf-8") as handle:
        return {row["joint_name"]: row for row in csv.DictReader(handle)}


def load_urdf_limits(urdf: Path) -> dict[str, dict[str, float]]:
    import xml.etree.ElementTree as ET

    root = ET.parse(urdf).getroot()
    out = {}
    for joint in root.findall("joint"):
        if joint.get("type") in {"fixed", "floating"}:
            continue
        limit = joint.find("limit")
        if limit is None:
            continue
        lower = float(limit.get("lower"))
        upper = float(limit.get("upper"))
        mid = 0.5 * (lower + upper)
        half = 0.5 * (upper - lower) * SOFT_LIMIT_FACTOR
        out[joint.get("name")] = {
            "lower": lower,
            "upper": upper,
            "soft_lower": mid - half,
            "soft_upper": mid + half,
            "velocity": float(limit.get("velocity")),
            "effort": float(limit.get("effort")),
        }
    return out


def torque_limit(envelope: dict[str, float], speed: np.ndarray) -> np.ndarray:
    v = np.abs(np.asarray(speed, dtype=np.float64))
    t0 = envelope["plateau_torque_nm"]
    vb = envelope["plateau_end_rad_s"]
    vz = envelope["zero_torque_rad_s"]
    out = np.full_like(v, t0)
    ramp = (v > vb) & (v < vz)
    out[ramp] = t0 * (vz - v[ramp]) / (vz - vb)
    out[v >= vz] = 0.0
    return out


# ----------------------------------------------------------------------------------------
# dynamics
# ----------------------------------------------------------------------------------------


class Plant:
    """MuJoCo model wrapper that returns the four torque terms separately."""

    def __init__(self, mjcf: Path, joint_names: list[str]) -> None:
        try:
            import mujoco
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise FeasibilityError("this analysis requires MuJoCo") from exc
        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(mjcf))
        # See the module docstring: with no root state in the bank the feet are driven through the
        # floor, and contact forces would swamp the very quantity being measured.
        self.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        self.data = mujoco.MjData(self.model)
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.names = joint_names
        self.qadr = []
        self.vadr = []
        for name in joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise FeasibilityError(f"MJCF lacks joint {name!r}")
            self.qadr.append(int(self.model.jnt_qposadr[jid]))
            self.vadr.append(int(self.model.jnt_dofadr[jid]))

    def _write_state(self, q: np.ndarray, v: np.ndarray) -> None:
        for i in range(len(self.names)):
            self.data.qpos[self.qadr[i]] = q[i]
            self.data.qvel[self.vadr[i]] = v[i]

    def terms(self, q: np.ndarray, v: np.ndarray, a: np.ndarray) -> dict[str, np.ndarray]:
        """Gravity, Coriolis, own-acceleration and cross-coupling torque, per joint."""
        mj = self.mujoco
        n = len(self.names)

        # gravity alone: zero velocity, zero acceleration
        self._write_state(q, np.zeros_like(v))
        mj.mj_forward(self.model, self.data)
        self.data.qacc[:] = 0.0
        mj.mj_inverse(self.model, self.data)
        gravity = np.array([self.data.qfrc_inverse[k] for k in self.vadr])

        # bias: real velocity, zero acceleration -> gravity + Coriolis/centrifugal
        self._write_state(q, v)
        mj.mj_forward(self.model, self.data)
        self.data.qacc[:] = 0.0
        mj.mj_inverse(self.model, self.data)
        bias = np.array([self.data.qfrc_inverse[k] for k in self.vadr])

        # full: real acceleration too. qacc is written AFTER mj_forward on purpose.
        for i in range(n):
            self.data.qacc[self.vadr[i]] = a[i]
        mj.mj_inverse(self.model, self.data)
        total = np.array([self.data.qfrc_inverse[k] for k in self.vadr])

        # own-acceleration term and effective inertia, one joint at a time
        own = np.zeros(n)
        inertia = np.zeros(n)
        for i in range(n):
            self.data.qacc[:] = 0.0
            self.data.qacc[self.vadr[i]] = a[i]
            mj.mj_inverse(self.model, self.data)
            own[i] = self.data.qfrc_inverse[self.vadr[i]] - bias[i]
            self.data.qacc[:] = 0.0
            self.data.qacc[self.vadr[i]] = 1.0
            mj.mj_inverse(self.model, self.data)
            inertia[i] = self.data.qfrc_inverse[self.vadr[i]] - bias[i]
        return {
            "gravity": gravity,
            "coriolis": bias - gravity,
            "own": own,
            "coupling": total - bias - own,
            "total": total,
            "bias": bias,
            "effective_inertia": inertia,
        }


def finite_difference_acceleration(qvel: np.ndarray, fps: float) -> np.ndarray:
    dt = 1.0 / float(fps)
    out = np.zeros_like(qvel)
    if qvel.shape[0] >= 3:
        out[1:-1] = (qvel[2:] - qvel[:-2]) / (2.0 * dt)
        out[0] = (qvel[1] - qvel[0]) / dt
        out[-1] = (qvel[-1] - qvel[-2]) / dt
    return out


# ----------------------------------------------------------------------------------------
# findings
# ----------------------------------------------------------------------------------------

REMEDIES = {
    "pose_out_of_range": (
        "the POSE itself is outside the joint's usable range -- adjust stance, grip, table height "
        "or ready position; swinging slower will not help"
    ),
    "speed_over_limit": (
        "this segment moves faster than the joint can turn -- shorten the backswing or slow the "
        "tempo through this part of the stroke"
    ),
    "torque_drive_limited": (
        "the joint's OWN acceleration dominates the torque, so the motor genuinely cannot drive "
        "this segment -- reduce the angular acceleration here, i.e. make the direction change less "
        "abrupt"
    ),
    "torque_reaction_limited": (
        "the joint is being flung or dragged: gravity, centrifugal or coupling dominates, so the "
        "torque is a REACTION to the rest of the body. Do not tell the performer to move this "
        "joint less. Either allow the joint to comply instead of tracking rigidly, or slow the "
        "PARENT segment that is throwing it"
    ),
    "frozen_joint_artifact": (
        "the retarget froze this joint, and holding it rigid against the body's swing costs torque "
        "-- this is a pipeline artifact, not something the performer did. Capture this joint's real "
        "motion, or accept it is not a capture finding"
    ),
    "parallel_unjudged": (
        "a parallel 2x2 mechanism whose motor-space torque needs a pose-dependent Jacobian we do "
        "not hold -- not judged, and never reported as passing"
    ),
}


def analyse_clip(
    path: Path,
    plant: Plant,
    order: list[str],
    envelope,
    joint_map,
    urdf_limits,
    hit_frame: Optional[int] = None,
    stride: int = 1,
) -> dict[str, Any]:
    payload = np.load(path, allow_pickle=True)
    qpos = np.asarray(payload["joint_pos"], dtype=np.float64)
    qvel = np.asarray(payload["joint_vel"], dtype=np.float64)
    fps = float(np.asarray(payload["fps"]).ravel()[0])
    frames = qpos.shape[0]
    if qpos.shape[1] != len(order):
        raise FeasibilityError(f"{path.name} has {qpos.shape[1]} joints, runtime order names {len(order)}")
    qacc = finite_difference_acceleration(qvel, fps)

    def phase_of(frame: int) -> str:
        if hit_frame is None:
            return "unknown"
        if frame < hit_frame - 1:
            return "pre_hit"
        if frame > hit_frame + 1:
            return "post_hit"
        return "hit"

    sampled = list(range(0, frames, max(1, stride)))
    terms = {k: np.zeros((len(sampled), len(order))) for k in
             ("gravity", "coriolis", "own", "coupling", "total", "bias", "effective_inertia")}
    for row, f in enumerate(sampled):
        got = plant.terms(qpos[f], qvel[f], qacc[f])
        for k in terms:
            terms[k][row] = got[k]

    findings = []
    per_joint = {}
    for i, name in enumerate(order):
        lim = urdf_limits[name]
        row = joint_map.get(name)
        span = float(qpos[:, i].max() - qpos[:, i].min())
        frozen = span < FROZEN_RANGE_RAD

        # --- pose ---
        under = lim["soft_lower"] - qpos[:, i]
        over = qpos[:, i] - lim["soft_upper"]
        breach = np.maximum(under, over)
        pose_worst = int(np.argmax(breach))
        # --- speed ---
        speed_ratio = np.abs(qvel[:, i]) / lim["velocity"]
        speed_worst = int(np.argmax(speed_ratio))
        # --- torque ---
        entry = {
            "range_rad": span,
            "frozen": frozen,
            "pose_margin_rad": float(-breach[pose_worst]),
            "speed_ratio": float(speed_ratio[speed_worst]),
        }
        if breach[pose_worst] > 0:
            findings.append({
                "joint": name, "category": "pose_out_of_range", "severity": "blocker",
                "frame": pose_worst, "phase": phase_of(pose_worst),
                "detail": f"{breach[pose_worst]:.4f} rad ({np.degrees(breach[pose_worst]):.1f} deg) past the soft limit",
                "remedy": REMEDIES["pose_out_of_range"],
            })
        if speed_ratio[speed_worst] > 1.0:
            findings.append({
                "joint": name, "category": "speed_over_limit", "severity": "blocker",
                "frame": speed_worst, "phase": phase_of(speed_worst),
                "detail": f"|speed| is {speed_ratio[speed_worst]:.2f}x the joint limit",
                "remedy": REMEDIES["speed_over_limit"],
            })

        if row is None or row["topology"] != "serial":
            entry["torque"] = "unjudged_parallel"
            findings.append({
                "joint": name, "category": "parallel_unjudged", "severity": "unknown",
                "frame": -1, "phase": "n/a",
                "detail": f"parallel pair {row['parallel_pair']}" if row else "not in the vendor motor map",
                "remedy": REMEDIES["parallel_unjudged"],
            })
            per_joint[name] = entry
            continue

        family = row["motor_family"]
        speeds = qvel[sampled, i]
        cap = torque_limit(envelope[family], speeds)
        magnitude = np.abs(terms["total"][:, i])
        ratio = np.where(cap > 0, magnitude / np.maximum(cap, 1e-12), np.inf)
        peak = int(np.argmax(np.where(np.isfinite(ratio), ratio, -1)))
        peak_frame = sampled[peak]
        headroom = (cap[peak] - np.abs(terms["bias"][peak, i])) / max(terms["effective_inertia"][peak, i], 1e-12)
        entry.update({
            "motor_family": family,
            "peak_torque_nm": float(magnitude[peak]),
            "torque_limit_at_that_speed_nm": float(cap[peak]),
            "torque_ratio": float(ratio[peak]),
            "peak_frame": peak_frame,
            "derived_accel_headroom_rad_s2": float(headroom),
            "actual_accel_rad_s2": float(abs(qacc[peak_frame, i])),
            "share": {
                k: float(np.abs(terms[k][:, i]).mean() / max(magnitude.mean(), 1e-12))
                for k in ("gravity", "coriolis", "own", "coupling")
            },
        })
        if ratio[peak] > 1.0:
            share = entry["share"]
            if frozen:
                category = "frozen_joint_artifact"
                severity = "artifact"
            elif share["own"] >= DOMINANCE_SHARE:
                category = "torque_drive_limited"
                severity = "blocker"
            else:
                category = "torque_reaction_limited"
                severity = "major"
            dominant = max(("gravity", "coriolis", "own", "coupling"), key=lambda k: share[k])
            findings.append({
                "joint": name, "category": category, "severity": severity,
                "frame": peak_frame, "phase": phase_of(peak_frame),
                "detail": (
                    f"{magnitude[peak]:.2f} N*m against a {cap[peak]:.2f} N*m envelope at "
                    f"{abs(speeds[peak]):.2f} rad/s ({ratio[peak]:.2f}x); dominant term "
                    f"{dominant} at {share[dominant]*100:.0f}%, own-acceleration {share['own']*100:.0f}%"
                ),
                "remedy": REMEDIES[category],
            })
        per_joint[name] = entry

    order_sev = {"blocker": 0, "major": 1, "artifact": 2, "unknown": 3}
    findings.sort(key=lambda f: (order_sev.get(f["severity"], 9), f["joint"]))
    return {
        "file": path.name,
        "frames": frames,
        "fps": fps,
        "hit_frame": hit_frame,
        "sampled_frames": len(sampled),
        "findings": findings,
        "per_joint": per_joint,
        "verdict": (
            "BLOCKED" if any(f["severity"] == "blocker" for f in findings)
            else "NEEDS_ATTENTION" if any(f["severity"] == "major" for f in findings)
            else "OK_FOR_JUDGED_JOINTS"
        ),
    }


def load_hit_frames(manifest: Optional[Path]) -> dict[str, int]:
    if manifest is None or not manifest.is_file():
        return {}
    doc = json.loads(manifest.read_text(encoding="utf-8"))
    rows = doc if isinstance(doc, list) else None
    if rows is None:
        for value in doc.values():
            if isinstance(value, list):
                rows = value
                break
    out = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        name = str(row.get("npz") or row.get("uid") or "")
        frame = row.get("hit_frame_50")
        if name and frame is not None:
            out[Path(name).stem.replace("hope_", "")] = int(frame)
    return out


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--motion", type=Path, required=True, help="one NPZ or a bank directory")
    parser.add_argument("--mjcf", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None, help="supplies hit_frame_50 for phase labels")
    parser.add_argument("--stride", type=int, default=2, help="analyse every Nth frame (torque is the cost)")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--brief", action="store_true", help="print the capture-session briefing")
    return parser.parse_args(argv)


def build_brief(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-clip findings into instructions for the next capture session."""
    by_cat: dict[str, dict[str, dict[str, Any]]] = {}
    for result in results:
        for f in result["findings"]:
            slot = by_cat.setdefault(f["category"], {}).setdefault(
                f["joint"], {"clips": 0, "worst": None, "phases": {}, "remedy": f["remedy"]}
            )
            slot["clips"] += 1
            slot["phases"][f["phase"]] = slot["phases"].get(f["phase"], 0) + 1
            if slot["worst"] is None:
                slot["worst"] = f["detail"]
    brief = {}
    for category, joints in by_cat.items():
        brief[category] = {
            "remedy": next(iter(joints.values()))["remedy"],
            "joints": dict(sorted(
                ((j, {k: v for k, v in d.items() if k != "remedy"}) for j, d in joints.items()),
                key=lambda kv: -kv[1]["clips"],
            )),
        }
    return brief


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        order = load_joint_order()
        envelope = load_envelope()
        joint_map = load_joint_map()
        urdf_limits = load_urdf_limits(args.urdf)
        hits = load_hit_frames(args.manifest)
        plant = Plant(args.mjcf, order)
        clips = sorted(args.motion.glob("*.npz")) if args.motion.is_dir() else [args.motion]
        results = []
        for path in clips:
            uid = path.stem.replace("hope_", "")
            results.append(
                analyse_clip(path, plant, order, envelope, joint_map, urdf_limits,
                             hits.get(uid), args.stride)
            )
        brief = build_brief(results)
        report = {
            "kind": "a3_motion_feasibility_v1",
            "motion": str(args.motion),
            "clips": len(results),
            "verdicts": {v: sum(1 for r in results if r["verdict"] == v) for v in
                         {r["verdict"] for r in results}},
            "capture_brief": brief,
            "clips_detail": results,
        }
        if args.output_json:
            args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if args.brief:
            print(f"clips: {report['clips']}   verdicts: {report['verdicts']}\n")
            for category, block in brief.items():
                print(f"[{category}]  {block['remedy']}")
                for joint, d in list(block["joints"].items())[:8]:
                    print(f"    {joint:28s} {d['clips']:3d} clips   phases={d['phases']}")
                    if d["worst"]:
                        print(f"        worst: {d['worst']}")
                print()
        else:
            print(json.dumps({k: v for k, v in report.items() if k != "clips_detail"}, ensure_ascii=False))
        return 0
    except (FeasibilityError, FileNotFoundError, KeyError, OSError, ValueError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
