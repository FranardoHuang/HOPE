#!/usr/bin/env python3
"""A-layer dynamic feasibility oracle (MuJoCo mj_inverse) for motion .npz clips.

WHAT THIS IS (design: docs/research/robot_centric_timing_2026-07-09.md section 七)
    The physical upper-bound screen of the three-tier feasibility ladder
    (A = inverse-dynamics oracle, B = PD replay, C = tracker prerun). Per frame
    of a reference clip it computes the generalized force required to realize
    the stored motion (mj_inverse -> qfrc_inverse) and grades two TRUE
    constraints — the ones no controller can negotiate around:

    1. ACTUATED-JOINT TORQUE: |tau_req| vs tau_max (MJCF/URDF actuatorfrcrange,
       6..320 Nm across the 31 joints). Exceeding = NO controller can track.
    2. GROUND-CONTACT FEASIBILITY: the 6 floating-base rows of qfrc_inverse are
       the external wrench the ground MUST supply. Three checks:
         a. unilaterality  : vertical force fz >= 0 (ground pushes, never pulls)
         b. CoP            : center of pressure inside the support polygon
                             (feet corner points from the foot collision hulls,
                             posed by body_pos_w / body_quat_w)
         c. friction cone  : |f_tangential| <= mu * fz  (mu default 0.8)
       "Push-off-window foot slip" = the reference demanding a tangential force
       the floor cannot provide — check (c) is exactly that, in Newtons.

    Acceleration limits are deliberately NOT a check here: acceleration is the
    shadow of torque/inertia (section 七), the torque check subsumes it.

FRAME CONVENTIONS (empirically pinned on the vendor MJCF, 2026-07-09)
    * MuJoCo free-joint translational DOFs are WORLD-frame; rotational DOFs are
      BODY-LOCAL. Hence f_world = qfrc_inverse[0:3] and
      tau_world = R_root @ qfrc_inverse[3:6] (moment about the root origin).
      Validated: static stand -> fz == subtree weight to 1e-9, CoP == CoM xy
      projection to 1e-4 m under a 60 deg root yaw (LOCAL matches, WORLD does
      not). Regression-tested in tests/test_feasibility_oracle.py.
    * npz quats (Isaac body_quat_w) and MuJoCo qpos quats are both wxyz.
    * qvel/qacc are finite-differenced from qpos with mj_differentiatePos so
      the quaternion tangent-space convention is MuJoCo's own by construction.

MODEL PREP
    Contacts and joint-limit constraints are DISABLED for the inverse pass:
    every Newton is billed to the actuators + the root residual, nothing hides
    in a limit constraint (position-limit violations are L0 audit territory,
    scripts/audit_motion_npz.py). Damping / frictionloss / armature stay ON —
    they are real physics the motors must pay for. The <motor> actuator layer
    is irrelevant to mj_inverse (it never reads ctrl).

VERDICT — DOSE-CALIBRATED (dose-curve backtest, 2026-07-09)
    Retargeted references are kinematic (GVHMR/GMR IK): they are NEVER
    perfectly dynamically consistent, so "any frame violates => reject" is
    uncalibrated — the 0709 backtest shows even v4rg (fall rate 0.02, the
    known-good floor) carries transient CoP excursions at the whip. What
    predicts training falls is the DOSE: the time-share of frames whose
    required ground wrench is infeasible. Backtest on the backhand family
    (known fall rates v4rg 0.02 / v5syn 0.29 / v5hLt 0.86 / v5hLs 0.82):

        clip     CoP-out dose   fric>1 frames   fall rate
        v4rg        0.167            0             0.02
        v5syn       0.232            2             0.29
        v5syn35     0.316            3             (pending — oracle predicts
                                                    between v5syn and v5hLt)
        v5hLt       0.478            2             0.86
        v5hLs       0.667            4             0.82

    Ordering matches measured fall rates on every resolvable pair (v5hLs vs
    v5hLt, 0.82 vs 0.86, is a statistical tie the oracle flips). Calibrated
    gates (constants DOSE_*): FAIL when cop dose >= 0.35 or torque dose
    >= 0.10 or friction dose >= 0.05 or any sustained fz<0 / flight
    violation; WARN when cop dose >= 0.20 or any friction-cone frame or
    torque dose >= 0.02; else PASS. Per-frame physical thresholds
    (TORQUE_FAIL etc.) still define what counts as a violation frame and
    drive the reported violation windows; isolated single-frame torque
    spikes (v4rg fh carries a 172% one and trains fine) are demonstrably
    NOT fall-predictive at 50 fps.

    Exit code 0 / 1 / 2 for PASS / WARN / FAIL (CI-friendly, same contract
    as the L0 audit). All violation windows are reported relative to the
    annotated contact frame (--annotations, strike_annotations.yaml `phase`).

USAGE
    python scripts/feasibility_oracle.py CLIP.npz [CLIP2.npz ...] \
        [--mjcf PATH] [--annotations PATH] [--body-order LIST|FILE] \
        [--mu 0.8] [--support-band 0.03] [--md OUT.md] [--json OUT.json]

    Pod run (mjeval venv has mujoco; NEVER the training GPU env):
        /workspace/hope_mjeval_venv/bin/python scripts/feasibility_oracle.py \
            /workspace/franco/motion_work/motions/v5_height_fix/hope_backhand_v5hLt_cal.npz \
            --mjcf $REPO/agi/.../a3_pingpong.xml \
            --body-order /workspace/franco/body_order_isaac.txt

DEPENDENCIES
    numpy + mujoco (>=3.x). PyYAML optional for --annotations (falls back to a
    minimal parser for the `phase:` field, same contract as audit_motion_npz).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # imported lazily-ish so the pure-geometry helpers stay testable anywhere
    import mujoco
except ImportError:  # pragma: no cover - exercised only on hosts without mujoco
    mujoco = None

# ---------------------------------------------------------------------------
# Joint order of the npz joint_pos/joint_vel columns (Isaac articulation
# order, 31 DOF). Same provenance as scripts/audit_motion_npz.py (donor ONNX
# joint_names metadata; PINGPONG_DEPLOY_ALIGNMENT.md section 4).
# ---------------------------------------------------------------------------
ISAAC_JOINT_NAMES: List[str] = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "head_yaw_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "head_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]

FOOT_BODIES: Tuple[str, str] = ("left_ankle_roll_Link", "right_ankle_roll_Link")
ROOT_BODY = "pelvis_link"
BODY_ORDER_SIDECARS = ("body_order.txt", "body_order_isaac.txt")

# thresholds ---------------------------------------------------------------
TORQUE_FAIL = 1.0          # |tau|/tau_max above this = violation frame
TORQUE_WARN = 0.90         # margin excursion
FZ_FAIL_N = -1.0           # required vertical ground force below this = pulling
COP_FAIL_M = 0.02          # CoP outside the support hull by more than this
COP_WARN_M = 0.0           # outside at all = margin excursion
FRICTION_FAIL = 1.0        # |f_t| / (mu*fz)
FRICTION_WARN = 0.80
FLIGHT_FORCE_FRAC = 0.05   # flight frames: |f_req| must be < this * weight
SUSTAIN_RUN = 2            # violation runs >= this many frames = "sustained"

# dose gates — CALIBRATED on the 2026-07-09 backhand dose-curve backtest
# (see VERDICT in the module docstring; do not tune without re-running it)
DOSE_COP_FAIL = 0.35       # CoP-outside time-share; v5hLt 0.478 (fall 0.86)
DOSE_COP_WARN = 0.20       # v5syn 0.232 (fall 0.29); v4rg 0.167 (fall 0.02)
DOSE_FRIC_FAIL = 0.05      # friction-cone time-share; v5hLs 0.070
DOSE_TAU_FAIL = 0.10       # torque-overrun time-share
DOSE_TAU_WARN = 0.02
SOLE_BAND_M = 0.01         # foot-hull vertices within this of the lowest = sole
COP_MIN_FZ = 1.0           # CoP/friction undefined below this fz [N]
DEFAULT_MU = 0.8
DEFAULT_SUPPORT_BAND = 0.03  # foot is support when sole z < ground + band [m]

DEFAULT_MJCF = (
    "agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/"
    "a3_pingpong/a3_pingpong.xml"
)
DEFAULT_ANNOTATIONS = "hope_training/whole_body_tracking/cfg/strike_annotations.yaml"

CHECK_ORDER = ("torque", "fz", "cop", "friction", "flight")
CHECK_LABEL = {
    "torque": "关节力矩 |τ|/τ_max",
    "fz": "立力 fz≥0",
    "cop": "CoP∈支撑面",
    "friction": "摩擦锥 |f_t|≤μ·fz",
    "flight": "腾空帧外力≈0",
}


# ---------------------------------------------------------------------------
# pure-geometry helpers (numpy only; unit-tested without mujoco)
# ---------------------------------------------------------------------------

def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    """2D cross product (np.cross on 2-vectors is removed in numpy>=2)."""
    return float(a[0] * b[1] - a[1] * b[0])


def convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """Andrew monotone chain. points (N,2) -> CCW hull vertices (H,2)."""
    pts = np.unique(np.asarray(points, dtype=float).reshape(-1, 2), axis=0)
    if len(pts) <= 2:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def half(iterable):
        out: List[np.ndarray] = []
        for p in iterable:
            while len(out) >= 2 and _cross2(out[-1] - out[-2], p - out[-2]) <= 0:
                out.pop()
            out.append(p)
        return out

    lower = half(pts)
    upper = half(pts[::-1])
    return np.array(lower[:-1] + upper[:-1])


def signed_dist_to_hull(point: np.ndarray, hull: np.ndarray) -> float:
    """Signed distance from a 2D point to a CCW convex hull.

    Negative = inside (distance to nearest edge), positive = outside (exact
    Euclidean distance to the polygon). Degenerate hulls (<3 vertices) give
    distance to the point/segment (always >= 0 unless exactly on it).
    """
    p = np.asarray(point, dtype=float)
    hull = np.asarray(hull, dtype=float)
    if len(hull) == 0:
        return float("inf")
    if len(hull) == 1:
        return float(np.linalg.norm(p - hull[0]))
    seg_d, edge_signed = [], []
    n = len(hull)
    for i in range(n):
        a, b = hull[i], hull[(i + 1) % n]
        ab = b - a
        L2 = float(ab @ ab)
        t = 0.0 if L2 == 0 else float(np.clip((p - a) @ ab / L2, 0.0, 1.0))
        seg_d.append(float(np.linalg.norm(p - (a + t * ab))))
        if L2 > 0:
            edge_signed.append(_cross2(ab, p - a) / math.sqrt(L2))
    if len(hull) == 2:
        return seg_d[0]
    # CCW: cross > 0 means left of edge = inside
    inside = all(s >= 0 for s in edge_signed)
    return -min(seg_d) if inside else min(seg_d)


def cop_from_wrench(
    f_w: np.ndarray, tau_w_about_p: np.ndarray, p_w: np.ndarray, ground_z: float
) -> Tuple[float, float]:
    """CoP on the plane z=ground_z for wrench (f_w, tau about point p_w), world frame."""
    m0 = np.asarray(tau_w_about_p, float) + np.cross(np.asarray(p_w, float), np.asarray(f_w, float))
    fz = float(f_w[2])
    px = (ground_z * f_w[0] - m0[1]) / fz
    py = (m0[0] + ground_z * f_w[1]) / fz
    return float(px), float(py)


def runs_of(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Consecutive True runs of a boolean mask -> [(start, end_inclusive)]."""
    out: List[Tuple[int, int]] = []
    idx = np.flatnonzero(np.asarray(mask, bool))
    if idx.size == 0:
        return out
    start = prev = int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i == prev + 1:
            prev = i
            continue
        out.append((start, prev))
        start = prev = i
    out.append((start, prev))
    return out


# ---------------------------------------------------------------------------
# annotations (same minimal contract as audit_motion_npz: clips.<stem>.phase)
# ---------------------------------------------------------------------------

def load_annotations(path: Optional[str]) -> Dict[str, dict]:
    if not path or str(path).lower() == "none":
        return {}
    p = Path(path)
    if not p.is_file():
        return {}
    text = p.read_text()
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text) or {}
        clips = data.get("clips", {}) or {}
        return {k: v for k, v in clips.items() if isinstance(v, dict)}
    except ImportError:
        clips: Dict[str, dict] = {}
        cur: Optional[str] = None
        for raw in text.splitlines():
            line = raw.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip())
            body = line.strip()
            if indent == 2 and body.endswith(":"):
                cur = body[:-1]
                clips[cur] = {}
            elif indent >= 4 and cur and ":" in body:
                k, v = body.split(":", 1)
                clips[cur][k.strip()] = v.strip()
        return clips


def contact_frame_of(stem: str, annotations: Dict[str, dict], n_frames: int) -> Optional[int]:
    entry = annotations.get(stem)
    if not entry:
        return None
    try:
        phase = float(entry.get("phase"))
    except (TypeError, ValueError):
        return None
    return int(round(phase * (n_frames - 1)))


# ---------------------------------------------------------------------------
# model wrapper
# ---------------------------------------------------------------------------

@dataclass
class OracleModel:
    model: "mujoco.MjModel"
    joint_qposadr: np.ndarray      # (31,) qpos column per Isaac joint
    joint_dofadr: np.ndarray       # (31,) dof row per Isaac joint
    tau_max: np.ndarray            # (31,)
    foot_corners: Dict[str, np.ndarray]  # body-frame sole polygon corners (K,3)
    weight: float                  # total subtree weight [N]

    @property
    def nq(self) -> int:
        return self.model.nq

    @property
    def nv(self) -> int:
        return self.model.nv


def _foot_sole_corners(model: "mujoco.MjModel", body_name: str) -> np.ndarray:
    """Sole polygon corners of a foot body, in body frame.

    Collects vertices of all collision-capable geoms on the body (mesh verts
    posed by geom_pos/quat; primitive geoms via their corner/axis extents),
    keeps the ones within SOLE_BAND_M of the lowest point, hulls their xy.
    """
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        raise ValueError(f"body {body_name!r} not in model")
    verts: List[np.ndarray] = []
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] != bid:
            continue
        if model.geom_contype[gid] == 0 and model.geom_conaffinity[gid] == 0:
            continue  # visual-only geom
        gpos = model.geom_pos[gid]
        gmat = np.zeros(9)
        mujoco.mju_quat2Mat(gmat, model.geom_quat[gid])
        gmat = gmat.reshape(3, 3)
        if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = model.geom_dataid[gid]
            adr, num = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            v = model.mesh_vert[adr : adr + num].astype(float)
            # mesh vertices are stored relative to the mesh frame; geoms pose it
            verts.append(v @ gmat.T + gpos)
        else:
            sz = model.geom_size[gid]
            corners = np.array(
                [[sx, sy, sz_] for sx in (-sz[0], sz[0]) for sy in (-sz[1], sz[1]) for sz_ in (-sz[2], sz[2])]
            )
            verts.append(corners @ gmat.T + gpos)
    if not verts:
        raise ValueError(f"no collision geoms on {body_name!r} (foot polygon unavailable)")
    v = np.vstack(verts)
    sole = v[v[:, 2] < v[:, 2].min() + SOLE_BAND_M]
    hull_xy = convex_hull_2d(sole[:, :2])
    z = float(sole[:, 2].min())
    return np.column_stack([hull_xy, np.full(len(hull_xy), z)])


def load_oracle_model(mjcf_path: str) -> OracleModel:
    if mujoco is None:
        raise RuntimeError("mujoco is not installed in this environment")
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    # bill every Newton to actuators + root residual; keep passive physics on
    model.opt.disableflags |= (
        mujoco.mjtDisableBit.mjDSBL_CONTACT | mujoco.mjtDisableBit.mjDSBL_LIMIT
    )
    root_jnt = model.jnt_type[0]
    if root_jnt != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError(
            "model root joint is not FREE — this oracle needs a floating base "
            f"(got jnt_type[0]={int(root_jnt)}); build/find a free-root variant"
        )
    qposadr, dofadr, tau_max = [], [], []
    for name in ISAAC_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"joint {name!r} missing from MJCF")
        if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(f"joint {name!r} is not a hinge")
        qposadr.append(int(model.jnt_qposadr[jid]))
        dofadr.append(int(model.jnt_dofadr[jid]))
        if not model.jnt_actfrclimited[jid]:
            raise ValueError(f"joint {name!r} has no actuatorfrcrange (tau_max unknown)")
        tau_max.append(float(model.jnt_actfrcrange[jid][1]))
    feet = {b: _foot_sole_corners(model, b) for b in FOOT_BODIES}
    rid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ROOT_BODY)
    weight = float(model.body_subtreemass[rid]) * float(np.linalg.norm(model.opt.gravity))
    return OracleModel(
        model=model,
        joint_qposadr=np.array(qposadr),
        joint_dofadr=np.array(dofadr),
        tau_max=np.array(tau_max),
        foot_corners=feet,
        weight=weight,
    )


# ---------------------------------------------------------------------------
# npz -> (qpos, qvel, qacc) trajectories
# ---------------------------------------------------------------------------

def resolve_body_order(npz_path: Path, data, spec: Optional[str]) -> List[str]:
    if "body_names" in getattr(data, "files", []):
        return [str(x) for x in data["body_names"]]
    if spec:
        p = Path(spec)
        if p.is_file():
            return [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
        return [s.strip() for s in spec.split(",") if s.strip()]
    for sidecar in BODY_ORDER_SIDECARS:
        p = npz_path.parent / sidecar
        if p.is_file():
            return [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
    raise ValueError(
        f"{npz_path.name}: cannot resolve body order (no npz body_names, no "
        f"--body-order, no sidecar {BODY_ORDER_SIDECARS}) — refusing to guess"
    )


def build_qpos(om: OracleModel, joint_pos: np.ndarray, root_pos: np.ndarray, root_quat: np.ndarray) -> np.ndarray:
    T = joint_pos.shape[0]
    q = np.asarray(root_quat, dtype=float).copy()
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    for t in range(1, T):  # hemisphere continuity (avoid 2pi jumps in subQuat)
        if float(q[t] @ q[t - 1]) < 0.0:
            q[t] = -q[t]
    qpos = np.zeros((T, om.nq))
    qpos[:, 0:3] = root_pos
    qpos[:, 3:7] = q
    qpos[:, om.joint_qposadr] = joint_pos
    return qpos


def differentiate(om: OracleModel, qpos: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """qpos (T,nq) -> qvel (T,nv), qacc (T,nv) by central differences.

    Half-step velocities via mj_differentiatePos (quaternion-correct, MuJoCo's
    own tangent convention); frame velocity = mean of adjacent half-steps,
    acceleration = their difference / dt. Endpoint frames copy the nearest
    half-step velocity and are excluded from grading (eval mask).
    """
    T = qpos.shape[0]
    half = np.zeros((T - 1, om.nv))
    buf = np.zeros(om.nv)
    for t in range(T - 1):
        mujoco.mj_differentiatePos(om.model, buf, dt, qpos[t], qpos[t + 1])
        half[t] = buf
    qvel = np.zeros((T, om.nv))
    qacc = np.zeros((T, om.nv))
    qvel[1:-1] = 0.5 * (half[:-1] + half[1:])
    qvel[0], qvel[-1] = half[0], half[-1]
    qacc[1:-1] = (half[1:] - half[:-1]) / dt
    return qvel, qacc


# ---------------------------------------------------------------------------
# the oracle proper
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    name: str
    fail_runs: List[Tuple[int, int]] = field(default_factory=list)
    warn_frames: int = 0
    fail_frames: int = 0
    peak: float = 0.0            # worst metric value (meaning depends on check)
    peak_frame: int = -1
    detail: str = ""

    @property
    def sustained(self) -> bool:
        return any(e - s + 1 >= SUSTAIN_RUN for s, e in self.fail_runs)

    @property
    def isolated(self) -> bool:
        return self.fail_frames > 0 and not self.sustained


@dataclass
class ClipReport:
    path: str
    stem: str
    n_frames: int
    fps: float
    contact_frame: Optional[int]
    verdict: str = "PASS"
    checks: Dict[str, CheckResult] = field(default_factory=dict)
    # torque details
    util: Optional[np.ndarray] = None            # (T,31) |tau|/tau_max
    tau_req: Optional[np.ndarray] = None         # (T,31)
    top_joints: List[dict] = field(default_factory=list)
    binding_sequence: List[dict] = field(default_factory=list)
    # contact details
    fz: Optional[np.ndarray] = None
    cop_excess: Optional[np.ndarray] = None      # signed dist to hull (+ = outside)
    fric_ratio: Optional[np.ndarray] = None
    support_count: Optional[np.ndarray] = None
    ground_z: float = 0.0
    weight: float = 0.0
    fd_vs_stored_vel: float = float("nan")
    doses: Dict[str, float] = field(default_factory=dict)
    cop_peak_offset: Optional[Tuple[float, float]] = None  # world dx,dy from support centroid
    notes: List[str] = field(default_factory=list)

    def rel(self, f: int) -> str:
        if self.contact_frame is None:
            return f"f{f}"
        d = f - self.contact_frame
        return f"f{f}({'+' if d >= 0 else ''}{d})"


def analyze_clip(
    om: OracleModel,
    npz_path: Path,
    annotations: Dict[str, dict],
    body_order_spec: Optional[str],
    mu: float,
    support_band: float,
) -> ClipReport:
    data = np.load(npz_path, allow_pickle=True)
    fps = float(np.array(data["fps"]).reshape(-1)[0])
    dt = 1.0 / fps
    joint_pos = np.asarray(data["joint_pos"], dtype=float)
    joint_vel_stored = np.asarray(data["joint_vel"], dtype=float)
    body_pos = np.asarray(data["body_pos_w"], dtype=float)
    body_quat = np.asarray(data["body_quat_w"], dtype=float)
    T, J = joint_pos.shape
    if J != len(ISAAC_JOINT_NAMES):
        raise ValueError(f"{npz_path.name}: joint_pos has {J} columns, expected 31")

    body_names = resolve_body_order(npz_path, data, body_order_spec)
    if len(body_names) != body_pos.shape[1]:
        raise ValueError(
            f"{npz_path.name}: body order has {len(body_names)} names but "
            f"body_pos_w has {body_pos.shape[1]} bodies"
        )
    b_idx = {n: i for i, n in enumerate(body_names)}
    for need in (ROOT_BODY, *FOOT_BODIES):
        if need not in b_idx:
            raise ValueError(f"{npz_path.name}: body {need!r} missing from body order")

    stem = npz_path.name[: -len(".npz")] if npz_path.name.endswith(".npz") else npz_path.name
    rep = ClipReport(
        path=str(npz_path),
        stem=stem,
        n_frames=T,
        fps=fps,
        contact_frame=contact_frame_of(stem, annotations, T),
        weight=om.weight,
    )

    qpos = build_qpos(om, joint_pos, body_pos[:, b_idx[ROOT_BODY]], body_quat[:, b_idx[ROOT_BODY]])
    qvel, qacc = differentiate(om, qpos, dt)

    # cross-check: finite-difference joint velocity vs stored joint_vel
    rep.fd_vs_stored_vel = float(
        np.max(np.abs(qvel[1:-1][:, om.joint_dofadr] - joint_vel_stored[1:-1]))
    ) if T > 2 else float("nan")

    # inverse dynamics, frame by frame
    d = mujoco.MjData(om.model)
    qfrc = np.zeros((T, om.nv))
    for t in range(T):
        d.qpos[:] = qpos[t]
        d.qvel[:] = qvel[t]
        d.qacc[:] = qacc[t]
        mujoco.mj_inverse(om.model, d)
        qfrc[t] = d.qfrc_inverse

    eval_mask = np.zeros(T, dtype=bool)
    eval_mask[1:-1] = True  # endpoints have no central difference

    # ---- check 1: actuated torque utilization -----------------------------
    tau_req = qfrc[:, om.joint_dofadr]
    util = np.abs(tau_req) / om.tau_max[None, :]
    rep.tau_req, rep.util = tau_req, util
    util_e = np.where(eval_mask[:, None], util, 0.0)
    frame_peak = util_e.max(axis=1)
    tq = CheckResult("torque")
    tq.fail_frames = int(np.sum(frame_peak > TORQUE_FAIL))
    tq.warn_frames = int(np.sum((frame_peak > TORQUE_WARN) & (frame_peak <= TORQUE_FAIL)))
    tq.fail_runs = runs_of(frame_peak > TORQUE_FAIL)
    pk = int(np.argmax(frame_peak))
    tq.peak, tq.peak_frame = float(frame_peak[pk]), pk
    tq.detail = ISAAC_JOINT_NAMES[int(np.argmax(util_e[pk]))]
    rep.checks["torque"] = tq

    # per-joint peaks (top offenders table)
    jpk = util_e.max(axis=0)
    order = np.argsort(-jpk)
    rep.top_joints = [
        {
            "joint": ISAAC_JOINT_NAMES[j],
            "peak_util": float(jpk[j]),
            "tau_max": float(om.tau_max[j]),
            "peak_frame": int(np.argmax(util_e[:, j])),
            "frames_over_1": int(np.sum(util_e[:, j] > TORQUE_FAIL)),
        }
        for j in order[:8]
    ]
    # binding sequence: which joint binds (>90%) when, in time order
    bind_mask = util_e > TORQUE_WARN
    for j in range(len(ISAAC_JOINT_NAMES)):
        for s, e in runs_of(bind_mask[:, j]):
            rep.binding_sequence.append(
                {
                    "joint": ISAAC_JOINT_NAMES[j],
                    "start": int(s),
                    "end": int(e),
                    "peak_util": float(util_e[s : e + 1, j].max()),
                }
            )
    rep.binding_sequence.sort(key=lambda r: (r["start"], -r["peak_util"]))

    # ---- root residual -> required ground wrench --------------------------
    f_w = qfrc[:, 0:3]
    tau_local = qfrc[:, 3:6]
    tau_w = np.zeros_like(tau_local)
    Rbuf = np.zeros(9)
    for t in range(T):
        mujoco.mju_quat2Mat(Rbuf, qpos[t, 3:7])
        tau_w[t] = Rbuf.reshape(3, 3) @ tau_local[t]

    # support feet from npz foot poses + model sole corners
    foot_world: Dict[str, np.ndarray] = {}
    for fb in FOOT_BODIES:
        i = b_idx[fb]
        corners = om.foot_corners[fb]  # (K,3) body frame
        world = np.zeros((T, len(corners), 3))
        for t in range(T):
            mujoco.mju_quat2Mat(Rbuf, body_quat[t, i] / np.linalg.norm(body_quat[t, i]))
            world[t] = corners @ Rbuf.reshape(3, 3).T + body_pos[t, i]
        foot_world[fb] = world
    sole_z = {fb: foot_world[fb][:, :, 2].min(axis=1) for fb in FOOT_BODIES}
    ground_z = float(min(sole_z[fb].min() for fb in FOOT_BODIES))
    rep.ground_z = ground_z
    support = {fb: sole_z[fb] < ground_z + support_band for fb in FOOT_BODIES}
    rep.support_count = np.sum([support[fb] for fb in FOOT_BODIES], axis=0)

    fz = f_w[:, 2].copy()
    rep.fz = fz
    cop_excess = np.full(T, np.nan)
    fric_ratio = np.full(T, np.nan)
    flight_force = np.full(T, np.nan)
    cop_xy = np.full((T, 2), np.nan)
    hull_centroid = np.full((T, 2), np.nan)
    for t in range(T):
        if not eval_mask[t]:
            continue
        feet_now = [fb for fb in FOOT_BODIES if support[fb][t]]
        if not feet_now:
            flight_force[t] = float(np.linalg.norm(f_w[t])) / om.weight
            continue
        ft = float(np.hypot(f_w[t, 0], f_w[t, 1]))
        fric_ratio[t] = ft / (mu * max(fz[t], COP_MIN_FZ))
        if fz[t] > COP_MIN_FZ:
            px, py = cop_from_wrench(f_w[t], tau_w[t], qpos[t, 0:3], ground_z)
            hull = convex_hull_2d(
                np.vstack([foot_world[fb][t][:, :2] for fb in feet_now])
            )
            cop_excess[t] = signed_dist_to_hull(np.array([px, py]), hull)
            cop_xy[t] = (px, py)
            hull_centroid[t] = hull.mean(axis=0)
    rep.cop_excess, rep.fric_ratio = cop_excess, fric_ratio

    def grade(name: str, metric: np.ndarray, fail_at: float, warn_at: float) -> CheckResult:
        c = CheckResult(name)
        valid = ~np.isnan(metric)
        vm = np.where(valid, metric, -np.inf)
        c.fail_frames = int(np.sum(vm > fail_at))
        c.warn_frames = int(np.sum((vm > warn_at) & (vm <= fail_at)))
        c.fail_runs = runs_of(vm > fail_at)
        if valid.any():
            pk = int(np.argmax(vm))
            c.peak, c.peak_frame = float(vm[pk]), pk
        return c

    fzc = grade("fz", -fz * np.where(eval_mask, 1, np.nan), -FZ_FAIL_N, -FZ_FAIL_N)
    fzc.peak = -fzc.peak  # report as min fz
    rep.checks["fz"] = fzc
    rep.checks["cop"] = grade("cop", cop_excess, COP_FAIL_M, COP_WARN_M)
    rep.checks["friction"] = grade("friction", fric_ratio, FRICTION_FAIL, FRICTION_WARN)
    rep.checks["flight"] = grade("flight", flight_force, FLIGHT_FORCE_FRAC, FLIGHT_FORCE_FRAC)

    # CoP exit direction at the worst frame (binding localization aid)
    cpk = rep.checks["cop"].peak_frame
    if cpk >= 0 and not np.isnan(cop_xy[cpk]).any():
        rep.cop_peak_offset = (
            float(cop_xy[cpk, 0] - hull_centroid[cpk, 0]),
            float(cop_xy[cpk, 1] - hull_centroid[cpk, 1]),
        )

    # ---- verdict: dose-calibrated (see module docstring) -------------------
    n_eval = max(int(eval_mask.sum()), 1)
    dose_cop = float(np.sum(np.nan_to_num(cop_excess, nan=-np.inf) > COP_WARN_M)) / n_eval
    dose_fric = float(np.sum(np.nan_to_num(fric_ratio, nan=-np.inf) > FRICTION_FAIL)) / n_eval
    dose_tau = float(tq.fail_frames) / n_eval
    rep.doses = {"cop": dose_cop, "friction": dose_fric, "torque": dose_tau}
    hard_fail = rep.checks["fz"].sustained or rep.checks["flight"].sustained
    if hard_fail or dose_cop >= DOSE_COP_FAIL or dose_tau >= DOSE_TAU_FAIL or dose_fric >= DOSE_FRIC_FAIL:
        rep.verdict = "FAIL"
    elif (
        dose_cop >= DOSE_COP_WARN
        or dose_fric > 0
        or dose_tau >= DOSE_TAU_WARN
        or rep.checks["fz"].fail_frames > 0
        or rep.checks["flight"].fail_frames > 0
    ):
        rep.verdict = "WARN"
    else:
        rep.verdict = "PASS"
    return rep


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def _fmt_runs(rep: ClipReport, runs: List[Tuple[int, int]], limit: int = 6) -> str:
    if not runs:
        return "-"
    parts = [f"{rep.rel(s)}..{rep.rel(e)}" if e > s else rep.rel(s) for s, e in runs[:limit]]
    if len(runs) > limit:
        parts.append(f"(+{len(runs) - limit} runs)")
    return ", ".join(parts)


def clip_markdown(rep: ClipReport, mu: float) -> str:
    L: List[str] = []
    fc = "unannotated" if rep.contact_frame is None else f"f{rep.contact_frame}"
    L.append(f"### {rep.stem} — **{rep.verdict}**\n")
    L.append(
        f"- frames: {rep.n_frames} @ {rep.fps:.0f}fps | contact: {fc} | "
        f"weight: {rep.weight:.1f} N | ground z: {rep.ground_z:+.3f} m | "
        f"fd-vs-stored joint_vel max mismatch: {rep.fd_vs_stored_vel:.3f} rad/s"
    )
    tq = rep.checks["torque"]
    L.append(
        f"- torque: peak util **{tq.peak * 100:.0f}%** ({tq.detail} @ {rep.rel(tq.peak_frame)}), "
        f"frames>100%: {tq.fail_frames}, 90-100%: {tq.warn_frames} | FAIL runs: {_fmt_runs(rep, tq.fail_runs)}"
    )
    fzc, cop, fric, fl = (rep.checks[k] for k in ("fz", "cop", "friction", "flight"))
    off = ""
    if rep.cop_peak_offset is not None:
        off = f", exit dir world({rep.cop_peak_offset[0]:+.2f},{rep.cop_peak_offset[1]:+.2f})m"
    L.append(
        f"- contact: min fz {fzc.peak:.0f} N @ {rep.rel(fzc.peak_frame)} | "
        f"CoP max excursion {cop.peak * 100:+.1f} cm @ {rep.rel(cop.peak_frame)}{off} "
        f"(out-frames: {cop.fail_frames + cop.warn_frames}) | "
        f"friction(mu={mu}) peak {fric.peak * 100:.0f}% @ {rep.rel(fric.peak_frame)} "
        f"(frames>100%: {fric.fail_frames})"
    )
    L.append(
        f"- doses (violation time-share): CoP-out **{rep.doses.get('cop', 0):.3f}** "
        f"(gates: warn {DOSE_COP_WARN}, fail {DOSE_COP_FAIL}) | "
        f"friction {rep.doses.get('friction', 0):.3f} | torque {rep.doses.get('torque', 0):.3f}"
    )
    if fl.fail_frames:
        L.append(f"- flight-frame force residual FAILs: {fl.fail_frames} ({_fmt_runs(rep, fl.fail_runs)})")
    viol = [
        f"{CHECK_LABEL[k]}: {_fmt_runs(rep, rep.checks[k].fail_runs)}"
        for k in CHECK_ORDER
        if rep.checks.get(k) and rep.checks[k].fail_runs
    ]
    L.append(f"- violation windows (rel contact): {'; '.join(viol) if viol else 'none'}")
    if rep.top_joints:
        L.append("\n| joint | tau_max [Nm] | peak util | @frame | frames>100% |")
        L.append("| --- | --- | --- | --- | --- |")
        for r in rep.top_joints:
            if r["peak_util"] < 0.25 and r is not rep.top_joints[0]:
                continue
            L.append(
                f"| {r['joint']} | {r['tau_max']:.0f} | {r['peak_util'] * 100:.0f}% | "
                f"{rep.rel(r['peak_frame'])} | {r['frames_over_1']} |"
            )
    binding = [b for b in rep.binding_sequence if b["peak_util"] > TORQUE_WARN]
    if binding:
        L.append("\nbinding (>90%) sequence: " + "; ".join(
            f"{b['joint']} {rep.rel(b['start'])}..{rep.rel(b['end'])} ({b['peak_util'] * 100:.0f}%)"
            for b in binding[:10]
        ))
    for n in rep.notes:
        L.append(f"- note: {n}")
    L.append("")
    return "\n".join(L)


def summary_markdown(reps: List[ClipReport], mu: float) -> str:
    L = [
        "| clip | verdict | CoP dose | fric dose | τ dose | peak τ util | CoP max out [cm] | fric peak | min fz [N] |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in reps:
        tq, cop, fric, fzc = (r.checks[k] for k in ("torque", "cop", "friction", "fz"))
        L.append(
            f"| {r.stem} | {r.verdict} | **{r.doses.get('cop', 0):.3f}** | "
            f"{r.doses.get('friction', 0):.3f} | {r.doses.get('torque', 0):.3f} | "
            f"{tq.peak * 100:.0f}% | {max(cop.peak, 0) * 100:.1f} | "
            f"{fric.peak * 100:.0f}% | {fzc.peak:.0f} |"
        )
    return "\n".join(L)


def report_json(rep: ClipReport) -> dict:
    def cr(c: CheckResult) -> dict:
        return {
            "fail_frames": c.fail_frames,
            "warn_frames": c.warn_frames,
            "fail_runs": c.fail_runs,
            "sustained": c.sustained,
            "peak": c.peak,
            "peak_frame": c.peak_frame,
            "detail": c.detail,
        }

    return {
        "clip": rep.stem,
        "path": rep.path,
        "verdict": rep.verdict,
        "n_frames": rep.n_frames,
        "fps": rep.fps,
        "contact_frame": rep.contact_frame,
        "weight_N": rep.weight,
        "ground_z": rep.ground_z,
        "fd_vs_stored_vel": rep.fd_vs_stored_vel,
        "doses": rep.doses,
        "cop_peak_offset": rep.cop_peak_offset,
        "checks": {k: cr(v) for k, v in rep.checks.items()},
        "top_joints": rep.top_joints,
        "binding_sequence": rep.binding_sequence[:40],
        "util_p95": float(np.percentile(rep.util[1:-1].max(axis=1), 95)) if rep.util is not None and rep.n_frames > 2 else None,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("clips", nargs="+", help="motion .npz files")
    ap.add_argument("--mjcf", default=os.environ.get("FEAS_ORACLE_MJCF") or str(repo_root() / DEFAULT_MJCF))
    ap.add_argument("--annotations", default=os.environ.get("FEAS_ORACLE_ANNOTATIONS") or str(repo_root() / DEFAULT_ANNOTATIONS))
    ap.add_argument("--body-order", default=None, help="body-name list/file for body_pos_w columns")
    ap.add_argument("--mu", type=float, default=DEFAULT_MU, help="friction coefficient (default 0.8)")
    ap.add_argument("--support-band", type=float, default=DEFAULT_SUPPORT_BAND)
    ap.add_argument("--md", default=None, help="write a markdown report here")
    ap.add_argument("--json", dest="json_out", default=None, help="write full metrics json here")
    args = ap.parse_args(argv)

    om = load_oracle_model(args.mjcf)
    annotations = load_annotations(args.annotations)

    reps: List[ClipReport] = []
    for clip in args.clips:
        rep = analyze_clip(om, Path(clip), annotations, args.body_order, args.mu, args.support_band)
        reps.append(rep)
        tq, cop, fric = (rep.checks[k] for k in ("torque", "cop", "friction"))
        print(
            f"[{rep.verdict:4s}] {rep.stem}: CoP dose {rep.doses.get('cop', 0):.3f} "
            f"(max out {max(cop.peak, 0) * 100:.1f}cm) | fric dose {rep.doses.get('friction', 0):.3f} "
            f"(peak {fric.peak * 100:.0f}%) | tau dose {rep.doses.get('torque', 0):.3f} "
            f"(peak {tq.peak * 100:.0f}% {tq.detail})"
        )

    if args.md:
        lines = [
            "# A-layer feasibility oracle report (mj_inverse)",
            "",
            f"generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')} | "
            f"mjcf: `{args.mjcf}` | mu={args.mu} | support band {args.support_band} m",
            "",
            "## summary",
            "",
            summary_markdown(reps, args.mu),
            "",
            "## per-clip",
            "",
        ]
        lines += [clip_markdown(r, args.mu) for r in reps]
        Path(args.md).write_text("\n".join(lines))
        print(f"markdown -> {args.md}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps([report_json(r) for r in reps], indent=2))
        print(f"json -> {args.json_out}")

    worst = max((r.verdict for r in reps), key=("PASS", "WARN", "FAIL").index)
    return ("PASS", "WARN", "FAIL").index(worst)


if __name__ == "__main__":
    sys.exit(main())
