#!/usr/bin/env python3
"""L1 self-collision audit ("自碰撞检查器") for retargeted motion .npz clips.

WHAT THIS IS
    The acceptance-stack hole-patch decided by franco 2026-07-09 (TIMELINE 晚五):
    the L0 判炸器 (scripts/audit_motion_npz.py) grades a clip against URDF *joint*
    limits and foot skate, but it has no idea whether the robot's own links pass
    through each other. A stroke-deepening asset (extend_stroke v1) can therefore
    audit fully green while the racket is buried in the torso. This tool closes
    that hole: it replays the clip pose-by-pose in the VENDOR MJCF and asks MuJoCo
    itself, per frame, "are any two links of this robot interpenetrating?".

    franco's hint that motivated it (TIMELINE 晚五): "正手加引拍很方便,反手可能和
    躯干碰撞". The 晚六 correction narrowed the borrow-stroke DOFs to the arm chain
    (shoulder internal rotation + elbow lift) plus waist yaw, and gave this tool
    its second job: "管'肘抬高后小臂-躯干余隙'". Hence the two headline pairs:

        RACKET  <-> TORSO      (晚五 main item)
        FOREARM <-> TORSO      (晚六 elbow-lift clearance)

WHY A SIBLING SCRIPT AND NOT AN 8TH CHECK INSIDE audit_motion_npz.py
    The 判炸器's module docstring makes a dependency promise: "numpy + standard
    library only. No isaac, no torch." It is the gate that must run anywhere — CI,
    a laptop checkout, a bare CPU box — because nothing may enter the pipeline
    without it. Self-collision cannot be computed from the npz alone: it needs the
    robot's collision geometry, i.e. the vendor MJCF and a mujoco build. Folding it
    in would either (a) make the L0 gate un-runnable wherever mujoco is missing, or
    (b) make it silently skip its own check — the exact failure mode 判炸器 check 7
    was hardened against ("fail-loud, never a silent skip").

    So this is a SEPARATE TIER, modelled on scripts/feasibility_oracle.py (the other
    mujoco-dependent grader): same npz/annotations/body-order input contract, same
    PASS/WARN/FAIL vocabulary, same 0/1/2 exit code, same fail-loud posture. It
    lives next to the 判炸器 and REUSES its primitives (Finding, Suggestion,
    ISAAC_JOINT_NAMES, resolve_body_order, load_annotations, _ranges, _runs) rather
    than copying them, so the two tools cannot drift apart. audit_motion_npz.py is
    NOT modified by this branch beyond a docstring pointer: zero behaviour change.

WHAT COUNTS AS A SELF-COLLISION (i.e. which contacts are "legal")
    Per frame: write qpos from the clip, mj_forward, read d.contact. A contact with
    dist < 0 is an interpenetration. Legal contacts are excluded as follows:

    1. THE FLOOR. The only legal contact a reference clip may have with the world.
       Removed by zeroing the floor geom's contype/conaffinity at load, so MuJoCo
       never generates those contacts. (Foot-vs-floor pathology is 判炸器 check 7's
       job, not ours.)
    2. ADJACENT LINKS. Handled by MuJoCo itself: same-body geoms, parent-child body
       pairs, and the vendor's own 27 <exclude> entries never produce contacts.

    Nothing else is excluded, and that is a measured decision, not an assumption:
      * neutral poses (keyframe "stand", qpos0) produce ZERO robot-robot contacts
        — the model carries no always-on hull artifact that would need subtracting;
      * all 28 production *_cal clips produce ZERO robot-robot contacts (2026-07-09
        sweep, see the report). So no "chain-adjacency" heuristic exclusion list is
        warranted, and adding one would only blind the detector.
    If a future MJCF adds an <exclude> that hides the racket<->torso pair, load
    raises: the main item can never be silently skipped (see _assert_pairs_visible).

CHECKS (graded PASS / WARN / FAIL; process exit code 0 / 1 / 2)
    1. self_collision — per colliding body pair, over the whole clip.
         持续 (sustained): a run of >= SUSTAIN_RUN consecutive colliding frames
             => FAIL. A reference the policy must track is asking the robot to hold
             two links inside each other.
         深穿 (deep): any single frame with penetration depth >= DEEP_FAIL_M
             => FAIL. Not a graze — geometry is passing through.
         擦碰 (grazing): isolated, shallow contacts (< SUSTAIN_RUN frames AND
             < DEEP_FAIL_M deep) => WARN. Convex collision hulls are inflated
             approximations of the STL, so a one-frame sub-millimetre touch at a
             hull corner is a modelling artifact, not a motion pathology.
       The racket<->torso pair is reported explicitly even when clean, so a PASS
       report proves the main item actually ran.
    2. clearance — minimum distance, over the backswing window [0, contact frame],
       of RACKET<->TORSO and FOREARM<->TORSO (and the same pairs vs the pelvis).
       Below CLEARANCE_WARN_M => WARN ("余量太薄"). This is the quantitative form
       of franco's hint and the acceptance number for extend_stroke v2.
    3. backswing_frame — the deepest-drawback frame t*, reported with the clearance
       readings at that frame (the baseline table this branch was asked to produce).

    Repair suggestion, when the racket or forearm hits the torso: franco's 晚六
    ladder — redistribute the stroke over the ARM CHAIN (shoulder internal rotation
    + elbow lift) plus WAIST YAW; never waist pitch/roll or the legs (the oracle's
    measured torque binding), and keep CoM displacement ~= 0.

DEEPEST-DRAWBACK FRAME t* (definition, since none existed in the repo)
    Let p(t) be the racket site position expressed in the TORSO frame (so base
    translation/yaw cannot contaminate it) and c the annotated contact frame. Walk
    backwards from c while the racket keeps getting farther from where it will
    strike:  t* = the first local maximum of ||p(t) - p(c)|| encountered going back
    from c. That is the swing turnaround = the end of the backswing = the deepest
    draw. Frames before t* (ready pose, approach) are not part of the backswing and
    are deliberately not searched, which is why a plain argmax over [0, c] is wrong.

!! MuJoCo 3.10 CAVEAT — mj_geomDistance IS NOT TRUSTED HERE (measured 2026-07-09)
    For mesh<->mesh pairs, mj_geomDistance(..., distmax) sporadically returns
    exactly 0.0 once distmax exceeds the true distance, with a `fromto` segment
    whose length contradicts the returned value (v4rg backhand f35, racket vs torso:
    raw 0.0000, fromto length 0.3935, true distance 0.2532-0.2644 by an independent
    separating-axis bracket). Raising opt.ccd_iterations 35 -> 200 does not help.
    The failure only ever occurs for distmax > true distance; for distmax <= true
    distance the function saturates exactly (returns distmax).

    So clearances are computed WITHOUT trusting the returned value, using only its
    saturation predicate  far(dm) := (returned == dm)  <=>  true distance >= dm,
    bisected to CLEARANCE_TOL. Verified: the bisection reproduces the raw value at
    every frame where the raw value is self-consistent, and recovers the true
    distance at the frames where it is not. (This is also why the tool does NOT
    inflate geom_margin to harvest near-miss distances from mj_forward — that path
    runs straight through the same defect.)
    GRADING IS UNAFFECTED: FAIL/WARN read d.contact from mj_forward, i.e. MuJoCo's
    penetration path — the very code the vendor sim itself runs.

REAL-ASSET BASELINE (no motion npz in the laptop checkout — run on the pod):
    /workspace/hope_mjeval_venv/bin/python \
        hope_training/whole_body_tracking/scripts/audit_self_collision.py \
        /workspace/franco/motion_work/motions/regen_0708_candidates/hope_*_cal.npz \
        /workspace/franco/motion_work/motions/v5_height_fix/hope_*_cal.npz \
        --body-order /workspace/franco/body_order_isaac.txt --baseline-md BASE.md
    EXPECT: every production clip PASS, zero contacts (2026-07-09 baseline).

USAGE
    python scripts/audit_self_collision.py CLIP.npz [CLIP2.npz ...] \
        [--mjcf PATH] [--annotations cfg/strike_annotations.yaml] \
        [--body-order LIST|FILE] [--clearance-window backswing|all] \
        [--md OUT.md] [--json OUT.json] [--baseline-md OUT.md]

    Exit code: 0 = all PASS, 1 = worst is WARN, 2 = any FAIL (CI-friendly).

DEPENDENCIES
    numpy + mujoco (>=3.1.4, for mj_geomDistance) + the vendor MJCF. PyYAML optional
    (annotations fall back to the 判炸器's minimal parser). No isaac, no torch.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Reuse the 判炸器's primitives rather than copying them: one definition of the
# joint table, the grading vocabulary, the fail-loud body-order resolver and the
# annotations parser, so the two tiers cannot drift apart.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_motion_npz import (  # noqa: E402
    FAIL,
    PASS,
    WARN,
    ISAAC_JOINT_NAMES,
    Finding,
    Suggestion,
    _LEVEL_RANK,
    _ranges,
    _runs,
    _ann_phase,
    load_annotations,
    resolve_body_order,
)

try:  # keep the pure helpers importable on hosts without mujoco
    import mujoco
except ImportError:  # pragma: no cover - exercised only where mujoco is absent
    mujoco = None


# --- model wiring -----------------------------------------------------------
DEFAULT_MJCF_REL = (
    "agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/"
    "a3_pingpong/a3_pingpong.xml"
)
ROOT_BODY = "pelvis_link"
FLOOR_GEOM = "floor"

# The racket is a pair of collision geoms bolted to right_wrist_yaw_Link (face
# mesh + handle capsule); the "forearm" is the elbow link plus the wrist-roll
# link (wrist roll is forearm pronation, so it is forearm, not hand).
RACKET_GEOMS: Tuple[str, ...] = ("right_racket_collision", "right_racket_handle_collision")
FOREARM_GEOMS: Tuple[str, ...] = ("right_elbow_collision", "right_wrist_roll_collision")
TORSO_GEOMS: Tuple[str, ...] = ("torso_collision",)
PELVIS_GEOMS: Tuple[str, ...] = ("pelvis_collision",)
RACKET_SITE = "right_racket"
TORSO_BODY = "torso_Link"

# Clearance pairs of the baseline table. The first two are franco's headline
# items (晚五 racket-torso, 晚六 forearm-torso); the pelvis rows are carried
# because a right-handed backhand draws the racket across toward the left hip.
CLEARANCE_PAIRS: Tuple[Tuple[str, Tuple[str, ...], Tuple[str, ...]], ...] = (
    ("racket-torso", RACKET_GEOMS, TORSO_GEOMS),
    ("forearm-torso", FOREARM_GEOMS, TORSO_GEOMS),
    ("racket-pelvis", RACKET_GEOMS, PELVIS_GEOMS),
    ("forearm-pelvis", FOREARM_GEOMS, PELVIS_GEOMS),
)
# The pair that must never be silently skipped (the whole point of the branch).
REQUIRED_PAIR = ("racket-torso", RACKET_GEOMS, TORSO_GEOMS)

# --- thresholds -------------------------------------------------------------
SUSTAIN_RUN = 2          # colliding frames in a row -> 持续 -> FAIL (as feasibility_oracle)
DEEP_FAIL_M = 0.005      # single-frame penetration depth [m] that is not a graze
CLEARANCE_WARN_M = 0.02  # racket/forearm vs torso closer than this -> WARN (余量太薄)
CLEARANCE_DISTMAX = 0.6  # clearances above this are reported as ">= 0.6" (saturated)
CLEARANCE_TOL = 1e-4     # bisection precision [m]
BACKSWING_EPS = 1e-4     # monotonicity tolerance when walking back from contact [m]


@dataclass
class SelfColModel:
    model: "mujoco.MjModel"
    joint_qposadr: np.ndarray            # (31,) qpos column per Isaac joint
    geom_ids: Dict[str, int]             # collision-geom name -> id
    racket_site: int
    torso_body: int

    @property
    def nq(self) -> int:
        return self.model.nq


@dataclass
class PairHit:
    """One colliding body pair over the whole clip.

    Grouped by BODY for a readable report, but the participating GEOM pairs are
    carried too: the racket geoms and the hand geoms hang off the same body
    (right_wrist_yaw_Link), so body granularity alone cannot tell "the racket hit
    the torso" (franco's main item) from "the knuckles brushed the torso".
    """
    body1: str
    body2: str
    frames: List[int]
    depth_peak: float                    # max penetration depth [m]
    depth_peak_frame: int
    runs: List[Tuple[int, int]]
    geom_pairs: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def longest_run(self) -> int:
        return max((e - s + 1 for s, e in self.runs), default=0)

    @property
    def sustained(self) -> bool:
        return self.longest_run >= SUSTAIN_RUN

    @property
    def deep(self) -> bool:
        return self.depth_peak >= DEEP_FAIL_M

    @property
    def level(self) -> str:
        return FAIL if (self.sustained or self.deep) else WARN

    def involves(self, aset: Sequence[str], bset: Sequence[str]) -> bool:
        """True iff some contact of this pair was between a geom of each set."""
        a, b = set(aset), set(bset)
        return any((g1 in a and g2 in b) or (g2 in a and g1 in b) for g1, g2 in self.geom_pairs)

    def to_json(self) -> dict:
        return {
            "body1": self.body1,
            "body2": self.body2,
            "level": self.level,
            "n_frames": len(self.frames),
            "frames": _ranges(self.frames),
            "longest_run": self.longest_run,
            "depth_peak_mm": round(self.depth_peak * 1000.0, 3),
            "depth_peak_frame": self.depth_peak_frame,
            "geom_pairs": [f"{a}|{b}" for a, b in self.geom_pairs],
        }


@dataclass
class Clearance:
    """Minimum distance of one named pair over the scanned window."""
    name: str
    min_dist: float
    min_frame: int
    at_backswing: Optional[float] = None
    saturated: bool = False              # min_dist pinned at CLEARANCE_DISTMAX

    def to_json(self) -> dict:
        return {
            "pair": self.name,
            "min_dist_m": round(self.min_dist, 5),
            "min_frame": self.min_frame,
            "at_backswing_m": None if self.at_backswing is None else round(self.at_backswing, 5),
            "saturated": self.saturated,
        }


@dataclass
class SelfColReport:
    path: str
    stem: str
    frames: int = 0
    fps: int = 0
    verdict: str = PASS
    findings: List[Finding] = field(default_factory=list)
    suggestion: Suggestion = field(default_factory=lambda: Suggestion("none"))
    contact_frame: Optional[int] = None
    annotated_phase: Optional[float] = None
    backswing_frame: Optional[int] = None
    hits: List[PairHit] = field(default_factory=list)
    clearances: List[Clearance] = field(default_factory=list)
    window: str = "backswing"

    def add(self, finding: Finding) -> None:
        self.findings.append(finding)
        if _LEVEL_RANK[finding.level] > _LEVEL_RANK[self.verdict]:
            self.verdict = finding.level

    def clearance(self, name: str) -> Optional[Clearance]:
        return next((c for c in self.clearances if c.name == name), None)

    def to_json(self) -> dict:
        return {
            "path": self.path,
            "stem": self.stem,
            "frames": self.frames,
            "fps": self.fps,
            "verdict": self.verdict,
            "contact_frame": self.contact_frame,
            "annotated_phase": self.annotated_phase,
            "backswing_frame": self.backswing_frame,
            "clearance_window": self.window,
            "findings": [f.to_json() for f in self.findings if f.level != PASS],
            "collisions": [h.to_json() for h in self.hits],
            "clearances": [c.to_json() for c in self.clearances],
            "suggestion": self.suggestion.to_json(),
        }


# ---------------------------------------------------------------------------
# MuJoCo contact-filter replication (pure logic; unit-tested without a clip)
# ---------------------------------------------------------------------------

def geom_pair_enabled(model: "mujoco.MjModel", g1: int, g2: int) -> bool:
    """True iff MuJoCo's broadphase filter would ever let this geom pair collide.

    Mirrors engine_collision_driver.c: contype/conaffinity masks, same weld body,
    parent-weld body (world exempt), and the <contact><exclude> signature table.
    Used to prove at load time that the racket<->torso pair is visible; a future
    MJCF that excludes it must break loudly, not quietly.
    """
    m = model
    ct1, ca1 = int(m.geom_contype[g1]), int(m.geom_conaffinity[g1])
    ct2, ca2 = int(m.geom_contype[g2]), int(m.geom_conaffinity[g2])
    if not ((ct1 & ca2) or (ct2 & ca1)):
        return False

    w1, w2 = int(m.body_weldid[m.geom_bodyid[g1]]), int(m.body_weldid[m.geom_bodyid[g2]])
    if w1 == w2:
        return False
    parent_filtering = not (m.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT))
    if parent_filtering:
        pw1, pw2 = int(m.body_weldid[m.body_parentid[w1]]), int(m.body_weldid[m.body_parentid[w2]])
        if (w1 != 0 and w1 == pw2) or (w2 != 0 and w2 == pw1):
            return False

    b1, b2 = int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])
    lo, hi = min(b1, b2), max(b1, b2)
    signature = (lo << 16) + hi
    return not bool((m.exclude_signature[: m.nexclude] == signature).any())


# ---------------------------------------------------------------------------
# robust geom distance (see the mj_geomDistance caveat in the module docstring)
# ---------------------------------------------------------------------------

def _far(model, data, g1: int, g2: int, dm: float) -> bool:
    """far(dm) <=> true distance >= dm. Reads ONLY mj_geomDistance's saturation."""
    return float(mujoco.mj_geomDistance(model, data, g1, g2, dm, None)) >= dm - 1e-12


def geom_clearance(
    model, data, g1: int, g2: int,
    distmax: float = CLEARANCE_DISTMAX, tol: float = CLEARANCE_TOL,
) -> Tuple[float, bool]:
    """Signed distance between two geoms. Returns (distance, saturated).

    Never trusts mj_geomDistance's returned VALUE (MuJoCo 3.10 mesh-mesh defect:
    it can return 0.0 whenever distmax exceeds the true distance). Instead it
    bisects the saturation predicate, which is exact for distmax <= true distance.
    Penetration (distance < 0) is read directly: distmax=0 puts the call in its
    penetration branch, where the EPA depth is what the simulator itself uses.
    """
    d0 = float(mujoco.mj_geomDistance(model, data, g1, g2, 0.0, None))
    if d0 < 0.0:
        return d0, False
    if _far(model, data, g1, g2, distmax):
        return distmax, True
    lo, hi = 0.0, distmax        # invariant: far(lo) true, far(hi) false
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if _far(model, data, g1, g2, mid):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), False


def group_clearance(model, data, gset1: Sequence[int], gset2: Sequence[int]) -> Tuple[float, bool]:
    """Minimum clearance between two groups of geoms (e.g. racket vs torso)."""
    best, sat = float("inf"), True
    for a in gset1:
        for b in gset2:
            d, s = geom_clearance(model, data, a, b)
            if d < best:
                best, sat = d, s
    return best, sat


# ---------------------------------------------------------------------------
# model loading
# ---------------------------------------------------------------------------

def _gid(model, name: str) -> int:
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if i < 0:
        raise ValueError(f"collision geom {name!r} missing from MJCF")
    return i


def _assert_pairs_visible(model, geom_ids: Dict[str, int]) -> None:
    """Fail-loud guard: the racket<->torso pair must be collidable in this MJCF."""
    name, aset, bset = REQUIRED_PAIR
    for a in aset:
        for b in bset:
            if not geom_pair_enabled(model, geom_ids[a], geom_ids[b]):
                raise RuntimeError(
                    f"MJCF filters the required {name} pair ({a} <-> {b}): this audit's "
                    f"main item would be silently skipped. Check <contact><exclude> and "
                    f"the contype/conaffinity masks before trusting any verdict."
                )


def load_selfcol_model(mjcf_path: str) -> SelfColModel:
    if mujoco is None:
        raise RuntimeError("mujoco is not installed in this environment")
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    if model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError(
            "model root joint is not FREE — this audit needs a floating base "
            f"(got jnt_type[0]={int(model.jnt_type[0])})"
        )

    # LEGAL CONTACT #1: the ground. Zeroing the masks means MuJoCo never generates
    # floor contacts, so every contact this tool sees is robot-vs-robot by
    # construction (foot-vs-floor pathology belongs to 判炸器 check 7).
    floor = _gid(model, FLOOR_GEOM)
    model.geom_contype[floor] = 0
    model.geom_conaffinity[floor] = 0

    qposadr = []
    for name in ISAAC_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"joint {name!r} missing from MJCF")
        if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(f"joint {name!r} is not a hinge")
        qposadr.append(int(model.jnt_qposadr[jid]))

    names = set(RACKET_GEOMS) | set(FOREARM_GEOMS) | set(TORSO_GEOMS) | set(PELVIS_GEOMS)
    geom_ids = {n: _gid(model, n) for n in sorted(names)}
    _assert_pairs_visible(model, geom_ids)

    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, RACKET_SITE)
    if site < 0:
        raise ValueError(f"site {RACKET_SITE!r} missing from MJCF")
    tb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, TORSO_BODY)
    if tb < 0:
        raise ValueError(f"body {TORSO_BODY!r} missing from MJCF")

    return SelfColModel(
        model=model,
        joint_qposadr=np.array(qposadr),
        geom_ids=geom_ids,
        racket_site=site,
        torso_body=tb,
    )


# ---------------------------------------------------------------------------
# kinematics helpers
# ---------------------------------------------------------------------------

def build_qpos(sm: SelfColModel, joint_pos: np.ndarray, root_pos: np.ndarray,
               root_quat: np.ndarray) -> np.ndarray:
    """(T,31) joint columns + root pose -> (T,nq). Same convention as feasibility_oracle."""
    T = joint_pos.shape[0]
    q = np.asarray(root_quat, dtype=float).copy()
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    qpos = np.zeros((T, sm.nq))
    qpos[:, 0:3] = root_pos
    qpos[:, 3:7] = q
    qpos[:, sm.joint_qposadr] = joint_pos
    return qpos


def backswing_frame(racket_in_torso: np.ndarray, contact: int, eps: float = BACKSWING_EPS) -> int:
    """Deepest-drawback frame: walk back from contact while the racket keeps receding.

    See the module docstring. `racket_in_torso` is (T,3) racket-site positions in the
    torso frame. A plain argmax over [0, contact] would return the ready pose, not
    the swing turnaround, whenever the clip opens far from the strike point.
    """
    if contact <= 0:
        return 0
    d = np.linalg.norm(racket_in_torso - racket_in_torso[contact], axis=1)
    t = contact
    while t > 0 and d[t - 1] > d[t] + eps:
        t -= 1
    return int(t)


def collect_hits(sm: SelfColModel, data, qpos: np.ndarray) -> Tuple[List[PairHit], np.ndarray]:
    """Per-frame mj_forward + contact detection. Returns (pair hits, colliding mask)."""
    m = sm.model
    T = qpos.shape[0]
    bname = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(b))
    gname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(g))
    per_pair: Dict[Tuple[str, str], Dict[int, float]] = {}
    per_pair_geoms: Dict[Tuple[str, str], set] = {}
    colliding = np.zeros(T, dtype=bool)
    for t in range(T):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(m, data)
        for i in range(data.ncon):
            c = data.contact[i]
            if c.dist >= 0.0:
                continue  # near-miss inside a margin band; not an interpenetration
            b1, b2 = bname(m.geom_bodyid[c.geom1]), bname(m.geom_bodyid[c.geom2])
            g1, g2 = gname(c.geom1), gname(c.geom2)
            key = (b1, b2) if b1 <= b2 else (b2, b1)
            depth = float(-c.dist)
            slot = per_pair.setdefault(key, {})
            slot[t] = max(slot.get(t, 0.0), depth)
            per_pair_geoms.setdefault(key, set()).add((g1, g2) if g1 <= g2 else (g2, g1))
            colliding[t] = True

    hits: List[PairHit] = []
    for (b1, b2), by_frame in sorted(per_pair.items()):
        frames = sorted(by_frame)
        mask = np.zeros(T, dtype=bool)
        mask[frames] = True
        peak_frame = max(frames, key=lambda f: by_frame[f])
        hits.append(PairHit(
            body1=b1, body2=b2, frames=frames,
            depth_peak=by_frame[peak_frame], depth_peak_frame=peak_frame,
            runs=_runs(mask), geom_pairs=sorted(per_pair_geoms[(b1, b2)]),
        ))
    hits.sort(key=lambda h: (-_LEVEL_RANK[h.level], -h.depth_peak))
    return hits, colliding


# ---------------------------------------------------------------------------
# per-clip audit
# ---------------------------------------------------------------------------

def audit_clip(
    npz_path: str,
    sm: SelfColModel,
    annotations: Optional[Dict[str, dict]] = None,
    body_order: Optional[str] = None,
    window: str = "backswing",
) -> SelfColReport:
    stem = Path(npz_path).stem
    rep = SelfColReport(path=str(npz_path), stem=stem, window=window)
    entry = (annotations or {}).get(stem, {})
    rep.annotated_phase = _ann_phase(entry)

    try:
        data_npz = np.load(npz_path, allow_pickle=True)
        q = np.asarray(data_npz["joint_pos"], dtype=float)
        fps = int(np.asarray(data_npz["fps"]).reshape(-1)[0])
        body_pos = np.asarray(data_npz["body_pos_w"], dtype=float)
        body_quat = np.asarray(data_npz["body_quat_w"], dtype=float)
    except Exception as exc:  # malformed npz is a FAIL, not a crash
        rep.add(Finding("load", FAIL, f"cannot load/parse npz: {exc}"))
        rep.suggestion = Suggestion("regenerate", ["clip unreadable — regenerate"])
        return rep

    if not (np.isfinite(q).all() and np.isfinite(body_pos).all() and np.isfinite(body_quat).all()):
        rep.add(Finding("load", FAIL, "non-finite values (NaN/Inf) in joint_pos/body_pos_w/body_quat_w"))
        rep.suggestion = Suggestion("regenerate", ["corrupt arrays (NaN/Inf) — regenerate the clip"])
        return rep

    T, J = q.shape
    rep.frames, rep.fps = T, fps
    if J != len(ISAAC_JOINT_NAMES):
        rep.add(Finding("load", FAIL, f"npz joint dim {J} != {len(ISAAC_JOINT_NAMES)} (Isaac table)"))
        rep.suggestion = Suggestion("regenerate", ["joint table mismatch — fix inputs and re-audit"])
        return rep

    # The root pose comes from body column 0; prove it really is the pelvis rather
    # than trusting the convention (fail-loud, as 判炸器 check 7 does for the feet).
    try:
        names = resolve_body_order(body_order, str(npz_path), int(body_pos.shape[1]), data=data_npz)
        if names[0] != ROOT_BODY:
            raise ValueError(f"body column 0 is {names[0]!r}, expected the root {ROOT_BODY!r}")
    except (OSError, ValueError) as exc:
        rep.add(Finding("load", FAIL, f"body order unresolved ({exc}) — refusing to guess the root pose"))
        rep.suggestion = Suggestion("reaudit", [
            "body order unresolved — supply --body-order (or drop body_order.txt next to "
            "the npz) and RE-AUDIT; the root pose drives every contact in this check",
        ])
        return rep

    if rep.annotated_phase is not None:
        rep.contact_frame = int(round(rep.annotated_phase * (T - 1)))

    m = sm.model
    d = mujoco.MjData(m)
    qpos = build_qpos(sm, q, body_pos[:, 0], body_quat[:, 0])

    # -- check 1: per-frame contact detection --------------------------------
    rep.hits, _ = collect_hits(sm, d, qpos)
    for h in rep.hits:
        why = []
        if h.sustained:
            why.append(f"持续 {h.longest_run} 帧 >= {SUSTAIN_RUN}")
        if h.deep:
            why.append(f"深穿 {h.depth_peak * 1000:.2f} mm >= {DEEP_FAIL_M * 1000:g} mm")
        if not why:
            why.append(f"擦碰(孤立 {len(h.frames)} 帧,峰值 {h.depth_peak * 1000:.2f} mm)")
        geoms = ", ".join(f"{a}|{b}" for a, b in h.geom_pairs)
        rep.add(Finding(
            "self_collision", h.level,
            f"{h.body1} <-> {h.body2} interpenetrate ({geoms}): " + ";".join(why),
            joint=f"{h.body1}|{h.body2}", frames=h.frames,
            value=h.depth_peak, threshold=DEEP_FAIL_M,
        ))
    if not rep.hits:
        rep.add(Finding("self_collision", PASS,
                        f"no link interpenetration in {T} frames (floor excluded; "
                        f"{m.nexclude} vendor <exclude> pairs honoured)"))

    # The main item gets its own line whether or not it fired, so a PASS report is
    # evidence the check ran rather than evidence it was skipped. Matched on GEOMS,
    # not bodies: the racket and the hand share right_wrist_yaw_Link, and "the
    # knuckles brushed the torso" is not "the racket hit the torso".
    main_hit = next((h for h in rep.hits if h.involves(RACKET_GEOMS, TORSO_GEOMS)), None)
    if main_hit is None:
        rep.add(Finding("racket_torso", PASS,
                        "拍体-躯干 main item: clean (pair verified collidable at model load)"))

    # -- checks 2 & 3: backswing frame + clearances ---------------------------
    lo, hi = 0, T - 1
    if window == "backswing":
        if rep.contact_frame is None:
            rep.window = "all (no annotated contact frame)"
        else:
            hi = max(1, rep.contact_frame)
    scan = range(lo, hi + 1)

    prel = np.zeros((T, 3))
    for t in range(T):
        d.qpos[:] = qpos[t]
        mujoco.mj_forward(m, d)
        R = d.xmat[sm.torso_body].reshape(3, 3)
        prel[t] = R.T @ (d.site_xpos[sm.racket_site] - d.xpos[sm.torso_body])
    if rep.contact_frame is not None:
        rep.backswing_frame = backswing_frame(prel, rep.contact_frame)

    tracks: Dict[str, List[float]] = {name: [] for name, _, _ in CLEARANCE_PAIRS}
    sat_all: Dict[str, bool] = {name: True for name, _, _ in CLEARANCE_PAIRS}
    for t in scan:
        d.qpos[:] = qpos[t]
        mujoco.mj_forward(m, d)
        for name, aset, bset in CLEARANCE_PAIRS:
            dist, sat = group_clearance(m, d, [sm.geom_ids[g] for g in aset],
                                        [sm.geom_ids[g] for g in bset])
            tracks[name].append(dist)
            sat_all[name] = sat_all[name] and sat

    for name, _, _ in CLEARANCE_PAIRS:
        arr = np.asarray(tracks[name])
        i = int(np.argmin(arr))
        at_bs = None
        if rep.backswing_frame is not None and lo <= rep.backswing_frame <= hi:
            at_bs = float(arr[rep.backswing_frame - lo])
        rep.clearances.append(Clearance(
            name=name, min_dist=float(arr[i]), min_frame=int(i + lo),
            at_backswing=at_bs, saturated=bool(sat_all[name]),
        ))

    for c in rep.clearances:
        if not c.name.endswith("torso"):
            continue  # pelvis rows are carried for the table, not graded
        if c.min_dist < 0:
            continue  # already a self_collision FAIL/WARN; don't double-count
        if c.min_dist < CLEARANCE_WARN_M:
            rep.add(Finding(
                "clearance", WARN,
                f"{c.name} 最小余隙 {c.min_dist * 1000:.1f} mm < {CLEARANCE_WARN_M * 1000:g} mm "
                f"(余量太薄:未自撞,但引拍再深一点就会)",
                joint=c.name, frames=[c.min_frame], value=c.min_dist, threshold=CLEARANCE_WARN_M,
            ))
        else:
            rep.add(Finding(
                "clearance", PASS,
                f"{c.name} 最小余隙 {c.min_dist * 1000:.1f} mm @f{c.min_frame}"
                + ("" if not c.saturated else f" (>= {CLEARANCE_DISTMAX} m, saturated)"),
                joint=c.name, value=c.min_dist, threshold=CLEARANCE_WARN_M,
            ))

    rep.suggestion = _build_suggestion(rep)
    return rep


# the stroke surgery's own geometry: what extend_stroke moves, and what it may hit
ARM_CHAIN_GEOMS: Tuple[str, ...] = RACKET_GEOMS + FOREARM_GEOMS
TRUNK_GEOMS: Tuple[str, ...] = TORSO_GEOMS + PELVIS_GEOMS


def _build_suggestion(rep: SelfColReport) -> Suggestion:
    """Encode franco's 晚六 repair ladder for a self-colliding stroke."""
    fails = [h for h in rep.hits if h.level == FAIL]
    warns = [h for h in rep.hits if h.level == WARN]
    if not rep.hits:
        thin = [f for f in rep.findings if f.check == "clearance" and f.level == WARN]
        if thin:
            return Suggestion("tighten_stroke", [
                "无自撞,但躯干余隙已进警戒带——引拍若再加深必撞;"
                "行程要继续加就走 extend_stroke v2 的多关节分配,别再压这一路",
            ])
        return Suggestion("none", ["no interpenetration — clip is self-collision clean"])

    arm_torso = [h for h in fails + warns if h.involves(ARM_CHAIN_GEOMS, TRUNK_GEOMS)]
    if arm_torso:
        pairs = ", ".join(f"{h.body1}<->{h.body2}(帧 {_ranges(h.frames)})" for h in arm_torso)
        return Suggestion("redistribute_stroke", [
            f"反手引拍撞躯干:{pairs}",
            "按 franco 晚六法则改用 extend_stroke v2 的多关节行程分配:"
            "分配集=手臂链(肩内旋/肩俯仰/肘抬高)+ 腰偏航;"
            "禁用集=腰俯仰/侧滚 + 双腿(oracle 实测力矩 binding);"
            "约束=自碰撞 + 关节限位 + CoM 位移≈0(引拍机构不许带走质心)",
            "改完必须重跑本检查器 + 判炸器 + feasibility_oracle 三关",
        ])
    if fails:
        pairs = ", ".join(f"{h.body1}<->{h.body2}" for h in fails)
        return Suggestion("regenerate", [
            f"自撞发生在非拍臂链上({pairs})——这不是行程手术能修的,"
            f"查重定向/腿移植来源,或重新生成该 clip",
        ])
    return Suggestion("reaudit", [
        "只有擦碰级接触(孤立且浅):可能是凸包近似的伪影,但余量已经归零——"
        "复核该帧姿态,并在任何加深行程的手术前重跑本检查器",
    ])


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def _fmt_clear(c: Optional[Clearance], key: str = "min") -> str:
    if c is None:
        return "-"
    v = c.min_dist if key == "min" else c.at_backswing
    if v is None:
        return "-"
    if v < 0:
        return f"**{v * 1000:.1f}** (穿透)"
    txt = f"{v * 1000:.1f}"
    return txt + (" (≥sat)" if c.saturated and key == "min" else "")


def report_markdown(reports: List[SelfColReport], meta: dict) -> str:
    out: List[str] = []
    out.append("# Self-collision audit (L1, vendor MJCF)")
    out.append("")
    out.append(f"- generated: {meta['generated_utc']}")
    out.append(f"- mjcf: `{meta['mjcf']}`")
    out.append(f"- annotations: `{meta['annotations'] or '(none)'}`")
    out.append(f"- body order: `{meta.get('body_order') or '(auto: npz body_names / sidecar)'}`")
    out.append(f"- clearance window: {meta['window']}; "
               f"sustained>= {SUSTAIN_RUN} frames = FAIL, depth >= {DEEP_FAIL_M * 1000:g} mm = FAIL, "
               f"isolated+shallow = WARN (擦碰); clearance < {CLEARANCE_WARN_M * 1000:g} mm = WARN")
    out.append("")
    out.append("| clip | verdict | frames | colliding pairs | FAILs | WARNs | suggestion |")
    out.append("|---|---|---|---|---|---|---|")
    for r in reports:
        nf = sum(1 for f in r.findings if f.level == FAIL)
        nw = sum(1 for f in r.findings if f.level == WARN)
        out.append(f"| {r.stem} | **{r.verdict}** | {r.frames} | {len(r.hits)} | {nf} | {nw} "
                   f"| {r.suggestion.kind} |")
    out.append("")
    for r in reports:
        out.append(f"## {r.stem} — **{r.verdict}**")
        out.append("")
        out.append(f"- file: `{r.path}` ({r.frames} frames @ {r.fps} fps)")
        if r.contact_frame is not None:
            out.append(f"- annotated contact: frame {r.contact_frame} (phase {r.annotated_phase:.4f})")
        if r.backswing_frame is not None:
            out.append(f"- deepest-drawback frame t* = {r.backswing_frame}")
        out.append("")
        rows = [f for f in r.findings if f.level != PASS]
        if rows:
            out.append("| check | level | pair | frames | value | threshold | note |")
            out.append("|---|---|---|---|---|---|---|")
            for f in sorted(rows, key=lambda x: -_LEVEL_RANK[x.level]):
                msg = f.message.replace("|", "\\|")
                out.append(
                    f"| {f.check} | {f.level} | {f.joint or '-'} "
                    f"| {_ranges(f.frames) if f.frames else '-'} "
                    f"| {'' if f.value is None else format(f.value, '.4f')} "
                    f"| {'' if f.threshold is None else format(f.threshold, '.4f')} | {msg} |"
                )
        else:
            out.append("no interpenetration, no thin clearance — all checks PASS.")
        out.append("")
        out.append("| clearance pair | min [mm] | @frame | at t* [mm] |")
        out.append("|---|---|---|---|")
        for c in r.clearances:
            out.append(f"| {c.name} | {_fmt_clear(c)} | {c.min_frame} | {_fmt_clear(c, 'bs')} |")
        out.append("")
        out.append(f"**Repair suggestion**: `{r.suggestion.kind}`")
        out.append("")
        for ln in r.suggestion.lines:
            out.append(f"- {ln}")
        out.append("")
    return "\n".join(out)


def baseline_markdown(reports: List[SelfColReport], meta: dict) -> str:
    """The branch's headline deliverable: backswing clearance baseline per asset."""
    out: List[str] = []
    out.append("# 反手引拍最深帧 拍/小臂-躯干 最小距离基线表")
    out.append("")
    out.append(f"- generated: {meta['generated_utc']}  ·  mjcf: `{meta['mjcf']}`")
    out.append(f"- t* = 引拍最深帧(从触球帧回溯至拍体不再远离触球点的转向帧,躯干系)")
    out.append(f"- 扫描窗 = [0, 触球帧];距离 = MuJoCo 几何精确距离(二分,精度 {CLEARANCE_TOL * 1000:g} mm)")
    out.append("")
    out.append("| clip | verdict | T | 触球帧 c | 引拍最深帧 t* | 拍-躯干 @t* [mm] | 小臂-躯干 @t* [mm] "
               "| 拍-躯干 窗内最小 [mm] | @f | 小臂-躯干 窗内最小 [mm] | @f | 拍-骨盆 窗内最小 [mm] |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in reports:
        rt, ft = r.clearance("racket-torso"), r.clearance("forearm-torso")
        rp = r.clearance("racket-pelvis")
        out.append(
            f"| {r.stem} | {r.verdict} | {r.frames} "
            f"| {'-' if r.contact_frame is None else r.contact_frame} "
            f"| {'-' if r.backswing_frame is None else r.backswing_frame} "
            f"| {_fmt_clear(rt, 'bs')} | {_fmt_clear(ft, 'bs')} "
            f"| {_fmt_clear(rt)} | {'-' if rt is None else rt.min_frame} "
            f"| {_fmt_clear(ft)} | {'-' if ft is None else ft.min_frame} "
            f"| {_fmt_clear(rp)} |"
        )
    out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    # scripts/ -> whole_body_tracking/ -> hope_training/ -> repo root
    return Path(__file__).resolve().parents[3]


def _default_mjcf() -> Path:
    return _repo_root() / DEFAULT_MJCF_REL


def _default_annotations() -> Path:
    return Path(__file__).resolve().parents[1] / "cfg/strike_annotations.yaml"


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("npz", nargs="+", help="motion .npz file(s) to audit")
    p.add_argument("--mjcf", default=str(_default_mjcf()),
                   help="vendor A3 MJCF (default: repo agi/A3_MuJoCo_Sim/.../a3_pingpong.xml)")
    p.add_argument("--annotations", default=str(_default_annotations()),
                   help="strike annotations yaml (contact phase). Pass 'none' to disable.")
    p.add_argument("--body-order", default=None,
                   help="body-name order of the npz body_* columns: comma list or file. "
                        "Default: npz-embedded `body_names` key, else body_order.txt / "
                        "body_order_isaac.txt next to the npz. Unresolved = FAIL (fail-loud).")
    p.add_argument("--clearance-window", choices=("backswing", "all"), default="backswing",
                   help="frames scanned for the clearance table (default: [0, contact frame])")
    p.add_argument("--md", default=None, help="write the markdown report here")
    p.add_argument("--json", default=None, help="write the JSON summary here")
    p.add_argument("--baseline-md", default=None, help="write the backswing baseline table here")
    p.add_argument("--quiet", action="store_true", help="suppress stdout markdown")
    args = p.parse_args(argv)

    sm = load_selfcol_model(args.mjcf)
    ann = load_annotations(args.annotations)

    reports = [
        audit_clip(f, sm, annotations=ann, body_order=args.body_order,
                   window=args.clearance_window)
        for f in args.npz
    ]
    exit_code = max((_LEVEL_RANK[r.verdict] for r in reports), default=0)

    meta = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        "mjcf": str(args.mjcf),
        "annotations": None if str(args.annotations).lower() == "none" else str(args.annotations),
        "body_order": args.body_order,
        "window": args.clearance_window,
    }
    md = report_markdown(reports, meta)
    if not args.quiet:
        print(md)
    if args.md:
        Path(args.md).write_text(md)
    if args.baseline_md:
        Path(args.baseline_md).write_text(baseline_markdown(reports, meta))
    if args.json:
        Path(args.json).write_text(json.dumps(
            {**meta, "exit_code": exit_code, "clips": [r.to_json() for r in reports]}, indent=2))

    print(f"[selfcol] {len(reports)} clip(s): "
          + ", ".join(f"{r.stem}={r.verdict}" for r in reports)
          + f" -> exit {exit_code}", file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
