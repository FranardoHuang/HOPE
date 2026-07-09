#!/usr/bin/env python3
"""L0 static motion-feasibility audit ("判炸器") for retargeted motion .npz clips.

WHAT THIS IS
    An in-pipeline automatic detector (franco decision 2026-07-08) that grades every
    motion .npz against the AgiBot A3 URDF limits BEFORE the clip is allowed anywhere
    near training. It answers one question per clip: "will this reference blow up a
    policy that tries to track it?" — purely from the stored kinematics, no simulator.

PIPELINE CONTEXT (why clips get sick)
    video (30fps) -> GVHMR (whole-body SMPL) -> GMR (mink differential-IK retarget to
    A3, 31 joints) -> CSV -> csv_to_npz_mujoco --grip-rot -> hope_*_cal.npz (50fps).
    Forensics (two adversarial review rounds, CONFIRMED 2026-07-08): the first-frames
    glitch is GMR IK cold start — mink's Configuration initializes from qpos0 (upright
    zero pose) while the human is already crouched at frame 0, so frames ~0-3 are a
    convergence chase transient (leg-joint velocity peaks 7.4-15.9 rad/s across all
    seven production clips; only v5 backhand actually exceeds URDF velocity limits).
    GVHMR is innocent (~1 rad/s at the same frames); downstream copies faithfully.

REPAIR POLICY (franco, 2026-07-08 — encoded in the suggestion logic below)
    1. The PRIMARY fix is at the source: GMR warm-up + per-joint velocity limits.
    2. trim is a FALLBACK, only "when limits are actually exceeded" (超了才掐) —
       the videos are precision-cut, so trimming eats real motion. This tool only
       suggests a trim when FAIL-level violations are confined to the head/tail.
    3. Mid-clip velocity violations -> suggest slow-play (<= min(limit/peak));
       anything else -> reject / regenerate. Legal swing peaks (11.88-12.63 rad/s)
       must survive: never suggest a global velocity clamp.
    4. Foot-skate FAILs beyond the head window (franco/yikang 定案 2026-07-09) are
       leg-pose pathology (occlusion hallucination / retarget) — trim and slow-play
       CANNOT fix them; the only repairs are a leg transplant
       (transplant_legs.py --mode pinned, the v5hL/v5hLp recipe) or a reshoot.

CHECKS (graded PASS / WARN / FAIL; process exit code 0 / 1 / 2)
    1. joint position vs URDF hard limits (exceed = FAIL); soft-limit excursion
       (--soft-factor, default 0.9, matching training soft_joint_pos_limit_factor)
       = WARN; limit saturation (>=3 consecutive frames within 0.005 rad of a hard
       limit) = WARN — the signature of retarget-side clamping.
    2. joint velocity, TWO scopes (stored joint_vel AND forward-difference of
       joint_pos — either one exceeding counts): >100% URDF velocity limit = FAIL,
       >80% = WARN.
    3. acceleration heuristic: whole-body mean|acc| > 8 rad/s^2 = WARN.
       Calibration (production clips, 2026-07-08): healthy band hopex/v4/oblique
       2.5-3.5; trim6 (learnable but aggressive) 9.4.
    4. first-frame health: frame-0 max|joint_vel| > 2 rad/s = FAIL (downgraded to
       WARN by `first_frame_exempt: true` in the annotations registry — ALL legacy
       clips need this: they were filmed starting mid-swing, a hot start is their
       ground truth), 1-2 rad/s = WARN; frame-0 base linear speed > 0.5 m/s = WARN.
    5. stored joint_vel vs np.gradient(joint_pos)*fps: max mismatch > 1 rad/s = WARN
       (resampling-aliasing probe).
    6. repair suggestion (see REPAIR POLICY): head/tail-confined FAILs -> "trim K
       frames" with an equivalent python slice command AND the registry-phase shift
       reminder (new_phase = (contact-K)/(T-K-1)); mid-clip velocity-only FAILs ->
       "slow-play <= X" (X = min(limit/peak)); otherwise -> reject/regenerate.
       If a suggested trim would eat into the strike protection window (contact
       frame +-5, from --annotations), the suggestion escalates to regenerate.
       Foot-skate FAILs beyond the head window VETO both trim and slow-play
       (REPAIR POLICY 4) -> transplant_legs / reshoot.
    7. support-foot slip a.k.a. FOOT SKATE (franco/yikang 定案 2026-07-09: a
       nominally planted foot sliding horizontally is the #1 fall killer in
       reference tracking). Per foot (left/right_ankle_roll_Link columns of
       body_pos_w; body order via --body-order, an npz-embedded `body_names` key,
       or body_order.txt / body_order_isaac.txt next to the npz — an UNRESOLVED
       body order is a FAIL, fail-loud, never a silent skip): support frames are
       z < min(z) + 0.03 m; slip speed |v_xy| = finite difference * fps over
       intervals whose BOTH end frames are support. Support-slip peak
       > 0.15 m/s = FAIL, > 0.05 = WARN, <= 0.05 = PASS. Slip inside the first
       EDGE_WINDOW (10) frames is graded separately (`foot_skate_head`, the
       historical head-glitch band) and keeps the normal trim path; FAIL beyond
       the head window -> transplant/reshoot suggestion (see 6).
       Calibration (production clips, 定案 2026-07-09): v5 backhand imagined legs
       0.30-0.35 m/s MID-CLIP (deadly — inside the push-off window); v4 / v5hL
       true or pinned legs 0.010 (clean); v5rg forehand 0.034 (light, PASS).

REAL-ASSET REGRESSION (no motion npz in the laptop checkout — run on the pod):
    source /workspace/franco/env.sh && cd $HOPE_WBT && \
    python scripts/audit_motion_npz.py \
        /workspace/franco/motion_work/motions/regen_0708_candidates/hope_*hand_v4rg_cal.npz \
        /workspace/franco/motion_work/motions/regen_0708_candidates/hope_*hand_v5rg_cal.npz \
        /workspace/franco/motion_work/motions/v5_height_fix/hope_*hand_v5hL_cal.npz \
        --body-order /workspace/franco/body_order_isaac.txt
    EXPECT for check 7: hope_backhand_v5rg_cal FAIL (imagined-leg mid-clip slip
    0.30-0.35 m/s); hope_forehand_v5rg_cal PASS (0.034); all v4rg + v5hL PASS
    (true/pinned legs, ~0.010).

USAGE
    python scripts/audit_motion_npz.py CLIP.npz [CLIP2.npz ...] \
        [--urdf PATH] [--joint-names LIST|FILE] [--body-order LIST|FILE|none] \
        [--soft-factor 0.9] [--annotations cfg/strike_annotations.yaml] \
        [--md OUT.md] [--json OUT.json]

    Exit code: 0 = all PASS, 1 = worst is WARN, 2 = any FAIL (CI-friendly).

DEPENDENCIES
    numpy + standard library only. No isaac, no torch. URDF parsing = xml.etree.
    PyYAML is used for --annotations IF installed; otherwise a built-in minimal
    parser extracts the two fields this tool needs (`phase`, `first_frame_exempt`).

SISTER CHECK — SELF-COLLISION (L1, 2026-07-09, branch selfcol-check-0709)
    This tool cannot see the robot's own geometry: it grades joints and feet, never
    links passing through each other (franco 晚五: "判炸器不查自碰撞,v1 加深资产
    可能自撞全绿"). That hole is covered by the sibling scripts/audit_self_collision.py,
    which replays the clip in the vendor MJCF and grades interpenetration (racket-
    vs-torso is its main item) plus the backswing racket/forearm-to-torso clearance.
    It is a SEPARATE tier precisely because it needs mujoco + the MJCF, which this
    L0 gate deliberately does not. Run BOTH before letting a stroke-deepened asset
    (extend_stroke) anywhere near training.

FUTURE WIRING (documented on purpose, NOT implemented — institutional fix "修c",
pending franco sign-off; do not build these without it):
    * registry gate: cfg/strike_annotations.yaml gains a per-clip
      `feasibility_audit:` block ({verdict, date, report}) written from this tool's
      JSON output; a clip without a PASS/WARN audit cannot be registered.
    * gen_stage1_questions.py fail-closed: refuse to emit question banks for clips
      whose registry entry has no feasibility_audit verdict or has verdict FAIL.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple
from xml.etree import ElementTree

import numpy as np

# ---------------------------------------------------------------------------
# Joint order of the npz `joint_pos` / `joint_vel` columns (Isaac articulation
# order, 31 DOF).
#
# PROVENANCE (do not edit casually):
#   * source of truth: the donor ONNX `joint_names` metadata (policy-idx order),
#     tabulated in agi/a3_deploy_example/PINGPONG_DEPLOY_ALIGNMENT.md section 4
#     ("Action contract — 31-DOF", policy idx 0..30). Names there are without the
#     `_joint` suffix; the URDF names below add it back.
#   * cross-checked against source/whole_body_tracking/whole_body_tracking/robots/
#     agibot_a3.py — NOTE its AGIBOT_A3_JOINT_NAMES is the GMR-CSV/controller
#     order, NOT this order. The npz writers (scripts/csv_to_npz*.py) store
#     joint_pos in the Isaac ARTICULATION (breadth-first) order, which is the
#     interleaved waist/leg/arm order below.
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

# Time-indexed npz keys (same list as scripts/trim_motion_clip.py) — the keys a
# trim slice must cut together. `fps` is scalar and stays.
TIME_KEYS: Tuple[str, ...] = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)

# --- thresholds (constants; calibration noted where it exists) --------------
POS_EPS = 1e-6              # float32 slack on hard position limits [rad]
SAT_TOL = 0.005             # "pinned at the limit" tolerance [rad]
SAT_MIN_RUN = 3             # consecutive pinned frames -> WARN
VEL_WARN_FRAC = 0.8         # velocity WARN at 80% of the URDF limit
ACC_WARN_MEAN = 8.0         # mean|acc| WARN [rad/s^2]; healthy 2.5-3.5, trim6 9.4
FIRST_VEL_FAIL = 2.0        # frame-0 max|joint_vel| FAIL threshold [rad/s]
FIRST_VEL_WARN = 1.0        # frame-0 max|joint_vel| WARN threshold [rad/s]
FIRST_BASE_WARN = 0.5       # frame-0 base linear speed WARN threshold [m/s]
CONSISTENCY_WARN = 1.0      # stored-vs-gradient velocity mismatch WARN [rad/s]
EDGE_WINDOW = 10            # head/tail window: trims only up to K<=10 frames
STRIKE_PROTECT = 5          # protection margin around the annotated contact frame

# --- check 7: support-foot slip (foot skate), 定案 2026-07-09 ----------------
FOOT_BODIES: Tuple[str, str] = ("left_ankle_roll_Link", "right_ankle_roll_Link")
SKATE_SUPPORT_BAND = 0.03   # support frames: z < min(z) + band [m] (per foot)
SKATE_PEAK_FAIL = 0.15      # support-slip peak FAIL [m/s]; v5 bh imagined legs 0.30-0.35
SKATE_PEAK_WARN = 0.05      # support-slip peak WARN [m/s]; healthy v4/v5hL 0.010, v5rg fh 0.034
BODY_ORDER_SIDECARS = ("body_order.txt", "body_order_isaac.txt")  # looked up next to the npz

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
_LEVEL_RANK = {PASS: 0, WARN: 1, FAIL: 2}


@dataclass
class JointLimits:
    lower: float
    upper: float
    effort: Optional[float]
    velocity: Optional[float]


@dataclass
class Finding:
    check: str                   # e.g. "vel_limit_stored"
    level: str                   # PASS/WARN/FAIL (PASS findings are informational)
    message: str
    joint: Optional[str] = None
    frames: Optional[List[int]] = None
    value: Optional[float] = None
    threshold: Optional[float] = None

    def to_json(self) -> dict:
        return {
            "check": self.check,
            "level": self.level,
            "joint": self.joint,
            "frames": self.frames,
            "value": None if self.value is None else round(float(self.value), 4),
            "threshold": None if self.threshold is None else round(float(self.threshold), 4),
            "message": self.message,
        }


@dataclass
class Suggestion:
    kind: str                    # none | trim_head | trim_tail | trim_both | slow_down |
                                 # transplant_legs | reaudit | regenerate
    lines: List[str] = field(default_factory=list)
    trim_head: int = 0
    trim_tail: int = 0
    slow_factor: Optional[float] = None

    def to_json(self) -> dict:
        return {
            "kind": self.kind,
            "trim_head": self.trim_head,
            "trim_tail": self.trim_tail,
            "slow_factor": None if self.slow_factor is None else round(self.slow_factor, 4),
            "lines": self.lines,
        }


@dataclass
class ClipReport:
    path: str
    stem: str
    frames: int = 0
    fps: int = 0
    verdict: str = PASS
    findings: List[Finding] = field(default_factory=list)
    suggestion: Suggestion = field(default_factory=lambda: Suggestion("none"))
    contact_frame: Optional[int] = None
    annotated_phase: Optional[float] = None
    first_frame_exempt: bool = False

    @property
    def duration_s(self) -> float:
        return self.frames / self.fps if self.fps else 0.0

    def add(self, finding: Finding) -> None:
        self.findings.append(finding)
        if _LEVEL_RANK[finding.level] > _LEVEL_RANK[self.verdict]:
            self.verdict = finding.level

    def to_json(self) -> dict:
        return {
            "path": self.path,
            "stem": self.stem,
            "frames": self.frames,
            "fps": self.fps,
            "duration_s": round(self.duration_s, 3),
            "verdict": self.verdict,
            "contact_frame": self.contact_frame,
            "annotated_phase": self.annotated_phase,
            "first_frame_exempt": self.first_frame_exempt,
            "findings": [f.to_json() for f in self.findings if f.level != PASS],
            "suggestion": self.suggestion.to_json(),
        }


# ---------------------------------------------------------------------------
# URDF / annotations parsing
# ---------------------------------------------------------------------------

def parse_urdf_limits(urdf_path: str) -> Dict[str, JointLimits]:
    """Parse revolute/prismatic joint limits from a URDF via xml.etree (stdlib)."""
    root = ElementTree.parse(str(urdf_path)).getroot()
    limits: Dict[str, JointLimits] = {}
    for joint in root.iter("joint"):
        if joint.get("type") in (None, "fixed"):
            continue
        lim = joint.find("limit")
        if lim is None or lim.get("lower") is None or lim.get("upper") is None:
            continue
        eff = lim.get("effort")
        vel = lim.get("velocity")
        limits[joint.get("name")] = JointLimits(
            lower=float(lim.get("lower")),
            upper=float(lim.get("upper")),
            effort=float(eff) if eff is not None else None,
            velocity=float(vel) if vel is not None else None,
        )
    if not limits:
        raise ValueError(f"no joint limits found in URDF: {urdf_path}")
    return limits


def _mini_yaml_clips(text: str) -> Dict[str, dict]:
    """Fallback parser for the `clips:` section of cfg/strike_annotations.yaml.

    Handles exactly the subset this tool needs (top-level `clips:`, clip stems at
    2-space indent, scalar `key: value` fields at 4-space indent; comments and
    multi-line continuation values are skipped). Used only when PyYAML is absent.
    """
    clips: Dict[str, dict] = {}
    in_clips = False
    current: Optional[str] = None
    for raw in text.splitlines():
        if raw.lstrip().startswith("#") or not raw.strip():
            continue
        line = raw.split(" #", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        if indent == 0:
            in_clips = line.strip() == "clips:"
            current = None
            continue
        if not in_clips:
            continue
        stripped = line.strip()
        if indent == 2 and stripped.endswith(":") and ":" not in stripped[:-1]:
            current = stripped[:-1]
            clips[current] = {}
        elif indent == 4 and current is not None and ":" in stripped:
            key, _, val = stripped.partition(":")
            val = val.strip()
            if val:
                clips[current][key.strip()] = val
    return clips


def load_annotations(path: Optional[str]) -> Dict[str, dict]:
    """Load per-clip annotations ({stem: {phase, first_frame_exempt, ...}})."""
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
        return _mini_yaml_clips(text)


def _ann_phase(entry: dict) -> Optional[float]:
    try:
        return float(entry.get("phase"))
    except (TypeError, ValueError):
        return None


def _ann_exempt(entry: dict) -> bool:
    return str(entry.get("first_frame_exempt", "")).strip().lower() in ("true", "yes", "1")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _ranges(frames: Sequence[int]) -> str:
    """Compress [0,1,2,3,19,20] -> '0-3,19-20' for readable tables."""
    frames = sorted(set(int(f) for f in frames))
    if not frames:
        return "-"
    out, start, prev = [], frames[0], frames[0]
    for f in frames[1:]:
        if f == prev + 1:
            prev = f
            continue
        out.append(f"{start}-{prev}" if prev > start else f"{start}")
        start = prev = f
    out.append(f"{start}-{prev}" if prev > start else f"{start}")
    return ",".join(out)


def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Return [start, end] (inclusive) index pairs of True runs in a 1-D mask."""
    runs: List[Tuple[int, int]] = []
    start = None
    for i, v in enumerate(mask.tolist()):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def resolve_joint_names(spec: Optional[str], n_joints: int) -> List[str]:
    """--joint-names: None -> builtin table; else comma-list or file (one/line)."""
    if not spec:
        names = list(ISAAC_JOINT_NAMES)
    else:
        is_file = False
        if "," not in spec:
            try:
                is_file = Path(spec).is_file()
            except OSError:  # ENAMETOOLONG etc. — a long inline list is not a path
                is_file = False
        if is_file:
            names = [ln.strip() for ln in Path(spec).read_text().splitlines() if ln.strip() and not ln.startswith("#")]
        else:
            names = [n.strip() for n in spec.split(",") if n.strip()]
    if len(names) != n_joints:
        raise ValueError(f"joint-names count {len(names)} != npz joint dim {n_joints}")
    return names


def _limits_for(names: Sequence[str], limits: Dict[str, JointLimits]) -> List[JointLimits]:
    out = []
    for n in names:
        if n in limits:
            out.append(limits[n])
        elif n + "_joint" in limits:  # deploy-doc style names without the suffix
            out.append(limits[n + "_joint"])
        else:
            raise KeyError(f"joint '{n}' not found in URDF limits")
    return out


def _names_from_file(path: Path) -> List[str]:
    return [ln.strip() for ln in path.read_text().splitlines()
            if ln.strip() and not ln.startswith("#")]


def resolve_body_order(spec: Optional[str], npz_path: str, n_bodies: int, data=None) -> List[str]:
    """Resolve the body-name order of the npz body_* columns (check 7 input).

    The npz writers store body arrays in the Isaac ARTICULATION body order but do
    not record it in the file; the pipeline's source of truth is the donor-ONNX /
    --discover-map body list (csv_to_npz_mujoco.py writes it as body_order.txt,
    "MJ body names in Isaac column order"; the pod copy lives at
    /workspace/franco/body_order_isaac.txt).

    Priority: explicit `spec` (comma list, or a file with one name per line) >
    npz-embedded `body_names` key (newer writers) > a BODY_ORDER_SIDECARS file next
    to the npz. Raises ValueError when nothing resolves or the count mismatches —
    check 7 must fail LOUD, never silently skip.
    """
    names: Optional[List[str]] = None
    src = "?"
    if spec:
        is_file = False
        if "," not in spec:
            try:
                is_file = Path(spec).is_file()
            except OSError:  # ENAMETOOLONG etc. — a long inline list is not a path
                is_file = False
            if not is_file:
                raise ValueError(f"--body-order is neither a file nor a comma list: {spec!r}")
        names = _names_from_file(Path(spec)) if is_file else [n.strip() for n in spec.split(",") if n.strip()]
        src = spec if is_file else "(inline list)"
    elif data is not None and "body_names" in getattr(data, "files", []):
        raw = np.asarray(data["body_names"]).reshape(-1).tolist()
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in raw]
        src = "npz body_names key"
    else:
        for cand in BODY_ORDER_SIDECARS:
            p = Path(npz_path).resolve().parent / cand
            if p.is_file():
                names, src = _names_from_file(p), str(p)
                break
    if names is None:
        raise ValueError(
            "body order unknown — pass --body-order (comma list or file; the "
            "csv_to_npz_mujoco --discover-map output) or place "
            f"{'/'.join(BODY_ORDER_SIDECARS)} next to the npz"
        )
    if len(names) != n_bodies:
        raise ValueError(f"body-order count {len(names)} != npz body dim {n_bodies} (source: {src})")
    return names


# ---------------------------------------------------------------------------
# check 7: support-foot slip (foot skate)
# ---------------------------------------------------------------------------

def _check_foot_skate(
    rep: ClipReport,
    data,
    body_pos: Optional[np.ndarray],
    body_order: Optional[str],
    npz_path: str,
    fps: int,
    n_frames: int,
    fail_frames: Set[int],
) -> Tuple[Set[int], Optional[str]]:
    """Check 7 (定案 2026-07-09): grade horizontal slip of the nominally planted feet.

    Adds findings to `rep` and FAIL frames to `fail_frames` (head-window FAILs keep
    the normal trim path). Returns (skate_fail_beyond_head, note):
      * skate_fail_beyond_head — FAIL frames outside the head window; a non-empty set
        VETOES trim/slow-play in the suggestion builder (leg-pose pathology);
      * note — None, or 'unresolved' (body order missing: audit incomplete, re-audit
        after fixing inputs) or 'corrupt' (body_pos_w unusable: regenerate).
    """
    if body_order is not None and str(body_order).strip().lower() == "none":
        rep.add(Finding("foot_skate", PASS,
                        "foot-skate check explicitly disabled (--body-order none)"))
        return set(), None
    if body_pos is None:
        rep.add(Finding("foot_skate", FAIL,
                        "npz has no body_pos_w — support-foot slip cannot be audited "
                        "(fail-loud: check 7 is never skipped silently)"))
        return set(), "corrupt"
    if body_pos.shape[0] != n_frames:
        rep.add(Finding("foot_skate", FAIL,
                        f"body_pos_w frame count {body_pos.shape[0]} != joint_pos {n_frames}"))
        return set(), "corrupt"
    if not np.isfinite(body_pos).all():
        rep.add(Finding("foot_skate", FAIL, "non-finite values (NaN/Inf) in body_pos_w"))
        return set(), "corrupt"

    try:
        names = resolve_body_order(body_order, npz_path, int(body_pos.shape[1]), data=data)
        feet_cols = []
        for foot in FOOT_BODIES:
            if foot not in names:
                raise ValueError(f"body order does not contain '{foot}'")
            feet_cols.append(names.index(foot))
    except (OSError, ValueError) as exc:
        rep.add(Finding(
            "foot_skate", FAIL,
            f"body order unresolved ({exc}) — FAIL-LOUD: foot skate is the #1 fall "
            f"killer and check 7 is never skipped silently",
        ))
        return set(), "unresolved"

    if n_frames < 2:
        rep.add(Finding("foot_skate", PASS, "single-frame clip — no slip to measure"))
        return set(), None

    veto: Set[int] = set()
    sample_idx = np.arange(n_frames - 1)
    head_mask = (sample_idx + 1) < EDGE_WINDOW  # both end frames inside the head window
    for foot, col in zip(FOOT_BODIES, feet_cols):
        p = body_pos[:, col, :]
        z = p[:, 2]
        support = z < (float(z.min()) + SKATE_SUPPORT_BAND)
        # sample i spans frames i -> i+1; slip counts only while the foot is
        # nominally planted at BOTH ends (a lifting/landing foot is a legal step)
        planted = support[:-1] & support[1:]
        v_xy = np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1) * fps
        for check, seg, label in (
            ("foot_skate_head", planted & head_mask,
             f"in the head window (first {EDGE_WINDOW} frames)"),
            ("foot_skate", planted & ~head_mask, "beyond the head window"),
        ):
            if not seg.any():
                if check == "foot_skate":
                    rep.add(Finding(check, PASS,
                                    "no planted support interval beyond the head window",
                                    joint=foot))
                continue
            speeds = v_xy[seg]
            peak, mean = float(speeds.max()), float(speeds.mean())
            if peak > SKATE_PEAK_FAIL:
                over = seg & (v_xy > SKATE_PEAK_FAIL)
                fr = sorted({f for i in np.flatnonzero(over).tolist() for f in (i, i + 1)})
                if check == "foot_skate":
                    msg = (f"support-foot horizontal slip peak {peak:.3f} m/s (mean {mean:.3f}) "
                           f"{label} > {SKATE_PEAK_FAIL:g} m/s — FOOT SKATE: the nominally "
                           f"planted foot slides during stance (定案 calibration: v5 bh imagined "
                           f"legs 0.30-0.35 = deadly, inside the push-off window; healthy "
                           f"v4/v5hL band 0.010)")
                    veto.update(fr)
                else:
                    msg = (f"support-foot slip peak {peak:.3f} m/s (mean {mean:.3f}) {label} > "
                           f"{SKATE_PEAK_FAIL:g} m/s — historical head-glitch band (GMR IK cold "
                           f"start); if FAILs stay confined here the trim path applies")
                rep.add(Finding(check, FAIL, msg, joint=foot, frames=fr,
                                value=peak, threshold=SKATE_PEAK_FAIL))
                fail_frames.update(fr)
            elif peak > SKATE_PEAK_WARN:
                over = seg & (v_xy > SKATE_PEAK_WARN)
                fr = sorted({f for i in np.flatnonzero(over).tolist() for f in (i, i + 1)})
                rep.add(Finding(
                    check, WARN,
                    f"support-foot slip peak {peak:.3f} m/s (mean {mean:.3f}) {label} in the "
                    f"{SKATE_PEAK_WARN:g}-{SKATE_PEAK_FAIL:g} m/s band (healthy band ~0.010; "
                    f"light-but-PASS example v5rg fh 0.034)",
                    joint=foot, frames=fr, value=peak, threshold=SKATE_PEAK_WARN,
                ))
            elif check == "foot_skate":
                rep.add(Finding(check, PASS,
                                f"support-foot slip peak {peak:.3f} m/s (mean {mean:.3f}) {label}",
                                joint=foot, value=peak, threshold=SKATE_PEAK_WARN))
    return veto, None


# ---------------------------------------------------------------------------
# per-clip audit
# ---------------------------------------------------------------------------

def audit_clip(
    npz_path: str,
    limits: Dict[str, JointLimits],
    joint_names: Optional[str] = None,
    soft_factor: float = 0.9,
    annotations: Optional[Dict[str, dict]] = None,
    body_order: Optional[str] = None,
) -> ClipReport:
    stem = Path(npz_path).stem
    rep = ClipReport(path=str(npz_path), stem=stem)

    entry = (annotations or {}).get(stem, {})
    rep.first_frame_exempt = _ann_exempt(entry)
    rep.annotated_phase = _ann_phase(entry)

    try:
        data = np.load(npz_path, allow_pickle=True)
        q = np.asarray(data["joint_pos"], dtype=np.float64)
        dq = np.asarray(data["joint_vel"], dtype=np.float64)
        fps = int(np.asarray(data["fps"]).reshape(-1)[0])
        base_lin = None
        if "body_lin_vel_w" in getattr(data, "files", []):
            # body index 0 = root link (pelvis) in the articulation body order
            base_lin = np.linalg.norm(np.asarray(data["body_lin_vel_w"], dtype=np.float64)[:, 0, :], axis=1)
        body_pos = None
        if "body_pos_w" in getattr(data, "files", []):
            body_pos = np.asarray(data["body_pos_w"], dtype=np.float64)
    except Exception as exc:  # malformed npz is a FAIL, not a crash
        rep.add(Finding("load", FAIL, f"cannot load/parse npz: {exc}"))
        rep.suggestion = Suggestion("regenerate", ["clip unreadable — regenerate"])
        return rep

    if not (np.isfinite(q).all() and np.isfinite(dq).all()):
        rep.add(Finding("load", FAIL, "non-finite values (NaN/Inf) in joint_pos/joint_vel"))
        rep.suggestion = Suggestion("regenerate", ["corrupt arrays (NaN/Inf) — regenerate the clip"])
        return rep

    T, J = q.shape
    rep.frames, rep.fps = T, fps
    if rep.annotated_phase is not None:
        rep.contact_frame = int(round(rep.annotated_phase * (T - 1)))

    try:
        names = resolve_joint_names(joint_names, J)
        jlims = _limits_for(names, limits)
    except (ValueError, KeyError) as exc:
        rep.add(Finding("load", FAIL, f"joint-name/limit resolution failed: {exc}"))
        rep.suggestion = Suggestion("regenerate", ["joint table mismatch — fix inputs and re-audit"])
        return rep

    lo = np.array([l.lower for l in jlims])
    hi = np.array([l.upper for l in jlims])
    vlim = np.array([l.velocity if l.velocity is not None else np.inf for l in jlims])
    mid, half = (lo + hi) / 2.0, (hi - lo) / 2.0
    soft_lo, soft_hi = mid - half * soft_factor, mid + half * soft_factor

    fail_frames: Set[int] = set()          # frames carrying FAIL-level violations
    vel_fail_ratios: List[float] = []      # peak/limit for slow-down sizing

    # -- check 1: position hard limits / soft limits / saturation ------------
    for j, name in enumerate(names):
        col = q[:, j]
        hard = (col < lo[j] - POS_EPS) | (col > hi[j] + POS_EPS)
        if hard.any():
            fr = np.flatnonzero(hard)
            worst = float(col[fr][np.argmax(np.maximum(lo[j] - col[fr], col[fr] - hi[j]))])
            rep.add(Finding(
                "pos_limit", FAIL,
                f"position outside URDF hard limits [{lo[j]:.3f}, {hi[j]:.3f}] rad",
                joint=name, frames=fr.tolist(), value=worst, threshold=float(hi[j]),
            ))
            fail_frames.update(fr.tolist())
        else:
            soft = (col < soft_lo[j]) | (col > soft_hi[j])
            if soft.any():
                fr = np.flatnonzero(soft)
                worst = float(col[fr][np.argmax(np.abs(col[fr] - mid[j]))])
                rep.add(Finding(
                    "pos_soft", WARN,
                    f"position beyond soft limit (factor {soft_factor:g}) "
                    f"[{soft_lo[j]:.3f}, {soft_hi[j]:.3f}] rad",
                    joint=name, frames=fr.tolist(), value=worst, threshold=float(soft_hi[j]),
                ))
        near = (col >= hi[j] - SAT_TOL) | (col <= lo[j] + SAT_TOL)
        sat_runs = [(a, b) for a, b in _runs(near) if b - a + 1 >= SAT_MIN_RUN]
        if sat_runs:
            fr = [f for a, b in sat_runs for f in range(a, b + 1)]
            rep.add(Finding(
                "pos_saturation", WARN,
                f"pinned within {SAT_TOL:g} rad of a hard limit for >= {SAT_MIN_RUN} "
                f"consecutive frames (retarget clamping signature)",
                joint=name, frames=fr, value=float(col[fr[0]]), threshold=float(hi[j]),
            ))

    # -- check 2: velocity, two scopes (stored + forward difference) ----------
    def _vel_scope(vel: np.ndarray, scope: str, frame_of) -> None:
        nonlocal vel_fail_ratios
        for j, name in enumerate(names):
            if not np.isfinite(vlim[j]):
                continue
            a = np.abs(vel[:, j])
            over = a > vlim[j]
            if over.any():
                idx = np.flatnonzero(over)
                peak = float(a[idx].max())
                fr = sorted({f for i in idx.tolist() for f in frame_of(i)})
                rep.add(Finding(
                    f"vel_limit_{scope}", FAIL,
                    f"|velocity| ({scope}) exceeds URDF limit {vlim[j]:.2f} rad/s",
                    joint=name, frames=fr, value=peak, threshold=float(vlim[j]),
                ))
                fail_frames.update(fr)
                vel_fail_ratios.append(peak / float(vlim[j]))
            else:
                warn = a > VEL_WARN_FRAC * vlim[j]
                if warn.any():
                    idx = np.flatnonzero(warn)
                    peak = float(a[idx].max())
                    fr = sorted({f for i in idx.tolist() for f in frame_of(i)})
                    rep.add(Finding(
                        f"vel_limit_{scope}", WARN,
                        f"|velocity| ({scope}) > {VEL_WARN_FRAC:.0%} of URDF limit "
                        f"{vlim[j]:.2f} rad/s",
                        joint=name, frames=fr, value=peak, threshold=float(VEL_WARN_FRAC * vlim[j]),
                    ))

    _vel_scope(dq, "stored", lambda i: (i,))
    if T >= 2:
        fd = np.diff(q, axis=0) * fps  # sample i spans frames i -> i+1: blame both
        _vel_scope(fd, "fd", lambda i: (i, i + 1))

    # -- check 3: acceleration heuristic --------------------------------------
    if T >= 3:
        acc = np.diff(dq, axis=0) * fps
        mean_acc = float(np.abs(acc).mean())
        if mean_acc > ACC_WARN_MEAN:
            rep.add(Finding(
                "acc_mean", WARN,
                f"whole-body mean|acc| {mean_acc:.2f} rad/s^2 > {ACC_WARN_MEAN:g} "
                f"(healthy band 2.5-3.5; trim6 'learnable but aggressive' = 9.4)",
                value=mean_acc, threshold=ACC_WARN_MEAN,
            ))
        else:
            rep.add(Finding("acc_mean", PASS, f"whole-body mean|acc| {mean_acc:.2f} rad/s^2",
                            value=mean_acc, threshold=ACC_WARN_MEAN))

    # -- check 4: first-frame health ------------------------------------------
    v0 = float(np.abs(dq[0]).max())
    v0_joint = names[int(np.abs(dq[0]).argmax())]
    if v0 > FIRST_VEL_FAIL:
        if rep.first_frame_exempt:
            rep.add(Finding(
                "first_frame_vel", WARN,
                f"frame-0 max|joint_vel| {v0:.2f} rad/s > {FIRST_VEL_FAIL:g} (ghost start) "
                f"— DOWNGRADED to WARN by first_frame_exempt (legacy clip filmed mid-swing)",
                joint=v0_joint, frames=[0], value=v0, threshold=FIRST_VEL_FAIL,
            ))
        else:
            rep.add(Finding(
                "first_frame_vel", FAIL,
                f"frame-0 max|joint_vel| {v0:.2f} rad/s > {FIRST_VEL_FAIL:g} "
                f"(IK cold-start ghost; no annotation exemption)",
                joint=v0_joint, frames=[0], value=v0, threshold=FIRST_VEL_FAIL,
            ))
            fail_frames.add(0)
    elif v0 > FIRST_VEL_WARN:
        rep.add(Finding(
            "first_frame_vel", WARN,
            f"frame-0 max|joint_vel| {v0:.2f} rad/s in the {FIRST_VEL_WARN:g}-{FIRST_VEL_FAIL:g} band",
            joint=v0_joint, frames=[0], value=v0, threshold=FIRST_VEL_WARN,
        ))
    if base_lin is not None and float(base_lin[0]) > FIRST_BASE_WARN:
        rep.add(Finding(
            "first_frame_base", WARN,
            f"frame-0 base linear speed {base_lin[0]:.2f} m/s > {FIRST_BASE_WARN:g}",
            frames=[0], value=float(base_lin[0]), threshold=FIRST_BASE_WARN,
        ))

    # -- check 5: stored velocity vs np.gradient (resampling probe) -----------
    if T >= 2:
        grad = np.gradient(q, axis=0) * fps
        mismatch = float(np.abs(dq - grad).max())
        if mismatch > CONSISTENCY_WARN:
            jj = int(np.unravel_index(np.abs(dq - grad).argmax(), dq.shape)[1])
            rep.add(Finding(
                "vel_consistency", WARN,
                f"stored joint_vel vs np.gradient(joint_pos)*fps max mismatch "
                f"{mismatch:.2f} rad/s > {CONSISTENCY_WARN:g} (resampling-aliasing probe)",
                joint=names[jj], value=mismatch, threshold=CONSISTENCY_WARN,
            ))

    # -- check 7: support-foot slip (foot skate) -------------------------------
    skate_veto, skate_note = _check_foot_skate(
        rep, data, body_pos, body_order, str(npz_path), fps, T, fail_frames
    )

    # -- check 6: repair suggestion --------------------------------------------
    rep.suggestion = _build_suggestion(
        rep, fail_frames, vel_fail_ratios, T, fps, dq, base_lin, skate_veto
    )
    if skate_note and rep.suggestion.kind == "none":
        # a FAIL finding exists but carries no frames — replace the misleading
        # "nothing to trim" note with the actionable one
        if skate_note == "unresolved":
            rep.suggestion = Suggestion("reaudit", [
                "body order unresolved — check 7 (foot skate, the #1 fall killer) did NOT "
                "run; supply --body-order (or drop body_order.txt next to the npz) and "
                "RE-AUDIT before trusting this clip",
            ])
        else:
            rep.suggestion = Suggestion("regenerate", [
                "body_pos_w missing/corrupt — regenerate the clip (check 7 could not run)",
            ])
    return rep


def _build_suggestion(
    rep: ClipReport,
    fail_frames: Set[int],
    vel_fail_ratios: List[float],
    T: int,
    fps: int,
    dq: np.ndarray,
    base_lin: Optional[np.ndarray],
    skate_veto: Optional[Set[int]] = None,
) -> Suggestion:
    """Encode franco's repair ladder: trim only when limits are exceeded."""
    if not fail_frames:
        note = "no FAIL-level violations — nothing to trim (trim only when limits are exceeded)"
        return Suggestion("none", [note])

    # REPAIR POLICY 4 (定案 2026-07-09): foot skate beyond the head window is
    # leg-pose pathology — it vetoes trim AND slow-play, whatever else FAILed.
    if skate_veto:
        return Suggestion("transplant_legs", [
            "支撑脚滑移 FAIL 在头部窗口之外(帧 " + _ranges(sorted(skate_veto)) + ")——"
            "腿姿病(遮挡幻觉/重定向),trim/慢放救不了,走腿移植"
            "(transplant_legs.py --mode pinned)或重拍",
            "mid-clip support-foot skate = leg-pose pathology (occlusion hallucination / "
            "retarget): the reference's planted foot slides inside its stance window, so a "
            "tracking policy inherits a sliding contact and falls. Trimming cannot remove "
            "mid-clip frames and slow-play only slows the same slide — fix the LEGS: "
            "transplant healthy stance legs (transplant_legs.py --mode pinned, the "
            "v5hL/v5hLp recipe) or reshoot the source video",
        ])

    head = {f for f in fail_frames if f < EDGE_WINDOW}
    tail = {f for f in fail_frames if f >= T - EDGE_WINDOW}
    mid = fail_frames - head - tail

    pos_fail_frames: Set[int] = set()
    for f in rep.findings:
        if f.check == "pos_limit" and f.level == FAIL and f.frames:
            pos_fail_frames.update(f.frames)

    def _healthy(k: int) -> bool:
        ok = float(np.abs(dq[k]).max()) < FIRST_VEL_FAIL
        if ok and base_lin is not None:
            ok = float(base_lin[k]) < FIRST_BASE_WARN
        return ok

    if not mid:
        lines: List[str] = []
        trim_head = trim_tail = 0
        if head:
            k = max(head) + 1
            while k <= EDGE_WINDOW and k < T and not _healthy(k):
                k += 1
            if k > EDGE_WINDOW or k >= T:
                return Suggestion("regenerate", [
                    f"head FAIL cluster {_ranges(sorted(head))} has no healthy restart frame "
                    f"within the first {EDGE_WINDOW} frames — transient too long to trim; "
                    f"reject / regenerate at the source (GMR warm-up)",
                ])
            trim_head = k
            lines.append(
                f"suggest TRIM FIRST {k} FRAMES: FAIL violations confined to frames "
                f"{_ranges(sorted(head))}; frame {k} is the first healthy restart frame "
                f"(max|joint_vel| {float(np.abs(dq[k]).max()):.2f} rad/s"
                + (f", base speed {float(base_lin[k]):.2f} m/s" if base_lin is not None else "")
                + ")"
            )
        if tail:
            trim_tail = T - min(tail)
            lines.append(
                f"suggest TRIM LAST {trim_tail} FRAMES: FAIL violations confined to frames "
                f"{_ranges(sorted(tail))}"
            )

        if trim_head + trim_tail >= T:
            return Suggestion("regenerate", [
                f"suggested trims (head {trim_head} + tail {trim_tail}) cover the whole "
                f"{T}-frame clip (short clip: head/tail edge windows overlap) — nothing would "
                f"remain; reject / regenerate at the source",
            ])

        # strike protection window: refuse a trim that eats into contact +- margin
        if rep.contact_frame is not None:
            c = rep.contact_frame
            if (trim_head and trim_head > c - STRIKE_PROTECT) or (
                trim_tail and (T - 1 - trim_tail) < c + STRIKE_PROTECT
            ):
                return Suggestion("regenerate", [
                    f"trim (head {trim_head} / tail {trim_tail}) would eat into the strike "
                    f"protection window [{c - STRIKE_PROTECT}, {c + STRIKE_PROTECT}] "
                    f"(contact frame {c}) — do NOT trim; reject / regenerate at the source",
                ])

        stop = T - trim_tail
        slice_txt = f"{trim_head}:{stop if trim_tail else ''}"
        in_p, out_p = rep.path, str(Path(rep.path).with_name(Path(rep.path).stem + "_trim.npz"))
        lines.append(
            "equivalent slice:  python -c \"import numpy as np; "
            f"d=dict(np.load('{in_p}')); "
            f"[d.update({{k: d[k][{slice_txt}]}}) for k in {list(TIME_KEYS)} if k in d]; "
            f"np.savez('{out_p}', **d)\""
        )
        lines.append(
            "(or scripts/trim_motion_clip.py for a strike-centered window cut)"
        )
        # registry-phase shift reminder
        if rep.contact_frame is not None and rep.annotated_phase is not None:
            c, new_T = rep.contact_frame, T - trim_head - trim_tail
            new_phase = (c - trim_head) / (new_T - 1) if new_T > 1 else float("nan")
            lines.append(
                f"REGISTRY REMINDER: shift the annotated phase — contact frame {c} -> "
                f"{c - trim_head}; phase {rep.annotated_phase:.3f} -> {new_phase:.3f} "
                f"(= (contact-K_head)/(T-K_head-K_tail-1)). Update cfg/strike_annotations.yaml "
                f"AND any strike_phase_per_clip training overrides for the trimmed file."
            )
        else:
            lines.append(
                "REGISTRY REMINDER: if this clip has an annotated contact phase, it must shift: "
                "new_phase = (contact_frame - K_head) / (T - K_head - K_tail - 1). Update "
                "cfg/strike_annotations.yaml AND any strike_phase_per_clip overrides."
            )
        kind = "trim_both" if (trim_head and trim_tail) else ("trim_head" if trim_head else "trim_tail")
        return Suggestion(kind, lines, trim_head=trim_head, trim_tail=trim_tail)

    # mid-clip FAILs: slow-play only fixes velocity, never position
    if not pos_fail_frames and vel_fail_ratios:
        factor = 1.0 / max(vel_fail_ratios)  # = min(limit/peak)
        return Suggestion("slow_down", [
            f"mid-clip velocity FAILs — suggest SLOW-PLAY to <= {factor:.2f}x speed "
            f"(time-stretch >= {1.0 / factor:.2f}x, i.e. resample {T} frames @ {fps} fps onto a "
            f"longer timeline; X = min(URDF limit / peak) over all velocity FAILs)",
            "caveat: slow-play dilutes strike dynamics for the WHOLE clip (legal swing peaks "
            "included) — prefer the source-level fix (GMR warm-up + per-joint velocity limits)",
        ], slow_factor=factor)

    return Suggestion("regenerate", [
        f"FAIL violations include mid-clip frames {_ranges(sorted(mid))} that neither trim "
        f"(head/tail only) nor slow-play (position limits are time-invariant) can fix — "
        f"reject the clip / regenerate at the source",
    ])


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def report_markdown(reports: List[ClipReport], meta: dict) -> str:
    out: List[str] = []
    out.append("# Motion feasibility audit (L0 static)")
    out.append("")
    out.append(f"- generated: {meta['generated_utc']}")
    out.append(f"- urdf: `{meta['urdf']}`")
    out.append(f"- soft position-limit factor: {meta['soft_factor']}")
    out.append(f"- annotations: `{meta['annotations'] or '(none)'}`")
    out.append(f"- body order (check 7): "
               f"`{meta.get('body_order') or '(auto: npz body_names key / sidecar next to npz)'}`")
    out.append("")
    out.append("| clip | verdict | frames | fps | dur [s] | FAILs | WARNs | suggestion |")
    out.append("|---|---|---|---|---|---|---|---|")
    for r in reports:
        nf = sum(1 for f in r.findings if f.level == FAIL)
        nw = sum(1 for f in r.findings if f.level == WARN)
        out.append(
            f"| {r.stem} | **{r.verdict}** | {r.frames} | {r.fps} | {r.duration_s:.2f} "
            f"| {nf} | {nw} | {r.suggestion.kind} |"
        )
    out.append("")
    for r in reports:
        out.append(f"## {r.stem} — **{r.verdict}**")
        out.append("")
        out.append(f"- file: `{r.path}`  ({r.frames} frames @ {r.fps} fps = {r.duration_s:.2f} s)")
        if r.contact_frame is not None:
            out.append(
                f"- annotated contact: frame {r.contact_frame} (phase {r.annotated_phase:.3f}); "
                f"strike protection window [{r.contact_frame - STRIKE_PROTECT}, "
                f"{r.contact_frame + STRIKE_PROTECT}]"
            )
        if r.first_frame_exempt:
            out.append("- first_frame_exempt: true (legacy clip filmed mid-swing)")
        out.append("")
        rows = [f for f in r.findings if f.level != PASS]
        if rows:
            out.append("| check | level | joint | frames | value | threshold | note |")
            out.append("|---|---|---|---|---|---|---|")
            for f in sorted(rows, key=lambda x: -_LEVEL_RANK[x.level]):
                msg = f.message.replace("|", "\\|")  # keep GFM table cells intact
                out.append(
                    f"| {f.check} | {f.level} | {f.joint or '-'} "
                    f"| {_ranges(f.frames) if f.frames else '-'} "
                    f"| {'' if f.value is None else format(f.value, '.3f')} "
                    f"| {'' if f.threshold is None else format(f.threshold, '.3f')} "
                    f"| {msg} |"
                )
        else:
            out.append("no violations — all checks PASS.")
        out.append("")
        out.append("**Repair suggestion** (policy: trim only when limits are exceeded):")
        out.append("")
        out.append(f"- kind: `{r.suggestion.kind}`")
        for ln in r.suggestion.lines:
            out.append(f"- {ln}")
        out.append("")
    return "\n".join(out)


def summary_json(reports: List[ClipReport], meta: dict, exit_code: int) -> dict:
    return {
        **meta,
        "exit_code": exit_code,
        "clips": [r.to_json() for r in reports],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    # scripts/ -> whole_body_tracking/ -> hope_training/ -> repo root
    return Path(__file__).resolve().parents[3]


def _default_urdf() -> Path:
    return _repo_root() / "agi/URDF/A3T2.5-URDF-std-pingpang/urdf/URDF-JOINT-LINK.urdf"


def _default_annotations() -> Path:
    return Path(__file__).resolve().parents[1] / "cfg/strike_annotations.yaml"


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("npz", nargs="+", help="motion .npz file(s) to audit")
    p.add_argument("--urdf", default=str(_default_urdf()),
                   help="A3 URDF with joint limits (default: repo agi/URDF/.../URDF-JOINT-LINK.urdf)")
    p.add_argument("--joint-names", default=None,
                   help="override joint order: comma-separated names or a file (one per line); "
                        "default = built-in 31-joint Isaac articulation table")
    p.add_argument("--body-order", default=None,
                   help="body-name order of the npz body_* columns for check 7 (foot skate): "
                        "comma list or a file with one name per line (the csv_to_npz_mujoco "
                        "--discover-map body_order.txt; pod copy "
                        "/workspace/franco/body_order_isaac.txt). Default: npz-embedded "
                        "`body_names` key, else body_order.txt / body_order_isaac.txt next to "
                        "the npz. Unresolved = FAIL (fail-loud). Pass 'none' to explicitly "
                        "disable check 7.")
    p.add_argument("--soft-factor", type=float, default=0.9,
                   help="soft position-limit factor (WARN band), matches training "
                        "soft_joint_pos_limit_factor (default 0.9)")
    p.add_argument("--annotations", default=str(_default_annotations()),
                   help="strike annotations yaml (contact phase -> protection window; "
                        "first_frame_exempt flag). Pass 'none' to disable.")
    p.add_argument("--md", default=None, help="write the markdown report here")
    p.add_argument("--json", default=None, help="write the JSON summary here")
    p.add_argument("--quiet", action="store_true", help="suppress stdout markdown")
    args = p.parse_args(argv)

    limits = parse_urdf_limits(args.urdf)
    ann = load_annotations(args.annotations)

    reports = [
        audit_clip(f, limits, joint_names=args.joint_names,
                   soft_factor=args.soft_factor, annotations=ann,
                   body_order=args.body_order)
        for f in args.npz
    ]
    exit_code = max((_LEVEL_RANK[r.verdict] for r in reports), default=0)

    meta = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        "urdf": str(args.urdf),
        "soft_factor": args.soft_factor,
        "annotations": None if str(args.annotations).lower() == "none" else str(args.annotations),
        "body_order": args.body_order,
    }
    md = report_markdown(reports, meta)
    if not args.quiet:
        print(md)
    if args.md:
        Path(args.md).write_text(md)
    if args.json:
        Path(args.json).write_text(json.dumps(summary_json(reports, meta, exit_code), indent=2))

    print(f"[audit] {len(reports)} clip(s): "
          + ", ".join(f"{r.stem}={r.verdict}" for r in reports)
          + f" -> exit {exit_code}", file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
