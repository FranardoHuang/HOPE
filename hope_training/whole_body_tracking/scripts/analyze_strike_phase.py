"""Find the correct strike phase for the forehand & backhand reference clips.

Analyzes the RETARGETED ROBOT REFERENCE MOTION (artifacts/hope_*:v1/motion.npz),
NOT the trained policy. The npz already stores world-frame body kinematics for all
32 bodies, so no FK / Isaac is needed:
    body_pos_w     (T, 32, 3)   world position
    body_quat_w    (T, 32, 4)   world orientation (wxyz)
    body_lin_vel_w (T, 32, 3)   world linear velocity
    body_ang_vel_w (T, 32, 3)   world angular velocity

Body 0  = pelvis_link (root).   Body 31 = right_wrist_yaw_Link (the WRIST — the
fixed massless racket links pingpang_red_Link etc. are MERGED by PhysX and are
NOT separate bodies in the npz; 32 bodies = pelvis + 31 actuated links).

RACKET BLADE (default, 2026-07-02 fix): the training strike metric
(RacketTargetCommand in mdp/hope_commands.py, wrist_offset FK mode) measures the
BLADE = wrist frame * T_mount with mount_offset=(0.21021, 0.032078, 0.032036) m
and identity mount_quat. This script used to analyze the bare wrist point, which
planned targets ~0.1-0.2 m short and ~0.5-0.9 m/s slow of true blade contact.
Default is now the blade:
    blade_pos = wrist_pos + R(wrist_quat) @ mount_offset
    blade_vel = wrist_lin_vel + wrist_ang_vel x (R(wrist_quat) @ mount_offset)
Pass --wrist-point to reproduce the old wrist-point behavior.

Racket face normal = R(wrist_quat)[:, 1]  (+Y blade face, red/forehand face;
mount_quat is identity so wrist +Y == blade +Y). For video/GVHMR clips this is
only a wrist-frame proxy; when cfg/strike_annotations.yaml marks
face_normal_reliable: false, do not quote it as the physical paddle face normal.
analyze(grip_rot=grip_rotation(alpha_z, beta_x)) applies the registry grip
calibration (the solved HUMAN blade = R_wrist @ Rg @ y_hat, `grip:` block) to the
face normal AND the blade position/velocity FK — the single grip application
point used by gen_stage1_questions --grip registry.

For each frame it computes, in the robot-root (pelvis-yaw) frame:
  frame index, normalized phase, racket world pos, racket world lin vel, speed,
  racket vx along HOPE +X (root forward), racket face normal, distance to the
  fixed strike plane (x = 0.40 m in front of the robot), and a phase label
  (backswing / forward-strike / follow-through).

Then it applies hand-aligned strike annotations from cfg/strike_annotations.yaml.
The speed-peak detector is diagnostic only; it is a known trap on real-play video
where the post-contact whip can be faster than contact.

    python scripts/analyze_strike_phase.py
    python scripts/analyze_strike_phase.py \
        --clip forehand:artifacts/hope_forehand:v2/motion.npz \
        --clip backhand:artifacts/hope_backhand:v2/motion.npz \
        --out-dir analysis/strike_timing_v2
"""
from __future__ import annotations

import argparse
import os
import pathlib

import numpy as np

try:
    import yaml
except ImportError:  # keep the script importable in minimal numpy-only shells
    yaml = None

HERE = os.path.dirname(os.path.abspath(__file__))
WBT = os.path.dirname(HERE)

RACKET_BODY = 31           # right_wrist_yaw_Link (racket links are merged fixed links)
PELVIS_BODY = 0            # pelvis_link
NORMAL_AXIS = 1            # racket-local +Y is the blade face normal
STRIKE_PLANE_X = 0.40      # HITTER fixed striking plane, 0.40 m in front of robot

# Wrist -> blade-center mount transform. MUST match RacketTargetCommandCfg in
# mdp/hope_commands.py (mount_offset / mount_quat). mount_quat is identity, so
# only the position offset matters here.
MOUNT_OFFSET = np.array([0.210211399202899, 0.0320784994676765, 0.0320358706296689])
# Training-parity clean strike velocity: centered FD of the blade FK position
# over +-CLEAN_VEL_WINDOW frames (RacketTargetCommand clean_reference_strike_velocity).
CLEAN_VEL_WINDOW = 2

CLIPS = {
    "forehand": os.path.join(WBT, "artifacts/hope_forehand:v1/motion.npz"),
    "backhand": os.path.join(WBT, "artifacts/hope_backhand:v1/motion.npz"),
}
DEFAULT_ANNOTATIONS = os.path.join(WBT, "cfg", "strike_annotations.yaml")


def quat_to_rot(q):
    """wxyz quats (..,4) -> rotation matrices (..,3,3)."""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - w * z)
    R[..., 0, 2] = 2 * (x * z + w * y)
    R[..., 1, 0] = 2 * (x * y + w * z)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - w * x)
    R[..., 2, 0] = 2 * (x * z - w * y)
    R[..., 2, 1] = 2 * (y * z + w * x)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def grip_rotation(alpha_z_deg, beta_x_deg):
    """Registry grip calibration rotation (strike_annotations.yaml ``grip:``, franco 2026-07-06).

    The video pipeline never sees the paddle; the HUMAN blade orientation was inferred from the
    rally prior as a constant WRIST-LOCAL rotation, right-multiplied onto the wrist frame:

        Rg = Rz(alpha_z) @ Rx(beta_x)
        blade_normal_world = R_wrist_world @ Rg @ y_hat

    Same convention (and matrix order) as csv_to_npz_mujoco.apply_grip_rotation — the bake path
    that re-solves the wrist joints so the robot blade points along R_wrist_old @ Rg.
    """
    ca, sa = np.cos(np.deg2rad(alpha_z_deg)), np.sin(np.deg2rad(alpha_z_deg))
    cb, sb = np.cos(np.deg2rad(beta_x_deg)), np.sin(np.deg2rad(beta_x_deg))
    Rz = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cb, -sb], [0.0, sb, cb]])
    return Rz @ Rx


def yaw_only_rot(R):
    """Project a rotation onto its yaw (heading) about world +Z -> (..,3,3)."""
    fwd = R[..., :, 0].copy()      # body +X axis in world
    fwd[..., 2] = 0.0
    n = np.linalg.norm(fwd, axis=-1, keepdims=True)
    n = np.where(n < 1e-8, 1.0, n)
    fwd = fwd / n
    up = np.zeros_like(fwd)
    up[..., 2] = 1.0
    left = np.cross(up, fwd)
    Ry = np.stack([fwd, left, up], axis=-1)   # columns = root x,y,z in world
    return Ry


def analyze(name, path, use_blade=True, grip_rot=None):
    d = np.load(path)
    fps = int(d["fps"][0])
    P = d["body_pos_w"].astype(np.float64)        # (T,32,3)
    Q = d["body_quat_w"].astype(np.float64)
    V = d["body_lin_vel_w"].astype(np.float64)
    A = d["body_ang_vel_w"].astype(np.float64)
    T = P.shape[0]
    phase = np.arange(T) / T

    # Root (pelvis) yaw frame: world -> root rotation = Ry^T applied to world vectors.
    Rroot = yaw_only_rot(quat_to_rot(Q[:, PELVIS_BODY]))     # (T,3,3) cols=root axes in world
    pelvis_xy = P[:, PELVIS_BODY].copy()

    racket_R = quat_to_rot(Q[:, RACKET_BODY])     # wrist orientation (== blade, identity mount_quat)
    if grip_rot is not None:
        # GRIP CALIBRATION — the SINGLE application point in the whole stage-1 pipeline
        # (gen_stage1_questions self-check asserts this). The human blade frame is the wrist
        # frame right-multiplied by Rg = grip_rotation(alpha_z, beta_x); face normal, blade
        # POSITION mount offset and blade velocity below all use the rotated frame — the same
        # convention as the csv_to_npz_mujoco --grip-rot bake (registry note 2026-07-06: blade
        # positions shift up to ~13 cm with the wrist rotation).
        racket_R = racket_R @ np.asarray(grip_rot, dtype=np.float64)
    if use_blade:
        # Blade point = wrist FK'd through the mount (matches RacketTargetCommand wrist_offset mode).
        offset_w = np.einsum("tij,j->ti", racket_R, MOUNT_OFFSET)          # (T,3)
        racket_w = P[:, RACKET_BODY] + offset_w                            # blade world pos
        racket_v_w = V[:, RACKET_BODY] + np.cross(A[:, RACKET_BODY], offset_w)  # + omega x r
    else:
        racket_w = P[:, RACKET_BODY]              # legacy: bare wrist point
        racket_v_w = V[:, RACKET_BODY]
    normal_w = racket_R[:, :, NORMAL_AXIS]        # world face normal

    # Training-parity CLEAN strike velocity: centered finite difference of the (blade) position over
    # +-CLEAN_VEL_WINDOW frames with edge clamping, exactly like _ensure_reference_strike_state().
    W = CLEAN_VEL_WINDOW
    lo = np.clip(np.arange(T) - W, 0, T - 1)
    hi = np.clip(np.arange(T) + W, 0, T - 1)
    clean_v_w = (racket_w[hi] - racket_w[lo]) / (2.0 * W / fps)

    # Express racket pos/vel/normal in the root (pelvis-yaw, origin at pelvis) frame.
    rel = racket_w - pelvis_xy                    # (T,3)
    racket_root = np.einsum("tij,tj->ti", np.transpose(Rroot, (0, 2, 1)), rel)
    racket_v_root = np.einsum("tij,tj->ti", np.transpose(Rroot, (0, 2, 1)), racket_v_w)
    normal_root = np.einsum("tij,tj->ti", np.transpose(Rroot, (0, 2, 1)), normal_w)

    speed = np.linalg.norm(racket_v_w, axis=1)
    vx = racket_v_root[:, 0]                       # along HOPE +X (forward)
    dist_plane = np.abs(racket_root[:, 0] - STRIKE_PLANE_X)

    # Phase labels. Forward strike = vx>0 region around the forward-swing peak.
    fwd_peak = int(np.argmax(vx))                 # largest forward velocity
    label = np.empty(T, dtype=object)
    for i in range(T):
        if vx[i] > 0.3 and speed[i] > 0.4 * speed.max():
            label[i] = "forward-strike"
        elif i < fwd_peak:
            label[i] = "backswing"
        else:
            label[i] = "follow-through"

    # --- Strike-frame selection -------------------------------------------------
    # Auto candidate only. On real-play video the fastest forward frame can be the
    # post-contact whip/pull-up, so a hand annotation must override it before the
    # value is used for training or evaluation.
    fwd_mask = vx > 0.3
    cand = np.where(fwd_mask)[0]
    if len(cand) == 0:                             # degenerate fallback
        cand = np.array([fwd_peak])
    auto_strike = int(cand[np.argmax(speed[cand])])

    info = dict(
        name=name, T=T, fps=fps, phase=phase, speed=speed, vx=vx,
        path=path, dist_plane=dist_plane, racket_root=racket_root, racket_v_root=racket_v_root,
        normal_root=normal_root, normal_w=normal_w, label=label, strike=auto_strike, auto_strike=auto_strike,
        fwd_peak=fwd_peak, speed_peak=int(np.argmax(speed)),
        racket_w=racket_w, racket_v_w=racket_v_w, clean_v_w=clean_v_w,
        grip_applied=grip_rot is not None,
        point="blade" if use_blade else "wrist", annotation=None, annotation_key=None,
        annotation_phase=None, face_normal_reliable=True, selection_source="auto-unverified",
    )
    return info



def _annotation_keys(name, path):
    p = pathlib.Path(path)
    keys = [p.stem, name]
    if p.name == "motion.npz":
        keys.insert(0, p.parent.name)
    return [k for i, k in enumerate(keys) if k and k not in keys[:i]]


def _load_annotations(path):
    if not path or not os.path.exists(path):
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML is required to read strike annotations")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("clips", {}) or {}


def _load_grip_block(path):
    """The registry ``grip:`` block ({session: {alpha_z_deg, beta_x_deg, ...}}), {} if absent."""
    if not path or not os.path.exists(path):
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML is required to read strike annotations")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("grip", {}) or {}


def _annotation_frame(ann, T):
    if ann.get("frame") is not None:
        return int(ann["frame"])
    if ann.get("phase") is None:
        return None
    return int(round(float(ann["phase"]) * (T - 1)))


def _apply_annotation(info, key, ann):
    frame = _annotation_frame(ann, info["T"])
    if frame is None:
        return
    frame = max(0, min(int(frame), info["T"] - 1))
    info["strike"] = frame
    info["annotation"] = ann
    info["annotation_key"] = key
    info["annotation_phase"] = float(ann["phase"]) if ann.get("phase") is not None else None
    info["face_normal_reliable"] = bool(ann.get("face_normal_reliable", True))
    info["selection_source"] = "annotation"


def _sampling_box_lines(info, s):
    pos = info["racket_w"][s]
    vel = info["clean_v_w"][s]
    clip = info["name"]
    axes = "xyz"
    for axis, center in zip(axes, pos):
        yield f'task.racket.pos_range_per_clip.{clip}.{axis}=[{center - 0.10:.2f},{center + 0.10:.2f}]'
    for axis, center in zip(axes, vel):
        yield f'task.racket.vel_range_per_clip.{clip}.{axis}=[{center - 0.50:.2f},{center + 0.50:.2f}]'


def report(info):
    name, T = info["name"], info["T"]
    s = info["strike"]
    auto = info["auto_strike"]
    ann = info.get("annotation")
    print("\n" + "=" * 74)
    key = info.get("annotation_key") or pathlib.Path(info["path"]).stem
    print(f"{name.upper()}  ({T} frames @ {info['fps']} Hz)  [tracked point: {info['point'].upper()}]  [key: {key}]")
    print("=" * 74)
    print(f"  racket SPEED peak : frame {info['speed_peak']:3d}  phase {info['speed_peak']/T:.3f}  "
          f"|v|={info['speed'][info['speed_peak']]:.2f}  vx={info['vx'][info['speed_peak']]:+.2f}")
    print(f"  forward VX peak   : frame {info['fwd_peak']:3d}  phase {info['fwd_peak']/T:.3f}  "
          f"|v|={info['speed'][info['fwd_peak']]:.2f}  vx={info['vx'][info['fwd_peak']]:+.2f}")
    print(f"  AUTO pick (trap!) : frame {auto:3d}  phase {auto/T:.3f}")
    if ann:
        status = ann.get("status", "?")
        who = ann.get("annotator", "?") or "?"
        date = ann.get("date", "-") or "-"
        phase = info.get("annotation_phase")
        phase_s = f"{phase:.3f}" if phase is not None else f"{s/(T - 1):.3f}"
        print(f"  HAND ANNOTATION   : frame {s:3d}  phase {phase_s}  [{status}, {who}, {date}] <- SELECTED")
        if auto != s:
            print(f"  ** AUTO != ANNOTATION by {abs(auto - s)} frames - speed peak can be post-contact whip/pull-up; trust the annotation. **")
        if status != "verified":
            print("  ** UNVERIFIED annotation - scrub the source video frame-by-frame before accepting this clip. **")
    else:
        print("  ** NO HAND ANNOTATION - auto pick is diagnostic only. Add this clip to cfg/strike_annotations.yaml before training. **")
    print(f"  --> SELECTED STRIKE FRAME {s} <--")
    if info.get("annotation_phase") is not None:
        print(f"      annotation phase = {info['annotation_phase']:.3f}")
        print(f"      frame phase      = {s}/{T}={s/T:.3f}, {s}/(T-1)={s/(T - 1):.3f}")
    else:
        print(f"      phase            = {s/(T - 1):.3f}  (frame/(T-1); auto-unverified)")
    print(f"      racket speed     = {info['speed'][s]:.3f} m/s")
    print(f"      racket vx (+X)   = {info['vx'][s]:+.3f} m/s")
    print(f"      racket pos(root) = ({info['racket_root'][s,0]:+.3f}, {info['racket_root'][s,1]:+.3f}, {info['racket_root'][s,2]:+.3f})")
    print(f"      pos (WORLD)      = ({info['racket_w'][s,0]:+.3f}, {info['racket_w'][s,1]:+.3f}, {info['racket_w'][s,2]:+.3f})   <- YAML pos_range_per_clip center")
    print(f"      vel (WORLD)      = ({info['racket_v_w'][s,0]:+.3f}, {info['racket_v_w'][s,1]:+.3f}, {info['racket_v_w'][s,2]:+.3f})")
    print(f"      vel (WORLD,clean)= ({info['clean_v_w'][s,0]:+.3f}, {info['clean_v_w'][s,1]:+.3f}, {info['clean_v_w'][s,2]:+.3f})   "
          f"<- +-{CLEAN_VEL_WINDOW}-frame FD, training-parity; YAML vel_range_per_clip center")
    print(f"      dist to x=0.40   = {info['dist_plane'][s]:.3f} m")
    face_note = ""
    if not info.get("face_normal_reliable", True):
        face_note = "  ** UNRELIABLE: video/GVHMR wrist +Y proxy; do NOT quote as physical paddle face direction **"
    print(f"      face normal(root)= ({info['normal_root'][s,0]:+.3f}, {info['normal_root'][s,1]:+.3f}, {info['normal_root'][s,2]:+.3f}){face_note}")
    print(f"      label            = {info['label'][s]}")
    print("      paste-ready sampling box (pos +-0.10 / clean vel +-0.50):")
    for line in _sampling_box_lines(info, s):
        print(f'        "{line}"')

    # candidate comparison (backhand of interest, but print for both)
    print("\n  candidate-phase table:")
    print(f"    {'phase':>6} {'frame':>5} {'|v|':>6} {'vx':>7} {'dist0.4':>8} {'label':>15}")
    for ph in (0.36, 0.50, 0.59, 0.74):
        f = int(round(ph * (T - 1)))
        f = min(max(f, 0), T - 1)
        print(f"    {ph:6.2f} {f:5d} {info['speed'][f]:6.2f} {info['vx'][f]:+7.2f} "
              f"{info['dist_plane'][f]:8.3f} {info['label'][f]:>15}")


def plot(infos, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncols = len(infos)
    fig, axes = plt.subplots(4, ncols, figsize=(7 * ncols, 14), sharex=True,
                             squeeze=False)
    for col, info in enumerate(infos):
        ph = info["phase"]
        s = info["strike"]
        name = info["name"]
        for row, (y, ylab, ref) in enumerate([
            (info["vx"], "racket vx (+X) [m/s]", 0.0),
            (info["speed"], "racket speed [m/s]", None),
            (info["dist_plane"], "dist to x=0.40 plane [m]", None),
            (np.degrees(np.arccos(np.clip(info["racket_v_root"][:, 0] /
             np.maximum(np.linalg.norm(info["racket_v_root"], axis=1), 1e-6), -1, 1))),
             "vel angle off +X [deg]", None),
        ]):
            ax = axes[row, col]
            ax.plot(ph, y, "-o", ms=2, lw=1)
            if ref is not None:
                ax.axhline(ref, color="gray", ls=":", lw=0.8)
            ax.axvline(s / info["T"], color="r", ls="--", lw=1.2, label=f"strike {s/info['T']:.3f}")
            for ph_c, c in [(0.36, "g"), (0.50, "m"), (0.59, "orange"), (0.74, "brown")]:
                ax.axvline(ph_c, color=c, ls=":", lw=0.8, alpha=0.6)
            ax.set_ylabel(ylab)
            if row == 0:
                ax.set_title(f"{name}  (strike frame {s}, phase {s/info['T']:.3f})")
                ax.legend(fontsize=8)
            if row == 3:
                ax.set_xlabel("normalized phase")
            ax.grid(alpha=0.3)
    fig.suptitle("Reference racket kinematics over phase  "
                 "(dotted: candidates g=0.36 m=0.50 o=0.59 brown=0.74; red=selected)", y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"\n[plot] wrote {out}")


def _resolve(path):
    """Allow both absolute and WBT-relative clip paths."""
    return path if os.path.isabs(path) else os.path.join(WBT, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clip", action="append", default=None,
                    help="name:path.npz  (repeatable). Path may be absolute or "
                         "relative to the whole_body_tracking dir. Defaults to the "
                         "hardcoded v1 forehand+backhand clips.")
    # Convenience single-clip flags matching the requested interface.
    ap.add_argument("--motion-file", default=None, help="single clip npz path")
    ap.add_argument("--clip-name", default="clip", help="name for --motion-file")
    ap.add_argument("--out-dir", default=None,
                    help="output dir for plots (default: <WBT>/diag_videos)")
    ap.add_argument("--annotations", default=DEFAULT_ANNOTATIONS,
                    help="hand-aligned strike annotation YAML. Relative paths are resolved from the "
                         "whole_body_tracking dir. Use '' to disable.")
    ap.add_argument("--wrist-point", action="store_true",
                    help="analyze the bare wrist body point (legacy pre-2026-07-02 behavior) "
                         "instead of the racket BLADE (wrist + mount FK, matches the training "
                         "strike metric in mdp/hope_commands.py)")
    args = ap.parse_args()

    if args.motion_file:
        clips = {args.clip_name: _resolve(args.motion_file)}
    elif args.clip:
        clips = {}
        for spec in args.clip:
            name, _, path = spec.partition(":")
            if not path:
                ap.error(f"--clip must be name:path, got {spec!r}")
            clips[name] = _resolve(path)
    else:
        clips = CLIPS

    ann_path = _resolve(args.annotations) if args.annotations else None
    annotations = _load_annotations(ann_path)
    if ann_path and annotations:
        print(f"[annotations] loaded {len(annotations)} clips from {ann_path}")

    infos = []
    for n, p in clips.items():
        info = analyze(n, p, use_blade=not args.wrist_point)
        for key in _annotation_keys(n, p):
            if key in annotations:
                _apply_annotation(info, key, annotations[key])
                break
        infos.append(info)
    for info in infos:
        report(info)
    if len(infos) > 1:
        phases = [info["annotation_phase"] if info.get("annotation_phase") is not None else info["strike"] / (info["T"] - 1) for info in infos]
        sources = ["ann" if info.get("annotation") else "auto-unverified" for info in infos]
        print(f"\n  strike_phase_per_clip override (order = {'/'.join(i['name'] for i in infos)}; source = {','.join(sources)}):")
        print(f"    \"task.racket.strike_phase_per_clip=[{','.join(f'{p:.3f}' for p in phases)}]\"")

    outdir = args.out_dir if args.out_dir else os.path.join(WBT, "diag_videos")
    outdir = _resolve(outdir)
    os.makedirs(outdir, exist_ok=True)
    tag = "_".join(i["name"] for i in infos)
    try:
        plot(infos, os.path.join(outdir, f"strike_phase_{tag}.png"))
    except ModuleNotFoundError as exc:
        print(f"\n[plot] skipped ({exc}); report above is still valid")


if __name__ == "__main__":
    main()
