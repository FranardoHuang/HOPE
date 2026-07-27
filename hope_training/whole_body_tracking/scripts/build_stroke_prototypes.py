#!/usr/bin/env python3
"""Derive the STROKE PROTOTYPE file — the one source of truth for selector + adapter.

人话:每个动作片段(fh_loop / bh_loop_c / fh_block_syn / bh_block / s0_highpress)在触球那一帧
究竟"往哪挥、挥多快、最快能挥多快、球能落在哪个高度/左右范围、准备要多久"——全部从片段自己的
npz + 编译清单里量出来,写成一个 JSON。规划器(numpy)和训练器(torch)读同一份字节、校验同一个
sha256,所以两边永远不可能各拿一套动作参数。

Nothing here is hand-tuned except the fields explicitly marked HUMAN in the record: they are
carried through verbatim and re-emitted, so a reviewer can see every judgement call in one place.

Frames (spec §0 — three frames exist and are silently different; do not invent a fourth):
  * ``W_floor``  env-local world, z = 0 at the FLOOR.  The clip npz body arrays are already in it.
  * ``B_yaw``    base(pelvis) origin, yaw-only rotation.  ``p_b = R_z(-yaw)^T (p_w - p_base_w)``.
  Contact-region x/y bands are B_yaw; the z band stays W_floor (height above the FLOOR) because
  that is the frame every consumer of a contact height uses (``vb_table_surface_z`` etc.).

THE CONTACT RULE (spec §2.4, thresholds are OPEN QUESTION 8.1 — surfaced as CLI flags so a change
to them is visible in the emitted provenance):
    contact_frame = argmax over f in [entry_frame, exit_frame] of the FORWARD (+x, B_yaw)
    component of the racket-site velocity, among frames with site z_W_floor >= --min-contact-z
    and v_z >= --min-contact-vz.  Ties -> lowest frame index.

Usage::

    python scripts/build_stroke_prototypes.py \
        --clip-dir vendor_assets/motion_finalize_20260724/probes/candidates_v2_noearly_probe2 \
        --library configs/canonical_motion_library_v2_20260724.json \
        --out configs/stroke_prototypes_v1_20260727.json

Re-running on unchanged inputs must reproduce the file byte-for-byte (``--check`` asserts it);
that is what makes a hand-edited AUTO field fail CI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import sys
import xml.etree.ElementTree as ET

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from racket_geometry_contract import RACKET_SITE_OFFSET_WRIST_M  # noqa: E402

# --- constants that are hardware/deploy facts, not tuning -------------------------------------
WRIST_BODY = "right_wrist_yaw_Link"
BASE_BODY = "pelvis_link"
# DEPLOY GATE, REPORTED — NOT APPLIED.  pp_policy.hpp:234 ``gate_speed_max``: the on-robot C++
# runner rejects (pp_policy.hpp:2459) any commanded |v_racket| above this.  It is a property of
# the DEPLOY runner's admission test, not of the stroke, so it does not belong inside the
# prototype's physical speed ceiling — the prototype ceiling is
# ``min(nominal * retime_s_max, sqrt(2 a_max L_deep))`` and nothing else (owner's ruling
# 2026-07-27: the old third term ``3.5 - 0.1`` was an unnamed literal folded into that ``min``,
# and the 0.1 margin had no derivation at all).  What the build does with it now: compute the
# headroom per stroke, emit it into ``provenance.deploy_gate`` so it is visible in the receipt,
# and print a WARN when a stroke's physical ceiling sits above the gate.  Nothing is clamped.
DEPLOY_GATE_SPEED_MAX_MPS = 3.5
# gen_stage1_questions.py:726 --stroke-budget-scale default.
STROKE_BUDGET_SCALE = 1.5

SCHEMA_VERSION = 1

# HUMAN fields (spec §2.2 D/C/G).  Every judgement call in the file lives in this table.
HUMAN_DEFAULTS = {
    "v_dir_tol_deg": 10.0,
    "slack_b_xy_m": 0.15,
    "slack_z_w_m": 0.10,
}
# Selector priority (spec §3.3, OPEN QUESTION 8.3): attack when there is time and the ball can be
# lifted; press a high ball; block only when nothing better qualifies.
HUMAN_PRIORITY = {
    "fh_loop": 0,
    "bh_loop_c": 0,
    "s0_highpress": 1,
    "bh_block": 2,
    "fh_block_syn": 2,
}
# fh_block_syn is bh_block played slower with a +Y face flip (identical source_sha256); it must not
# be selectable until it is rebuilt from a real forehand-block source (spec §7, defect 2).
HUMAN_ENABLED = {
    "fh_loop": True,
    "bh_loop_c": True,
    "s0_highpress": True,
    "bh_block": True,
    "fh_block_syn": False,
}


# ------------------------------------------------------------------ small math helpers --- #
def quat_to_rot(q_wxyz: np.ndarray) -> np.ndarray:
    """(..., 4) wxyz -> (..., 3, 3) rotation matrices."""
    q = np.asarray(q_wxyz, dtype=np.float64)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
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


def yaw_of(q_wxyz: np.ndarray) -> np.ndarray:
    """Yaw angle (rad) of a wxyz quaternion, the same convention as ``quat_rotate_inverse``
    applied with a yaw-only quaternion (pp_policy.hpp:2362)."""
    q = np.asarray(q_wxyz, dtype=np.float64)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def to_b_yaw(p_w: np.ndarray, base_p_w: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """World (W_floor) -> base-yaw-aligned B_yaw, per-frame."""
    d = np.asarray(p_w, dtype=np.float64) - np.asarray(base_p_w, dtype=np.float64)
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1], d[..., 2]], -1)


def rot_b_yaw(v_w: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """Rotate a world VECTOR into B_yaw (no translation)."""
    v = np.asarray(v_w, dtype=np.float64)
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([c * v[..., 0] + s * v[..., 1], -s * v[..., 0] + c * v[..., 1], v[..., 2]], -1)


def central_diff(p: np.ndarray, fps: float) -> np.ndarray:
    """+/-2-frame centred finite difference (spec §2.4), edge-clamped."""
    n = p.shape[0]
    idx = np.arange(n)
    hi = np.clip(idx + 2, 0, n - 1)
    lo = np.clip(idx - 2, 0, n - 1)
    dt = (hi - lo).astype(np.float64) / float(fps)
    dt[dt <= 0] = 1.0 / float(fps)
    return (p[hi] - p[lo]) / dt[:, None]


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha256(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


# ------------------------------------------------------------------ joint limit tables --- #
def urdf_velocity_limits(urdf_path) -> dict:
    root = ET.parse(str(urdf_path)).getroot()
    out = {}
    for joint in root.iter("joint"):
        lim = joint.find("limit")
        if lim is None:
            continue
        v = lim.get("velocity")
        if v is None:
            continue
        try:
            fv = float(v)
        except ValueError:
            continue
        if fv > 0.0:
            out[joint.get("name")] = fv
    return out


def accel_limits(env_path) -> dict:
    d = json.load(open(env_path))
    names, accs = d["joint_names"], d["acceleration_rad_s2"]
    if len(names) != len(accs):
        raise SystemExit(
            f"{env_path}: joint_names ({len(names)}) and acceleration_rad_s2 ({len(accs)}) "
            f"disagree — cannot build a retiming ceiling from it"
        )
    return {str(n): float(a) for n, a in zip(names, accs)}


def retime_ceiling(joint_names, q_vel, fps, v_lim, a_lim):
    """(s_max, (joint, "vel"|"acc")) — the uniform time-scale ceiling of this clip.

    Uniform retiming by s multiplies every joint velocity by s and every acceleration by s^2, so
    ``s_max = min_j min(v_lim_j / max|qdot_j|, sqrt(a_lim_j / max|qddot_j|))``.
    """
    q_acc = np.gradient(np.asarray(q_vel, dtype=np.float64), 1.0 / float(fps), axis=0)
    best, binding = math.inf, ("", "")
    for j, name in enumerate(joint_names):
        vmax = float(np.max(np.abs(q_vel[:, j])))
        amax = float(np.max(np.abs(q_acc[:, j])))
        if name in v_lim and vmax > 1e-9:
            s = v_lim[name] / vmax
            if s < best:
                best, binding = s, (name, "vel")
        if name in a_lim and amax > 1e-9:
            s = math.sqrt(a_lim[name] / amax)
            if s < best:
                best, binding = s, (name, "acc")
    if not math.isfinite(best):
        raise SystemExit(
            "retime_ceiling: no joint matched either the URDF velocity table or the acceleration "
            "envelope — the joint-name order file and the clip disagree; refusing to guess"
        )
    return float(best), binding


def deep_frame_and_L(blade: np.ndarray, c: int):
    """Backswing runway: (deepest frame before contact, arc length from it to contact).

    Same ledger as ``scripts/extend_stroke.deep_frame_and_L`` (kept inline so this builder has no
    import-time dependency on that CLI module).
    """
    dist = np.linalg.norm(blade[: c + 1] - blade[c], axis=1)
    d = int(np.argmax(dist))
    seg = blade[d : c + 1]
    L = float(np.sum(np.linalg.norm(np.diff(seg, axis=0), axis=1))) if seg.shape[0] > 1 else 0.0
    return d, L


# ------------------------------------------------------------------ per-clip derivation --- #
def derive_one(npz_path, window_start_s, window_end_s, motion_id, scope, source_sha, args,
               v_lim, a_lim, joint_names):
    z = np.load(str(npz_path))
    fps = float(np.asarray(z["fps"]).reshape(-1)[0])
    names = [str(s) for s in z["body_names"]]
    if WRIST_BODY not in names or BASE_BODY not in names:
        raise SystemExit(f"{npz_path}: missing {WRIST_BODY!r}/{BASE_BODY!r} in body_names")
    wi, bi = names.index(WRIST_BODY), names.index(BASE_BODY)

    body_pos = np.asarray(z["body_pos_w"], dtype=np.float64)
    body_quat = np.asarray(z["body_quat_w"], dtype=np.float64)
    T = body_pos.shape[0]
    entry = int(round(float(window_start_s) * fps))
    exit_ = int(round(float(window_end_s) * fps))
    if not (0 <= entry <= exit_ < T):
        raise SystemExit(
            f"{npz_path}: compiled contact window {window_start_s:.4f}..{window_end_s:.4f} s at "
            f"{fps:g} fps resolves to frames [{entry},{exit_}], outside the clip's {T} frames"
        )

    R_wrist = quat_to_rot(body_quat[:, wi])
    site_w = body_pos[:, wi] + np.einsum("tij,j->ti", R_wrist, RACKET_SITE_OFFSET_WRIST_M)
    v_site_w = central_diff(site_w, fps)

    base_p = body_pos[:, bi]
    yaw = yaw_of(body_quat[:, bi])
    site_b = to_b_yaw(site_w, base_p, yaw)
    v_site_b = rot_b_yaw(v_site_w, yaw)

    # --- THE CONTACT RULE (spec §2.4) ---
    win = np.arange(int(entry), int(exit_) + 1)
    ok = (site_w[win, 2] >= args.min_contact_z) & (v_site_b[win, 2] >= args.min_contact_vz)
    gated = win[ok]
    if gated.size == 0:
        raise SystemExit(
            f"{npz_path}: no frame in the compiled contact window [{entry},{exit_}] satisfies the "
            f"contact rule gates (site z >= {args.min_contact_z} m and blade v_z >= "
            f"{args.min_contact_vz} m/s). Window z range "
            f"[{site_w[win, 2].min():.3f}, {site_w[win, 2].max():.3f}] m, v_z range "
            f"[{v_site_b[win, 2].min():.3f}, {v_site_b[win, 2].max():.3f}] m/s — fix the clip or "
            f"the thresholds (--min-contact-z / --min-contact-vz), do not guess a contact frame"
        )
    fwd = v_site_b[gated, 0]
    contact = int(gated[int(np.argmax(fwd))])

    # --- face sign: the sign that makes the PHYSICAL striking face opponent-facing at contact ---
    face_axis_w = R_wrist[:, :, 1]                 # world image of the local +Y face normal
    face_sign = 1.0 if face_axis_w[contact, 0] > 0.0 else -1.0
    n_hat_w = face_sign * face_axis_w[contact]
    n_hat_b = rot_b_yaw(n_hat_w, yaw[contact])
    n_hat_b = n_hat_b / (np.linalg.norm(n_hat_b) + 1e-12)

    v_c = v_site_b[contact]
    speed_nominal = float(np.linalg.norm(v_c))
    if speed_nominal < 1e-6:
        raise SystemExit(f"{npz_path}: blade speed at contact frame {contact} is ~0 — bad window")
    v_hat_b = v_c / speed_nominal

    vw = v_site_b[win]
    vwn = vw / (np.linalg.norm(vw, axis=-1, keepdims=True) + 1e-12)
    cone = float(np.degrees(np.max(np.arccos(np.clip(vwn @ v_hat_b, -1.0, 1.0)))))

    # --- timing ---
    cycle_s = (T - 1) / fps
    t_prepare = contact / fps
    t_recover = (T - 1 - contact) / fps
    s_max, binding = retime_ceiling(joint_names, np.asarray(z["joint_vel"], dtype=np.float64),
                                    fps, v_lim, a_lim)
    s_min = float(args.retime_s_min)

    # --- runway ceiling: v* <= sqrt(2 a_max L_deep)  (StrokeGuard, gen_stage1_questions:566) ---
    deep, L_deep = deep_frame_and_L(site_w, contact)
    acc = np.gradient(np.gradient(site_w, 1.0 / fps, axis=0), 1.0 / fps, axis=0)
    a_max = float(np.max(np.linalg.norm(acc, axis=-1))) * STROKE_BUDGET_SCALE
    v_star_cap = float(math.sqrt(max(2.0 * a_max * L_deep, 0.0)))

    # TWO physical ceilings, both measured off this clip: the retime ceiling (how fast the clip
    # may be replayed before a joint runs out of velocity/acceleration) and the runway ceiling
    # (how fast the blade can be going at contact given the backswing travel it has to do it in).
    # The deploy gate is NOT a third term here — see DEPLOY_GATE_SPEED_MAX_MPS.
    speed_max = min(speed_nominal * s_max, v_star_cap)
    speed_min = speed_nominal * s_min
    if speed_min > speed_max:
        speed_min = min(speed_min, speed_max)

    # --- contact region, MEASURED over the window (never a shared half-width) ---
    band_b_x = (float(site_b[win, 0].min()), float(site_b[win, 0].max()))
    band_b_y = (float(site_b[win, 1].min()), float(site_b[win, 1].max()))
    band_z_w = (float(site_w[win, 2].min()), float(site_w[win, 2].max()))

    # FAMILY. Authority is the motion id (fh_/bh_); the measured contact side is recorded beside
    # it so the two can be compared. They disagree on the `full` scope for bh_block /
    # fh_block_syn — those clips contact on the -y side of the pelvis even though the stroke is a
    # backhand — which is exactly why the side is recorded and not silently used.
    y_b = float(site_b[contact, 1])
    measured = "forehand" if y_b < 0.0 else "backhand"
    if motion_id.startswith("fh_"):
        family = "forehand"
    elif motion_id.startswith("bh_"):
        family = "backhand"
    else:
        family = measured

    return {
        # A. identity / provenance
        "motion_id": motion_id,
        "scope": scope,
        "family": family,
        "family_measured_side": measured,
        "y_b_at_contact_m": y_b,
        "frames": int(T),
        "fps": fps,
        "npz_sha256": sha256_file(npz_path),
        "source_sha256": source_sha,
        # B. timing
        "cycle_s": cycle_s,
        "contact_window_frames": [int(entry), int(exit_)],
        "contact_frame": contact,
        "strike_phase": contact / (T - 1),
        "t_prepare_s": t_prepare,
        "t_recover_s": t_recover,
        "deep_frame": int(deep),
        "L_deep_m": L_deep,
        "t_backswing_s": (contact - deep) / fps,
        "t_prepare_min_s": t_prepare / s_max,
        "t_prepare_max_s": t_prepare / s_min,
        # C. velocity identity
        "v_hat_b": [float(x) for x in v_hat_b],
        "elevation_deg": float(np.degrees(np.arcsin(np.clip(v_hat_b[2], -1.0, 1.0)))),
        "window_dir_cone_deg": cone,
        "speed_nominal_mps": speed_nominal,
        "speed_max_mps": float(speed_max),
        "speed_min_mps": float(speed_min),
        "retime_range": [s_min, s_max],
        "retime_binding": list(binding),
        "v_star_cap_mps": v_star_cap,
        "v_dir_tol_deg": HUMAN_DEFAULTS["v_dir_tol_deg"],
        # D. contact region
        "p_contact_b": [float(x) for x in site_b[contact]],
        "band_b_x": list(band_b_x),
        "band_b_y": list(band_b_y),
        "band_z_w": list(band_z_w),
        "base_height_at_contact_m": float(base_p[contact, 2]),
        "slack_b_xy_m": HUMAN_DEFAULTS["slack_b_xy_m"],
        "slack_z_w_m": HUMAN_DEFAULTS["slack_z_w_m"],
        # E. face
        "face_sign": face_sign,
        "n_hat_b": [float(x) for x in n_hat_b],
        "cos_nv": float(np.dot(n_hat_b, v_hat_b)),
        # G. selection policy
        "priority": HUMAN_PRIORITY[motion_id],
        "enabled": HUMAN_ENABLED[motion_id],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip-dir", required=True)
    ap.add_argument("--library", default="configs/canonical_motion_library_v2_20260724.json")
    ap.add_argument("--urdf", default="agi/URDF/A3T2.5-URDF-std-pingpang/urdf/URDF-JOINT-LINK.urdf")
    ap.add_argument("--accel-envelope",
                    default="vendor_assets/motion_finalize_20260724/evidence/acceleration/"
                            "source_diagonal_acceleration_envelope.json")
    ap.add_argument("--joint-order", default="configs/a3_runtime_articulation_joint_order.txt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-contact-z", type=float, default=0.88,
                    help="contact-rule gate: racket-site height above the FLOOR (OPEN QUESTION 8.1)")
    ap.add_argument("--min-contact-vz", type=float, default=-0.30,
                    help="contact-rule gate: minimum blade v_z (OPEN QUESTION 8.1)")
    ap.add_argument("--retime-s-min", type=float, default=0.6,
                    help="slowest uniform time-scale a clip may be played at")
    ap.add_argument("--check", action="store_true",
                    help="do not write; fail when --out differs from the freshly derived bytes")
    args = ap.parse_args(argv)

    clip_dir = pathlib.Path(args.clip_dir)
    manifest_path = clip_dir / "BUILD_MANIFEST.json"
    manifest = json.load(open(manifest_path))
    library = json.load(open(args.library))
    source_sha = {m["motion_id"]: m.get("source_sha256", "") for m in library["motion_specs"]}

    joint_names = [
        ln.strip() for ln in open(args.joint_order)
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    v_lim = urdf_velocity_limits(args.urdf)
    a_lim = accel_limits(args.accel_envelope)

    scopes = {}
    for out in manifest["outputs"]:
        # BUILD_MANIFEST entry_frame/exit_frame index the SOURCE clip, not the retimed output
        # (fh_loop: source frames 46..57 against a 80-frame output). The seconds fields are the
        # ones expressed in the OUTPUT clip's own time base, so the window is resolved from them.
        rec = derive_one(
            clip_dir / out["filename"],
            float(out["contact_window_start_s"]), float(out["contact_window_end_s"]),
            str(out["motion_id"]), str(out["scope"]), source_sha.get(str(out["motion_id"]), ""),
            args, v_lim, a_lim, joint_names,
        )
        rec["npz_filename"] = out["filename"]
        rec["source_window_frames"] = [int(out["entry_frame"]), int(out["exit_frame"])]
        scopes.setdefault(rec["scope"], []).append(rec)

    for scope, recs in scopes.items():
        recs.sort(key=lambda r: list(HUMAN_PRIORITY).index(r["motion_id"]))
        for i, r in enumerate(recs):
            r["clip_index"] = i

    # DEPLOY-GATE RECEIPT.  The gate is reported, never applied (see DEPLOY_GATE_SPEED_MAX_MPS).
    # ``headroom_mps`` < 0 means this stroke's physical ceiling is above what the on-robot runner
    # will admit, so a command near that ceiling would be dropped by pp_policy.hpp:2459.  That is
    # a fact the reader has to be able to see; it is not a reason to shrink the stroke.
    gate_headroom = {
        scope: {r["motion_id"]: round(DEPLOY_GATE_SPEED_MAX_MPS - float(r["speed_max_mps"]), 6)
                for r in sorted(scopes[scope], key=lambda x: x["clip_index"])}
        for scope in sorted(scopes)
    }
    over_gate = [f"{s}/{m}" for s, d in gate_headroom.items() for m, h in d.items() if h < 0.0]

    doc = {
        "schema_version": SCHEMA_VERSION,
        "prototype_set_id": pathlib.Path(args.out).stem,
        "contact_rule": {
            "name": "max_forward_blade_speed_in_compiled_window",
            "min_site_z_w_m": args.min_contact_z,
            "min_blade_vz_mps": args.min_contact_vz,
            "note": "spec §2.4; the two thresholds are OPEN QUESTION 8.1 (owner's call)",
        },
        "provenance": {
            "clip_dir": str(clip_dir).replace(os.sep, "/"),
            "build_manifest_sha256": sha256_file(manifest_path),
            "library_sha256": sha256_file(args.library),
            "accel_envelope_sha256": sha256_file(args.accel_envelope),
            "deploy_gate": {
                "applied": False,
                "source": "pp_policy.hpp:234 gate_speed_max (reject at pp_policy.hpp:2459)",
                "speed_max_mps": DEPLOY_GATE_SPEED_MAX_MPS,
                "headroom_mps": gate_headroom,
                "over_gate": over_gate,
            },
            "stroke_budget_scale": STROKE_BUDGET_SCALE,
            "retime_s_min": args.retime_s_min,
        },
        "scopes": {k: scopes[k] for k in sorted(scopes)},
    }
    doc["derived_sha256"] = canonical_sha256(doc["scopes"])
    text = json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n"

    out_path = pathlib.Path(args.out)
    if args.check:
        if not out_path.exists():
            raise SystemExit(f"--check: {out_path} does not exist")
        have = out_path.read_text(encoding="utf-8")
        if have != text:
            raise SystemExit(
                f"--check: {out_path} is NOT reproducible from its inputs — a field was "
                f"hand-edited, or an input changed without a rebuild"
            )
        print(f"ok: {out_path} reproduces byte-for-byte (derived_sha256={doc['derived_sha256']})")
        _warn_over_deploy_gate(over_gate, gate_headroom)
        return 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"wrote {out_path} ({sum(len(v) for v in scopes.values())} records, "
          f"derived_sha256={doc['derived_sha256']})")
    for scope in sorted(scopes):
        for r in scopes[scope]:
            print(f"  [{scope}] {r['motion_id']:<13} clip={r['clip_index']} "
                  f"frame={r['contact_frame']:>3} phase={r['strike_phase']:.3f} "
                  f"fam={r['family']:<9} elev={r['elevation_deg']:+6.1f}deg "
                  f"v0={r['speed_nominal_mps']:.2f} vmax={r['speed_max_mps']:.2f} "
                  f"tprep={r['t_prepare_s']:.2f}s enabled={r['enabled']}")
    _warn_over_deploy_gate(over_gate, gate_headroom)
    return 0


def _warn_over_deploy_gate(over_gate, gate_headroom) -> None:
    """Say it out loud when a stroke's physical ceiling is above the deploy runner's gate.

    Printed to stderr so a launch summary that greps for WARN catches it.  This is the visible
    replacement for the old silent ``min(..., 3.4)``: the build no longer shrinks the stroke, so
    the only thing that keeps this honest is that the reader is told.
    """
    if not over_gate:
        return
    detail = ", ".join(
        f"{s}/{m} by {-h:.3f} m/s"
        for s, d in gate_headroom.items() for m, h in d.items() if h < 0.0
    )
    print(
        f"WARN deploy_gate: {len(over_gate)} stroke(s) have a physical speed ceiling above "
        f"pp_policy.hpp:234 gate_speed_max={DEPLOY_GATE_SPEED_MAX_MPS} m/s ({detail}). "
        f"Commands near those ceilings would be rejected on-robot (pp_policy.hpp:2459). "
        f"NOT clamped — the ceiling is what the clip can physically do.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    raise SystemExit(main())
