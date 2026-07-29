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
    contact_frame = argmax over f in [entry_frame, exit_frame] of the table/opponent-forward
    (+x, W_floor)
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

from racket_geometry_contract import (  # noqa: E402
    GEOMETRY_SOURCE_SHA256,
    RACKET_SITE_OFFSET_WRIST_M,
    face_center_from_site_local,
)

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

SCHEMA_VERSION = 2
VELOCITY_POINT_SEMANTICS = "selected_rubber_face_center"
CONTROL_POINT_SEMANTICS = "official_racket_site"

FRESH_N5_UPPER_MOTION_IDS = (
    "bh_loop_c",
    "v12_forehand_block",
    "bh_block",
    "s0_highpress",
    "fh_loop_high",
)
FRESH_N5_FORBIDDEN_MOTION_IDS = frozenset(
    {"fh_loop", "fh_block_syn"}
)
FRESH_N5_BANK_MOTION_IDS = (
    "fh_loop",
    "bh_loop_c",
    "fh_block_syn",
    "bh_block",
    "s0_highpress",
    "fh_loop_high",
    "v12_forehand_block",
)
CANONICAL_READY_V1_SHA256 = (
    "cb0a05ca9f7220686acfde1010c28ed04558fb2aa47ef2cfb2284d576ecd15b0"
)

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
    "fh_loop_high": 0,
    "v12_forehand_block": 2,
    "s0_highpress": 1,
    "bh_block": 2,
    "fh_block_syn": 2,
}
# fh_block_syn is bh_block played slower with a +Y face flip (identical source_sha256); it must not
# be selectable until it is rebuilt from a real forehand-block source (spec §7, defect 2).
HUMAN_ENABLED = {
    "fh_loop": True,
    "bh_loop_c": True,
    "fh_loop_high": True,
    "v12_forehand_block": True,
    "s0_highpress": True,
    "bh_block": True,
    "fh_block_syn": False,
}
DECLARED_FAMILY = {
    "fh_loop": "forehand",
    "fh_loop_high": "forehand",
    "v12_forehand_block": "forehand",
    "bh_loop_c": "backhand",
    "bh_block": "backhand",
    "fh_block_syn": "forehand",
    "s0_highpress": "backhand",
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


def runtime_clean_site_diff(
    p: np.ndarray,
    fps: float,
    half_window: int = 2,
) -> np.ndarray:
    """Mirror ``_ensure_reference_strike_state`` including edge clamping.

    Runtime clamps the two sampled frame indices to the clip but deliberately
    keeps the denominator ``2*W*dt``.  This differs from a conventional
    one-sided finite difference at the first/last ``W`` frames.
    """

    points = np.asarray(p, dtype=np.float64)
    window = int(half_window)
    if points.ndim != 2 or points.shape[1] != 3 or window < 1:
        raise ValueError("expected (T,3) points and half_window >= 1")
    count = points.shape[0]
    index = np.arange(count)
    hi = np.clip(index + window, 0, count - 1)
    lo = np.clip(index - window, 0, count - 1)
    return (
        (points[hi] - points[lo])
        / (2.0 * window / float(fps))
    )


def face_center_velocity_from_site_twist(
    site_velocity_w: np.ndarray,
    angular_velocity_w: np.ndarray,
    racket_rotation_w: np.ndarray,
    face_sign: int | float,
) -> np.ndarray:
    """Map the runtime's site-velocity and angular-velocity authorities.

    The runtime does *not* finite-difference the face-centre position.  It
    finite-differences the official site over ``clean_strike_vel_window=2``
    and combines that with the NPZ wrist angular velocity using the exact
    rigid-point identity.  A second finite difference of face position is
    physically reasonable but numerically different; that difference was
    enough to push native centre tasks above ``teacher_rate_max=1``.
    """

    site_velocity = np.asarray(site_velocity_w, dtype=np.float64)
    angular_velocity = np.asarray(
        angular_velocity_w,
        dtype=np.float64,
    )
    rotation = np.asarray(racket_rotation_w, dtype=np.float64)
    if (
        site_velocity.shape[-1:] != (3,)
        or angular_velocity.shape != site_velocity.shape
        or rotation.shape != site_velocity.shape[:-1] + (3, 3)
    ):
        raise ValueError(
            "site/angular/rotation arrays must have matching (...,3), "
            "(...,3), (...,3,3) shapes"
        )
    face_offset_w = np.einsum(
        "...ij,j->...i",
        rotation,
        face_center_from_site_local(face_sign),
    )
    return site_velocity + np.cross(angular_velocity, face_offset_w)


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


def _require_sha256(value, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SystemExit(f"{label} must be one lowercase SHA-256 digest")
    return value


def _fresh_n5_library_source_sha(library: dict) -> dict[str, str]:
    """Validate the exact seven-motion compiler input without promoting it.

    The canonical five remain an immutable compiler prefix.  The fresh training
    view is a separate five-row selection over exact upper outputs, so the two
    forbidden legacy rows remain available only as compiler provenance.
    """

    if not isinstance(library, dict):
        raise SystemExit("fresh N5 library must be one JSON object")
    if library.get("training_authorized") is not False:
        raise SystemExit("fresh N5 compiler library may not authorize training")
    if library.get("hardware_authorized") is not False:
        raise SystemExit("fresh N5 compiler library may not authorize hardware")
    specs = library.get("motion_specs")
    if not isinstance(specs, list):
        raise SystemExit("fresh N5 library motion_specs must be a list")
    motion_ids = tuple(
        row.get("motion_id") if isinstance(row, dict) else None
        for row in specs
    )
    if motion_ids != FRESH_N5_BANK_MOTION_IDS:
        raise SystemExit(
            "fresh N5 compiler library must contain the immutable canonical "
            f"five followed by fh_loop_high and v12_forehand_block; got "
            f"{list(motion_ids)}"
        )
    matrix = library.get("required_output_matrix")
    if not isinstance(matrix, dict) or matrix != {
        "motion_ids": list(FRESH_N5_BANK_MOTION_IDS),
        "scopes": ["upper", "full"],
        "candidate_count": 2 * len(FRESH_N5_BANK_MOTION_IDS),
    }:
        raise SystemExit(
            "fresh N5 compiler library must request the exact 7x2 output matrix"
        )
    ready = library.get("canonical_ready")
    if (
        not isinstance(ready, dict)
        or ready.get("sha256") != CANONICAL_READY_V1_SHA256
    ):
        raise SystemExit(
            "fresh N5 currently requires the exact registered legacy canonical "
            "ready v1; "
            "an unadopted grounded-ready candidate is not admissible"
        )
    source_sha: dict[str, str] = {}
    for index, row in enumerate(specs):
        if not isinstance(row, dict):
            raise SystemExit(f"fresh N5 motion_specs[{index}] must be an object")
        motion_id = str(row["motion_id"])
        source_sha[motion_id] = _require_sha256(
            row.get("source_sha256"),
            f"fresh N5 motion_specs[{index}].source_sha256",
        )
    return source_sha


def _fresh_n5_upper_outputs(manifest: dict) -> list[dict]:
    """Select the exact downstream five upper rows from a complete 7x2 bank."""

    if not isinstance(manifest, dict):
        raise SystemExit("fresh N5 BUILD_MANIFEST must be one JSON object")
    matrix = manifest.get("output_matrix")
    if not isinstance(matrix, dict) or matrix != {
        "motion_ids": list(FRESH_N5_BANK_MOTION_IDS),
        "scopes": ["upper", "full"],
        "candidate_count": 2 * len(FRESH_N5_BANK_MOTION_IDS),
    }:
        raise SystemExit(
            "fresh N5 BUILD_MANIFEST must bind the exact complete 7x2 bank"
        )
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list):
        raise SystemExit("fresh N5 BUILD_MANIFEST outputs must be a list")
    by_key: dict[tuple[str, str], dict] = {}
    for index, row in enumerate(outputs):
        if not isinstance(row, dict):
            raise SystemExit(f"fresh N5 output[{index}] must be an object")
        key = (str(row.get("motion_id")), str(row.get("scope")))
        if key in by_key:
            raise SystemExit(f"fresh N5 BUILD_MANIFEST duplicates output {key}")
        by_key[key] = row
    expected_keys = {
        (motion_id, scope)
        for motion_id in FRESH_N5_BANK_MOTION_IDS
        for scope in ("upper", "full")
    }
    if set(by_key) != expected_keys:
        missing = sorted(expected_keys - set(by_key))
        extra = sorted(set(by_key) - expected_keys)
        raise SystemExit(
            "fresh N5 BUILD_MANIFEST output matrix is incomplete or contains "
            f"unexpected rows; missing={missing}, extra={extra}"
        )
    selected: list[dict] = []
    for motion_id in FRESH_N5_UPPER_MOTION_IDS:
        row = by_key[(motion_id, "upper")]
        expected_filename = f"{motion_id}_upper_canonical_v2.npz"
        if row.get("filename") != expected_filename:
            raise SystemExit(
                f"fresh N5 upper output {motion_id} filename must be "
                f"{expected_filename!r}"
            )
        _require_sha256(
            row.get("output_npz_sha256"),
            f"fresh N5 upper output {motion_id}.output_npz_sha256",
        )
        selected.append(row)
    if any(
        str(row["motion_id"]) in FRESH_N5_FORBIDDEN_MOTION_IDS
        for row in selected
    ):
        raise SystemExit("fresh N5 training view contains a forbidden legacy motion")
    return selected


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
    v_site_w = runtime_clean_site_diff(site_w, fps, half_window=2)

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
    # Opponent-forward is the venue/world +X axis.  It is not B_yaw +X for
    # side-on ready stances (fivebind reaches ~-114 degrees yaw).
    fwd = v_site_w[gated, 0]
    contact = int(gated[int(np.argmax(fwd))])

    # --- face sign: the sign that makes the PHYSICAL striking face opponent-facing at contact ---
    face_axis_w = R_wrist[:, :, 1]                 # world image of the local +Y face normal
    face_sign = 1.0 if face_axis_w[contact, 0] > 0.0 else -1.0
    n_hat_w = face_sign * face_axis_w[contact]
    # ActionBall rotates prototype directions by the frozen birth/ready base
    # yaw.  Express both velocity and face normal in that same frame; using
    # contact-time root yaw silently rotates actions that turn while preparing.
    ready_yaw = float(yaw[0])
    n_hat_b = rot_b_yaw(n_hat_w, ready_yaw)
    n_hat_b = n_hat_b / (np.linalg.norm(n_hat_b) + 1e-12)

    # The fixed-direction solver output is a PHYSICAL CONTACT velocity at the
    # selected rubber face centre.  Use the exact same two numeric authorities
    # as runtime: clean ±2-frame site-position FD plus stored wrist omega.
    # Directly FD'ing face-centre position is close but not identical to this
    # twist and can make a native centre task require rate > 1.
    face_center_w = site_w + np.einsum(
        "tij,j->ti",
        R_wrist,
        face_center_from_site_local(face_sign),
    )
    if "body_ang_vel_w" not in z:
        raise SystemExit(
            f"{npz_path}: missing body_ang_vel_w required by the exact "
            "face-centre velocity authority"
        )
    body_ang_vel_w = np.asarray(z["body_ang_vel_w"], dtype=np.float64)
    if body_ang_vel_w.shape != body_pos.shape:
        raise SystemExit(
            f"{npz_path}: body_ang_vel_w shape {body_ang_vel_w.shape} "
            f"does not match body_pos_w {body_pos.shape}"
        )
    v_face_center_w = face_center_velocity_from_site_twist(
        v_site_w,
        body_ang_vel_w[:, wi],
        R_wrist,
        face_sign,
    )
    v_face_center_b = rot_b_yaw(
        v_face_center_w,
        np.full_like(yaw, ready_yaw),
    )

    v_c = v_face_center_b[contact]
    speed_nominal = float(np.linalg.norm(v_c))
    if speed_nominal < 1e-6:
        raise SystemExit(
            f"{npz_path}: selected rubber face-centre speed at contact "
            f"frame {contact} is ~0 — bad window"
        )
    v_hat_b = v_c / speed_nominal

    vw = v_face_center_b[win]
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
    deep, L_deep = deep_frame_and_L(face_center_w, contact)
    acc = np.gradient(
        np.gradient(face_center_w, 1.0 / fps, axis=0),
        1.0 / fps,
        axis=0,
    )
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
    family = DECLARED_FAMILY.get(motion_id, measured)

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
        # C. selected-rubber FACE-CENTRE velocity identity.  The explicit
        # field names are intentional: schema v1's generic v_hat/speed fields
        # were official-site values and must never be silently reinterpreted.
        "racket_face_center_velocity_hat_b": [
            float(x) for x in v_hat_b
        ],
        "racket_face_center_elevation_deg": float(
            np.degrees(np.arcsin(np.clip(v_hat_b[2], -1.0, 1.0)))
        ),
        "racket_face_center_window_dir_cone_deg": cone,
        "racket_face_center_speed_nominal_mps": speed_nominal,
        "racket_face_center_speed_max_mps": float(speed_max),
        "racket_face_center_speed_min_mps": float(speed_min),
        "retime_range": [s_min, s_max],
        "retime_binding": list(binding),
        "racket_face_center_v_star_cap_mps": v_star_cap,
        "racket_face_center_v_dir_tol_deg": HUMAN_DEFAULTS[
            "v_dir_tol_deg"
        ],
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
        "racket_face_center_cos_normal_velocity": float(
            np.dot(n_hat_b, v_hat_b)
        ),
        # G. selection policy
        # The historical five actions retain their declared selector policy.
        # Arbitrary admitted motion batches (for example ChingMu73) are
        # training prototypes, not a predesigned selector: keep them enabled
        # with neutral priority and let measured capability decide later.
        "priority": HUMAN_PRIORITY.get(motion_id, 0),
        "enabled": HUMAN_ENABLED.get(motion_id, True),
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
    ap.add_argument(
        "--fresh-n5-upper",
        action="store_true",
        help=(
            "build only the exact admitted upper training view in order "
            "bh_loop_c,v12_forehand_block,bh_block,s0_highpress,fh_loop_high "
            "from a complete seven-motion append bank"
        ),
    )
    ap.add_argument("--check", action="store_true",
                    help="do not write; fail when --out differs from the freshly derived bytes")
    args = ap.parse_args(argv)

    clip_dir = pathlib.Path(args.clip_dir)
    manifest_path = clip_dir / "BUILD_MANIFEST.json"
    manifest = json.load(open(manifest_path))
    library = json.load(open(args.library))
    if args.fresh_n5_upper:
        source_sha = _fresh_n5_library_source_sha(library)
        manifest_outputs = _fresh_n5_upper_outputs(manifest)
    else:
        source_sha = {
            m["motion_id"]: m.get("source_sha256", "")
            for m in library["motion_specs"]
        }
        manifest_outputs = manifest["outputs"]

    joint_names = [
        ln.strip() for ln in open(args.joint_order)
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    v_lim = urdf_velocity_limits(args.urdf)
    a_lim = accel_limits(args.accel_envelope)

    scopes = {}
    for out in manifest_outputs:
        npz_path = clip_dir / out["filename"]
        expected_output_sha = out.get("output_npz_sha256")
        if expected_output_sha is not None:
            expected_output_sha = _require_sha256(
                expected_output_sha,
                f"{out['motion_id']}/{out['scope']} output_npz_sha256",
            )
            actual_output_sha = sha256_file(npz_path)
            if actual_output_sha != expected_output_sha:
                raise SystemExit(
                    f"{out['motion_id']}/{out['scope']} compiled NPZ SHA drifted: "
                    f"expected {expected_output_sha}, got {actual_output_sha}"
                )
        # BUILD_MANIFEST entry_frame/exit_frame index the SOURCE clip, not the retimed output
        # (fh_loop: source frames 46..57 against a 80-frame output). The seconds fields are the
        # ones expressed in the OUTPUT clip's own time base, so the window is resolved from them.
        rec = derive_one(
            npz_path,
            float(out["contact_window_start_s"]), float(out["contact_window_end_s"]),
            str(out["motion_id"]), str(out["scope"]), source_sha.get(str(out["motion_id"]), ""),
            args, v_lim, a_lim, joint_names,
        )
        rec["npz_filename"] = out["filename"]
        rec["source_window_frames"] = [int(out["entry_frame"]), int(out["exit_frame"])]
        scopes.setdefault(rec["scope"], []).append(rec)

    for scope, recs in scopes.items():
        # BUILD_MANIFEST encounter order is the action identity authority.
        # Sorting by the historical five-action priority silently swapped
        # bh_block/s0_highpress relative to ActionBall's ordered manifest and
        # made the resulting prototype impossible to cross-bind.
        for i, r in enumerate(recs):
            r["clip_index"] = i
    if args.fresh_n5_upper:
        if tuple(scopes) != ("upper",):
            raise SystemExit("fresh N5 prototype must contain upper scope only")
        actual_order = tuple(
            row["motion_id"] for row in scopes["upper"]
        )
        if actual_order != FRESH_N5_UPPER_MOTION_IDS:
            raise SystemExit(
                "fresh N5 upper prototype order drifted: "
                f"expected {list(FRESH_N5_UPPER_MOTION_IDS)}, "
                f"got {list(actual_order)}"
            )

    # DEPLOY-GATE RECEIPT.  The gate is reported, never applied (see DEPLOY_GATE_SPEED_MAX_MPS).
    # ``headroom_mps`` < 0 means this stroke's physical ceiling is above what the on-robot runner
    # will admit, so a command near that ceiling would be dropped by pp_policy.hpp:2459.  That is
    # a fact the reader has to be able to see; it is not a reason to shrink the stroke.
    gate_headroom = {
        scope: {
            r["motion_id"]: round(
                DEPLOY_GATE_SPEED_MAX_MPS
                - float(r["racket_face_center_speed_max_mps"]),
                6,
            )
                for r in sorted(scopes[scope], key=lambda x: x["clip_index"])}
        for scope in sorted(scopes)
    }
    over_gate = [f"{s}/{m}" for s, d in gate_headroom.items() for m, h in d.items() if h < 0.0]

    doc = {
        "schema_version": SCHEMA_VERSION,
        "prototype_set_id": pathlib.Path(args.out).stem,
        "velocity_contract": {
            "direction_and_speed_point": VELOCITY_POINT_SEMANTICS,
            "policy_control_point": CONTROL_POINT_SEMANTICS,
            "mapping": (
                "v_face_center=v_site+omega_world_cross_"
                "r_face_center_from_site_world"
            ),
            "site_velocity_authority": (
                "centered_position_fd_half_window_2_clamped_per_clip"
            ),
            "angular_velocity_authority": (
                "npz_body_ang_vel_w_at_right_wrist_yaw_Link"
            ),
            "direction_frame_authority": (
                "canonical_ready_root_yaw_at_frame_0"
            ),
            "geometry_source_sha256": GEOMETRY_SOURCE_SHA256,
        },
        "contact_rule": {
            "name": (
                "max_table_opponent_forward_site_speed_in_compiled_window"
            ),
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
                  f"fam={r['family']:<9} "
                  f"elev={r['racket_face_center_elevation_deg']:+6.1f}deg "
                  f"v0={r['racket_face_center_speed_nominal_mps']:.2f} "
                  f"vmax={r['racket_face_center_speed_max_mps']:.2f} "
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
