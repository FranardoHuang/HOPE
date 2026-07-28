#!/usr/bin/env python3
"""Build + verify action-conditioned ball-first manifests (schema v3) from a ChingMu unit batch.

人话:把逐 unit 的 ChingMu 素材表(击球帧、站位、来球速度、球位)翻译成 action-ball 严格
manifest——每个动作一份"以实测最佳球为中心"的来球分布,再用正式 loader/adapter/sampler
在 CPU 上验证。只产 JSON,不发射训练,不授权任何 admission。

Producer order bound by the output: ``action -> incoming-ball sample -> frozen solve -> attempt``.

Subcommands
    build   read the batch manifest + clips, emit one manifest JSON (+ .sha256 sidecar
            + .buildreport.json sidecar), then re-load it through the strict loader.
    verify  load an emitted manifest, adapt to sampler profiles, and smoke-sample
            birth+sample rounds per action with validity checks and a reject histogram.

Measured centres (per action)
    contact_offset  = R_z(-yaw_before) @ (ball_pos_hit - station) in B_yaw, z above env floor
    incoming_speed  = |v_in_fit_hope_ms|
    incoming_dir    = R_z(-yaw_before) @ normalize(v_in) (B_yaw; inbound cone from -X axis)
    base_spawn      = station in env W frame (hope -> env via table_frame translation)
    racket speed    = physical right_racket site (wrist FK + RACKET_SITE_OFFSET_WRIST_M),
                      +/-2-frame central difference at hit_frame_50 (suggest_face_sign 口径)
    mount sign      = sign(n . v) at the hit frame (suggest_face_sign 口径), cross-checked
                      against the family expectation FH:+1 / BH:-1

Deliberate deviations from the verbal spec (fail-loud, reported in the build report):
    * time_to_contact centre: the requested 1.0 s centre is infeasible for most units because
      the loader requires min >= t_hit / teacher_rate_min + reaction_margin.  Default policy
      centres the window midpoint; --ttc-center-s clamps a requested centre into the window.
    * teacher_rate_min is raised above the CLI default per action when the certified
      time-to-contact window would otherwise be narrower than --min-ttc-window-s.
    * solver/physics profile SHA-256 default to test-fixture placeholders; a launch must
      re-pin them from the runtime contract (the runtime cross-check will refuse otherwise).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT_DEFAULT = SCRIPTS_DIR.parents[2]
MDP_DIR_REL = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp"
)

TABLE_LENGTH = 2.74
TABLE_WIDTH = 1.525
DEFAULT_EXCLUDE = ("Take_085_unit00_FH",)
FAMILY_MAP = {"FH": "forehand", "BH": "backhand"}
FAMILY_EXPECTED_SIGN = {"FH": 1, "BH": -1}
FPS = 50.0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_module(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _mdp_modules(repo_root: Path):
    """Import the real manifest/adapter/sampler modules as plain top-level modules."""
    mdp_dir = repo_root / MDP_DIR_REL
    if not mdp_dir.is_dir():
        raise SystemExit(f"mdp dir not found: {mdp_dir}")
    if str(mdp_dir) not in sys.path:
        sys.path.insert(0, str(mdp_dir))
    manifest_mod = _load_module("action_ball_manifest", mdp_dir / "action_ball_manifest.py")
    sampling_mod = _load_module("action_ball_sampling", mdp_dir / "action_ball_sampling.py")
    curriculum_mod = _load_module("action_ball_curriculum", mdp_dir / "action_ball_curriculum.py")
    adapter_mod = _load_module(
        "action_ball_profile_adapter", mdp_dir / "action_ball_profile_adapter.py"
    )
    return manifest_mod, sampling_mod, curriculum_mod, adapter_mod


def _rot_z(yaw_rad: float, x: float, y: float):
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    return (c * x - s * y, s * x + c * y)


def _normalize3(v):
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if n <= 1e-9:
        raise ValueError("cannot normalize a near-zero vector")
    return (v[0] / n, v[1] / n, v[2] / n)


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _tangent_frame(center):
    """Right-handed orthonormal (u, v) with cross(u, v) == center (unit)."""
    up = (0.0, 0.0, 1.0)
    if abs(_dot(center, up)) > 0.99:
        up = (1.0, 0.0, 0.0)
    u = _normalize3(_cross(up, center))
    v = _normalize3(_cross(center, u))
    return u, v


def _vec2(text: str, name: str):
    parts = [float(p) for p in text.split(",")]
    if len(parts) != 2:
        raise SystemExit(f"{name} must be 'x,y'")
    return parts


def _vec3(text: str, name: str):
    parts = [float(p) for p in text.split(",")]
    if len(parts) != 3:
        raise SystemExit(f"{name} must be 'x,y,z'")
    return parts


def _runtime_style_racket_site_speed(npz_path: Path, strike_frame: int, window: int) -> float:
    """Replicate the runtime's clean reference strike speed as closely as possible.

    hope_commands._ensure_reference_strike_state computes, in float32 motion buffers:
        blade[t] = wrist_pos[t] + R(wrist_quat[t]) @ mount_offset
        clean_lin = (blade[clamp(s+W)] - blade[clamp(s-W)]) / (2*W*dt)
    Note the denominator stays 2*W*dt even when the index clamps at a segment edge,
    unlike suggest_face_sign's span-corrected difference.  The runtime cross-checks
    the manifest pin at abs_tol=1e-6, so we keep the math in float32.
    """
    import numpy as np

    data = np.load(str(npz_path))
    names = [str(n) for n in data["body_names"]] if "body_names" in data.files else None
    widx = names.index("right_wrist_yaw_Link") if names else 31
    pos = np.asarray(data["body_pos_w"], dtype=np.float32)[:, widx]
    quat = np.asarray(data["body_quat_w"], dtype=np.float32)[:, widx]
    total = pos.shape[0]
    offset = np.array([0.21021, 0.032078, 0.032036], dtype=np.float32)

    def blade(frame: int) -> "np.ndarray":
        frame = min(max(frame, 0), total - 1)
        w, x, y, z = (quat[frame, i] for i in range(4))
        rot = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float32,
        )
        return pos[frame] + rot @ offset

    dt = np.float32(1.0 / FPS)
    diff = blade(strike_frame + window) - blade(strike_frame - window)
    vel = diff / (np.float32(2.0) * np.float32(window) * dt)
    return float(np.linalg.norm(vel.astype(np.float32)))


def _build_action(unit, args, face_row, report_rows):
    uid_raw = unit["uid"]
    action_id = uid_raw.lower()
    family_raw = unit["family"]
    family = FAMILY_MAP[family_raw]
    motion_sha256 = unit["npz_sha256"]

    # Native runtime truth (hope_commands._initialize_action_ball_runtime):
    #   t_hit  = round(strike_phase*(T-1)) * step_dt  == hit_frame_50 / 50
    #   t_cycle = (T-1) * step_dt                     == (T-1) / 50   (NOT T/50)
    # and the strike frame must lie strictly inside the segment.
    hit_frame = int(unit["hit_frame_50"])
    total = int(unit["T"])
    if not 0 < hit_frame < total - 1:
        raise SystemExit(f"{uid_raw}: hit frame {hit_frame} not strictly inside segment T={total}")
    if round(unit["strike_phase"] * (total - 1)) != hit_frame:
        raise SystemExit(
            f"{uid_raw}: rounded strike_phase*(T-1) disagrees with hit_frame_50; "
            "the runtime would target a different strike frame"
        )
    t_hit = hit_frame / FPS
    t_cycle = (total - 1) / FPS
    if not t_cycle > t_hit:
        raise SystemExit(f"{uid_raw}: (T-1)/50 must exceed hit_frame_50/50")

    # --- certified teacher-rate window and time-to-contact domain -----------------
    margin = args.reaction_margin_s
    rate_max = args.teacher_rate_max
    denom = t_hit + 1.0 - margin - args.min_ttc_window_s
    if denom <= t_hit:
        raise SystemExit(f"{uid_raw}: min TTC window leaves no feasible teacher_rate_min")
    rate_min_needed = t_hit / denom * (1.0 + 1e-9)
    rate_min = max(args.teacher_rate_min, rate_min_needed)
    rate_min_bumped = rate_min > args.teacher_rate_min
    if not rate_min <= 1.0 <= rate_max:
        raise SystemExit(f"{uid_raw}: teacher rate range [{rate_min}, {rate_max}] excludes 1.0")

    ttc_min = t_hit / rate_min + margin
    ttc_max = t_hit / rate_max + 1.0
    if not ttc_min < ttc_max:
        raise SystemExit(f"{uid_raw}: empty time-to-contact window [{ttc_min}, {ttc_max}]")
    if args.ttc_center_s is None:
        ttc_center = 0.5 * (ttc_min + ttc_max)
    else:
        ttc_center = min(max(args.ttc_center_s, ttc_min), ttc_max)
    ttc_lower_max = ttc_center - ttc_min
    ttc_upper_max = ttc_max - ttc_center
    ttc_lower_initial = min(args.ttc_std_initial_s, ttc_lower_max)
    ttc_upper_initial = min(args.ttc_std_initial_s, ttc_upper_max)

    # --- measured ball centre, rotated into B_yaw ---------------------------------
    yaw_rad = math.radians(unit["yaw_before_deg"])
    station = unit["station_xy_hope_m"]
    ball = unit["ball_pos_hit_hope_m"]
    dx, dy = ball[0] - station[0], ball[1] - station[1]
    off_x, off_y = _rot_z(-yaw_rad, dx, dy)
    off_z = ball[2] + args.surface_z  # hope z is above table surface; B_yaw z above env floor
    contact_center = (off_x, off_y, off_z)
    contact_lower_max = tuple(args.contact_std_max)
    contact_upper_max = tuple(args.contact_std_max)
    contact_lower_initial = tuple(args.contact_std_initial)
    contact_upper_initial = tuple(args.contact_std_initial)
    contact_min = tuple(c - w for c, w in zip(contact_center, contact_lower_max))
    contact_max = tuple(c + w for c, w in zip(contact_center, contact_upper_max))

    v_in = unit["v_in_fit_hope_ms"]
    speed_center = math.sqrt(v_in[0] ** 2 + v_in[1] ** 2 + v_in[2] ** 2)
    d_hope = _normalize3(v_in)
    dir_x, dir_y = _rot_z(-yaw_rad, d_hope[0], d_hope[1])
    incoming_center = _normalize3((dir_x, dir_y, d_hope[2]))
    incoming_u, incoming_v = _tangent_frame(incoming_center)

    inbound_axis = (-1.0, 0.0, 0.0)
    center_to_axis_deg = math.degrees(
        math.acos(max(-1.0, min(1.0, _dot(incoming_center, inbound_axis))))
    )
    radius_deg = math.hypot(args.dir_std_max_deg, args.dir_std_max_deg)
    min_cosine = args.inbound_min_cosine
    limit_deg = math.degrees(math.acos(min_cosine))
    relaxed = False
    if center_to_axis_deg + radius_deg > limit_deg - args.inbound_safety_deg:
        needed = center_to_axis_deg + radius_deg + args.inbound_safety_deg
        if needed >= 89.5:
            raise SystemExit(
                f"{uid_raw}: incoming direction {center_to_axis_deg:.1f} deg off the inbound "
                "axis; cannot certify an inbound cone (min_cosine would need to be < 0)"
            )
        min_cosine = math.cos(math.radians(needed))
        relaxed = True

    speed_min = 0.4 * speed_center  # loader requires exactly 0.4x
    speed_lower_max = min(args.speed_lower_max_frac * speed_center, speed_center - speed_min)
    speed_upper_max = args.speed_upper_max
    speed_max = speed_center + speed_upper_max
    speed_lower_initial = min(args.speed_std_initial, speed_lower_max)
    speed_upper_initial = min(args.speed_std_initial, speed_upper_max)

    spin_center = args.spin_mag_center
    spin_min = 0.0
    spin_max = args.spin_mag_max
    spin_lower_max = min(args.spin_mag_lower_std_max, spin_center - spin_min)
    spin_lower_initial = min(args.spin_mag_lower_std_initial, spin_lower_max)
    spin_upper_max = min(args.spin_mag_upper_std_max, spin_max - spin_center)
    spin_upper_initial = min(args.spin_mag_upper_std_initial, spin_upper_max)
    # canonical no-spin direction frame (test-fixture convention)
    spin_dir_center = (0.0, 1.0, 0.0)
    spin_dir_u = (0.0, 0.0, 1.0)
    spin_dir_v = (1.0, 0.0, 0.0)

    spawn_x = station[0] + args.near_x
    spawn_y = station[1] + TABLE_WIDTH / 2.0
    spawn_center = (spawn_x, spawn_y)
    spawn_span = tuple(args.base_spawn_span)
    spawn_std_initial = tuple(args.base_spawn_std_initial)
    spawn_std_max = tuple(args.base_spawn_std_max)
    spawn_min = tuple(c - w for c, w in zip(spawn_center, spawn_span))
    spawn_max = tuple(c + w for c, w in zip(spawn_center, spawn_span))

    dir_sides = {}
    for prefix, initial, maximum in (
        ("incoming_direction", args.dir_std_initial_deg, args.dir_std_max_deg),
        ("spin_direction", args.spin_dir_std_initial_deg, args.spin_dir_std_max_deg),
    ):
        for side in ("u_neg", "u_pos", "v_neg", "v_pos"):
            dir_sides[f"{prefix}_tangent_{side}_initial_deg"] = initial
            dir_sides[f"{prefix}_tangent_{side}_max_deg"] = maximum

    ball_profile = {
        "contact_offset_center_b_yaw_m": list(contact_center),
        "contact_offset_std_lower_initial_m": list(contact_lower_initial),
        "contact_offset_std_lower_max_m": list(contact_lower_max),
        "contact_offset_std_upper_initial_m": list(contact_upper_initial),
        "contact_offset_std_upper_max_m": list(contact_upper_max),
        "contact_offset_min_b_yaw_m": list(contact_min),
        "contact_offset_max_b_yaw_m": list(contact_max),
        "time_to_contact_center_s": ttc_center,
        "time_to_contact_std_lower_initial_s": ttc_lower_initial,
        "time_to_contact_std_lower_max_s": ttc_lower_max,
        "time_to_contact_std_upper_initial_s": ttc_upper_initial,
        "time_to_contact_std_upper_max_s": ttc_upper_max,
        "time_to_contact_min_s": ttc_min,
        "time_to_contact_max_s": ttc_max,
        "incoming_direction_center_b_yaw": list(incoming_center),
        "incoming_direction_tangent_u_b_yaw": list(incoming_u),
        "incoming_direction_tangent_v_b_yaw": list(incoming_v),
        "incoming_inbound_axis_b_yaw": list(inbound_axis),
        "incoming_inbound_min_cosine": min_cosine,
        "incoming_speed_center_mps": speed_center,
        "incoming_speed_std_lower_initial_mps": speed_lower_initial,
        "incoming_speed_std_lower_max_mps": speed_lower_max,
        "incoming_speed_std_upper_initial_mps": speed_upper_initial,
        "incoming_speed_std_upper_max_mps": speed_upper_max,
        "incoming_speed_min_mps": speed_min,
        "incoming_speed_max_mps": speed_max,
        "spin_direction_center_b_yaw": list(spin_dir_center),
        "spin_direction_tangent_u_b_yaw": list(spin_dir_u),
        "spin_direction_tangent_v_b_yaw": list(spin_dir_v),
        "spin_magnitude_center_radps": spin_center,
        "spin_magnitude_std_lower_initial_radps": spin_lower_initial,
        "spin_magnitude_std_lower_max_radps": spin_lower_max,
        "spin_magnitude_std_upper_initial_radps": spin_upper_initial,
        "spin_magnitude_std_upper_max_radps": spin_upper_max,
        "spin_magnitude_min_radps": spin_min,
        "spin_magnitude_max_radps": spin_max,
        "base_spawn_center_w_xy_m": list(spawn_center),
        "base_spawn_std_lower_initial_m": list(spawn_std_initial),
        "base_spawn_std_lower_max_m": list(spawn_std_max),
        "base_spawn_std_upper_initial_m": list(spawn_std_initial),
        "base_spawn_std_upper_max_m": list(spawn_std_max),
        "base_spawn_min_w_xy_m": list(spawn_min),
        "base_spawn_max_w_xy_m": list(spawn_max),
        "base_travel_center_b_yaw_xy_m": [0.0, 0.0],
        "base_travel_std_lower_initial_m": [0.0, 0.0],
        "base_travel_std_lower_max_m": [0.0, 0.0],
        "base_travel_std_upper_initial_m": [0.0, 0.0],
        "base_travel_std_upper_max_m": [0.0, 0.0],
        "base_travel_min_b_yaw_xy_m": [0.0, 0.0],
        "base_travel_max_b_yaw_xy_m": [0.0, 0.0],
    }
    ball_profile.update(dir_sides)

    mount_sign = int(face_row["suggested_sign"])
    racket_speed = _runtime_style_racket_site_speed(
        args._npz_path_for_unit, hit_frame, args.clean_vel_window
    )
    tool_speed = float(face_row["speed_clean"])
    if racket_speed <= 0.0:
        raise SystemExit(f"{uid_raw}: racket site speed at hit frame is not positive")

    report_rows.append(
        {
            "uid": uid_raw,
            "action_id": action_id,
            "family": family,
            "t_hit_s": t_hit,
            "t_cycle_s": t_cycle,
            "teacher_rate_min": rate_min,
            "teacher_rate_min_bumped": rate_min_bumped,
            "ttc_window_s": [ttc_min, ttc_max],
            "ttc_center_s": ttc_center,
            "contact_offset_center_b_yaw_m": list(contact_center),
            "incoming_speed_center_mps": speed_center,
            "incoming_dir_angle_to_inbound_axis_deg": center_to_axis_deg,
            "inbound_min_cosine": min_cosine,
            "inbound_min_cosine_relaxed": relaxed,
            "base_spawn_center_w_xy_m": list(spawn_center),
            "racket_site_speed_mps": racket_speed,
            "racket_site_speed_tool_f64_mps": tool_speed,
            "racket_site_speed_tool_delta_mps": abs(racket_speed - tool_speed),
            "mount_normal_sign": mount_sign,
            "mount_sign_matches_family": mount_sign == FAMILY_EXPECTED_SIGN[family_raw],
            "mount_sign_ambiguous": bool(face_row["ambiguous"]),
            "mount_sign_cos_clean": float(face_row["cos_clean"]),
            "mount_sign_raw_agrees": face_row.get("raw_agrees"),
        }
    )

    if args.motion_path_prefix:
        motion_path = args.motion_path_prefix.rstrip("/") + "/" + unit["npz"].split("/")[-1]
    else:
        motion_path = unit["npz"]
    return {
        "action_id": action_id,
        "family": family,
        "motion_path": motion_path,
        "motion_sha256": motion_sha256,
        "strike_phase": unit["strike_phase"],
        "reference_t_hit_s": t_hit,
        "reference_t_cycle_s": t_cycle,
        "reference_racket_site_speed_mps": racket_speed,
        "reaction_margin_s": margin,
        "teacher_rate_min": rate_min,
        "teacher_rate_max": rate_max,
        "mount_normal_sign": mount_sign,
        "ball_profile": ball_profile,
    }


def cmd_build(args) -> int:
    repo_root = Path(args.repo_root).resolve()
    batch_path = Path(args.batch_manifest).resolve()
    batch_root = Path(args.batch_root).resolve() if args.batch_root else batch_path.parent
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    batch_sha = _sha256_file(batch_path)

    exclude = set(args.exclude)
    units = [u for u in batch["units"] if u["uid"] not in exclude]
    if len(units) != args.expect_units:
        raise SystemExit(
            f"expected {args.expect_units} units after exclusions, got {len(units)} "
            "(pass --expect-units to override)"
        )

    if args.contact_std_initial[0] > args.contact_std_initial[1]:
        raise SystemExit("contact std initial x must be <= y")
    if args.contact_std_max[0] > args.contact_std_max[1]:
        raise SystemExit("contact std max x must be <= y")
    if args.contact_std_max[0] > 0.10:
        raise SystemExit("contact std max x must be <= 0.10 m (schema hard cap)")

    # face sign + racket site speed via the production offline tool
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    from suggest_face_sign import compute_face_sign  # noqa: E402

    manifest_mod, _, _, _ = _mdp_modules(repo_root)

    report_rows = []
    actions = []
    for unit in units:
        npz_path = (batch_root / unit["npz"]).resolve()
        if not npz_path.is_file():
            raise SystemExit(f"missing clip: {npz_path}")
        if not args.skip_npz_hash:
            actual = _sha256_file(npz_path)
            if actual != unit["npz_sha256"]:
                raise SystemExit(
                    f"{unit['uid']}: clip bytes drifted from batch manifest "
                    f"(expected {unit['npz_sha256']}, got {actual})"
                )
        face_row = compute_face_sign(str(npz_path), int(unit["hit_frame_50"]))
        args._npz_path_for_unit = npz_path
        action = _build_action(unit, args, face_row, report_rows)
        action["action_uid"] = manifest_mod.derive_action_ball_action_uid(
            action["action_id"], action["family"], action["motion_sha256"]
        )
        actions.append(action)

    ordered_keys = (
        "action_id",
        "action_uid",
        "motion_path",
        "motion_sha256",
        "strike_phase",
        "reference_t_hit_s",
        "reference_t_cycle_s",
        "reference_racket_site_speed_mps",
        "reaction_margin_s",
        "teacher_rate_min",
        "teacher_rate_max",
        "family",
        "mount_normal_sign",
        "ball_profile",
    )
    actions = [{key: action[key] for key in ordered_keys} for action in actions]

    prototype_path = repo_root / args.prototype_path
    if not prototype_path.is_file():
        raise SystemExit(f"prototype file not found: {prototype_path}")

    if args.landing_center_env is None:
        landing_center = [args.near_x + 0.75 * TABLE_LENGTH, 0.0]
    else:
        landing_center = list(args.landing_center_env)
    landing_span = list(args.landing_span)
    landing_std_initial = list(args.landing_std_initial)
    landing_std_max = list(args.landing_std_max)

    holdout_floor = max(args.min_proposals, args.min_safe_closed)
    if args.holdout_samples < holdout_floor:
        raise SystemExit(
            f"holdout samples {args.holdout_samples} below curriculum window {holdout_floor}"
        )

    document = {
        "schema_version": 3,
        "manifest_id": args.manifest_id,
        "mobility_mode": "no_move",
        "action_order": [action["action_id"] for action in actions],
        "prototype": {
            "path": args.prototype_path,
            "sha256": _sha256_file(prototype_path),
            "scope": args.prototype_scope,
        },
        "solver_profile_sha256": args.solver_profile_sha256,
        "physics_profile_sha256": args.physics_profile_sha256,
        "landing_aim": {
            "center_w_xy_m": landing_center,
            "std_lower_initial_m": landing_std_initial,
            "std_lower_max_m": landing_std_max,
            "std_upper_initial_m": landing_std_initial,
            "std_upper_max_m": landing_std_max,
            "min_w_xy_m": [c - w for c, w in zip(landing_center, landing_span)],
            "max_w_xy_m": [c + w for c, w in zip(landing_center, landing_span)],
        },
        "actions": actions,
        "curriculum": {
            "min_proposals": args.min_proposals,
            "min_safe_closed": args.min_safe_closed,
            "target_failure_rate": args.target_failure_rate,
            "failure_band_half_width": args.failure_band_half_width,
            "min_solver_admit_rate": 0.95,
            "min_install_rate": 0.95,
            "min_start_rate": 0.95,
            "min_close_rate": 0.95,
            "max_other_unsafe_rate": 0.02,
            "confidence_z": 1.96,
            "max_center_failures": 8,
        },
        "holdout": {
            "seed": args.holdout_seed,
            "samples_per_action": args.holdout_samples,
            "split_id": args.holdout_split_id,
        },
        "notes": (
            "Built by build_action_ball_manifest.py from "
            f"{batch_path.name} (sha256 {batch_sha}); motion_path and prototype path are "
            "relative to the training repo root. "
            + (
                "solver/physics profile pins were supplied by the caller (host-side "
                "computed from repository contract functions; runtime boot must confirm). "
                if (
                    args.solver_profile_sha256 != hashlib.sha256(b"solver").hexdigest()
                    or args.physics_profile_sha256 != hashlib.sha256(b"physics").hexdigest()
                )
                else "solver_profile_sha256/physics_profile_sha256 are PLACEHOLDER pins and "
                "must be re-pinned from the runtime contract before any launch. "
            )
            + "Metadata only; grants no motion admission."
        ),
    }

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    out_path.write_text(encoded, encoding="utf-8")

    loaded = manifest_mod.load_action_ball_manifest(out_path)
    file_sha = loaded.file_sha256
    (out_path.parent / (out_path.name + ".sha256")).write_text(
        f"{file_sha}  {out_path.name}\n", encoding="utf-8"
    )

    bumped = [r["uid"] for r in report_rows if r["teacher_rate_min_bumped"]]
    relaxed = [r["uid"] for r in report_rows if r["inbound_min_cosine_relaxed"]]
    sign_mismatch = [r["uid"] for r in report_rows if not r["mount_sign_matches_family"]]
    ambiguous = [r["uid"] for r in report_rows if r["mount_sign_ambiguous"]]
    speeds = sorted(r["racket_site_speed_mps"] for r in report_rows)
    policy_dt = 0.02
    episode_need = max(
        r["ttc_window_s"][1]
        + (r["t_cycle_s"] - r["t_hit_s"]) / r["teacher_rate_min"]
        + policy_dt
        for r in report_rows
    )
    report = {
        "builder": "build_action_ball_manifest.py",
        "batch_manifest": str(batch_path),
        "batch_manifest_sha256": batch_sha,
        "excluded_uids": sorted(exclude),
        "n_actions": len(actions),
        "out": str(out_path),
        "file_sha256": file_sha,
        "canonical_sha256": loaded.canonical_sha256,
        "racket_site_speed_mps": {
            "min": speeds[0],
            "median": speeds[len(speeds) // 2],
            "max": speeds[-1],
        },
        "min_episode_length_s_needed": episode_need,
        "teacher_rate_min_bumped_uids": bumped,
        "inbound_min_cosine_relaxed_uids": relaxed,
        "mount_sign_family_mismatch_uids": sign_mismatch,
        "mount_sign_ambiguous_uids": ambiguous,
        "per_action": report_rows,
    }
    report_path = out_path.parent / (out_path.name.replace(".json", "") + ".buildreport.json")
    report_path.write_text(
        json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"WROTE {out_path}")
    print(f"file_sha256      {file_sha}")
    print(f"canonical_sha256 {loaded.canonical_sha256}")
    print(f"actions          {len(actions)}")
    print(
        "racket site speed m/s min/median/max "
        f"{speeds[0]:.3f}/{speeds[len(speeds) // 2]:.3f}/{speeds[-1]:.3f}"
    )
    if bumped:
        print(f"WARN teacher_rate_min bumped above CLI default for: {bumped}")
    if relaxed:
        print(f"WARN inbound min_cosine relaxed for: {relaxed}")
    if sign_mismatch:
        print(f"WARN mount sign disagrees with family expectation for: {sign_mismatch}")
    if ambiguous:
        print(f"WARN mount sign ambiguous (|cos|<0.10 or slow) for: {ambiguous}")
    print(f"report           {report_path}")
    return 0


def _check_sample(sample, profile, min_cosine_by_uid):
    """Return a list of violated-reason strings for one sample (empty == legal)."""
    reasons = []

    def _in(lo, x, hi, name):
        if not (lo - 1e-9 <= x <= hi + 1e-9):
            reasons.append(name)

    for axis in range(3):
        _in(
            profile.contact_offset_min_b_yaw_m[axis],
            sample.contact_offset_from_base_goal_b_yaw_m[axis],
            profile.contact_offset_max_b_yaw_m[axis],
            f"contact_offset[{axis}]_out_of_bounds",
        )
    _in(
        profile.time_to_contact_min_s,
        sample.time_to_contact_s,
        profile.time_to_contact_max_s,
        "time_to_contact_out_of_bounds",
    )
    _in(
        profile.incoming_speed_min_mps,
        sample.incoming_speed_mps,
        profile.incoming_speed_max_mps,
        "incoming_speed_out_of_bounds",
    )
    _in(
        profile.spin_magnitude_min_radps,
        sample.spin_magnitude_radps,
        profile.spin_magnitude_max_radps,
        "spin_magnitude_out_of_bounds",
    )
    for axis in range(2):
        _in(
            profile.base_spawn_min_w_m[axis],
            sample.base_start_w_m[axis],
            profile.base_spawn_max_w_m[axis],
            f"base_spawn[{axis}]_out_of_bounds",
        )
        _in(
            profile.landing_aim_min_w_xy_m[axis],
            sample.landing_aim_w_xy_m[axis],
            profile.landing_aim_max_w_xy_m[axis],
            f"landing_aim[{axis}]_out_of_bounds",
        )
    direction = sample.incoming_direction_b_yaw
    norm = math.sqrt(sum(c * c for c in direction))
    if abs(norm - 1.0) > 1e-6:
        reasons.append("incoming_direction_not_unit")
    inbound = _dot(direction, profile.incoming_inbound_axis_b_yaw)
    if inbound < min_cosine_by_uid - 1e-9:
        reasons.append("incoming_direction_outside_inbound_cone")
    spin_dir = sample.spin_direction_b_yaw
    if abs(math.sqrt(sum(c * c for c in spin_dir)) - 1.0) > 1e-6:
        reasons.append("spin_direction_not_unit")
    if profile.mobility_mode == "no_move" and sample.base_goal_w_m != sample.base_start_w_m:
        reasons.append("no_move_base_goal_drifted")
    recomposed = tuple(
        sample.incoming_speed_mps * component for component in sample.incoming_direction_w
    )
    if any(abs(a - b) > 1e-9 for a, b in zip(recomposed, sample.incoming_velocity_w_mps)):
        reasons.append("incoming_velocity_inconsistent")
    try:
        sample.verify_sample_id()
    except ValueError:
        reasons.append("sample_id_mismatch")
    return reasons


def cmd_verify(args) -> int:
    repo_root = Path(args.repo_root).resolve()
    manifest_mod, sampling_mod, _, adapter_mod = _mdp_modules(repo_root)

    if args.assets_root:
        loaded = manifest_mod.load_action_ball_manifest(
            Path(args.manifest).resolve(),
            verify_referenced_assets=True,
            repo_root=Path(args.assets_root).resolve(),
        )
        print(f"REFERENCED ASSETS VERIFIED under {args.assets_root}")
    else:
        loaded = manifest_mod.load_action_ball_manifest(Path(args.manifest).resolve())
    manifest = loaded.manifest
    print(f"LOADED {args.manifest}")
    print(f"file_sha256      {loaded.file_sha256}")
    print(f"canonical_sha256 {loaded.canonical_sha256}")
    print(
        f"actions {len(manifest.actions)}  mobility {manifest.mobility_mode}  "
        f"target_failure_rate {manifest.curriculum.target_failure_rate}"
    )

    bundle = adapter_mod.adapt_action_ball_manifest(manifest)
    sampler = sampling_mod.ActionBallSampler(list(bundle.profiles), seed=args.seed)

    if args.action_ids:
        chosen = list(args.action_ids)
    else:
        by_family = {}
        for action in manifest.actions:
            by_family.setdefault(action.family, action.action_id)
        chosen = list(by_family.values())
        slowest = max(manifest.actions, key=lambda a: a.reference_t_hit_s)
        if slowest.action_id not in chosen:
            chosen.append(slowest.action_id)
        for action in manifest.actions:
            if len(chosen) >= args.n_actions:
                break
            if action.action_id not in chosen:
                chosen.append(action.action_id)
        chosen = chosen[: args.n_actions]

    uid_by_id = {a.action_id: a.action_uid for a in manifest.actions}
    profile_by_uid = {p.action_uid: p for p in bundle.profiles}
    min_cos_by_uid = {
        a.action_uid: a.ball_profile.incoming_inbound_min_cosine for a in manifest.actions
    }

    all_ok = True
    for action_id in chosen:
        uid = uid_by_id[action_id]
        profile = profile_by_uid[uid]
        for label, level in (("level0", 0.0), ("level1", 1.0)):
            n = args.samples if label == "level1" else args.samples_level0
            if n <= 0:
                continue
            levels = sampling_mod.DomainLevels(
                **{
                    field: level
                    for field in sampling_mod.DomainLevels.__dataclass_fields__
                }
            )
            histogram = {}
            legal = 0
            for _ in range(n):
                birth = sampler.reserve_birth(
                    action_uid=uid,
                    domain_epoch=args.domain_epoch,
                    levels=levels,
                    base_yaw_rad=args.base_yaw_rad,
                )
                sample = sampler.sample(
                    birth=birth,
                    action_uid=uid,
                    domain_epoch=args.domain_epoch,
                    levels=levels,
                    base_yaw_rad=args.base_yaw_rad,
                )
                reasons = _check_sample(sample, profile, min_cos_by_uid[uid])
                if reasons:
                    for reason in reasons:
                        histogram[reason] = histogram.get(reason, 0) + 1
                else:
                    legal += 1
            rate = legal / n
            print(
                f"SMOKE {action_id} {label} birth+sample n={n} legal={legal} "
                f"({rate:.1%}) rejects={json.dumps(histogram, sort_keys=True)}"
            )
            if legal != n:
                all_ok = False
    print("SMOKE RESULT:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="build one manifest JSON from the ChingMu batch")
    b.add_argument("--batch-manifest", required=True)
    b.add_argument("--batch-root", default=None, help="root for unit npz paths (default: manifest dir)")
    b.add_argument("--repo-root", default=str(REPO_ROOT_DEFAULT))
    b.add_argument("--out", required=True)
    b.add_argument("--manifest-id", required=True)
    b.add_argument("--exclude", nargs="*", default=list(DEFAULT_EXCLUDE))
    b.add_argument("--expect-units", type=int, default=73)
    b.add_argument("--skip-npz-hash", action="store_true")
    b.add_argument("--clean-vel-window", type=int, default=2,
                   help="frames W for the runtime-style clean strike velocity "
                        "(must equal racket.clean_strike_vel_window at launch)")
    b.add_argument("--motion-path-prefix", default=None,
                   help="rewrite motion_path to '<prefix>/<basename>' (e.g. a repo-root-"
                        "relative clips dir so launch referenced-asset verification passes)")
    b.add_argument("--reaction-margin-s", type=float, default=0.10)
    b.add_argument("--teacher-rate-min", type=float, default=0.6)
    b.add_argument("--teacher-rate-max", type=float, default=1.0)
    b.add_argument("--min-ttc-window-s", type=float, default=0.1)
    b.add_argument("--ttc-center-s", type=float, default=None,
                   help="requested TTC centre, clamped into the certified window "
                        "(default: window midpoint)")
    b.add_argument("--ttc-std-initial-s", type=float, default=0.05)
    b.add_argument("--contact-std-initial", type=lambda t: _vec3(t, "--contact-std-initial"),
                   default=[0.03, 0.05, 0.05])
    b.add_argument("--contact-std-max", type=lambda t: _vec3(t, "--contact-std-max"),
                   default=[0.08, 0.20, 0.15])
    b.add_argument("--speed-std-initial", type=float, default=0.15)
    b.add_argument("--speed-lower-max-frac", type=float, default=0.6)
    b.add_argument("--speed-upper-max", type=float, default=1.0)
    b.add_argument("--dir-std-initial-deg", type=float, default=3.0)
    b.add_argument("--dir-std-max-deg", type=float, default=15.0)
    b.add_argument("--spin-dir-std-initial-deg", type=float, default=3.0)
    b.add_argument("--spin-dir-std-max-deg", type=float, default=15.0)
    b.add_argument("--spin-mag-center", type=float, default=0.0)
    b.add_argument("--spin-mag-max", type=float, default=60.0)
    b.add_argument("--spin-mag-lower-std-initial", type=float, default=5.0)
    b.add_argument("--spin-mag-lower-std-max", type=float, default=40.0)
    b.add_argument("--spin-mag-upper-std-initial", type=float, default=5.0)
    b.add_argument("--spin-mag-upper-std-max", type=float, default=40.0)
    b.add_argument("--inbound-min-cosine", type=float, default=0.20)
    b.add_argument("--inbound-safety-deg", type=float, default=0.5)
    b.add_argument("--near-x", type=float, default=0.5)
    b.add_argument("--surface-z", type=float, default=0.76)
    b.add_argument("--landing-center-env", type=lambda t: _vec2(t, "--landing-center-env"),
                   default=None, help="default: opponent half centre (near_x + 2.055, 0)")
    b.add_argument("--landing-span", type=lambda t: _vec2(t, "--landing-span"),
                   default=[0.45, 0.60])
    b.add_argument("--landing-std-initial", type=lambda t: _vec2(t, "--landing-std-initial"),
                   default=[0.01, 0.01])
    b.add_argument("--landing-std-max", type=lambda t: _vec2(t, "--landing-std-max"),
                   default=[0.25, 0.45])
    b.add_argument("--base-spawn-span", type=lambda t: _vec2(t, "--base-spawn-span"),
                   default=[0.30, 0.40])
    b.add_argument("--base-spawn-std-initial", type=lambda t: _vec2(t, "--base-spawn-std-initial"),
                   default=[0.01, 0.01])
    b.add_argument("--base-spawn-std-max", type=lambda t: _vec2(t, "--base-spawn-std-max"),
                   default=[0.15, 0.25])
    b.add_argument("--prototype-path", default="configs/stroke_prototypes_v1_20260727.json")
    b.add_argument("--prototype-scope", default="full")
    b.add_argument("--solver-profile-sha256",
                   default=hashlib.sha256(b"solver").hexdigest(),
                   help="PLACEHOLDER default; re-pin from the runtime solver contract")
    b.add_argument("--physics-profile-sha256",
                   default=hashlib.sha256(b"physics").hexdigest(),
                   help="PLACEHOLDER default; re-pin from the runtime physics contract")
    b.add_argument("--min-proposals", type=int, default=256)
    b.add_argument("--min-safe-closed", type=int, default=256)
    b.add_argument("--target-failure-rate", type=float, default=0.10)
    b.add_argument("--failure-band-half-width", type=float, default=0.025)
    b.add_argument("--holdout-seed", type=int, default=20260728)
    b.add_argument("--holdout-samples", type=int, default=512)
    b.add_argument("--holdout-split-id", default="heldout_ball_chingmu73_v1")
    b.set_defaults(func=cmd_build)

    v = sub.add_parser("verify", help="strict-load a manifest and smoke birth+sample")
    v.add_argument("--manifest", required=True)
    v.add_argument("--repo-root", default=str(REPO_ROOT_DEFAULT))
    v.add_argument("--assets-root", default=None,
                   help="also verify prototype + motion bytes under this trusted root")
    v.add_argument("--seed", type=int, default=20260728)
    v.add_argument("--n-actions", type=int, default=3)
    v.add_argument("--action-ids", nargs="*", default=None)
    v.add_argument("--samples", type=int, default=100, help="birth+sample rounds at level 1.0")
    v.add_argument("--samples-level0", type=int, default=20,
                   help="extra birth+sample rounds at level 0.0")
    v.add_argument("--domain-epoch", type=int, default=0)
    v.add_argument("--base-yaw-rad", type=float, default=0.0)
    v.set_defaults(func=cmd_verify)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
