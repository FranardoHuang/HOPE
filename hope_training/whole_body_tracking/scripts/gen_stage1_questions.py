#!/usr/bin/env python3
"""Stage-1 question bank: fixed strike point, varying incoming speed, no spin, fixed landing.

Three-stage curriculum (NOW.md 2026-07-05, yikang): stage 1 teaches "return balls of
different speeds to a FIXED landing point from a FIXED strike point, with the CORRECT
inverse-solved face + racket velocity". This script builds that question bank offline:

  for each sampled incoming ball (speed in range, no spin, arriving at the fixed
  strike point) -> StrikeSpecPlanner.solve() (franco's LM inverse over the venue
  contact+Magnus chain) -> demanded face normal + racket velocity = the ANSWER.

Only answerable questions are kept (LM converged <=5 mm, speed budget respected);
the solvability rate is reported — if it is low, the fixed strike point is badly
chosen (move the point, do not train on unanswerable questions).

The fixed strike point defaults to the clip's HAND-ANNOTATED strike-frame blade
position (cfg/strike_annotations.yaml via analyze_strike_phase conventions), so the
motion prior and the question are automatically consistent.

Self-test (--check, default on): every kept answer is replayed through the TRAINING-
side physics (tracking/mdp/virtual_ball.py, torch venue port): predict_paddle_contact
+ coarse_landing must land within tolerance of the target — a cross-implementation
closed loop (planner numpy solver vs trainer torch physics).

Frames: the tracking env frame has the table near edge at x=vb_table_near_x and the
table CENTER line at y=0; the planner HOPE frame has the near-left corner at the
origin (x in [0, 2.74], y in [-1.525, 0], and z=0 at the TABLE SURFACE — the env
frame has the surface at z=+0.76). Positions convert by a pure translation
(env->hope: x -= near_x, y -= 0.7625, z -= 0.76); vectors are frame-parallel and
need no conversion.

PHASE SCAN MODE (--phase-scan; merged from claude's 0d scratch scanner 2026-07-05 so there
is ONE stage-1 tool): instead of solving questions at the annotated frame only, walk EVERY frame
of the clip and ask "if contact happened here — face pinned to this frame's face, racket velocity
constrained to a cone around this frame's OWN clip velocity (--cone-deg/--cone-mag), contact at
this frame's blade point — what fraction of the sampled incoming balls has at least one legal
return?" (legal = lands on the opponent half past the depth guard AND clears the net; trainer-side
torch venue physics, same closed loop as --check).

SEMANTICS BOUNDARY (yikang 2026-07-05): the annotation registry records VIDEO TRUTH (when the
human touched the ball); this scan produces the TRAINING-OPTIMAL phase (which frame's kinematics
best suit robot returns). The two may legitimately differ (hopex clips are dry swings — only the
latter exists). The scan therefore NEVER writes the registry: it prints suggested
``train_phase_candidates`` for a HUMAN to copy into the yaml as an independent field.

TRAIN/EXAM SPLIT (--split {train,exam,all}, default all = current behavior): membership is a
deterministic hash of each question's incoming velocity (question_split, ~80/20), NOT a seed/order
partition — so a --split train bank and a --split exam bank are disjoint at ANY generation seeds.
Exams are scored only on the exam split (docs/stage_curriculum_v1.md).

Run (Mac base env or pod venv; numpy+torch+yaml):
    python scripts/gen_stage1_questions.py \
        --clip forehand:/workspace/shared/motions/hope_forehand_hopex.npz \
        --clip backhand:/workspace/shared/motions/hope_backhand_hopex.npz \
        --n 512 --out cfg/stage1_questions.npz
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WBT = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(WBT))

sys.path.insert(0, os.path.join(REPO, "hope_ws", "src", "hope_planner"))

TABLE_HALF_W = 0.7625   # env y=0 is the table centre line; hope_y = env_y - TABLE_HALF_W
TABLE_SURFACE_Z = 0.76  # env table surface height; hope_z = env_z - TABLE_SURFACE_Z


def _load_mod(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_analyzer():
    return _load_mod("s1_asp", os.path.join(HERE, "analyze_strike_phase.py"))


def strike_state_from_clip(asp, name: str, path: str, annotations: dict):
    """Fixed strike point (env frame) + clip reference normal from the annotated frame."""
    info = asp.analyze(name, path, use_blade=True)
    keys = asp._annotation_keys(name, path)
    ann = next((annotations[k] for k in keys if k in annotations), None)
    if ann is None:
        raise SystemExit(f"{name}: no entry in strike_annotations.yaml (keys tried: {keys}) — "
                         "annotate first, do not fall back to the speed peak")
    asp._apply_annotation(info, next(k for k in keys if k in annotations), ann)
    s = info["strike"]
    return dict(
        contact_pos_env=info["racket_w"][s].copy(),      # tracking env frame (root at origin)
        clip_normal=info["normal_w"][s].copy(),   # WORLD-frame face proxy (frame-mix fix
        # 2026-07-05, franco's catch: normal_root is PELVIS-frame while positions/velocities
        # are world — mid-swing pelvis rotation rotated every "face" by tens of degrees)
        clip_vel=info["clean_v_w"][s].copy(),
        strike_frame=int(s),
        n_frames=int(info["T"]),
    )


def sample_incoming(rng, n, speed_lo, speed_hi, vy_max, vz_lo, vz_hi):
    """Incoming ball velocity AT the strike point (env/world axes, toward the robot = -x)."""
    speed = rng.uniform(speed_lo, speed_hi, size=n)
    vy = rng.uniform(-vy_max, vy_max, size=n)
    vz = rng.uniform(vz_lo, vz_hi, size=n)
    vx_sq = np.maximum(speed**2 - vy**2 - vz**2, 0.25)  # keep a real toward-robot component
    vx = -np.sqrt(vx_sq)
    return np.stack([vx, vy, vz], axis=1)


EXAM_FRAC = 0.2  # deterministic hash-split target: ~80% train / ~20% exam


def question_split(v_in_row) -> str:
    """Deterministic train/exam membership of a question, a pure function of its incoming velocity.

    blake2b over the ROUNDED (1e-4 m/s) v_in bytes -> uniform in [0, 1); < EXAM_FRAC = exam.
    Membership depends ONLY on the question content, never on seed/order/solver outcome, so a
    train bank and an exam bank generated at ANY seeds can never share a question — the exam is
    guaranteed disjoint from training.
    """
    import hashlib

    key = np.round(np.asarray(v_in_row, dtype=np.float64), 4).tobytes()
    h = hashlib.blake2b(key, digest_size=8).digest()
    return "exam" if int.from_bytes(h, "big") / 2.0**64 < EXAM_FRAC else "train"


def phase_scan(asp, name: str, path: str, annotations: dict, args) -> None:
    """Per-frame returnability under the frame's own swing-velocity cone (torch batch/frame)."""
    import torch

    vb = _load_mod("s1_vb", os.path.join(
        WBT, "source", "whole_body_tracking", "whole_body_tracking",
        "tasks", "tracking", "mdp", "virtual_ball.py"))
    prm = vb.load_venue_params(os.path.join(REPO, "configs", "ball_physics_venue.yaml"))
    torch.set_default_dtype(torch.float64)

    info = asp.analyze(name, path, use_blade=True)
    keys = asp._annotation_keys(name, path)
    ann = next((annotations[k] for k in keys if k in annotations), None)
    label_frame = None
    if ann is not None:
        asp._apply_annotation(info, next(k for k in keys if k in annotations), ann)
        label_frame = int(info["strike"])
    T = int(info["T"])

    rng = np.random.default_rng(args.seed)
    v_in = sample_incoming(rng, args.scan_balls, *args.speed_range, args.vy_max, *args.vz_range)
    M, K = args.scan_balls, args.cone_vels

    net_x = args.near_x + 1.37
    far_x = args.near_x + 2.74
    half_w = TABLE_HALF_W
    net_top = TABLE_SURFACE_Z + 0.1525
    ball_r = float(prm.ball_radius)
    depth_x = net_x + 0.3

    t64 = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64)  # noqa: E731
    scores = np.zeros(T)
    for t in range(T):
        v_clip = np.asarray(info["clean_v_w"][t], float)
        speed = float(np.linalg.norm(v_clip))
        if speed < args.scan_min_speed:
            continue
        v_dir = v_clip / speed
        ang = np.deg2rad(rng.uniform(0.0, args.cone_deg, size=K))
        aux = rng.standard_normal((K, 3))
        aux -= (aux @ v_dir)[:, None] * v_dir[None, :]
        aux /= np.linalg.norm(aux, axis=1, keepdims=True) + 1e-12
        dirs = np.cos(ang)[:, None] * v_dir[None, :] + np.sin(ang)[:, None] * aux
        v_r = dirs * (speed * rng.uniform(*args.cone_mag, size=K))[:, None]

        n_face = np.asarray(info["normal_w"][t], float)   # world frame (frame-mix fix)
        p_env = np.asarray(info["racket_w"][t], float)
        vv_in = t64(np.repeat(v_in, K, axis=0))               # (M*K,3)
        vv_r = t64(np.tile(v_r, (M, 1)))
        nn = t64(np.tile(n_face, (M * K, 1)))
        pp = t64(np.tile(p_env, (M * K, 1)))
        v_plus, w_plus = vb.predict_paddle_contact(vv_in, vv_r, nn, torch.zeros(M * K, 3), prm)
        land = vb.coarse_landing(pp, v_plus, w_plus, prm,
                                 surface_z=TABLE_SURFACE_Z + ball_r, net_x=net_x,
                                 h=0.01, n_steps=200)
        lx, ly = land["land_xy"][:, 0], land["land_xy"][:, 1]
        legal = (land["land_valid"] & land["net_valid"]
                 & (land["net_z"] > net_top + ball_r)
                 & (lx > depth_x) & (lx <= far_x) & (ly.abs() <= half_w))
        scores[t] = legal.view(M, K).any(dim=1).double().mean().item()

    top = np.argsort(-scores)[:3]
    band = [f"{t/(T-1):.2f}" for t in range(T) if scores[t] > 0.5]
    lbl = (f"registry frame {label_frame} (phase {label_frame/(T-1):.3f}) score "
           f"{scores[label_frame]:.0%}" if label_frame is not None else "no registry entry")
    print(f"[scan:{name}] T={T} | {lbl}")
    print(f"  top frames: " + ", ".join(
        f"f{t} (phase {t/(T-1):.3f}) {scores[t]:.0%}" for t in top))
    band_txt = (", ".join(band) if band else
                "NONE — no frame of this clip can return the sampled balls under its own cone")
    print(f"  >50% band: {band_txt}")
    cands = sorted({int(t) for t in top if scores[t] >= 0.5})
    if cands:
        print(f"  suggested (copy into strike_annotations.yaml by hand, independent field):\n"
              f"    train_phase_candidates: {[round(t/(T-1), 3) for t in cands]}"
              f"   # returnability-optimal, NOT video contact truth")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", action="append", required=True,
                    help="name:path.npz (repeatable; name = forehand/backhand)")
    ap.add_argument("--n", type=int, default=512, help="questions per clip")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--speed-range", type=float, nargs=2, default=[2.0, 5.0],
                    help="incoming ball speed at contact (m/s)")
    ap.add_argument("--vy-max", type=float, default=0.6)
    ap.add_argument("--vz-range", type=float, nargs=2, default=[-2.0, 0.3])
    ap.add_argument("--landing-env", type=float, nargs=2, default=[2.555, 0.0],
                    help="fixed landing target, env frame (default P2 half centre)")
    ap.add_argument("--near-x", type=float, default=0.5,
                    help="table near edge x in the env frame (cfg vb_table_near_x)")
    ap.add_argument("--speed-budget", type=float, default=4.0,
                    help="max |racket contact-point velocity| passed to the solver (m/s)")
    ap.add_argument("--out", default=os.path.join(WBT, "cfg", "stage1_questions.npz"))
    ap.add_argument("--no-check", action="store_true",
                    help="skip the torch virtual-ball closed-loop self-test")
    ap.add_argument("--split", choices=("train", "exam", "all"), default="all",
                    help="keep only this side of the deterministic ~80/20 hash split on the "
                         "incoming-velocity bytes (exam questions are disjoint from training at "
                         "ANY seed); 'all' = no split (current behavior)")
    # --- phase-scan mode (per-frame returnability under the clip's own swing-velocity cone) ---
    ap.add_argument("--phase-scan", action="store_true",
                    help="scan EVERY frame's returnability instead of building the bank; prints "
                         "suggested train_phase_candidates (never writes the registry)")
    ap.add_argument("--cone-deg", type=float, default=25.0,
                    help="[scan] max angle between candidate racket velocity and the frame's "
                         "clip velocity")
    ap.add_argument("--cone-mag", type=float, nargs=2, default=[0.6, 1.4],
                    help="[scan] candidate speed as a multiple of the frame's clip speed")
    ap.add_argument("--cone-vels", type=int, default=150, help="[scan] velocity candidates/frame")
    ap.add_argument("--scan-balls", type=int, default=24, help="[scan] incoming balls (fixed set)")
    ap.add_argument("--scan-min-speed", type=float, default=0.3,
                    help="[scan] frames with clip speed below this are unstrikeable (score 0)")
    args = ap.parse_args()

    from hope_planner.strike_spec_planner import StrikeSpecPlanner

    asp = _load_analyzer()
    annotations = asp._load_annotations(asp.DEFAULT_ANNOTATIONS)

    if args.phase_scan:
        for spec in args.clip:
            name, _, path = spec.partition(":")
            phase_scan(asp, name, path, annotations, args)
        return 0

    planner = StrikeSpecPlanner()
    rng = np.random.default_rng(args.seed)

    env2hope = np.array([-args.near_x, -TABLE_HALF_W, -TABLE_SURFACE_Z])
    landing_hope = np.asarray(args.landing_env) + env2hope[:2]

    banks = {}
    for spec in args.clip:
        name, _, path = spec.partition(":")
        st = strike_state_from_clip(asp, name, path, annotations)
        p_env = st["contact_pos_env"]
        p_hope = p_env + env2hope

        v_in = sample_incoming(rng, args.n, *args.speed_range, args.vy_max, *args.vz_range)
        # Deterministic train/exam membership per question (hash of v_in — see question_split);
        # counted BEFORE solving so both counts and the answerable-fraction denominator are per split.
        splits = np.array([question_split(v) for v in v_in])
        n_train, n_exam = int((splits == "train").sum()), int((splits == "exam").sum())
        sel = np.arange(args.n) if args.split == "all" else np.where(splits == args.split)[0]
        kept, normals, vels, diffs, residuals, landings, in_cone = [], [], [], [], [], [], []
        for i in sel:
            s = planner.solve(p_hope, v_in[i], None, landing_hope, args.speed_budget)
            if s is None or s.residual_m > planner.TOL_M:
                continue
            kept.append(i)
            cn = st["clip_normal"] / np.linalg.norm(st["clip_normal"])
            # Store the normal SIGN-ALIGNED to the clip's FK +Y red face: contact physics is
            # sign-agnostic (orient_normal flips internally) but the face-tracking reward is
            # sign-sensitive — the raw solver sign is arbitrary and could demand a 180-deg flip.
            nn = s.n if np.dot(s.n, cn) >= 0 else -s.n
            normals.append(nn)
            vels.append(s.v_r)
            residuals.append(s.residual_m)
            landings.append(s.landing_xy)
            diffs.append(np.degrees(np.arccos(np.clip(np.dot(nn, cn), -1, 1))))
            cv = st["clip_vel"]; cs = np.linalg.norm(cv)
            ang = np.degrees(np.arccos(np.clip(
                np.dot(s.v_r, cv) / (np.linalg.norm(s.v_r) * cs + 1e-12), -1, 1)))
            in_cone.append(bool(ang <= 25.0 and 0.6 * cs <= np.linalg.norm(s.v_r) <= 1.4 * cs))

        kept = np.array(kept, dtype=int)
        # The answerable-fraction denominator is the SELECTED split's question count, not args.n —
        # with --split the un-selected side is never solved, so args.n would understate the rate.
        rate = len(kept) / max(len(sel), 1)
        d = np.array(diffs)
        print(f"[{name}] strike point env={np.round(p_env, 3)} (clip frame {st['strike_frame']}/"
              f"{st['n_frames']}), split={args.split} (sampled {n_train} train / {n_exam} exam), "
              f"solvable {len(kept)}/{len(sel)} = {rate:.0%}")
        if rate < 0.5:
            print(f"  ** WARNING: solvability <50% — the fixed strike point is badly placed for "
                  f"this speed range; move the point / relax the budget instead of training on this **")
        if len(kept):
            print(f"  difficulty (face vs clip normal): med={np.median(d):.1f} deg  "
                  f"p90={np.percentile(d, 90):.1f}  max={d.max():.1f}  |  answers inside the "
                  f"clip swing-velocity cone (25deg/0.6-1.4x): {np.mean(in_cone):.0%}")
        banks[name] = dict(
            contact_pos_env=p_env, clip_normal=st["clip_normal"], clip_vel=st["clip_vel"],
            incoming_vel=v_in[kept], demanded_normal=np.array(normals),
            demanded_vel=np.array(vels), difficulty_deg=d,
            residual_m=np.array(residuals), landing_xy_hope=np.array(landings),
            in_cone=np.array(in_cone, dtype=bool),
        )

    # --- torch virtual-ball closed-loop self-test (trainer-side venue physics) ----------
    if not args.no_check:
        import torch
        vb = _load_mod("s1_vb", os.path.join(
            WBT, "source", "whole_body_tracking", "whole_body_tracking",
            "tasks", "tracking", "mdp", "virtual_ball.py"))
        prm = vb.load_venue_params(os.path.join(REPO, "configs", "ball_physics_venue.yaml"))
        torch.set_default_dtype(torch.float64)
        for name, b in banks.items():
            if not len(b["incoming_vel"]):
                continue
            N = len(b["incoming_vel"])
            t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64)
            p0 = t(np.tile(b["contact_pos_env"], (N, 1)))
            v_plus, w_plus = vb.predict_paddle_contact(
                t(b["incoming_vel"]), t(b["demanded_vel"]), t(b["demanded_normal"]),
                torch.zeros(N, 3), prm)
            land = vb.coarse_landing(
                p0, v_plus, w_plus, prm,
                surface_z=0.76 + prm.ball_radius,  # physical contact plane (surface + R)
                net_x=args.near_x + 1.37, h=0.01, n_steps=200)
            tgt = torch.tensor(args.landing_env)
            err = torch.linalg.norm(land["land_xy"] - tgt, dim=-1)
            ok = land["land_valid"]
            print(f"[{name}] torch closed-loop: landed {int(ok.sum())}/{N}, "
                  f"|err| med={err[ok].median():.3f} m  p90={err[ok].quantile(0.9):.3f} m  "
                  f"(planner residual med={np.median(b['residual_m'])*1000:.1f} mm)")

    meta = dict(seed=args.seed, speed_range=args.speed_range, vy_max=args.vy_max,
                vz_range=args.vz_range, landing_env=args.landing_env, near_x=args.near_x,
                speed_budget=args.speed_budget, split=args.split, exam_frac=EXAM_FRAC)
    flat = {f"{n}/{k}": v for n, b in banks.items() for k, v in b.items()}
    flat["meta_json"] = np.frombuffer(repr(meta).encode(), dtype=np.uint8)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, **flat)
    print(f"[bank] wrote {args.out}  (clips: {list(banks)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
