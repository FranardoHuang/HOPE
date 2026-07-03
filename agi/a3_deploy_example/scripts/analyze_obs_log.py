#!/usr/bin/env python3
"""Health-check + audit the ping-pong runner's CSV logs, with PASS/FAIL + a report.

Handles two kinds of CSV the runner writes (newer runs also log legs_passive):
  * OBS  CSV  (--obs-csv)  : tick,ts,mode,loc_mode,oracle_fresh,oracle_age_s,
                             sync_miss,legs_passive, obs_0..179, act_0..30
  * TRACE CSV (--trace-csv): tick,ts,mode,level,gain,swing,legs_passive,gravx,gravy,
                             gravz, des_<joint>..., q_<joint>..., qd_<joint>...,
                             kp_<joint>..., kd_<joint>...

For each file it auto-detects the type, runs checks, prints a PASS/FAIL summary,
and writes a small text report next to the CSV ("<file>.report.txt").

OBS checks   : obs dim==180; no NaN/Inf; projected_gravity sane; motion_anchor_pos_b
               ~0 in perfect_tracking; joint_pos_rel / joint_vel / action bounded;
               sync_miss==0; legs_passive (if logged); + a waist_roll audit (is the
               big command from the baked REFERENCE or from the policy ACTION?).
TRACE checks : q_des within joint limits (clamp really applied); kp/kd ranges;
               waist_roll clamp frequency; + a FULL-BODY (31-DOF) audit — legs
               driven (legs_passive), per-joint cmd/meas range + tracking, leg/waist
               amplitude, leg qd bounded, top movers/mis-trackers, rail-bound joints,
               HIP tracking, UPRIGHT (gravz~-1), and capture hygiene (warns if level 0
               and level 1, or multiple gain_scale, are MIXED in one capture).
               Hoisted leg TRACKING is expected poor (feet unloaded) -> WARN not FAIL.
NOT in CSV   : halts, foot-slip, support-rope load. Read halts from the [status]
               line / status.log; foot-slip + rope-load are VISUAL observations.

Usage:
  python3 scripts/analyze_obs_log.py /tmp/pp_obs.csv
  python3 scripts/analyze_obs_log.py /tmp/pp_obs.csv /tmp/pp_trace.csv
  python3 scripts/analyze_obs_log.py A=fabricated.csv B=perfect.csv C=oracle.csv
"""
import csv
import math
import sys
from statistics import mean, median

# ---- contract constants (from the ONNX metadata / PINGPONG_DEPLOY_ALIGNMENT.md) ----
N_ACT = 31
# Isaac policy index of waist_roll, its action_scale, default pose, A3 limit.
WAIST_ROLL_ISAAC_IDX = 5
WAIST_ROLL_SCALE = 0.230
WAIST_ROLL_DEFAULT = 0.0
WAIST_ROLL_LIMIT = 0.34907  # +/- rad, A3 MJCF (pp_joint_limits.hpp slot 1)


# obs block spans per contract (see pp_obs_builder.hpp / a3_pingpong_main.cpp blks175/blks180).
# 180 = legacy FULL layout; 175 = deploy_parity (drops motion_anchor_pos_b[62:65] +
# base_target_pos_b[170:172], racket target reframed relative to the racket FK).
def obs_blocks(n_obs_cols):
    if n_obs_cols == 180:
        return {"anchor_pos": (62, 65), "joint_pos_rel": (74, 105), "joint_vel": (105, 136),
                "proj_grav": (167, 170), "racket_tgt": (172, 175), "tts": 178, "swing": 179}
    if n_obs_cols == 175:
        return {"anchor_pos": None, "joint_pos_rel": (71, 102), "joint_vel": (102, 133),
                "proj_grav": (164, 167), "racket_tgt": (167, 170), "tts": 173, "swing": 174}
    return None

# A3 joint POSITION limits in backend(SDK) order, verbatim from pp_joint_limits.hpp
# (kSdkJointPosLo/Hi). Used by the trace full-body audit to flag rail-bound q_des.
JOINT_LIMITS = {
    "waist_yaw_joint": (-2.61799, 2.61799), "waist_roll_joint": (-0.34907, 0.34907),
    "waist_pitch_joint": (-0.48869, 0.41888), "head_yaw_joint": (-1.04720, 1.04720),
    "head_pitch_joint": (-0.43633, 0.26180),
    "left_shoulder_pitch_joint": (-2.87979, 2.87979), "left_shoulder_roll_joint": (-0.08727, 2.61799),
    "left_shoulder_yaw_joint": (-2.79253, 2.79253), "left_elbow_joint": (-0.95993, 1.74533),
    "left_wrist_roll_joint": (-2.79253, 2.79253), "left_wrist_pitch_joint": (-1.62316, 1.62316),
    "left_wrist_yaw_joint": (-1.62316, 1.62316),
    "right_shoulder_pitch_joint": (-2.87979, 2.87979), "right_shoulder_roll_joint": (-2.61799, 0.08727),
    "right_shoulder_yaw_joint": (-2.79253, 2.79253), "right_elbow_joint": (-0.95993, 1.74533),
    "right_wrist_roll_joint": (-2.79253, 2.79253), "right_wrist_pitch_joint": (-1.62316, 1.62316),
    "right_wrist_yaw_joint": (-1.62316, 1.62316),
    "left_hip_pitch_joint": (-2.51327, 2.93215), "left_hip_roll_joint": (-0.52360, 1.60570),
    "left_hip_yaw_joint": (-2.72271, 2.72271), "left_knee_joint": (-0.12217, 2.49582),
    "left_ankle_pitch_joint": (-0.90757, 0.52360), "left_ankle_roll_joint": (-0.34907, 0.34907),
    "right_hip_pitch_joint": (-2.51327, 2.93215), "right_hip_roll_joint": (-1.60570, 0.52360),
    "right_hip_yaw_joint": (-2.72271, 2.72271), "right_knee_joint": (-0.12217, 2.49582),
    "right_ankle_pitch_joint": (-0.90757, 0.52360), "right_ankle_roll_joint": (-0.34907, 0.34907),
}


def _per_joint_ranges(rowset, joints, prefix):
    """name -> (lo, hi, range) over rowset for column prefix+name; skip if empty."""
    out = {}
    for j in joints:
        vals = fcol(rowset, f"{prefix}{j}")
        if vals:
            out[j] = (min(vals), max(vals), max(vals) - min(vals))
    return out


def _tracking_error(rowset, joints):
    """name -> max|des-meas| over rowset (rows where both columns parse)."""
    out = {}
    for j in joints:
        worst, seen = 0.0, False
        for r in rowset:
            d, q = r.get(f"des_{j}"), r.get(f"q_{j}")
            if d in (None, "") or q in (None, ""):
                continue
            try:
                worst = max(worst, abs(float(d) - float(q)))
                seen = True
            except ValueError:
                continue
        if seen:
            out[j] = worst
    return out


def load(path):
    with open(path) as f:
        r = csv.DictReader(f)
        return list(r), (r.fieldnames or [])


def _row_floats(row, prefix, n):
    """Return [float]*n for row[prefix0..n-1], or None if any cell is missing /
    blank / unparseable (e.g. the last line of a still-being-written CSV)."""
    out = []
    for i in range(n):
        v = row.get(f"{prefix}{i}")
        if v is None or v == "":
            return None
        try:
            out.append(float(v))
        except ValueError:
            return None
    return out


def fcol(rows, name):
    out = []
    for row in rows:
        v = row.get(name)
        if v is None or v == "":
            continue
        try:
            out.append(float(v))
        except ValueError:
            continue
    return out


def vnorm(v):
    return sum(x * x for x in v) ** 0.5


class Report:
    def __init__(self, label, path):
        self.label, self.path = label, path
        self.lines, self.checks = [], []  # checks: (name, status, detail)

    def info(self, s):
        self.lines.append(s)

    def check(self, name, status, detail):
        self.checks.append((name, status, detail))

    def overall(self):
        st = [c[1] for c in self.checks]
        return "FAIL" if "FAIL" in st else ("WARN" if "WARN" in st else "PASS")

    def render(self):
        out = [f"========== [{self.label}] {self.path} ==========", *self.lines, ""]
        for name, status, detail in self.checks:
            out.append(f"  [{status:4}] {name:28} {detail}")
        out.append(f"\n  OVERALL: {self.overall()}")
        return "\n".join(out)


def analyze_obs(label, path, rows, cols):
    rep = Report(label, path)
    n_obs_cols = sum(1 for c in cols if c.startswith("obs_"))
    has_act = any(c.startswith("act_") for c in cols)

    # 1. obs dimension (175 deploy_parity or 180 full)
    blocks = obs_blocks(n_obs_cols)
    rep.check("obs_dim in (175,180)", "PASS" if blocks is not None else "FAIL",
              f"found {n_obs_cols}")
    if blocks is None:
        rep.info(f"  ticks logged : {len(rows)}")
        return rep
    OBS_DIM = n_obs_cols
    B_ANCHOR_POS = blocks["anchor_pos"]
    B_PROJ_GRAV = blocks["proj_grav"]
    B_JOINT_POS_REL = blocks["joint_pos_rel"]
    B_JOINT_VEL = blocks["joint_vel"]
    B_RACKET_TGT = blocks["racket_tgt"]
    B_SWING = blocks["swing"]

    # Build obs/act from COMPLETE rows only (a live/streaming CSV may have a
    # partial last line -> DictReader yields None cells; skip those).
    obs, act, grows = [], ([] if has_act else None), []
    dropped = 0
    for r in rows:
        o = _row_floats(r, "obs_", OBS_DIM)
        if o is None:
            dropped += 1
            continue
        if has_act:
            a = _row_floats(r, "act_", N_ACT)
            if a is None:
                dropped += 1
                continue
            act.append(a)
        obs.append(o)
        grows.append(r)
    rep.info(f"  ticks logged : {len(rows)} (complete: {len(obs)}, dropped partial: {dropped})"
             f"   obs_cols={n_obs_cols}  act_cols={'yes' if has_act else 'no'}")
    if not obs:
        rep.check("has complete rows", "FAIL", "no complete obs rows (file still writing?)")
        return rep
    loc_mode = grows[0].get("loc_mode", "?")
    rep.info(f"  loc_mode enum: {loc_mode}  (0=fabricated, 1=perfect_tracking, 2=oracle)")

    # 2. NaN / Inf
    bad = sum(1 for o in obs for x in o if not math.isfinite(x))
    if act:
        bad += sum(1 for a in act for x in a if not math.isfinite(x))
    rep.check("no NaN/Inf", "PASS" if bad == 0 else "FAIL", f"{bad} non-finite values")

    # 3. projected gravity sane (|g|~1 every tick)
    pg = [o[B_PROJ_GRAV[0]:B_PROJ_GRAV[1]] for o in obs]
    pg_norms = [vnorm(v) for v in pg]
    gz = [v[2] for v in pg]
    worst = max(abs(n - 1.0) for n in pg_norms)
    rep.check("projected_gravity |g|~1", "PASS" if worst < 0.1 else "FAIL",
              f"max|‖g‖-1|={worst:.3f}  meanGz={mean(gz):+.3f} (upright≈-1)")

    # 4. motion_anchor_pos_b ~ 0 in perfect_tracking (180-D layout only; 175 drops the term)
    if B_ANCHOR_POS is not None:
        an = [vnorm(o[B_ANCHOR_POS[0]:B_ANCHOR_POS[1]]) for o in obs]
        amax = max(an)
        if loc_mode == "1":
            st = "PASS" if amax < 1e-3 else ("WARN" if amax < 0.05 else "FAIL")
            rep.check("anchor_pos_b~0 (perfect)", st, f"max|.|={amax:.5f}")
        else:
            rep.check("anchor_pos_b (non-perfect)", "INFO", f"max|.|={amax:.4f} (only ~0 expected in perfect_tracking)")
    else:
        rep.info("  anchor_pos_b: n/a (175-D deploy_parity layout has no world anchor-pos term)")

    # 5. joint_pos_rel bounded
    jpr = [max(abs(x) for x in o[B_JOINT_POS_REL[0]:B_JOINT_POS_REL[1]]) for o in obs]
    m = max(jpr)
    rep.check("joint_pos_rel bounded", "PASS" if m < 2.5 else ("WARN" if m < 5 else "FAIL"),
              f"max|q-q_def|={m:.3f} rad")

    # 6. joint_vel bounded
    jv = [max(abs(x) for x in o[B_JOINT_VEL[0]:B_JOINT_VEL[1]]) for o in obs]
    m = max(jv)
    rep.check("joint_vel bounded", "PASS" if m < 25 else ("WARN" if m < 60 else "FAIL"),
              f"max|qd|={m:.2f} rad/s")

    # 7. action bounded (raw pre-scale; legit swing peaks ~25, pathological >>)
    if act:
        amag = [vnorm(a) for a in act]
        amx = max(max(abs(x) for x in a) for a in act)
        st = "PASS" if amx < 30 else ("WARN" if amx < 100 else "FAIL")
        rep.check("action bounded", st, f"max|act|={amx:.2f}  mean‖act‖={mean(amag):.2f}")
        # buzz proxy: tick-to-tick action jerk
        dj = [vnorm([act[i][j] - act[i-1][j] for j in range(N_ACT)]) for i in range(1, len(act))]
        if dj:
            rep.info(f"  |Δaction| jerk (buzz proxy): mean={mean(dj):.3f} max={max(dj):.3f}")

    # racket target / tts / swing sanity
    rk = [o[B_RACKET_TGT[0]:B_RACKET_TGT[1]] for o in obs]
    rkx = [v[0] for v in rk]
    rep.info(f"  racket_target_pos_b x: min={min(rkx):+.3f} max={max(rkx):+.3f}  "
             + ("(175-D: target RELATIVE TO RACKET FK; shrinks toward 0 through the swing)"
                if OBS_DIM == 175 else
                "(180-D: forehand expects ~+0.4 FRONT; negative => yaw-frame wrong)"))
    sw = set(round(o[B_SWING]) for o in obs)
    rep.info(f"  swing_type values: {sorted(sw)}  (+1 forehand / -1 backhand)")
    sm = [int(r["sync_miss"]) for r in grows
          if r.get("sync_miss") not in (None, "") and r["sync_miss"].lstrip("-").isdigit()]
    sm_max = max(sm) if sm else 0
    rep.info(f"  cumulative sync_miss : {sm_max}")
    # sync_miss must stay 0; a few at startup (pre-alignment) are benign -> WARN,
    # a large/growing count means the state stream is dropping -> FAIL.
    rep.check("sync_miss==0", "PASS" if sm_max == 0 else ("WARN" if sm_max < 20 else "FAIL"),
              f"cumulative sync_miss={sm_max}" + (" (a few at startup are benign)" if sm_max else ""))

    # legs_passive (logged in newer runs): true => legs HELD nominal (NOT full-body).
    lp = set(int(round(v)) for v in fcol(grows, "legs_passive"))
    if lp:
        rep.info(f"  legs_passive (logged): {'true' if 1 in lp else 'false'}"
                 + ("  -> legs HELD nominal; this is NOT a full-body test" if 1 in lp
                    else "  -> legs policy-driven (full-body command path active)"))

    # ---- waist_roll audit (is the over-limit command from REFERENCE or ACTION?) ----
    ref_wr = [o[WAIST_ROLL_ISAAC_IDX] for o in obs]  # command block = ref joint_pos (Isaac order)
    rep.info("")
    rep.info("  --- waist_roll audit (A3 limit = ±%.3f rad) ---" % WAIST_ROLL_LIMIT)
    rep.info(f"  reference waist_roll (obs_{WAIST_ROLL_ISAAC_IDX}) : "
             f"min={min(ref_wr):+.3f} max={max(ref_wr):+.3f}")
    if act:
        a_wr = [a[WAIST_ROLL_ISAAC_IDX] for a in act]
        qdes_wr = [WAIST_ROLL_DEFAULT + x * WAIST_ROLL_SCALE for x in a_wr]
        over = [q for q in qdes_wr if abs(q) > WAIST_ROLL_LIMIT]
        frac = len(over) / len(qdes_wr)
        max_viol = max((abs(q) - WAIST_ROLL_LIMIT for q in qdes_wr if abs(q) > WAIST_ROLL_LIMIT),
                       default=0.0)
        rep.info(f"  policy q_des waist_roll = default + act*scale (act_{WAIST_ROLL_ISAAC_IDX}*{WAIST_ROLL_SCALE}): "
                 f"min={min(qdes_wr):+.3f} max={max(qdes_wr):+.3f}")
        rep.info(f"  q_des over ±{WAIST_ROLL_LIMIT:.3f}: {100*frac:.0f}% of ticks, max violation {max_viol:.3f} rad")
        ref_over = max(abs(x) for x in ref_wr) > WAIST_ROLL_LIMIT
        if frac < 0.01:
            src = "negligible — waist_roll is NOT a problem in this run"
        elif ref_over:
            src = ("the baked REFERENCE clip itself exceeds the A3 waist_roll limit "
                   "(embodiment mismatch: training URDF allowed more roll). Fix = soften "
                   "the reference/retarget, or accept+document the clamp.")
        else:
            src = ("the POLICY ACTION overshoots (reference is in-range) — action/ONNX, "
                   "not the clip. Clamp keeps it safe; consider lower gain on waist_roll.")
        rep.check("waist_roll within A3 limit",
                  "PASS" if frac < 0.01 else "WARN", f"{100*frac:.0f}% over-limit; source: {src}")
    return rep


def analyze_trace_fullbody(rep, rows, joints):
    """31-DOF (legs + waist) command-generation audit for the hoisted full-body
    verification. Reports per-joint command/measured ranges + tracking, leg/waist
    amplitudes, top movers/mis-trackers, and rail-bound joints, with PASS/WARN/FAIL.
    On a HOIST poor leg TRACKING is expected (feet bear no load) -> WARN, not FAIL;
    the goal is to confirm leg/waist commands are PRODUCED, bounded, and not railing."""
    LEG = [j for j in joints if any(k in j for k in ("hip", "knee", "ankle"))]
    HIP = [j for j in LEG if "hip" in j]
    WAIST = [j for j in joints if j.startswith("waist")]
    LLEG = [j for j in LEG if j.startswith("left")]
    RLEG = [j for j in LEG if j.startswith("right")]
    poly = [r for r in rows if r.get("mode") in ("2", "3")]  # SHADOW+MOTION: cmd generated
    motion = [r for r in rows if r.get("mode") == "3"]        # MOTION: robot actually driven
    base = poly or rows

    # capture hygiene (ground-contact analysis): which swing level(s) and gain(s)
    # appear in the policy rows. Mixing L0(frozen-windup, twitchy)+L1(swing), or
    # multiple gains, in one capture pools incomparable data -> WARN to capture clean.
    def _f(s):
        try:
            return float(s)
        except (TypeError, ValueError):
            return 0.0
    levels = sorted({r.get("level") for r in poly if r.get("level") not in (None, "")})
    gains = sorted({r.get("gain") for r in poly if r.get("gain") not in (None, "")}, key=_f)

    rep.info("")
    rep.info("  ===== full-body (31-DOF) verification =====")
    rep.info(f"  policy rows (SHADOW+MOTION)={len(poly)}  MOTION rows={len(motion)}  "
             f"(cmd ranges use policy rows; tracking/qd use MOTION rows)")

    # legs_passive: logged column preferred; else inferred from leg q_des activity.
    lp_vals = set(int(round(v)) for v in fcol(rows, "legs_passive"))
    legs_passive = (1 in lp_vals) if lp_vals else None
    src = "logged" if lp_vals else "inferred"

    des_rng = _per_joint_ranges(base, joints, "des_")     # name -> (lo,hi,range)
    meas_rng = _per_joint_ranges(motion, joints, "q_")

    def gmax_des(names):
        m = 0.0
        for j in names:
            if j in des_rng:
                m = max(m, abs(des_rng[j][0]), abs(des_rng[j][1]))
        return m

    def gmax_qd(names):
        m = 0.0
        for j in names:
            v = fcol(motion, f"qd_{j}")
            if v:
                m = max(m, max(abs(x) for x in v))
        return m

    lleg_des, rleg_des = gmax_des(LLEG), gmax_des(RLEG)
    lleg_qd, rleg_qd = gmax_qd(LLEG), gmax_qd(RLEG)
    waist_des = gmax_des(WAIST)

    # leg q_des "changing" %: ticks where any leg joint deviates >0.01 rad from its median.
    meds = {j: median(fcol(base, f"des_{j}")) for j in LEG if fcol(base, f"des_{j}")}
    moved, tot = 0, 0
    for r in base:
        tot += 1
        for j in LEG:
            x = r.get(f"des_{j}")
            if j in meds and x not in (None, ""):
                try:
                    if abs(float(x) - meds[j]) > 0.01:
                        moved += 1
                        break
                except ValueError:
                    pass
    leg_change_pct = 100.0 * moved / tot if tot else 0.0

    # waist rail-tick count (any waist joint des at/over its A3 limit).
    def rail_ticks(names):
        c = 0
        for r in base:
            for j in names:
                lo, hi = JOINT_LIMITS.get(j, (None, None))
                if lo is None:
                    continue
                x = r.get(f"des_{j}")
                if x in (None, ""):
                    continue
                try:
                    xf = float(x)
                except ValueError:
                    continue
                if xf <= lo + 1e-4 or xf >= hi - 1e-4:
                    c += 1
                    break
        return c
    waist_rail = rail_ticks(WAIST)

    # per-joint rail frequency (for the "frequently railed" check).
    railed = {}
    for j in joints:
        lo, hi = JOINT_LIMITS.get(j, (None, None))
        if lo is None:
            continue
        v = fcol(base, f"des_{j}")
        if not v:
            continue
        f = sum(1 for x in v if x <= lo + 1e-4 or x >= hi - 1e-4) / len(v)
        if f > 0.01:
            railed[j] = f
    freq_railed = {j: f for j, f in railed.items() if f > 0.20}

    # ---- info dumps ----
    rep.info("  per-joint (waist + legs): desR | measR | trk%")
    for j in WAIST + LLEG + RLEG:
        drr = des_rng.get(j, (0, 0, 0))[2]
        mrr = meas_rng.get(j, (0, 0, 0))[2]
        trk = 100.0 * mrr / drr if drr > 1e-3 else 0.0
        rep.info(f"    {j:28} desR={drr:5.3f}  measR={mrr:5.3f}  trk={trk:3.0f}%")
    rep.info(f"  leg q_des changing : {leg_change_pct:.0f}% of policy ticks (>0.01 rad from median)")
    rep.info(f"  left  leg          : max|q_des|={lleg_des:.3f}  max|qd|={lleg_qd:.2f}")
    rep.info(f"  right leg          : max|q_des|={rleg_des:.3f}  max|qd|={rleg_qd:.2f}")
    rep.info(f"  waist              : max|q_des|={waist_des:.3f}  rail-ticks={waist_rail}")
    rep.info(f"  capture hygiene    : levels={levels or ['?']}  gain_scale={gains or ['?']}  "
             f"(0=hold/windup, 1=swing)")
    top_cmd = sorted(des_rng.items(), key=lambda kv: kv[1][2], reverse=True)[:10]
    rep.info("  top 10 command amplitude (desR, rad):")
    for j, (lo, hi, rng) in top_cmd:
        rep.info(f"    {j:28} {rng:5.3f}  [{lo:+.3f},{hi:+.3f}]")
    terr = _tracking_error(motion, joints)
    if terr:
        rep.info("  top 10 tracking error (|des-meas| max, MOTION, rad):")
        for j, e in sorted(terr.items(), key=lambda kv: kv[1], reverse=True)[:10]:
            rep.info(f"    {j:28} {e:5.3f}")

    # ---- checks (PASS/WARN/FAIL) ----
    leg_des_present = any(des_rng.get(j, (0, 0, 0))[2] > 0.01 for j in LEG)
    if legs_passive is True:
        rep.check("full-body: legs driven", "WARN",
                  f"legs_passive=true ({src}) -> legs HELD nominal; NOT a full-body test "
                  f"(re-run legs_passive=false). leg desR max={max((des_rng.get(j,(0,0,0))[2] for j in LEG), default=0):.3f}")
    elif leg_des_present:
        rep.check("full-body: legs driven", "PASS",
                  f"legs_passive={'false' if legs_passive is False else '?'} ({src}); leg q_des present "
                  f"(L max|des|={lleg_des:.3f} R max|des|={rleg_des:.3f}, changing {leg_change_pct:.0f}% of ticks)")
    else:
        st = "FAIL" if legs_passive is False else "WARN"
        rep.check("full-body: legs driven", st,
                  f"legs_passive={legs_passive} ({src}) but leg q_des ~flat (range<0.01) -> legs NOT driven")

    leg_qd = max(lleg_qd, rleg_qd)
    rep.check("leg qd bounded", "PASS" if leg_qd < 25 else ("WARN" if leg_qd < 45 else "FAIL"),
              f"max|leg qd|={leg_qd:.2f} rad/s (L={lleg_qd:.2f} R={rleg_qd:.2f})")

    if motion:
        trks = [min(1.0, meas_rng[j][2] / des_rng[j][2]) for j in LEG
                if des_rng.get(j, (0, 0, 0))[2] > 0.02 and j in meas_rng]
        if trks:
            mt = 100.0 * mean(trks)
            rep.check("leg tracking (hoist)", "PASS" if mt >= 50 else "WARN",
                      f"mean leg trk={mt:.0f}%" + ("" if mt >= 50 else " -- poor, EXPECTED on a hoist (not a fail)"))
        else:
            rep.check("leg tracking (hoist)", "INFO", "leg cmd range too small to assess tracking")
    else:
        rep.check("leg tracking (hoist)", "INFO", "no MOTION rows (shadow-only) -> tracking not evaluated")

    if freq_railed:
        lst = ", ".join(f"{j} {100*f:.0f}%" for j, f in sorted(freq_railed.items(), key=lambda kv: -kv[1]))
        rep.check("no joint frequently railed", "WARN",
                  f"clamp firing >20% of ticks on: {lst} (safe: clamped; tuning flag)")
    else:
        rep.check("no joint frequently railed", "PASS",
                  f"no joint clamped >20% of ticks ({len(railed)} clamped occasionally)")

    # --- hip tracking (the weakest leg group; matters most on the ground, loaded) ---
    if motion and HIP:
        hip_trks = [min(1.0, meas_rng[j][2] / des_rng[j][2]) for j in HIP
                    if des_rng.get(j, (0, 0, 0))[2] > 0.02 and j in meas_rng]
        if hip_trks:
            ht = 100.0 * mean(hip_trks)
            rep.check("hip tracking", "PASS" if ht >= 40 else "WARN",
                      f"mean hip trk={ht:.0f}%" + ("" if ht >= 40 else
                      " -- poor (hips carry the most load/cmd); WARN if bounded, FAIL only on qd-spike/rail"))

    # --- upright: gravz~-1 from the trace IMU (ground-contact balance proxy) ---
    gz = fcol(motion or poly, "gravz")
    if gz:
        mean_gz = mean(gz)
        gx, gy = fcol(motion or poly, "gravx"), fcol(motion or poly, "gravy")
        tilt = (max((x * x + y * y) ** 0.5 for x, y in zip(gx, gy))
                if gx and gy and len(gx) == len(gy) else 0.0)
        st = "PASS" if mean_gz <= -0.9 else ("WARN" if mean_gz <= -0.75 else "FAIL")
        rep.check("upright (gravz~-1)", st,
                  f"mean gravz={mean_gz:+.3f} (upright≈-1), max horiz tilt={tilt:.2f}" +
                  ("" if mean_gz <= -0.9 else " -- LEANING; check support/balance"))

    # --- capture hygiene: warn if level 0 and level 1 (or multiple gains) are mixed ---
    if len(levels) > 1:
        rep.check("single level in capture", "WARN",
                  f"capture MIXES levels {levels}: tracking pools L0(frozen-windup, twitchy) + "
                  f"L1(swing). Capture each level separately for clean ground-contact numbers.")
    else:
        rep.check("single level in capture", "PASS",
                  f"level={levels[0] if levels else '?'} throughout")
    if len(gains) > 1:
        rep.check("single gain in capture", "WARN",
                  f"gain_scale CHANGED during capture ({gains}): pools incomparable gains. "
                  f"Log each gain level separately (Stage G5).")
    else:
        rep.check("single gain in capture", "PASS",
                  f"gain_scale={gains[0] if gains else '?'} throughout")


def analyze_trace(label, path, rows, cols):
    rep = Report(label, path)
    joints = [c[len("des_"):] for c in cols if c.startswith("des_")]
    rep.info(f"  ticks logged : {len(rows)}   joints={len(joints)}")
    if not joints:
        rep.check("trace has des_ columns", "FAIL", "no des_<joint> columns")
        return rep

    # MOTION rows only for gain/clamp checks (mode enum: 0=PASSIVE 1=PD_STAND
    # 2=SHADOW 3=MOTION 4=REF). PASSIVE is limp (kp=kd=0 by design) and PD_STAND
    # uses flat stand gains -> including them would give a bogus kp/kd range.
    motion = [r for r in rows if r.get("mode") == "3"]
    rep.info(f"  rows: total={len(rows)}  MOTION={len(motion)}  (kp/kd & clamp checks use MOTION only)")

    # 1. q_des finite + bounded (post-clamp, so within joint limits)
    bad = 0
    max_des = 0.0
    for row in rows:
        for j in joints:
            v = row.get(f"des_{j}")
            if v in (None, ""):
                continue
            try:
                x = float(v)
            except ValueError:
                bad += 1
                continue
            if not math.isfinite(x):
                bad += 1
            max_des = max(max_des, abs(x))
    rep.check("q_des finite", "PASS" if bad == 0 else "FAIL", f"{bad} non-finite des values")
    rep.info(f"  max|q_des| over all joints: {max_des:.3f} rad")

    # 2. kp / kd ranges — un-scaled by the per-row gain, on MOTION rows, so they
    #    recover the TRAINING gains regardless of --gain-scale. (raw published =
    #    training * gain; PASSIVE rows legitimately publish 0 and are excluded.)
    def unscaled_range(prefix):
        vals = []
        for r in motion:
            g = r.get("gain")
            try:
                g = float(g)
            except (TypeError, ValueError):
                continue
            if g <= 1e-6:
                continue
            for j in joints:
                v = r.get(f"{prefix}{j}")
                if v in (None, ""):
                    continue
                try:
                    vals.append(float(v) / g)
                except ValueError:
                    continue
        return (min(vals), max(vals)) if vals else (None, None)
    if motion:
        kp_lo, kp_hi = unscaled_range("kp_")
        kd_lo, kd_hi = unscaled_range("kd_")
        if kp_lo is not None:
            ok = kp_lo >= 19 and kp_hi <= 260
            rep.check("kp in [20,250] (un-scaled)", "PASS" if ok else "WARN",
                      f"training-gain range [{kp_lo:.0f}, {kp_hi:.0f}]")
        if kd_lo is not None:
            ok = kd_lo >= 1.5 and kd_hi <= 9
            rep.check("kd in [2,8] (un-scaled)", "PASS" if ok else "WARN",
                      f"training-gain range [{kd_lo:.1f}, {kd_hi:.1f}]")
    else:
        rep.check("kp/kd checked", "INFO", "no MOTION rows in this trace")

    # 3. waist_roll clamp frequency on MOTION rows (post-clamp des pinned at ±limit)
    lim = WAIST_ROLL_LIMIT
    des = fcol(motion, "des_waist_roll_joint")
    if des:
        railed = sum(1 for v in des if abs(abs(v) - lim) < 1e-4 or abs(v) > lim)
        frac = railed / len(des)
        meas = fcol(motion, "q_waist_roll_joint")
        detail = f"{100*frac:.0f}% of MOTION ticks at the ±{lim:.3f} rail (clamp firing)"
        if meas:
            detail += f"; measured range [{min(meas):+.3f},{max(meas):+.3f}]"
        rep.check("waist_roll not rail-bound", "PASS" if frac < 0.10 else "WARN", detail)

    # full-body (legs + waist) command audit for the hoisted full-body verification
    analyze_trace_fullbody(rep, rows, joints)
    return rep


def analyze(label, path):
    rows, cols = load(path)
    if not rows:
        rep = Report(label, path)
        rep.check("non-empty", "FAIL", "0 rows")
        return rep
    if any(c.startswith("obs_") for c in cols):
        return analyze_obs(label, path, rows, cols)
    if any(c.startswith("des_") for c in cols):
        return analyze_trace(label, path, rows, cols)
    rep = Report(label, path)
    rep.check("recognized csv", "FAIL", "neither obs_ nor des_ columns present")
    return rep


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return
    reps = []
    for a in args:
        label, path = (a.split("=", 1) if "=" in a else ("log", a))
        rep = analyze(label, path)
        text = rep.render()
        print("\n" + text)
        report_path = path + ".report.txt"
        try:
            with open(report_path, "w") as f:
                f.write(text + "\n")
            print(f"  (report saved: {report_path})")
        except OSError as e:
            print(f"  (could not save report: {e})")
        reps.append(rep)
    print("\n==================== SUMMARY ====================")
    for rep in reps:
        print(f"  {rep.overall():4}  {rep.label:6} {rep.path}")
    print("\nNotes:")
    print("  * obs CSV: racket_target_pos_b x ~ +0.4 (front) confirms the yaw fix; "
          "anchor_pos_b~0 confirms perfect_tracking.")
    print("  * A/B/C: fabricated (A) buzzes (high |Δaction|, large anchor); oracle (C) clean "
          "proves the policy; perfect (B) ~ C is the hardware-safe approximation.")
    print("  * kp/kd come from the TRACE csv / first-tick dump (not the obs csv).")
    print("  * full-body: a HOIST cannot prove balance. 'legs driven' PASS + bounded qd + no")
    print("    rail = full-body COMMANDS are safe; poor leg tracking is EXPECTED (WARN) on a")
    print("    hoist. True balance needs a later ground-contact test (rope, low gain, level 0).")
    print("  * ground-contact: also watch 'upright (gravz~-1)', 'hip tracking', and capture")
    print("    hygiene (don't mix level 0/1 or gains in one log). halts/foot-slip/rope-load are")
    print("    NOT in the CSV -> read halts from [status]; foot-slip + rope-load are visual.")
    # non-zero exit if any file FAILed (handy in scripts/CI)
    sys.exit(1 if any(r.overall() == "FAIL" for r in reps) else 0)


if __name__ == "__main__":
    main()
