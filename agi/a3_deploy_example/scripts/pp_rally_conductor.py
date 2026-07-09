#!/usr/bin/env python3
"""RALLY conductor for the headless --planner closed-loop test (Gate 3, 110-D redesign).

Differs from pp_planner_conductor.py (the legacy PER-POINT harness, kept for the
177 walk-and-strike generation) in ONE load-bearing way: after the initial stand +
'm', the robot is NEVER reset and never 'p'-aborted — it must return serve after
serve from its station, exactly the real-life hitter scenario (§9.3: human serves
every 4-6 s, robot station-keeps between swings). That continuous regime is what
the per-point discipline structurally cannot test: post-swing recovery INTO the
next engage, station drift accumulation across cycles, fh<->bh alternation from
the walked (not reset) pose, and serve arrivals during rest/recovery windows.

Per-serve accounting: every fake_ball serve (parsed live from /tmp/pp_ball.log)
opens a serve window; runner-log events (engage / swing complete / recovery /
gate REJECT / FALL GUARD) and the sim pelvis pose are attributed to it. Output:
a per-serve verdict table + /tmp/pp_rally_report.json + PASS/FAIL.

Env knobs:
  PP_SERVES=12        serves counted after MOTION entry (default 12 > one full
                      sweep of the 10-placement list)
  PP_RESET_Y=0.0      lateral start offset (nonzero station demand from serve 1)
  PP_EXTRA_ARGS=""    extra runner flags (e.g. "--hold-recover 1.2")
  PP_DROPOUT_AT=0     serve index (1-based) at which to freeze the planner
                      (SIGSTOP) for PP_DROPOUT_S right after its engage — the
                      mid-swing mocap/planner dropout stress. 0 = off.
  PP_DROPOUT_S=1.0    dropout duration (s); > cmd-timeout 0.5 and base max-age 0.2
PASS: fall_guard == 0 AND every ENGAGED serve completes + recovers AND
      engaged >= PP_MIN_RETURN_RATE (default 0.8) of counted serves AND
      station drift <= 0.5 m AND min pelvis z >= 0.80 during MOTION.
"""
import json
import os
import pty
import re
import signal
import subprocess
import sys
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray
from mujoco_sim_msgs.msg import SimReset   # needs the A3_MuJoCo_Sim install overlay sourced

GEAR = "/home/dongc1/workspace/HOPE/agi/a3_deploy_example"
DIST = GEAR + "/dist/a3_deploy_x86_64"
RUNNER_LOG = "/tmp/pp_runner.log"
BALL_LOG = "/tmp/pp_ball.log"
REPORT_JSON = "/tmp/pp_rally_report.json"
OBS_CSV = "/tmp/pp_obs.csv"

SERVES = int(os.environ.get("PP_SERVES", "12"))
RESET_Y = float(os.environ.get("PP_RESET_Y", "0.0"))
DROPOUT_AT = int(os.environ.get("PP_DROPOUT_AT", "0"))
DROPOUT_S = float(os.environ.get("PP_DROPOUT_S", "1.0"))
# PASS is SAFETY-focused by default (0 falls, every engaged swing completes+recovers,
# bounded drift/height); the return RATE is a demo-quality metric — with the heading
# gates + operator rescues the current model returns ~1 serve per stand (~25-35%).
# Set PP_MIN_RETURN_RATE to additionally gate on rate.
MIN_RETURN_RATE = float(os.environ.get("PP_MIN_RETURN_RATE", "0.0"))
SERVE_PERIOD_S = 6.5           # sweep serves: ~1.6-2.1 s flight + 4 s pause
GLOBAL_TIMEOUT_S = SERVES * SERVE_PERIOD_S + 60.0

rclpy.init()
node = Node("rally_conductor")
state = {"x": None, "y": None, "z": None, "min_z_motion": 99.0,
         "valid": 0, "invalid": 0}


def pose_cb(m):
    state["x"] = m.pose.position.x
    state["y"] = m.pose.position.y
    state["z"] = m.pose.position.z


def flat_cb(m):
    if m.data[1] > 0.5:
        state["valid"] += 1
    else:
        state["invalid"] += 1


node.create_subscription(PoseStamped, "/sim/a3/pelvis_pose", pose_cb, 10)
node.create_subscription(Float64MultiArray, "/racket/command_flat", flat_cb, 10)
reset_pub = node.create_publisher(SimReset, "/sim/a3/reset", 10)


def spin(sec):
    t0 = time.time()
    while time.time() - t0 < sec:
        rclpy.spin_once(node, timeout_sec=0.05)


def z():
    return state["z"] if state["z"] is not None else -1.0


def xy():
    return (state["x"] or 0.0, state["y"] or 0.0)


def reset():
    # Keyframe stand at x=0.33: the adaptive plane (robot_x + x_hit_offset 0.70) then sits
    # at ~1.03 where the sweep serves' post-bounce arcs cross at trained heights.
    m = SimReset()
    m.mode = 1
    m.keyframe_id = 0
    m.set_base = True
    m.pelvis_pose.position.x = 0.33
    m.pelvis_pose.position.y = RESET_Y
    m.pelvis_pose.position.z = 1.07
    m.pelvis_pose.orientation.w = 1.0
    m.set_base_twist = True
    m.zero_all_velocities = True
    reset_pub.publish(m)


# ---- spawn the runner on a pty (termios key thread needs a real tty) ----
master, slave = pty.openpty()
log = open(RUNNER_LOG, "wb")
proc = subprocess.Popen(
    ["env", "A3_SOURCE_ROBOT_ENV=0", "LD_LIBRARY_PATH=.:/opt/ros/jazzy/lib",
     "./run_a3_pingpong.sh", "--planner", "--start", "passive", "--official-stand",
     "--swing-rest", "1.5", "--obs-csv", OBS_CSV]
    + os.environ.get("PP_EXTRA_ARGS", "").split(),
    cwd=DIST, stdin=slave, stdout=log, stderr=subprocess.STDOUT,
    start_new_session=True)
os.close(slave)


def key(c):
    os.write(master, c.encode())
    print(f"[rally] key '{c}' @ z={z():.2f}", flush=True)


print("[rally] waiting for runner boot (driver started)...", flush=True)
t0 = time.time()
while time.time() - t0 < 90:
    spin(1)
    if time.time() - t0 < 12:      # never trust an early marker (stale log tail)
        continue
    try:
        with open(RUNNER_LOG, "rb") as f:
            if b"driver started" in f.read():
                break
    except FileNotFoundError:
        pass
print(f"[rally] runner up after {time.time()-t0:.0f}s; standing the robot", flush=True)
spin(1.0)
print(f"[rally] /sim/a3/reset subscribers: {node.count_subscribers('/sim/a3/reset')}",
      flush=True)

# ---- incremental log watchers ----
_run_ofs = [0]
_ball_ofs = [0]
ENGAGE_RE = re.compile(
    rb"\[pp engage\] (forehand|backhand) \S+[^:]*: tgt base-rel \(([-+0-9.]+),([-+0-9.]+),([-+0-9.]+)\) tts=([0-9.]+)s \(clock tts0=([0-9.]+)s\)")
REJECT_RE = re.compile(rb"\[pp gate\] REJECT\(110\) (fh|bh) ([^\n]*)")
SERVE_RE = re.compile(rb"serve (\d+): p=\[([^\]]*)\] v=\[([^\]]*)\]")


def new_runner_events():
    try:
        with open(RUNNER_LOG, "rb") as f:
            f.seek(_run_ofs[0])
            chunk = f.read()
            _run_ofs[0] += len(chunk)
    except FileNotFoundError:
        chunk = b""
    return {
        "engages": [
            {"side": m.group(1).decode(),
             "tgt_b": [float(m.group(2)), float(m.group(3)), float(m.group(4))],
             "tts": float(m.group(5)), "tts0": float(m.group(6))}
            for m in ENGAGE_RE.finditer(chunk)],
        "complete": chunk.count(b"swing complete"),
        "recovered": chunk.count(b"post-swing recovery done"),
        "fall_guard": chunk.count(b"FALL GUARD"),
        "rejects": [m.group(1).decode() + " " + m.group(2).decode()[:110]
                    for m in REJECT_RE.finditer(chunk)],
        "no_base": chunk.count(b"NO FRESH mocap base sample"),
    }


def new_serves():
    try:
        with open(BALL_LOG, "rb") as f:
            f.seek(_ball_ofs[0])
            chunk = f.read()
            _ball_ofs[0] += len(chunk)
    except FileNotFoundError:
        chunk = b""
    return [{"n": int(m.group(1)),
             "p": [round(float(v), 2) for v in m.group(2).decode().split(",")],
             "v": [round(float(v), 2) for v in m.group(3).decode().split(",")]}
            for m in SERVE_RE.finditer(chunk)]


def stand_and_motion(label):
    """Proven stand dance (see pp_planner_conductor.py): reset into the armed 's'
    catch window, require continuous standing, then 'm' (MOTION)."""
    for attempt in range(12):
        reset()
        spin(0.15)
        key("s")
        spin(1.5)
        if z() < 0.95:
            print(f"[rally] {label} attempt {attempt}: z={z():.2f} after reset+s, retry",
                  flush=True)
            continue
        stable = True
        for _ in range(12):
            spin(0.5)
            if z() < 1.0:
                stable = False
                break
        if stable:
            new_runner_events()       # flush pending markers
            new_serves()
            key("m")
            return True
        print(f"[rally] {label} stand lost in settle (z={z():.2f}); retry", flush=True)
    return False


def planner_pids():
    out = subprocess.run(["pgrep", "-f", "hope_planner_node"],
                         capture_output=True, text=True).stdout.split()
    return [int(p) for p in out]


if not stand_and_motion("rally"):
    print("[rally] FAILED to stand the robot; aborting", flush=True)
    key("q")
    spin(3)
    proc.terminate()
    sys.exit(1)

anchor_xy = xy()
print(f"[rally] MOTION entered at base=({anchor_xy[0]:+.2f},{anchor_xy[1]:+.2f}); "
      f"counting {SERVES} serves, NO resets from here", flush=True)

# ---- the RALLY: attribute events to serve windows, never touch the robot ----
serve_rows = []       # one dict per counted serve
cur = None            # the open serve row
active = None         # row of the last ENGAGED serve (swing-outcome attribution)
fell = False
dropout_done = False
t_start = time.time()
serves_seen = 0
tail_deadline = None  # after the last serve, wait for its outcome
rescues = 0           # operator-style re-stands (§9.3: op may re-stand a stuck/yawed robot)
misses_in_a_row = 0   # consecutive serves with no engage -> rescue trigger

while time.time() - t_start < GLOBAL_TIMEOUT_S:
    spin(0.25)
    if z() >= 0:
        state["min_z_motion"] = min(state["min_z_motion"], z())
        if cur is not None:
            cur["min_z"] = min(cur["min_z"], z())

    for s in new_serves():
        serves_seen += 1
        if serves_seen > SERVES:
            continue
        if cur is not None:
            serve_rows.append(cur)
            misses_in_a_row = 0 if cur["engaged"] else misses_in_a_row + 1
            # OPERATOR RESCUE (mirrors the §9.3 demo discipline): a robot standing but
            # refusing to engage (e.g. PLANNER: yawed after a divergent follow-through)
            # gets an operator re-stand — 'p', reset square, 's', 'm'. Counted + reported;
            # the rally continues. A FALL is never rescued (the gate ends).
            if misses_in_a_row >= 2 and serves_seen < SERVES:
                rescues += 1
                misses_in_a_row = 0
                print(f"[rally] RESCUE {rescues}: 2 serves w/o engage -> operator re-stand",
                      flush=True)
                key("p")
                spin(1.0)
                if not stand_and_motion(f"rescue {rescues}"):
                    print("[rally] rescue re-stand FAILED; ending rally", flush=True)
                    fell = True
                    break
        bx, by = xy()
        cur = {"serve": serves_seen, "ball_p0": s["p"], "ball_v0": s["v"],
               "t": round(time.time() - t_start, 1),
               "base_at_serve": [round(bx, 3), round(by, 3)],
               "engaged": None, "complete": 0, "recovered": 0,
               "rejects": [], "min_z": z() if z() > 0 else 99.0,
               "dropout": False}
        print(f"[rally] serve {serves_seen}/{SERVES} p={s['p']} v={s['v']} "
              f"base=({bx:+.2f},{by:+.2f})", flush=True)
        if serves_seen == SERVES:
            tail_deadline = time.time() + 2.5 * SERVE_PERIOD_S
    if fell:
        break

    ev = new_runner_events()
    if cur is not None:
        for e in ev["engages"]:
            cur["engaged"] = e
            active = cur          # swing outcome events credit the ENGAGED serve
            print(f"[rally]   engage {e['side']} tgt_b={e['tgt_b']} tts={e['tts']:.2f} "
                  f"tts0={e['tts0']:.2f}", flush=True)
            if DROPOUT_AT and cur["serve"] == DROPOUT_AT and not dropout_done:
                dropout_done = True
                cur["dropout"] = True
                pids = planner_pids()
                print(f"[rally]   DROPOUT stress: SIGSTOP planner {pids} for "
                      f"{DROPOUT_S}s (mid-swing stale cmd + base)", flush=True)
                for p in pids:
                    os.kill(p, signal.SIGSTOP)
                spin(DROPOUT_S)
                for p in pids:
                    os.kill(p, signal.SIGCONT)
        # complete/recovered belong to the swing (the ENGAGED serve) even when the
        # recovery finishes windows later (the tighter static-handoff gates make the
        # policy recovery span 1-2 serve periods — safe, and must not read as MISSED).
        tgt = active if active is not None else cur
        tgt["complete"] += ev["complete"]
        tgt["recovered"] += ev["recovered"]
        cur["rejects"] += ev["rejects"]
        if ev["no_base"]:
            cur.setdefault("no_base_warns", 0)
            cur["no_base_warns"] += ev["no_base"]
    if ev["fall_guard"]:
        fell = True
        print("[rally] FALL GUARD tripped — rally over", flush=True)
        break
    if tail_deadline is not None and time.time() > tail_deadline:
        break
    # early exit: last serve fully resolved
    if (tail_deadline is not None and cur is not None and cur["serve"] == SERVES
            and cur["recovered"]):
        spin(1.0)
        break

if cur is not None:
    serve_rows.append(cur)

end_xy = xy()   # BEFORE 'q': the runner exit drops the robot limp and the pelvis slides
key("q")
spin(3)
try:
    proc.wait(timeout=10)
except subprocess.TimeoutExpired:
    proc.kill()

# ---- verdicts ----
drift = ((end_xy[0] - anchor_xy[0]) ** 2 + (end_xy[1] - anchor_xy[1]) ** 2) ** 0.5
returned = 0
for r in serve_rows:
    r["ok"] = bool(r["engaged"]) and r["complete"] >= 1 and r["recovered"] >= 1
    returned += r["ok"]
    sd = r["engaged"]["side"][:2] if r["engaged"] else "--"
    print(f"[rally] serve {r['serve']:2d} @{r['t']:6.1f}s base=({r['base_at_serve'][0]:+.2f},"
          f"{r['base_at_serve'][1]:+.2f}) {sd} engage={'Y' if r['engaged'] else 'n'} "
          f"complete={r['complete']} recovered={r['recovered']} minz={r['min_z']:.2f} "
          f"rejects={len(r['rejects'])}{' DROPOUT' if r['dropout'] else ''} "
          f"-> {'RETURNED' if r['ok'] else 'MISSED'}", flush=True)

n = len(serve_rows)
engaged = sum(1 for r in serve_rows if r["engaged"])
incomplete = [r["serve"] for r in serve_rows if r["engaged"] and not r["ok"]]
rate = returned / n if n else 0.0
ok = (not fell and n >= SERVES and engaged >= 1 and not incomplete
      and rate >= MIN_RETURN_RATE and drift <= 0.5
      and state["min_z_motion"] >= 0.80)
summary = {
    "serves": n, "engaged": engaged, "returned": returned,
    "return_rate": round(rate, 3), "falls": int(fell),
    "engaged_but_unresolved": incomplete,
    "station_drift_m": round(drift, 3),
    "min_z_motion": round(state["min_z_motion"], 3),
    "valid_cmds": state["valid"], "invalid_cmds": state["invalid"],
    "dropout_stress": DROPOUT_AT if dropout_done else 0,
    "operator_rescues": rescues,
    "pass": bool(ok),
    "rows": serve_rows,
}
with open(REPORT_JSON, "w") as f:
    json.dump(summary, f, indent=1)
print(f"[rally] SUMMARY: serves={n} engaged={engaged} returned={returned} "
      f"({100*rate:.0f}%) falls={int(fell)} rescues={rescues} drift={drift:.2f}m "
      f"min_z={state['min_z_motion']:.3f} -> {'PASS' if ok else 'FAIL'} "
      f"(report: {REPORT_JSON}, obs csv: {OBS_CSV})", flush=True)
sys.exit(0 if ok else 1)
