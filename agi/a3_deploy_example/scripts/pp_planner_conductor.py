#!/usr/bin/env python3
"""State-driven conductor for the headless --planner closed-loop test.

Owns the runner on a real pty (so its termios key thread works), watches the sim
pelvis pose, and only presses 's'/'m' when the robot is VERIFIED standing —
retrying resets as needed. Open-loop sleep schedules kept missing the stand
window; this closes the loop.
"""
import os
import pty
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

rclpy.init()
node = Node("conductor")
state = {"z": None, "t": 0.0, "valid": 0, "invalid": 0, "min_z_motion": 99.0}


def pose_cb(m):
    state["z"] = m.pose.position.z
    state["t"] = time.time()


def flat_cb(m):
    if m.data[1] > 0.5:
        state["valid"] += 1
    else:
        state["invalid"] += 1


node.create_subscription(PoseStamped, "/sim/a3/pelvis_pose", pose_cb, 10)
node.create_subscription(Float64MultiArray, "/racket/command_flat", flat_cb, 10)
# Persistent reset publisher: per-call `ros2 topic pub --once` discovery proved flaky
# after repeated harness restarts (resets silently vanished; robot never stood).
reset_pub = node.create_publisher(SimReset, "/sim/a3/reset", 10)


def spin(sec):
    t0 = time.time()
    while time.time() - t0 < sec:
        rclpy.spin_once(node, timeout_sec=0.05)


def z():
    return state["z"] if state["z"] is not None else -1.0


def reset():
    # Keyframe stand + base placed at x=0.33: the adaptive x_hit (robot_x + 0.67) then sits
    # at 1.0 where the probed serve's post-bounce arc peaks at z=0.72 — inside the forehand
    # training box. (175-D obs has no world base position; placement is free.)
    m = SimReset()
    m.mode = 1
    m.keyframe_id = 0
    m.set_base = True
    m.pelvis_pose.position.x = 0.33
    m.pelvis_pose.position.y = 0.0
    m.pelvis_pose.position.z = 1.07
    m.pelvis_pose.orientation.w = 1.0
    m.set_base_twist = True
    m.zero_all_velocities = True
    reset_pub.publish(m)


# ---- spawn the runner on a pty ----
master, slave = pty.openpty()
log = open(RUNNER_LOG, "wb")
proc = subprocess.Popen(
    ["env", "A3_SOURCE_ROBOT_ENV=0", "LD_LIBRARY_PATH=.:/opt/ros/jazzy/lib",
     "./run_a3_pingpong.sh", "--planner", "--start", "passive", "--official-stand", "--swing-rest", "1.5"],
    cwd=DIST, stdin=slave, stdout=log, stderr=subprocess.STDOUT,
    start_new_session=True)
os.close(slave)


def key(c):
    os.write(master, c.encode())
    print(f"[conductor] key '{c}' @ z={z():.2f}", flush=True)


print("[conductor] waiting for runner boot (driver started)...", flush=True)
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
print(f"[conductor] runner up after {time.time()-t0:.0f}s; standing the robot", flush=True)
# confirm the sim actually subscribes our reset topic (discovery sanity)
spin(1.0)
print(f"[conductor] /sim/a3/reset subscribers: {node.count_subscribers('/sim/a3/reset')}",
      flush=True)

# ---- incremental runner-log watcher (per-point event detection) ----
_log_ofs = [0]


def new_log_events():
    """Return counts of interesting markers in log bytes appended since last call."""
    try:
        with open(RUNNER_LOG, "rb") as f:
            f.seek(_log_ofs[0])
            chunk = f.read()
            _log_ofs[0] += len(chunk)
    except FileNotFoundError:
        chunk = b""
    return {
        "engage": chunk.count(b"[pp engage]"),
        "complete": chunk.count(b"swing complete"),
        "recovered": chunk.count(b"post-swing recovery done"),
        "fall_guard": chunk.count(b"FALL GUARD"),
    }


def stand_and_motion(label):
    """STAND dance: per attempt = reset (teleport to the stand keyframe, limp) then
    IMMEDIATELY 's' (official gains catch it inside the ~0.5 s buckle window). Arming
    's' on a LYING robot instead trips the fall guard -> PASSIVE -> every later reset
    collapses limp; the next attempt's 's' re-arms PD_STAND. Ends with 'm' (MOTION)."""
    for attempt in range(12):
        reset()
        spin(0.15)
        key("s")
        spin(1.5)
        if z() < 0.95:
            print(f"[conductor] {label} attempt {attempt}: z={z():.2f} after reset+s, retry",
                  flush=True)
            continue
        # settle under PD_STAND, require CONTINUOUS standing (the gain-catch leaves
        # residual sway; 'm' too early -> the policy hold amplifies it and falls)
        stable = True
        for _ in range(12):
            spin(0.5)
            if z() < 1.0:
                stable = False
                break
        if stable:
            new_log_events()          # flush pending markers before the point starts
            key("m")                  # MOTION: planner-driven engage from here on
            return True
        print(f"[conductor] {label} stand lost in settle (z={z():.2f}); retry", flush=True)
    return False


if not stand_and_motion("point 1"):
    print("[conductor] FAILED to stand the robot; aborting", flush=True)
    key("q")
    spin(3)
    proc.terminate()
    sys.exit(1)

# ---- RUN phase: PER-POINT loop, mirroring the §9.3 hardware demo discipline ----
# The model holds ONE clean return per placement (known margin: the post-lunge static
# stand tips over after ~5-7 s, and the walked-forward robot puts later serves out of
# reach). So the honest closed-loop test = the demo loop: serve -> return -> recovery
# -> 'p' (operator abort / point over) -> reset to the start spot -> 's' -> 'm' ->
# next point. A fall BEFORE the deliberate 'p' is a real failure; the limp collapse
# AFTER 'p' is the operator catch, not a fall.
POINTS = 3
POINT_TIMEOUT_S = 22.0   # serves every 4 s; engage typically within ~8 s of 'm'
results = []
for point in range(1, POINTS + 1):
    t0 = time.time()
    got = {"engage": 0, "complete": 0, "recovered": 0, "fall_guard": 0}
    while time.time() - t0 < POINT_TIMEOUT_S:
        spin(0.3)
        if z() >= 0:
            state["min_z_motion"] = min(state["min_z_motion"], z())
        ev = new_log_events()
        for k in got:
            got[k] += ev[k]
        if got["fall_guard"]:
            break                      # fell during the point = real failure
        if got["recovered"]:
            break                      # clean return + balanced out = point over
    ok = bool(got["recovered"]) and not got["fall_guard"]
    results.append({"point": point, **got, "ok": ok,
                    "dur_s": round(time.time() - t0, 1)})
    print(f"[conductor] point {point}: engage={got['engage']} complete={got['complete']} "
          f"recovered={got['recovered']} fall_guard={got['fall_guard']} "
          f"-> {'OK' if ok else 'FAIL'}", flush=True)
    key("p")                           # point over: operator abort (limp; sim 'catch')
    spin(1.0)
    if point < POINTS:
        if not stand_and_motion(f"point {point + 1}"):
            print("[conductor] FAILED to re-stand between points; aborting", flush=True)
            break

key("q")
spin(3)
try:
    proc.wait(timeout=10)
except subprocess.TimeoutExpired:
    proc.kill()

n_ok = sum(1 for r in results if r["ok"])
print(f"[conductor] SUMMARY: points_ok={n_ok}/{len(results)} results={results} "
      f"min_z_motion={state['min_z_motion']:.3f} "
      f"valid_cmds={state['valid']} invalid_cmds={state['invalid']} "
      f"-> {'PASS' if n_ok == POINTS else 'FAIL'}", flush=True)
