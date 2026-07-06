#!/usr/bin/env python3
"""Gate 2.5 — automated TRANSITION-MATRIX test for the scripted C++ runner.

Gate 2 (pp_freebase_watch) only automates: stand -> level-1 swing -> hold; everything
else is human-keyed. This drives the full transition matrix deterministically:

  P1  stand -> MOTION at level 0        (m0 entry — the --planner entry path)
  P2  LONG level-0 hold (20 s)          (m0 endurance — never tested before)
  P3  m0 -> m1 ('1'), fh swing, complete, post-swing hold 8 s
  P4  dir switch AT HOLD ('b'), then m0 -> m1: bh swing from a walked position
  P5  mid-swing dir switch ('f' during the bh swing — must QUEUE, not snap)
  P6  post-queue swing (fh applied from the latch), complete
  P7  two more alternating cycles with short (3 s) holds

Each phase reports PASS/FAIL (fall guard or pelvis z < 0.5 = fail) so a fragile
transition is IDENTIFIED, not averaged away. Run under perfect_tracking (default)
or oracle (--oracle; start scripts/run_oracle.sh separately... the harness does it).
"""
import os
import pty
import subprocess
import sys
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from mujoco_sim_msgs.msg import SimReset

GEAR = "/home/dongc1/workspace/HOPE/agi/a3_deploy_example"
DIST = GEAR + "/dist/a3_deploy_x86_64"
RUNNER_LOG = "/tmp/pp_runner_g25.log"
ORACLE = "--oracle" in sys.argv

rclpy.init()
node = Node("gate25")
state = {"z": None}
node.create_subscription(PoseStamped, "/sim/a3/pelvis_pose",
                         lambda m: state.update(z=m.pose.position.z), 10)
reset_pub = node.create_publisher(SimReset, "/sim/a3/reset", 10)


def spin(sec):
    t0 = time.time()
    while time.time() - t0 < sec:
        rclpy.spin_once(node, timeout_sec=0.05)


def z():
    return state["z"] if state["z"] is not None else -1.0


def reset():
    m = SimReset()
    m.mode = 1
    m.keyframe_id = 0
    m.set_base = True
    m.pelvis_pose.position.z = 1.07
    m.pelvis_pose.orientation.w = 1.0
    m.set_base_twist = True
    m.zero_all_velocities = True
    reset_pub.publish(m)


def log_bytes():
    try:
        with open(RUNNER_LOG, "rb") as f:
            return f.read()
    except FileNotFoundError:
        return b""


def wait_marker(marker, timeout, offset):
    """Wait for marker to appear in the log AFTER byte `offset`. Returns (found, new_len)."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        spin(0.3)
        buf = log_bytes()
        if marker.encode() in buf[offset:]:
            return True, len(buf)
    return False, len(log_bytes())


results = []


def phase(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"[g25] {'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)


def fell():
    return z() < 0.5 or b"FALL GUARD" in log_bytes()[guard_base:]


# ---- runner on a pty (SCRIPTED mode — no planner; level 0 entry) ----
args = ["env", "A3_SOURCE_ROBOT_ENV=0", "LD_LIBRARY_PATH=.:/opt/ros/jazzy/lib",
        "./run_a3_pingpong.sh", "--start", "passive", "--official-stand",
        "--single-swing", "--level", "0"]
if ORACLE:
    args.append("--oracle-pelvis")
master, slave = pty.openpty()
log = open(RUNNER_LOG, "wb")
proc = subprocess.Popen(args, cwd=DIST, stdin=slave, stdout=log,
                        stderr=subprocess.STDOUT, start_new_session=True)
os.close(slave)


def key(c):
    os.write(master, c.encode())
    print(f"[g25] key '{c}' @ z={z():.2f}", flush=True)


print(f"[g25] loc mode: {'ORACLE (real base feedback)' if ORACLE else 'perfect_tracking'}",
      flush=True)
t0 = time.time()
while time.time() - t0 < 90:
    spin(1)
    if time.time() - t0 >= 12 and b"driver started" in log_bytes():
        break
print(f"[g25] runner up after {time.time()-t0:.0f}s", flush=True)
spin(1)

# stand: reset-into-armed-PD_STAND (proven pattern)
stood = False
for _ in range(12):
    reset()
    spin(0.15)
    key("s")
    spin(1.5)
    if z() < 0.95:
        continue
    stable = all((spin(0.5), z() > 1.0)[1] for _ in range(10))
    if stable:
        stood = True
        break
if not stood:
    print("[g25] could not stand the robot; abort", flush=True)
    key("q")
    sys.exit(1)

guard_base = len(log_bytes())   # falls counted only after MOTION entry

# P1: stand -> MOTION at level 0
key("m")
spin(2.0)
phase("P1 stand->m0 entry", not fell(), f"z={z():.2f}")

# P2: LONG level-0 hold
t0 = time.time()
ok = True
while time.time() - t0 < 20:
    spin(0.5)
    if fell():
        ok = False
        break
phase("P2 m0 hold 20s", ok, f"z={z():.2f} t={time.time()-t0:.0f}s")

# P3: m0 -> m1 fh swing -> complete -> post-swing hold 8s
if ok:
    off = len(log_bytes())
    key("1")
    done, off = wait_marker("swing complete", 8.0, off)
    swing_ok = done and not fell()
    phase("P3a m0->m1 fh swing+complete", swing_ok, f"z={z():.2f}")
    if swing_ok:
        t0 = time.time()
        hold_ok = True
        while time.time() - t0 < 8:
            spin(0.5)
            if fell():
                hold_ok = False
                break
        phase("P3b post-swing hold 8s", hold_ok, f"z={z():.2f}")
        ok = hold_ok

# P4: dir switch AT HOLD -> bh swing from the walked position
if ok:
    key("b")
    spin(2.0)
    sw_ok = not fell()
    phase("P4a dir switch at hold (f->b)", sw_ok, f"z={z():.2f}")
    if sw_ok:
        off = len(log_bytes())
        key("1")
        done, off = wait_marker("swing complete", 8.0, off)
        ok = done and not fell()
        phase("P4b bh swing from walked pos", ok, f"z={z():.2f}")

# P5+P6: mid-swing dir switch must QUEUE; queued dir applies; next swing = fh
if ok:
    spin(3.0)                      # settle in hold
    if not fell():
        key("1")                   # bh swing (dir still b)
        spin(0.6)                  # 0.6s INTO the swing:
        off_f = len(log_bytes())   # scan baseline: BEFORE the 'f' key (see P6 note)
        key("f")                   # mid-swing switch -> must QUEUE
        queued, _ = wait_marker("QUEUED", 2.0, off_f)
        done, off = wait_marker("swing complete", 8.0, off_f)
        p5_ok = queued and done and not fell()
        phase("P5 mid-swing f-switch queued + swing completes", p5_ok,
              f"queued={queued} complete={done} z={z():.2f}")
        if p5_ok:
            # The latch may LEGALLY apply BEFORE "swing complete": if 'f' lands while the
            # swing clock still sits at the windup clamp, the apply condition
            # (last_tts_at_windup_) fires immediately — QUEUED and applied print 2 lines
            # apart, then the (now-fh) swing completes (2026-07-06 g25 oracle log). Scan
            # from the pre-'f' offset: wait_marker returns the CURRENT log end, which by
            # then is already past the early apply line, so any later baseline misses it.
            applied, _ = wait_marker("queued swing dir applied", 4.0, off_f)
            spin(2.0)
            off = len(log_bytes())
            key("1")               # fh swing via the applied latch
            done, off = wait_marker("swing complete", 8.0, off)
            ok = applied and done and not fell()
            phase("P6 latched fh swing", ok, f"applied={applied} z={z():.2f}")
        else:
            ok = False

# P7: two more alternating cycles, short holds
if ok:
    for i, d in enumerate(["b", "f"]):
        spin(3.0)
        key(d)
        spin(1.5)
        off = len(log_bytes())
        key("1")
        done, off = wait_marker("swing complete", 8.0, off)
        c_ok = done and not fell()
        phase(f"P7.{i+1} cycle {d}", c_ok, f"z={z():.2f}")
        if not c_ok:
            ok = False
            break

key("q")
spin(3)
try:
    proc.wait(timeout=10)
except subprocess.TimeoutExpired:
    proc.kill()

print("\n[g25] ================== MATRIX ==================", flush=True)
for name, okk, detail in results:
    print(f"[g25]  {'PASS' if okk else 'FAIL'}  {name}  ({detail})", flush=True)
n_fail = sum(1 for _, okk, _ in results if not okk)
print(f"[g25] {len(results)-n_fail}/{len(results)} phases passed "
      f"({'ORACLE' if ORACLE else 'perfect_tracking'})", flush=True)
