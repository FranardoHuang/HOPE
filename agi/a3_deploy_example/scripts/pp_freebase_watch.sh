#!/usr/bin/env bash
# One-command free-base MuJoCo test WITH viewer + interactive runner.
#
# Why this exists: the manual 3-terminal flow fails because reset_sim.sh must be
# fired ONLY during the runner's PD_STAND warmup (it loads the stand keyframe and
# zeroes velocity). Firing it during MOTION snaps the robot back to the stand pose
# every time -> the swing never develops ("robot won't swing"). This script fires
# the reset automatically at +3/+5/+7 s (inside a 9 s warmup) and then leaves the
# sim alone, so MOTION runs uninterrupted.
#
# Run:  distrobox enter hope -- bash /home/dongc1/workspace/HOPE/agi/a3_deploy_example/scripts/pp_freebase_watch.sh
#   (no display? prefix MUJOCO_GL=egl, but then there's no window to watch)
# Keys once it swings: 1=swing / 0=hold, f=forehand, b=backhand, ,/.=slower/faster, q=quit.
# Do NOT run reset_sim.sh by hand after warmup — it will stop the swing.
set -o pipefail
GEAR=/home/dongc1/workspace/HOPE/agi/a3_deploy_example
DIST="$GEAR/dist/a3_deploy_x86_64"
WARMUP="${WARMUP:-9}"
set +u; source /opt/ros/jazzy/setup.bash 2>/dev/null || true; set +u

echo "[watch] cleanup stale sim/runner"
pkill -9 -f "aimrt_main.*a3_pingpong" 2>/dev/null || true
pkill -9 -f "a3_deploy_onnx_ref_pingpong" 2>/dev/null || true
pkill -9 -x iox-roudi 2>/dev/null || true
rm -f /dev/shm/iox1_0_* 2>/dev/null || true
sleep 1

echo "[watch] starting sim (viewer) ..."
cd "$GEAR"
setsid bash -c "source /opt/ros/jazzy/setup.bash 2>/dev/null; A3_SIM_FLAVOR=auto A3_SIM_CFG=a3_pingpong_iceoryx_cfg.yaml ./scripts/run_sim.sh" >/tmp/pp_sim.log 2>&1 &
for i in $(seq 1 40); do
  grep -qiE "will wait for shutdown|Sim Start|gui" /tmp/pp_sim.log 2>/dev/null && break
  sleep 1
done
echo "[watch] sim up (log: /tmp/pp_sim.log)"

# Auto-reset ONLY during warmup: load the stand keyframe + zero velocity at +3/+5/+7 s,
# then stop. This is the piece that is easy to get wrong by hand.
( sleep 3; "$GEAR/scripts/reset_sim.sh"
  sleep 2; "$GEAR/scripts/reset_sim.sh"
  sleep 2; "$GEAR/scripts/reset_sim.sh"
  echo "[watch] warmup resets done; leaving sim alone so MOTION can swing" ) >/tmp/pp_reset.log 2>&1 &

echo "[watch] starting runner (interactive). It holds PD_STAND for ${WARMUP}s, then swings."
echo "[watch] watch the viewer; keys: 1/0 swing/hold, f/b forehand/backhand, q quit."
cd "$DIST"
A3_SOURCE_ROBOT_ENV=0 A3_TRANSPORT=iceoryx LD_LIBRARY_PATH=".:/opt/ros/jazzy/lib" \
  ./run_a3_pingpong.sh --start motion --level 1 --official-stand --auto-leg-hold --warmup-sec "$WARMUP"

echo "[watch] runner exited; cleaning up sim"
pkill -9 -f "aimrt_main.*a3_pingpong" 2>/dev/null || true
pkill -9 -x iox-roudi 2>/dev/null || true
rm -f /dev/shm/iox1_0_* 2>/dev/null || true