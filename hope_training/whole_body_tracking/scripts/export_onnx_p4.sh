#!/usr/bin/env bash
# Export policy.onnx from the p4 deploy-parity checkpoint, then KILL play.py (it enters an infinite
# sim-step loop after the export, so it never exits on its own in headless mode).
# Run: distrobox enter grasping -- bash scripts/export_onnx_p4.sh
set -uo pipefail
cd "$HOME/workspace/HOPE/hope_training/whole_body_tracking"
source setup_train_env.sh >/dev/null 2>&1

RUN="logs/rsl_rl/agibot_a3_hope_realsensor/2026-07-01_23-12-20_footwork_ft_p4_swingreach"
CKPT="$RUN/model_12100.pt"
FH="$PWD/artifacts/hope_forehand_hopex/motion.npz"
BH="$PWD/artifacts/hope_backhand_hopex/motion.npz"
ONNX="$RUN/exported/policy.onnx"
LOG="/tmp/export_p4.log"

rm -f "$ONNX" "$LOG"
echo "[export] launching play.py (setsid); ONNX -> $ONNX"

# setsid = own process group so we can kill the whole Isaac tree after the onnx is written.
# source inside the setsid subshell — hope_isaac_py is a shell FUNCTION, not inherited by `bash -c`.
setsid bash -c "cd '$PWD' && source setup_train_env.sh >/dev/null 2>&1 && \
  hope_isaac_py scripts/play.py task=HOPEPingPongDeployParity algo=ppo headless=true \
  num_envs=2 checkpoint='$CKPT' 'motion_file=[$FH,$BH]'" >"$LOG" 2>&1 &
PGID=$!
echo "[export] pgid=$PGID  (log: $LOG)"

# poll up to 8 min for the onnx to appear, then give 5s for metadata + kill the group.
for i in $(seq 1 96); do
  if [ -f "$ONNX" ]; then
    echo "[export] policy.onnx appeared after ~$((i*5))s; waiting 6s for metadata then killing."
    sleep 6
    break
  fi
  # bail early if the process died before producing the onnx
  if ! kill -0 "$PGID" 2>/dev/null; then
    echo "[export] play.py process group exited before onnx appeared — see $LOG"; break
  fi
  sleep 5
done

# kill the whole process group (negative PID), then any stragglers.
kill -TERM -"$PGID" 2>/dev/null
sleep 3
kill -KILL -"$PGID" 2>/dev/null
pkill -f "scripts/play.py" 2>/dev/null

echo ""
if [ -f "$ONNX" ]; then
  echo "[export] SUCCESS: $ONNX"
  ls -la "$ONNX" "$RUN/exported/"
else
  echo "[export] FAILED — no onnx. tail of log:"; tail -30 "$LOG"
fi
