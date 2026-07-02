#!/usr/bin/env bash
# Export policy.onnx from the explicitpd_ft model_25700, then KILL play.py (infinite sim loop after export).
# Run: distrobox enter grasping -- bash scripts/export_onnx_explicitpd.sh
set -uo pipefail
cd "$HOME/workspace/HOPE/hope_training/whole_body_tracking"
source setup_train_env.sh >/dev/null 2>&1

RUN="logs/rsl_rl/agibot_a3_hope_realsensor/2026-07-02_03-08-38_explicitpd_ft"
CKPT="$RUN/model_25700.pt"
FH="$PWD/artifacts/hope_forehand_hopex/motion.npz"
BH="$PWD/artifacts/hope_backhand_hopex/motion.npz"
ONNX="$RUN/exported/policy.onnx"
LOG="/tmp/export_explicitpd.log"

rm -f "$ONNX" "$LOG"
echo "[export] launching play.py (setsid); ONNX -> $ONNX"
setsid bash -c "cd '$PWD' && source setup_train_env.sh >/dev/null 2>&1 && \
  hope_isaac_py scripts/play.py task=HOPEPingPongDeployParity algo=ppo headless=true \
  num_envs=2 checkpoint='$CKPT' 'motion_file=[$FH,$BH]'" >"$LOG" 2>&1 &
PGID=$!
echo "[export] pgid=$PGID  log=$LOG"
for i in $(seq 1 96); do
  if [ -f "$ONNX" ]; then echo "[export] onnx appeared ~$((i*5))s; +6s for metadata then kill."; sleep 6; break; fi
  kill -0 "$PGID" 2>/dev/null || { echo "[export] play.py exited before onnx — see $LOG"; break; }
  sleep 5
done
kill -TERM -"$PGID" 2>/dev/null; sleep 3; kill -KILL -"$PGID" 2>/dev/null; pkill -f "scripts/play.py" 2>/dev/null
echo ""
[ -f "$ONNX" ] && { echo "[export] SUCCESS: $ONNX"; ls -la "$ONNX"; } || { echo "[export] FAILED. tail:"; tail -25 "$LOG"; }
