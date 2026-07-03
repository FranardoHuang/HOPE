#!/usr/bin/env bash
# Export policy.onnx from a checkpoint via play.py, then KILL play.py (it enters an
# infinite sim loop after the export).
#
# Portable: resolves the whole_body_tracking dir from its own location; every
# machine-specific value is an env override (defaults document the 2026-07-02
# explicitpd_ft model_25700 export):
#
#   HOPE_EXPORT_RUN=<run dir relative to wbt or absolute>
#   HOPE_EXPORT_CKPT=<model_*.pt path>            default: $HOPE_EXPORT_RUN/model_25700.pt
#   HOPE_EXPORT_TASK=HOPEPingPongDeployParity
#   HOPE_EXPORT_FH / HOPE_EXPORT_BH=<motion .npz> default: artifacts/hope_{forehand,backhand}_hopex
#   HOPE_EXPORT_LOG=/tmp/export_explicitpd.log
#
# Run inside the GPU/Isaac environment: bash scripts/export_onnx_explicitpd.sh
set -uo pipefail

WBT_DIR="${HOPE_WBT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$WBT_DIR"
source setup_train_env.sh >/dev/null 2>&1

RUN="${HOPE_EXPORT_RUN:-logs/rsl_rl/agibot_a3_hope_realsensor/2026-07-02_03-08-38_explicitpd_ft}"
CKPT="${HOPE_EXPORT_CKPT:-$RUN/model_25700.pt}"
TASK="${HOPE_EXPORT_TASK:-HOPEPingPongDeployParity}"
FH="${HOPE_EXPORT_FH:-$PWD/artifacts/hope_forehand_hopex/motion.npz}"
BH="${HOPE_EXPORT_BH:-$PWD/artifacts/hope_backhand_hopex/motion.npz}"
ONNX="$RUN/exported/policy.onnx"
LOG="${HOPE_EXPORT_LOG:-/tmp/export_explicitpd.log}"

[ -f "$CKPT" ] || { echo "[FATAL] checkpoint not found: $CKPT (set HOPE_EXPORT_RUN/HOPE_EXPORT_CKPT)"; exit 1; }
for f in "$FH" "$BH"; do
    [ -f "$f" ] || { echo "[FATAL] motion file not found: $f (set HOPE_EXPORT_FH/HOPE_EXPORT_BH)"; exit 1; }
done

rm -f "$ONNX" "$LOG"
echo "[export] launching play.py (setsid); ONNX -> $ONNX"
setsid bash -c "cd '$PWD' && source setup_train_env.sh >/dev/null 2>&1 && \
  hope_isaac_py scripts/play.py task=$TASK algo=ppo headless=true \
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
