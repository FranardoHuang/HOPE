#!/usr/bin/env bash
# Export policy.onnx from a HOPEPingPongHitter (177-D hitter_footwork) checkpoint, then KILL
# play.py (it enters an infinite sim-step loop after the export, so it never exits on its own
# in headless mode).
#
# Usage (inside the `grasping` distrobox, from whole_body_tracking/):
#   bash scripts/export_onnx_hitter.sh <run_dir> [model_XXXX.pt]
# Examples:
#   bash scripts/export_onnx_hitter.sh logs/rsl_rl/agibot_a3_hope_hitter/2026-07-06_00-56-46
#   bash scripts/export_onnx_hitter.sh logs/rsl_rl/agibot_a3_hope_hitter/2026-07-06_00-56-46 model_11000.pt
# With no checkpoint arg, the NEWEST model_*.pt in the run dir is used.
#
# MOTION CLIPS: pinned to the hopex RE-GROUNDED clips (identical content to registry source
# artifacts hope_forehand:v2 / hope_backhand:v1 — the 2026-07-06_00-56-46 hitter run bakes
# exactly these, verified in its wandb debug.log). Override with FH=/path BH=/path for other
# lineages. The export MUST bake the same clips as training or the reference motion is wrong.
# ALWAYS verify against the run's wandb config:
#   grep -oE "'motion_file': \[[^]]*\]" hope_training/wandb/wandb/run-*<id>/logs/debug.log | sort -u
#
# 177-specific: the exporter bakes actor_obs_contract=hitter_footwork + ref_reach_offset_xy
# (per-clip base->racket reach at the strike frame) — the C++ runner derives the deploy-time
# base station from these and REFUSES a 177 model without them.
# Output: <run_dir>/exported/policy.onnx  (+ contract check via inspect_a3_deploy_contract.py)
set -uo pipefail
cd "$(dirname "$0")/.."

RUN="${1:?usage: export_onnx_hitter.sh <run_dir> [model_XXXX.pt]}"
if [ -n "${2:-}" ]; then
  CKPT="$RUN/$2"
else
  CKPT="$(ls -t "$RUN"/model_*.pt | head -1)"
fi
[ -f "$CKPT" ] || { echo "[export] checkpoint not found: $CKPT"; exit 1; }

FH="${FH:-$PWD/artifacts/hope_forehand_hopex/motion.npz}"
BH="${BH:-$PWD/artifacts/hope_backhand_hopex/motion.npz}"
[ -f "$FH" ] && [ -f "$BH" ] || { echo "[export] motion clips missing: $FH / $BH"; exit 1; }

ONNX="$RUN/exported/policy.onnx"
LOG="/tmp/export_hitter.log"
rm -f "$ONNX" "$LOG"
echo "[export] checkpoint: $CKPT"
echo "[export] clips: $FH + $BH"
echo "[export] launching play.py (setsid); ONNX -> $ONNX"

# setsid = own process group so we can kill the whole Isaac tree after the onnx is written.
# source inside the setsid subshell — hope_isaac_py is a shell FUNCTION, not inherited by `bash -c`.
setsid bash -c "cd '$PWD' && source setup_train_env.sh >/dev/null 2>&1 && \
  hope_isaac_py scripts/play.py task=HOPEPingPongHitter algo=ppo headless=true \
  num_envs=2 checkpoint='$CKPT' 'motion_file=[$FH,$BH]'" >"$LOG" 2>&1 &
PGID=$!
echo "[export] pgid=$PGID  (log: $LOG)"

# poll up to 10 min for the onnx to appear, then give 6s for metadata + kill the group.
for i in $(seq 1 120); do
  if [ -f "$ONNX" ]; then
    echo "[export] policy.onnx appeared after ~$((i*5))s; waiting 6s for metadata then killing."
    sleep 6
    break
  fi
  if ! kill -0 "$PGID" 2>/dev/null; then
    echo "[export] play.py process group exited before onnx appeared — see $LOG"; break
  fi
  sleep 5
done

kill -TERM -"$PGID" 2>/dev/null
sleep 3
kill -KILL -"$PGID" 2>/dev/null

if [ -f "$ONNX" ]; then
  echo "[export] SUCCESS: $ONNX"
  ls -la "$RUN/exported/"
  # contract gate: 177-D hitter_footwork, clip metadata + ref_reach_offset_xy present, positive gains
  python3 scripts/inspect_a3_deploy_contract.py --onnx "$ONNX" 2>/dev/null \
    || echo "[export] (inspect_a3_deploy_contract.py needs onnx python env — run it manually)"
  python3 - "$ONNX" <<'PY' 2>/dev/null || echo "[export] (onnx python env missing — check ref_reach_offset_xy manually)"
import sys, onnx
m = onnx.load(sys.argv[1])
md = {p.key: p.value for p in m.metadata_props}
missing = [k for k in ("actor_obs_contract", "ref_reach_offset_xy") if k not in md]
if missing:
    print(f"[export] WARNING: 177 deploy metadata MISSING: {missing} — C++ runner will refuse this model")
else:
    print(f"[export] 177 metadata OK: actor_obs_contract={md['actor_obs_contract']} "
          f"ref_reach_offset_xy={md['ref_reach_offset_xy']}")
PY
else
  echo "[export] FAILED — no onnx. tail of log:"; tail -30 "$LOG"; exit 1
fi
