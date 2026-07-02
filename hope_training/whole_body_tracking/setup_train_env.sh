# =============================================================================
# HOPE WBC training environment. SOURCE this inside the Isaac/GPU shell before
# running scripts/train.py / scripts/play.py (reimplement.md Step 14.1):
#
#   cd /workspace/yikang/nohope/hope_training/whole_body_tracking
#   source setup_train_env.sh
#
# It (1) puts the working-tree whole_body_tracking package first on PYTHONPATH,
# (2) defines the `hope_isaac_py` launcher for the current IsaacLab install, and
# (3) exports the wandb team/org/project.
#
# MUST be SOURCED, not executed (`./setup_train_env.sh` would set everything in a
# subshell that then exits, leaving your shell unchanged). Re-source it in every
# new terminal. Safe to re-source. Idempotent.
# =============================================================================

# Absolute path to this script's dir (.../hope_training/whole_body_tracking), so
# sourcing works from any cwd and any clone location.
_WBT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Current machine paths verified on 2026-07-02. /workspace/isaacsim/python.sh
# and /opt/drone_venv are legacy paths and are not present on this machine.
export HOPE_ISAAC_VENV=${HOPE_ISAAC_VENV:-/workspace/hope_isaac_venv}
export ISAACLAB_PATH=${ISAACLAB_PATH:-/workspace/IsaacLab}

# ORDER MATTERS: the working-tree `source/whole_body_tracking` MUST come FIRST.
# If a stale installed copy exists in the Isaac venv, source-first guarantees the
# working tree wins.
export HOPE_WBT_PYTHONPATH=$_WBT_DIR/source/whole_body_tracking

# Launch IsaacLab's wrapper with the venv first on PATH. HOPE_ISAAC_TERM
# defaults to xterm so the wrapper's `tabs` helper does not fail under TERM=dumb.
hope_isaac_py () {
  TERM="${HOPE_ISAAC_TERM:-xterm}" \
    PATH="$HOPE_ISAAC_VENV/bin:$PATH" \
    PYTHONPATH="${HOPE_WBT_PYTHONPATH}${PYTHONPATH:+:$PYTHONPATH}" \
    "$ISAACLAB_PATH/isaaclab.sh" -p "$@"
}

# wandb: runs log to your TEAM; the motion registry is read from your ORG. These
# are DIFFERENT — using the team for the registry fails ("Unable to find
# organization for entity ..."). See reimplement.md Step 12.5.
export WANDB_ENTITY=BerkeleyPingPong                                      # team/entity for training runs
export WANDB_REGISTRY_ORG=dongc_1-university-of-california-berkeley-org   # org for motions registry
export WANDB_PROJECT=hope_wbc                                             # training project
export WANDB_MOTION_PROJECT=csv_to_npz                                    # motion-upload project
export WANDB_DIR=${WANDB_DIR:-/workspace/yikang/nohope/hope_training/wandb}
mkdir -p "$WANDB_DIR"

unset _WBT_DIR

echo "[hope] training env ready."
echo "[hope]   hope_isaac_py -> $ISAACLAB_PATH/isaaclab.sh -p (venv: $HOPE_ISAAC_VENV)"
echo "[hope]   WANDB_ENTITY=$WANDB_ENTITY  WANDB_REGISTRY_ORG=$WANDB_REGISTRY_ORG  WANDB_PROJECT=$WANDB_PROJECT  WANDB_DIR=$WANDB_DIR"
