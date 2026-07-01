# =============================================================================
# HOPE WBC training environment. SOURCE this inside the `grasping` distrobox
# before running scripts/train.py / scripts/play.py (reimplement.md Step 14.1):
#
#   distrobox enter grasping
#   cd ~/workspace/HOPE/hope_training/whole_body_tracking
#   source setup_train_env.sh
#
# It (1) sets the PYTHONPATH that lets Isaac's bundled python see hydra/omegaconf
# (installed in /opt/drone_venv) plus isaaclab/isaaclab_rl, (2) defines the
# `hope_isaac_py` launcher, and (3) exports the wandb team/org/project.
#
# MUST be SOURCED, not executed (`./setup_train_env.sh` would set everything in a
# subshell that then exits, leaving your shell unchanged). Re-source it in every
# new terminal. Safe to re-source. Idempotent.
# =============================================================================

# Absolute path to this script's dir (.../hope_training/whole_body_tracking), so
# sourcing works from any cwd and any clone location.
_WBT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# hydra/omegaconf live in /opt/drone_venv (Isaac's own python has neither);
# isaaclab_* are required for training (replay/csv_to_npz did not need isaaclab_rl).
#
# ORDER MATTERS: the working-tree `source/whole_body_tracking` MUST come FIRST, before
# /opt/drone_venv/.../site-packages. If a NON-editable `pip install .` ever put a copy of
# whole_body_tracking into the Isaac venv, a site-packages-first order would import that STALE copy and
# silently ignore your edits (cfg classes, reward/command code AND the cfg/task YAML override targets) —
# the runtime would diverge from your tree with no error. Source-first guarantees the working tree wins.
# train.py prints "whole_body_tracking imported from: ..." so you can confirm it points here, not there.
export HOPE_WBT_PYTHONPATH=$_WBT_DIR/source/whole_body_tracking:/opt/drone_venv/lib/python3.11/site-packages:/workspace/omni_drones/third_party/IsaacLab/source/isaaclab:/workspace/omni_drones/third_party/IsaacLab/source/isaaclab_tasks:/workspace/omni_drones/third_party/IsaacLab/source/isaaclab_assets:/workspace/omni_drones/third_party/IsaacLab/source/isaaclab_rl

# Launch Isaac's python with the training PYTHONPATH. The `${VAR:-<default>}` form
# falls back to a built-in absolute path if HOPE_WBT_PYTHONPATH is ever unset, so
# an empty PYTHONPATH (the `ModuleNotFoundError: No module named 'hydra'` cause)
# can't happen even if only the function got redefined in a fresh shell.
hope_isaac_py () {
  local _wbt=$HOME/workspace/HOPE/hope_training/whole_body_tracking/source/whole_body_tracking
  PYTHONPATH="${HOPE_WBT_PYTHONPATH:-$_wbt:/opt/drone_venv/lib/python3.11/site-packages:/workspace/omni_drones/third_party/IsaacLab/source/isaaclab:/workspace/omni_drones/third_party/IsaacLab/source/isaaclab_tasks:/workspace/omni_drones/third_party/IsaacLab/source/isaaclab_assets:/workspace/omni_drones/third_party/IsaacLab/source/isaaclab_rl}" \
    /workspace/isaacsim/python.sh "$@"
}

# wandb: runs log to your TEAM; the motion registry is read from your ORG. These
# are DIFFERENT — using the team for the registry fails ("Unable to find
# organization for entity ..."). See reimplement.md Step 12.5.
export WANDB_ENTITY=BerkeleyPingPong                                      # team (run logging)
export WANDB_REGISTRY_ORG=dongc_1-university-of-california-berkeley-org   # org  (motion registry)
export WANDB_PROJECT=hope_wbc

unset _WBT_DIR

echo "[hope] training env ready."
echo "[hope]   hope_isaac_py -> /workspace/isaacsim/python.sh (+ hydra/omegaconf + isaaclab_rl)"
echo "[hope]   WANDB_ENTITY=$WANDB_ENTITY  WANDB_REGISTRY_ORG=$WANDB_REGISTRY_ORG  WANDB_PROJECT=$WANDB_PROJECT"
