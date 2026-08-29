# =============================================================================
# HOPE WBC training environment. SOURCE this inside the GPU/Isaac shell before
# running scripts/csv_to_npz.py / scripts/train.py / scripts/play.py:
#
#   cd /workspace/yikang/nohope/hope_training/whole_body_tracking
#   source setup_train_env.sh
#
# It (1) sets source-first PYTHONPATH, (2) defines the `hope_isaac_py`
# launcher, and (3) exports overridable WandB defaults for optional registry or
# remote logging use. Put machine-specific paths and private WandB identities in
# setup_train_env.local.sh, which is git-ignored.
#
# MUST be SOURCED, not executed (`./setup_train_env.sh` would set everything in a
# subshell that then exits, leaving your shell unchanged). Re-source it in every
# new terminal. Safe to re-source. Idempotent.
# =============================================================================

# Absolute path to this script's dir (.../hope_training/whole_body_tracking), so
# sourcing works from any cwd and any clone location.
_WBT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Optional local override (real machine paths + WandB identity).
if [ -f "${_WBT_DIR}/setup_train_env.local.sh" ]; then
  # shellcheck disable=SC1091
  source "${_WBT_DIR}/setup_train_env.local.sh"
fi

# Runtime identity must come from the caller or the ignored local file.  Path
# discovery used to silently select whichever old Isaac/Python tree happened to
# exist first, so the same source commit could denote two different trainers.
if [ -z "${HOPE_ISAAC_PYTHON:-}" ]; then
  echo "[hope] ERROR: set HOPE_ISAAC_PYTHON explicitly; runtime path discovery is forbidden." >&2
  unset _WBT_DIR
  return 1 2>/dev/null || exit 1
fi
if [ ! -x "${HOPE_ISAAC_PYTHON}" ]; then
  echo "[hope] ERROR: HOPE_ISAAC_PYTHON is not executable: ${HOPE_ISAAC_PYTHON}" >&2
  unset _WBT_DIR
  return 1 2>/dev/null || exit 1
fi
if [ -z "${HOPE_ISAACLAB_ROOT:-}" ]; then
  echo "[hope] ERROR: set HOPE_ISAACLAB_ROOT explicitly; runtime path discovery is forbidden." >&2
  unset _WBT_DIR
  return 1 2>/dev/null || exit 1
fi
if [ ! -d "${HOPE_ISAACLAB_ROOT}/source" ]; then
  echo "[hope] ERROR: HOPE_ISAACLAB_ROOT has no source directory: ${HOPE_ISAACLAB_ROOT}" >&2
  unset _WBT_DIR
  return 1 2>/dev/null || exit 1
fi
if [ -n "${HOPE_ISAAC_VENV_SITE:-}" ] && [ ! -d "${HOPE_ISAAC_VENV_SITE}" ]; then
  echo "[hope] ERROR: HOPE_ISAAC_VENV_SITE is not a directory: ${HOPE_ISAAC_VENV_SITE}" >&2
  unset _WBT_DIR
  return 1 2>/dev/null || exit 1
fi
export HOPE_ISAAC_PYTHON HOPE_ISAACLAB_ROOT HOPE_ISAAC_VENV_SITE

# ORDER MATTERS: the working-tree source must come first so local edits win over
# any stale site-packages install.
_il="${HOPE_ISAACLAB_ROOT}/source"
HOPE_WBT_PYTHONPATH="${_WBT_DIR}/source/whole_body_tracking"
if [ -n "${HOPE_ISAAC_VENV_SITE:-}" ]; then
  HOPE_WBT_PYTHONPATH="${HOPE_WBT_PYTHONPATH}:${HOPE_ISAAC_VENV_SITE}"
fi
HOPE_WBT_PYTHONPATH="${HOPE_WBT_PYTHONPATH}:${_il}/isaaclab:${_il}/isaaclab_tasks:${_il}/isaaclab_assets:${_il}/isaaclab_rl"
export HOPE_WBT_PYTHONPATH

# Launch Isaac's python with the training PYTHONPATH.
hope_isaac_py () {
  PYTHONPATH="${HOPE_WBT_PYTHONPATH}" "${HOPE_ISAAC_PYTHON}" "$@"
}

# wandb: runs log to your TEAM; the motion registry is read from your ORG. These
# are DIFFERENT — using the team for the registry fails ("Unable to find
# organization for entity ..."). See reimplement.md Step 12.5. All values below
# are overridable via the environment or setup_train_env.local.sh; local runs
# can use logger=tensorboard and motion_file=... without WandB.
export WANDB_ENTITY="${WANDB_ENTITY:-BerkeleyPingPong}"                                    # team (run logging)
export WANDB_REGISTRY_ORG="${WANDB_REGISTRY_ORG:-dongc_1-university-of-california-berkeley-org}"  # org (motion registry)
export WANDB_PROJECT="${WANDB_PROJECT:-hope_wbc}"                                          # training project
export WANDB_MOTION_PROJECT="${WANDB_MOTION_PROJECT:-csv_to_npz}"                          # motion-upload project
# Keep wandb run files off small home dirs (default: <repo>/hope_training/wandb).
export WANDB_DIR="${WANDB_DIR:-$(dirname "${_WBT_DIR}")/wandb}"
mkdir -p "$WANDB_DIR"

unset _WBT_DIR _il

echo "[hope] training env ready."
echo "[hope]   hope_isaac_py -> ${HOPE_ISAAC_PYTHON}"
echo "[hope]   WANDB_ENTITY=$WANDB_ENTITY  WANDB_REGISTRY_ORG=$WANDB_REGISTRY_ORG  WANDB_PROJECT=$WANDB_PROJECT  WANDB_DIR=$WANDB_DIR"
