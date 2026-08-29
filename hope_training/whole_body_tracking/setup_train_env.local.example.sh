# Copy to setup_train_env.local.sh and replace paths with the exact runtime on
# this machine.  The local file is ignored because paths and optional service
# identities are machine-specific; the version contract is tracked in
# docs/operations/action_ball_isaac51_environment_identity_20260818.md.
HOPE_ISAAC_PYTHON=/workspace/isaacsim-5.1.0/python.sh
HOPE_ISAACLAB_ROOT=/workspace/IsaacLab-8320e0be
HOPE_ISAAC_VENV_SITE=/workspace/hope_drone_venv/lib/python3.11/site-packages

# Optional, never commit credentials here.
# WANDB_ENTITY=BerkeleyPingPong
# WANDB_REGISTRY_ORG=dongc_1-university-of-california-berkeley-org
