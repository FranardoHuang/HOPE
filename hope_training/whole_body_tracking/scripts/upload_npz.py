import os

import wandb

REGISTRY_NAME = os.environ.get("WANDB_REGISTRY_NAME", "motions")
COLLECTION_NAME = os.environ.get("WANDB_COLLECTION_NAME", "lafan_kungfu")
MOTION_FILE = os.environ.get("WANDB_MOTION_FILE", "./motions/motion.npz")

run = wandb.init(
    entity=os.environ.get("WANDB_ENTITY") or None,
    project=os.environ.get("WANDB_MOTION_PROJECT", "csv_to_npz"),
    name=COLLECTION_NAME,
)

artifact = wandb.Artifact(name=COLLECTION_NAME, type=REGISTRY_NAME)
artifact.add_file(MOTION_FILE, name="motion.npz")
logged_artifact = run.log_artifact(artifact)

run.link_artifact(artifact=logged_artifact, target_path=f"wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}")
run.finish()

org = os.environ.get("WANDB_REGISTRY_ORG", os.environ.get("WANDB_ENTITY", ""))
if org:
    print(f"Expected registry path: {org}/wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}:latest")
