"""Upload the RE-GROUNDED hopex clips as the new registry latest (2026-07-03).

The registry :latest had been silently taken over by the RAW v4 clips (frame-0 pelvis yaw
+82/+86 deg, never re-grounded), which trained the undeployable turn-and-walk model_9000
(see hope-backward-jump-verdict). This publishes artifacts/hope_{forehand,backhand}_hopex
(frame-0 yaw exactly 0, same 139/132 frame counts) as new versions of the SAME collections
(hope_forehand / hope_backhand), so the task YAML's registry defaults resolve to the
re-grounded lineage again and plain `train.py task=HOPEPingPongDeployParity` is safe.

Run with any python that has wandb + numpy (e.g. hope_isaac_py, or /opt/drone_venv python
in the grasping box). Verifies frame-0 pelvis yaw == 0 BEFORE uploading — refuses to
publish a non-re-grounded clip.
"""

import math
import os

import numpy as np
import wandb

REGISTRY_NAME = "motions"
CLIPS = {
    "hope_forehand": "artifacts/hope_forehand_hopex/motion.npz",
    "hope_backhand": "artifacts/hope_backhand_hopex/motion.npz",
}


def frame0_pelvis_yaw_deg(npz_path: str) -> float:
    d = np.load(npz_path)
    q = d["body_quat_w"][0, 0]  # frame 0, pelvis, wxyz
    return math.degrees(
        math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
    )


def main():
    for name, path in CLIPS.items():
        assert os.path.isfile(path), f"missing {path}"
        yaw = frame0_pelvis_yaw_deg(path)
        frames = int(np.load(path)["joint_pos"].shape[0])
        print(f"[upload] {name}: {path}  frames={frames}  frame-0 pelvis yaw={yaw:+.3f} deg")
        assert abs(yaw) < 1.0, (
            f"{path} is NOT re-grounded (yaw {yaw:+.1f} deg) — refusing to publish. "
            "Run scripts/reground_hope_frame.py first."
        )

    for name, path in CLIPS.items():
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "csv_to_npz"),
            name=f"{name}_hopex_reground_0703",
            notes="re-grounded (+X, frame-0 pelvis yaw=0) hopex lineage; supersedes raw v4",
        )
        art = run.log_artifact(artifact_or_path=path, name=name, type=REGISTRY_NAME)
        run.link_artifact(artifact=art, target_path=f"wandb-registry-{REGISTRY_NAME}/{name}")
        run.finish()
        print(f"[upload] {name}: logged + linked to wandb-registry-{REGISTRY_NAME}/{name}")

    print("[upload] DONE — verify with scripts/verify_registry_latest_0703.py")


if __name__ == "__main__":
    main()
