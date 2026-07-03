"""Verify registry :latest resolves to the re-grounded hopex clips (train.py's exact path)."""
import math
import os
import pathlib

import numpy as np
import wandb

ORG = os.environ.get("WANDB_REGISTRY_ORG", "dongc_1-university-of-california-berkeley-org")
api = wandb.Api()
ok = True
for name, want_frames in (("hope_forehand", 139), ("hope_backhand", 132)):
    art = api.artifact(f"{ORG}/wandb-registry-motions/{name}:latest")
    p = pathlib.Path(art.download()) / "motion.npz"
    d = np.load(p)
    q = d["body_quat_w"][0, 0]
    yaw = math.degrees(math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))
    frames = int(d["joint_pos"].shape[0])
    good = abs(yaw) < 1.0 and frames == want_frames
    ok &= good
    print(f"{name}:latest -> {art.source_qualified_name} (v{art.version.lstrip('v')})  "
          f"frames={frames}  frame-0 pelvis yaw={yaw:+.3f} deg  {'OK' if good else 'BAD'}")
print("REGISTRY-LATEST:", "RE-GROUNDED ✓" if ok else "STILL RAW ✗")
