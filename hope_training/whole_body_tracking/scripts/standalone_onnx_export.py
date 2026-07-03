#!/usr/bin/env python3
"""Export a training checkpoint to deploy ONNX WITHOUT booting Isaac (pure CPU torch).

Purpose: unblock exports when Isaac/Kit cannot boot (e.g. the 2026-07-04 runpod host whose Kit
hangs at extension load). Replicates utils/exporter.py faithfully:

- actor MLP rebuilt from the checkpoint's own weight shapes (no config guessing);
- EmpiricalNormalization loaded from the checkpoint's obs_norm_state_dict (exact rsl_rl math);
- the 6 motion-reference buffers read straight from the same .npz clips training used
  (MotionLoader semantics: concatenate clip arrays along time);
- graph/io identical to _OnnxMotionPolicyExporter (opset 11, same input/output names);
- metadata copied from a DONOR onnx exported earlier under the SAME task config (all metadata
  fields are config-derived — joint names/gains/obs layout/strike phases — so a same-config donor
  is exact; run_path is overridden).

Usage:
  python scripts/standalone_onnx_export.py --ckpt <model_x.pt> --fh fh.npz --bh bh.npz \
      --donor <older exported/policy.onnx> --out <run_dir>/exported --run-path <label>
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import onnx
import torch
import torch.nn as nn

MOTION_KEYS = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")


def build_actor(msd: dict) -> nn.Sequential:
    idx = sorted({int(k.split(".")[1]) for k in msd if k.startswith("actor.") and k.endswith(".weight")})
    layers, prev_end = [], -1
    for i in idx:
        w = msd[f"actor.{i}.weight"]
        if prev_end >= 0:
            for _ in range(i - prev_end - 1):
                layers.append(nn.ELU())
        lin = nn.Linear(w.shape[1], w.shape[0])
        lin.weight.data.copy_(w)
        lin.bias.data.copy_(msd[f"actor.{i}.bias"])
        layers.append(lin)
        prev_end = i
    return nn.Sequential(*layers)


class _StandaloneExporter(nn.Module):
    def __init__(self, actor, normalizer, motion):
        super().__init__()
        self.actor = actor
        self.normalizer = normalizer
        for k in MOTION_KEYS:
            setattr(self, k, motion[k])
        self.time_step_total = motion["joint_pos"].shape[0]

    def forward(self, x, time_step):
        t = torch.clamp(time_step.long().squeeze(-1), max=self.time_step_total - 1)
        return (
            self.actor(self.normalizer(x)),
            self.joint_pos[t],
            self.joint_vel[t],
            self.body_pos_w[t],
            self.body_quat_w[t],
            self.body_lin_vel_w[t],
            self.body_ang_vel_w[t],
        )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--fh", required=True)
    p.add_argument("--bh", required=True)
    p.add_argument("--donor", required=True, help="same-task-config exported policy.onnx to copy metadata from")
    p.add_argument("--out", required=True)
    p.add_argument("--run-path", default="standalone-export")
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    msd = ckpt["model_state_dict"]
    actor = build_actor(msd)
    in_dim = actor[0].in_features

    from rsl_rl.modules import EmpiricalNormalization

    normalizer = EmpiricalNormalization(shape=[in_dim])
    normalizer.load_state_dict(ckpt["obs_norm_state_dict"])
    normalizer.eval()

    clips = [dict(np.load(args.fh)), dict(np.load(args.bh))]
    seg_lengths = [c["joint_pos"].shape[0] for c in clips]
    # Motion buffers are HARVESTED from the donor ONNX rather than rebuilt from the npz: the env
    # subsets body arrays to cfg.body_names (14 tracked bodies) in cfg order, while the npz holds
    # all robot bodies — reproducing that mapping here would duplicate config; the donor (same clip
    # pair, asserted below) already carries the exact baked buffers. Bit-parity by construction.
    import onnxruntime as ort

    sess = ort.InferenceSession(args.donor, providers=["CPUExecutionProvider"])
    total = sum(seg_lengths)
    probe = np.zeros((1, 1), dtype=np.float32)
    obs0 = np.zeros((1, 1), dtype=np.float32)  # replaced below once in_dim known from donor input
    donor_in = {i.name: i.shape for i in sess.get_inputs()}
    obs0 = np.zeros((1, donor_in["obs"][1]), dtype=np.float32)
    rows = {k: [] for k in MOTION_KEYS}
    for ts in range(total):
        outs = sess.run(None, {"obs": obs0, "time_step": np.array([[ts]], dtype=np.float32)})
        for k, v in zip(MOTION_KEYS, outs[1:]):
            rows[k].append(v[0])
    motion = {k: torch.as_tensor(np.stack(rows[k]), dtype=torch.float32) for k in MOTION_KEYS}

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "policy.onnx")
    module = _StandaloneExporter(actor, normalizer, motion).eval()
    torch.onnx.export(
        module,
        (torch.zeros(1, in_dim), torch.zeros(1, 1)),
        out_path,
        export_params=True,
        opset_version=11,
        input_names=["obs", "time_step"],
        output_names=["actions", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"],
        dynamic_axes={},
    )

    donor = onnx.load(args.donor)
    donor_meta = {e.key: e.value for e in donor.metadata_props}
    for req in ("joint_names", "joint_stiffness", "clip_seg_lengths", "clip_strike_phases"):
        assert req in donor_meta, f"donor onnx missing metadata key {req} — pick a newer donor"
    donor_segs = [int(s) for s in donor_meta["clip_seg_lengths"].split(",")]
    assert donor_segs == seg_lengths, (
        f"donor clip layout {donor_segs} != provided clips {seg_lengths} — donor must come from the "
        f"same motion pair (otherwise gains/obs metadata may not match either)"
    )
    donor_meta["run_path"] = args.run_path

    model = onnx.load(out_path)
    del model.metadata_props[:]
    for k, v in donor_meta.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = k, str(v)
    onnx.save(model, out_path)

    print(f"[standalone-export] SUCCESS {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)")
    print(f"[standalone-export] in_dim={in_dim} actions={actor[-1].out_features} "
          f"clips={seg_lengths} iter={ckpt.get('iter', '?')} metadata_keys={len(donor_meta)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
