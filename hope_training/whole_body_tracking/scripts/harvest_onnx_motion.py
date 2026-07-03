#!/usr/bin/env python3
"""Harvest the baked motion-reference buffers out of an exported policy.onnx (onnxruntime venv).

Companion to standalone_onnx_export.py: runs the donor across every time_step and saves the six
motion outputs, so the torch-venv export step can bake bit-identical buffers without onnxruntime.

Usage: python scripts/harvest_onnx_motion.py --donor policy.onnx --total 271 --out harvest.npz
"""

from __future__ import annotations

import argparse

import numpy as np
import onnxruntime as ort

MOTION_KEYS = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--donor", required=True)
    p.add_argument("--total", type=int, required=True, help="total frames across the clip pair")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    sess = ort.InferenceSession(args.donor, providers=["CPUExecutionProvider"])
    obs0 = np.zeros((1, sess.get_inputs()[0].shape[1]), dtype=np.float32)
    rows = {k: [] for k in MOTION_KEYS}
    for ts in range(args.total):
        outs = sess.run(None, {"obs": obs0, "time_step": np.array([[ts]], dtype=np.float32)})
        for k, v in zip(MOTION_KEYS, outs[1:]):
            rows[k].append(v[0])
    np.savez(args.out, total=args.total, **{k: np.stack(v) for k, v in rows.items()})
    print(f"[harvest] {args.out}: total={args.total}, shapes=" +
          ", ".join(f"{k}{np.stack(v).shape}" for k, v in rows.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
