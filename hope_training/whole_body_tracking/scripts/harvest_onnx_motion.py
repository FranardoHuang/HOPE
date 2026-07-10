#!/usr/bin/env python3
"""Harvest the baked motion-reference buffers out of an exported policy.onnx (onnxruntime venv).

Companion to standalone_onnx_export.py: runs the donor across every time_step and saves the six
motion outputs, so the torch-venv export step can bake bit-identical buffers without onnxruntime.

Usage: python scripts/harvest_onnx_motion.py --donor policy.onnx --total 271 --out harvest.npz
"""

from __future__ import annotations

import argparse
import hashlib
import os

import numpy as np
import onnxruntime as ort

MOTION_KEYS = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--donor", required=True)
    p.add_argument("--total", type=int, required=True, help="total frames across the clip pair")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if args.total <= 0:
        raise SystemExit("[FATAL] --total must be positive")

    sess = ort.InferenceSession(args.donor, providers=["CPUExecutionProvider"])
    inputs = sess.get_inputs()
    outputs = sess.get_outputs()
    if [item.name for item in inputs] != ["obs", "time_step"]:
        raise SystemExit(f"[FATAL] donor inputs are not obs,time_step: {[x.name for x in inputs]}")
    expected_outputs = ["actions", *MOTION_KEYS]
    if [item.name for item in outputs] != expected_outputs:
        raise SystemExit(
            f"[FATAL] donor outputs are not {expected_outputs}: {[x.name for x in outputs]}"
        )
    obs_dim = inputs[0].shape[1]
    if not isinstance(obs_dim, int) or obs_dim <= 0:
        raise SystemExit(f"[FATAL] donor obs dimension is not fixed positive: {obs_dim!r}")
    obs0 = np.zeros((1, obs_dim), dtype=np.float32)
    rows = {k: [] for k in MOTION_KEYS}
    for ts in range(args.total):
        outs = sess.run(None, {"obs": obs0, "time_step": np.array([[ts]], dtype=np.float32)})
        for k, v in zip(MOTION_KEYS, outs[1:]):
            rows[k].append(v[0])
    stacked = {key: np.stack(value) for key, value in rows.items()}
    if any(not np.isfinite(value).all() for value in stacked.values()):
        raise SystemExit("[FATAL] donor emitted NaN/Inf motion references")
    donor_sha256 = _sha256_file(args.donor)
    tmp = args.out + ".tmp.npz"
    np.savez(
        tmp,
        total=args.total,
        donor_sha256=np.asarray(donor_sha256),
        donor_obs_dim=np.int64(obs_dim),
        **stacked,
    )
    os.replace(tmp, args.out)
    print(f"[harvest] {args.out}: total={args.total}, shapes=" +
          ", ".join(f"{k}{v.shape}" for k, v in stacked.items()) +
          f", donor_sha256={donor_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
