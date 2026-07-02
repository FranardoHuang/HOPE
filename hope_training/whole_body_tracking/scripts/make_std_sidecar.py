"""Extract the policy's learned action std from an rsl_rl checkpoint -> exported/learned_std.npy.

scripts/play.py exports policy.onnx but NOT the learned std, and mujoco_eval_onnx.py needs the std
sidecar for any dither mode (noise_scale > 0). This pulls the state-independent std parameter out of a
checkpoint and saves it as a (31,) float32 .npy next to the ONNX.

IMPORTANT: run this on the SAME checkpoint you exported the ONNX from, so the std matches the policy.
Needs torch (run with the Isaac env python `hope_isaac_py`, or any env that has torch + numpy — NOT the
mujoco-only hope-motion-py310, which lacks torch).

    hope_isaac_py scripts/make_std_sidecar.py \
        --checkpoint logs/rsl_rl/agibot_a3_hope/<RUN>/model_<N>.pt
    # -> writes logs/rsl_rl/agibot_a3_hope/<RUN>/exported/learned_std.npy  (override with --out)
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch


def _find_std(state_dict):
    """Return (key, std_vector) from an rsl_rl ActorCritic state dict.

    Handles the common parameter names: `std` (raw), `log_std` (-> exp), `action_std`, possibly nested
    under a submodule prefix (matches by the last dotted component). Raises if none is found.
    """
    # exact top-level names first, then suffix match (e.g. "actor.std", "policy.log_std")
    for name in ("std", "action_std", "log_std"):
        if name in state_dict:
            v = state_dict[name]
            return name, (v.exp() if name == "log_std" else v)
    for k, v in state_dict.items():
        leaf = k.split(".")[-1]
        if leaf in ("std", "action_std", "log_std"):
            return k, (v.exp() if leaf == "log_std" else v)
    raise SystemExit(
        "[FATAL] no std/log_std/action_std parameter found in the checkpoint's model_state_dict.\n"
        "        keys: " + ", ".join(list(state_dict.keys())[:40]))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="path to model_<N>.pt (the one the ONNX came from)")
    p.add_argument("--out", default=None,
                   help="output .npy path (default: <checkpoint_dir>/exported/learned_std.npy)")
    p.add_argument("--expect-dim", type=int, default=31, help="expected std length (warn-only if mismatch)")
    args = p.parse_args()

    if not os.path.isfile(args.checkpoint):
        raise SystemExit(f"[FATAL] checkpoint not found: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    key, std = _find_std(sd)
    std = std.detach().cpu().numpy().reshape(-1).astype(np.float32)
    if std.shape[0] != args.expect_dim:
        print(f"[WARN] std length {std.shape[0]} != expected {args.expect_dim} — check the checkpoint.")

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), "exported", "learned_std.npy")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.save(out, std)
    print(f"[make_std_sidecar] checkpoint = {args.checkpoint}")
    print(f"[make_std_sidecar] std param  = '{key}'  shape={std.shape}  "
          f"mean={std.mean():.4f} min={std.min():.4f} max={std.max():.4f}")
    print(f"[make_std_sidecar] saved -> {out}")


if __name__ == "__main__":
    main()
