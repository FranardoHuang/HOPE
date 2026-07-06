"""Warm-start a 177-D HITTER-footwork run from a 175-D deploy-parity checkpoint (2026-07-05).

Weight surgery: insert N zero COLUMNS into the actor/critic first-layer weight matrices at the
column index where the restored ``base_target_pos_b`` term sits in each obs layout (indices come
from ``verify_hitter_task.py``'s layout print — do NOT guess). Zero columns mean the padded policy
is byte-equivalent to the source checkpoint until training grows weights into the new channel —
the safest possible warm start. The optimizer state is DROPPED (shapes changed); iteration is
reset to 0 so the new run's wandb x-axis starts clean.

    hope_isaac_py scripts/make_hitter_warmstart.py \
        --ckpt logs/rsl_rl/agibot_a3_hope_deploy_parity/2026-07-05_13-04-02/model_19400.pt \
        --out  logs/rsl_rl/warmstarts/model_19400_hitter177.pt \
        --actor-insert 167 --critic-insert <from layout print> --n 2
"""

import argparse
import os

import torch

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--out", required=True)
p.add_argument("--actor-insert", type=int, required=True,
               help="column index of base_target_pos_b in the NEW 177-D actor layout")
p.add_argument("--critic-insert", type=int, required=True,
               help="column index of base_target_pos_b in the NEW critic layout")
p.add_argument("--n", type=int, default=2, help="channel width to insert (base_target_pos_b = 2)")
args = p.parse_args()

ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
sd = ckpt["model_state_dict"]


def pad_cols(key: str, insert_at: int, n: int):
    w = sd[key]
    assert w.dim() == 2, (key, w.shape)
    assert 0 <= insert_at <= w.shape[1], (key, w.shape, insert_at)
    zeros = torch.zeros(w.shape[0], n, dtype=w.dtype)
    sd[key] = torch.cat([w[:, :insert_at], zeros, w[:, insert_at:]], dim=1)
    print(f"[warmstart] {key}: {tuple(w.shape)} -> {tuple(sd[key].shape)} (zeros @ col {insert_at})")


# rsl_rl ActorCritic MLPs: first Linear of each head. Fail loudly if the naming drifts.
actor_keys = [k for k in sd if k.startswith("actor.") and k.endswith("0.weight")]
critic_keys = [k for k in sd if k.startswith("critic.") and k.endswith("0.weight")]
assert len(actor_keys) == 1 and len(critic_keys) == 1, (actor_keys, critic_keys)
pad_cols(actor_keys[0], args.actor_insert, args.n)
pad_cols(critic_keys[0], args.critic_insert, args.n)

# Optimizer state shapes no longer match; a fresh optimizer is correct for a warm start.
ckpt.pop("optimizer_state_dict", None)
ckpt["iter"] = 0

os.makedirs(os.path.dirname(args.out), exist_ok=True)
torch.save(ckpt, args.out)
print(f"[warmstart] wrote {args.out}")
