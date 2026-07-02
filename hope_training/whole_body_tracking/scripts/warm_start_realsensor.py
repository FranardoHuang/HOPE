"""Partial warm-start: model_15200 (`full`, 180-D actor) -> `real_sensor_only` (175-D actor).

The real_sensor_only actor obs drops two world-frame base-position terms and reframes a third
(see the obs-redesign report). Only the ACTOR-input-facing parameters change shape:

    actor.0.weight              [512, 180] -> [512, 175]   (column-select the kept obs dims)
    actor_obs_normalizer.*      [.., 180]  -> [.., 175]     (running_mean / running_var)

Everything else (deeper actor layers, the ENTIRE critic, which is privileged and unchanged) keeps
its learned weights. The first actor layer is COLUMN-REMAPPED rather than reinitialized: the columns
for the kept obs dims are copied from the trained weight, the dropped columns are removed, and the
reframed racket_target_pos_b columns (same slot, new meaning) are reused as a warm init. The optimizer
is reset (fresh Adam) — standard for fine-tuning a resized network.

OLD actor obs layout (180), with the dropped/kept index ranges:
    command              [  0: 62]  keep
    motion_anchor_pos_b  [ 62: 65]  DROP  (needs world base position)
    motion_anchor_ori_b  [ 65: 71]  keep
    base_ang_vel         [ 71: 74]  keep
    joint_pos            [ 74:105]  keep
    joint_vel            [105:136]  keep
    actions              [136:167]  keep
    projected_gravity    [167:170]  keep
    base_target_pos_b    [170:172]  DROP  (needs world base position)
    racket_target_pos_b  [172:175]  keep  (reframed to racket-FK-relative; columns reused as init)
    racket_target_vel_w  [175:178]  keep
    time_to_strike       [178:179]  keep
    swing_type           [179:180]  keep
=> dropped input indices = {62, 63, 64, 170, 171};  175 kept.

USAGE (run inside the Isaac/torch training env):

    # 1) inspect the checkpoint first (no writes) — confirm the remap targets:
    python scripts/warm_start_realsensor.py --old-ckpt <.../model_15200.pt> --dry-run

    # 2) write the warm-started checkpoint:
    python scripts/warm_start_realsensor.py \
        --old-ckpt logs/rsl_rl/agibot_a3_hope/2026-06-28_23-01-24_phase050_perclippos_scratch/model_15200.pt \
        --out      logs/rsl_rl/warmstart/model_15200_realsensor.pt

    # 3) fine-tune from it (fresh run; the env provides the 175-D actor obs):
    python scripts/train.py task=HOPEPingPongRealSensor algo=ppo headless=true \
        checkpoint_path=logs/rsl_rl/warmstart/model_15200_realsensor.pt
"""

from __future__ import annotations

import argparse


# OLD (`full`) actor obs dim and the input indices to DROP for `real_sensor_only`.
OLD_ACTOR_DIM = 180
DROP_INDICES = (62, 63, 64, 170, 171)  # motion_anchor_pos_b[62:65] + base_target_pos_b[170:172]
# Keys whose tensors may carry the actor-obs axis. We only remap these (never the critic).
ACTOR_KEY_HINTS = ("actor", "normaliz")


def _keep_indices(old_dim: int, drop) -> list[int]:
    drop = set(drop)
    return [i for i in range(old_dim) if i not in drop]


def _is_tensor(x) -> bool:
    import torch

    return isinstance(x, torch.Tensor)


def _remap_tensor(key: str, val, keep_idx, old_dim: int, log: list):
    """Return a (possibly column-selected) tensor + record the action. Only slices the LAST axis,
    and only when this is an actor-obs-facing param (key hint + the last axis equals old_dim)."""
    import torch

    if not _is_tensor(val) or val.dim() == 0:
        return val
    klow = key.lower()
    is_actor_obs_facing = any(h in klow for h in ACTOR_KEY_HINTS) and val.shape[-1] == old_dim
    if is_actor_obs_facing:
        new = val.index_select(-1, torch.tensor(keep_idx, device=val.device))
        log.append(("REMAP", key, tuple(val.shape), tuple(new.shape)))
        return new
    # Safety: a non-actor tensor that happens to carry the old actor dim on its last axis — do NOT
    # slice it (would corrupt the critic), but flag it so a surprising layout is never silent.
    if val.shape[-1] == old_dim and not any(h in klow for h in ACTOR_KEY_HINTS):
        log.append(("SKIP-180", key, tuple(val.shape), tuple(val.shape)))
    return val


def _walk_and_remap(obj, keep_idx, old_dim, log, prefix=""):
    """Recurse into the checkpoint (dicts of tensors / nested state_dicts) and remap in place."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            kk = f"{prefix}{k}"
            if _is_tensor(v):
                out[k] = _remap_tensor(kk, v, keep_idx, old_dim, log)
            elif isinstance(v, dict):
                out[k] = _walk_and_remap(v, keep_idx, old_dim, log, prefix=kk + ".")
            else:
                out[k] = v
        return out
    return obj


def _reset_optimizer(ckpt: dict, log: list):
    """Fresh Adam for the warm-start: keep the param-group structure (param count is unchanged) but
    clear the per-parameter moments (they would mismatch the resized actor-input layer)."""
    if "optimizer_state_dict" in ckpt and isinstance(ckpt["optimizer_state_dict"], dict):
        pg = ckpt["optimizer_state_dict"].get("param_groups", [])
        ckpt["optimizer_state_dict"] = {"state": {}, "param_groups": pg}
        log.append(("OPT-RESET", "optimizer_state_dict", "moments cleared", f"{len(pg)} param_group(s) kept"))


def main():
    import torch

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old-ckpt", required=True, help="path to the `full` rsl_rl checkpoint (model_15200.pt)")
    ap.add_argument("--out", default=None, help="path to write the warm-started checkpoint")
    ap.add_argument("--old-actor-dim", type=int, default=OLD_ACTOR_DIM)
    ap.add_argument("--drop-indices", default=",".join(map(str, DROP_INDICES)),
                    help="comma-separated actor-obs input indices to drop")
    ap.add_argument("--reset-iter", action="store_true", default=True, help="reset the iteration counter to 0")
    ap.add_argument("--dry-run", action="store_true", help="print the checkpoint layout + planned remap, write nothing")
    args = ap.parse_args()

    drop = tuple(int(x) for x in str(args.drop_indices).split(",") if x != "")
    keep_idx = _keep_indices(args.old_actor_dim, drop)
    new_dim = len(keep_idx)

    print(f"[warm-start] old actor dim = {args.old_actor_dim}  drop = {drop}  -> new actor dim = {new_dim}")
    assert new_dim == args.old_actor_dim - len(set(drop)), "keep/drop bookkeeping mismatch"

    ckpt = torch.load(args.old_ckpt, map_location="cpu", weights_only=False)
    print(f"[warm-start] loaded {args.old_ckpt}")
    print(f"[warm-start] top-level keys: {list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt).__name__}")

    if args.dry_run:
        # Full layout dump so the remap targets can be eyeballed before writing anything.
        def _dump(d, prefix=""):
            for k, v in d.items():
                kk = f"{prefix}{k}"
                if _is_tensor(v):
                    star = "  <-- carries 180 axis" if (hasattr(v, "shape") and args.old_actor_dim in tuple(v.shape)) else ""
                    print(f"    {kk:55s} {tuple(v.shape)}{star}")
                elif isinstance(v, dict):
                    print(f"    {kk}/ (dict, {len(v)} entries)")
                    _dump(v, kk + ".")
        if isinstance(ckpt, dict):
            _dump(ckpt)
        print("[warm-start] DRY RUN — nothing written. Re-run without --dry-run (and with --out) to write.")
        return

    if args.out is None:
        ap.error("--out is required unless --dry-run")

    log: list = []
    new_ckpt = _walk_and_remap(ckpt, keep_idx, args.old_actor_dim, log)
    _reset_optimizer(new_ckpt, log)
    if args.reset_iter and isinstance(new_ckpt, dict) and "iter" in new_ckpt:
        new_ckpt["iter"] = 0
        log.append(("ITER-RESET", "iter", "-> 0", ""))

    import os
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    torch.save(new_ckpt, args.out)

    print("\n[warm-start] actions:")
    n_remap = 0
    for kind, key, a, b in log:
        print(f"  {kind:10s} {key:50s} {a} -> {b}")
        n_remap += kind == "REMAP"
    if n_remap == 0:
        print("  !! WARNING: no actor-obs-facing param was remapped. Check --old-ckpt / --old-actor-dim "
              "with --dry-run; the layout may differ from the expected 180-D actor.")
    skipped = [l for l in log if l[0] == "SKIP-180"]
    if skipped:
        print(f"  !! NOTE: {len(skipped)} non-actor tensor(s) also carry a {args.old_actor_dim} axis and were "
              f"left untouched (see SKIP-180 above) — verify none is actually an actor-input layer.")
    print(f"\n[warm-start] wrote {args.out}")
    print("[warm-start] resume (fresh fine-tune run):")
    print(f"    python scripts/train.py task=HOPEPingPongRealSensor algo=ppo headless=true "
          f"checkpoint_path={args.out}")


if __name__ == "__main__":
    main()
