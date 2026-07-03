"""Shape-tolerant checkpoint loading for inference/export entry points.

2026-07-03: the DeployParity CRITIC obs dropped the vestigial ``base_target_pos_b`` (2 dims,
``HOPECriticDeployParityCfg``), so every pre-change DeployParity/RealSensor checkpoint (including the
deployed p4 lineage) fails rsl_rl's strict ``load_state_dict`` on the critic first layer when loaded
into the current env cfg. play.py (ONNX re-export) and eval_deterministic.py only ever need the ACTOR,
so they fall back to an actor-preserving partial load instead of dying. train.py deliberately does NOT
use this helper — resuming training from a shape-mismatched checkpoint must stay a loud error.
"""

from __future__ import annotations


def load_actor_tolerant(runner, path: str) -> None:
    """``runner.load(path)`` with a shape-tolerant fallback for inference-only callers.

    Tries the normal strict load first (current-generation checkpoints, full state incl. optimizer).
    On a shape mismatch (pre-2026-07-03 critic layout), re-loads the raw checkpoint, drops every
    tensor whose name/shape does not match the current policy, and loads the rest non-strictly —
    the actor (and its normalizer, if any) survive intact; the critic re-initializes. Loudly warns,
    because a policy loaded this way must never be used to RESUME TRAINING (fresh critic + no
    optimizer state), only for rollout/eval/export.
    """
    try:
        runner.load(path)
        return
    except RuntimeError as e:
        import torch

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        cur = runner.alg.policy.state_dict()
        keep = {k: v for k, v in sd.items() if k in cur and cur[k].shape == v.shape}
        dropped = sorted(set(sd) - set(keep))
        actor_dropped = [k for k in dropped if not k.startswith("critic")]
        if actor_dropped:
            # The mismatch is NOT confined to the critic — a partial load would silently corrupt the
            # actor. Re-raise the original strict error instead.
            raise RuntimeError(
                f"checkpoint/actor shape mismatch (not just the critic): {actor_dropped}"
            ) from e
        runner.alg.policy.load_state_dict(keep, strict=False)
        print(
            f"[compat] strict checkpoint load failed (pre-2026-07-03 critic layout: base_target_pos_b "
            f"was removed from the DeployParity critic). Loaded {len(keep)}/{len(sd)} tensors, dropped "
            f"{dropped}. ACTOR intact — do NOT resume training from this load.",
            flush=True,
        )
