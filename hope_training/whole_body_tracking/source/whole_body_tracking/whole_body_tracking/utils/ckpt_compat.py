"""Shape-tolerant checkpoint loading for inference/export entry points.

2026-07-03: the DeployParity CRITIC obs dropped the vestigial ``base_target_pos_b`` (2 dims,
``HOPECriticDeployParityCfg``), so every pre-change DeployParity/RealSensor checkpoint (including the
deployed p4 lineage) fails rsl_rl's strict ``load_state_dict`` on the critic first layer when loaded
into the current env cfg. play.py (ONNX re-export) and eval_deterministic.py only ever need the ACTOR,
so they fall back to an actor-preserving partial load instead of dying. train.py deliberately does NOT
use this helper — resuming training from a shape-mismatched checkpoint must stay a loud error.
"""

from __future__ import annotations


def _validate_and_restore_actor_normalizer(runner, checkpoint: dict, *, restore: bool) -> bool:
    """Validate checkpoint/config normalization truth and optionally restore its state.

    In the rsl_rl version used by HOPE the actor normalizer belongs to the runner, not the
    ActorCritic module.  A shape-tolerant policy load therefore has to restore the top-level
    ``obs_norm_state_dict`` explicitly; otherwise export silently bakes a fresh normalizer.
    """
    import torch

    trained_with_obs_norm = isinstance(checkpoint, dict) and "obs_norm_state_dict" in checkpoint
    configured = bool(getattr(runner, "empirical_normalization", False))
    if trained_with_obs_norm != configured:
        raise RuntimeError(
            "checkpoint/config observation-normalization mismatch: "
            f"checkpoint trained_with_obs_norm={trained_with_obs_norm}, "
            f"runner empirical_normalization={configured}. Refusing a semantic actor mismatch."
        )
    if not trained_with_obs_norm:
        return False

    normalizer = getattr(runner, "obs_normalizer", None)
    if normalizer is None:
        raise RuntimeError("normalized checkpoint but runner has no obs_normalizer")
    state = checkpoint["obs_norm_state_dict"]
    required = ("_mean", "_std", "count")
    missing = [key for key in required if key not in state]
    if missing:
        raise RuntimeError(f"obs_norm_state_dict missing keys: {missing}")
    mean = state["_mean"]
    std = state["_std"]
    count = state["count"]
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise RuntimeError("obs_norm_state_dict contains NaN/Inf")
    if mean.numel() == 0 or mean.shape != std.shape or not torch.all(std > 0):
        raise RuntimeError(
            f"invalid obs normalizer shapes/scale: mean={tuple(mean.shape)} std={tuple(std.shape)}"
        )
    if not torch.isfinite(count).all() or float(count.item()) <= 0.0:
        raise RuntimeError(f"invalid obs normalizer count: {count!r}")
    current = normalizer.state_dict()
    if "_mean" not in current or current["_mean"].shape != mean.shape:
        got = tuple(current.get("_mean", torch.empty(0)).shape)
        raise RuntimeError(
            f"checkpoint/runner obs normalizer dimension mismatch: checkpoint={tuple(mean.shape)} "
            f"runner={got}"
        )
    if restore:
        normalizer.load_state_dict(state)
    return True


def load_actor_tolerant(runner, path: str) -> bool:
    """``runner.load(path)`` with a shape-tolerant fallback for inference-only callers.

    Tries the normal strict load first (current-generation checkpoints, full state incl. optimizer).
    On a shape mismatch (pre-2026-07-03 critic layout), re-loads the raw checkpoint, drops every
    tensor whose name/shape does not match the current policy, and loads the rest non-strictly —
    the actor (and its normalizer, if any) survive intact; the critic re-initializes. Loudly warns,
    because a policy loaded this way must never be used to RESUME TRAINING (fresh critic + no
    optimizer state), only for rollout/eval/export.
    """
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    trained_with_obs_norm = _validate_and_restore_actor_normalizer(
        runner, ckpt, restore=False
    )
    try:
        runner.load(path)
        # Strict load restores runner.obs_normalizer. Validate its checkpoint state once more so
        # callers can safely pass that exact module to the ONNX exporter. Restore explicitly too:
        # runner lineages have differed on whether their generic load() owns the top-level state.
        _validate_and_restore_actor_normalizer(runner, ckpt, restore=True)
        return trained_with_obs_norm
    except RuntimeError as e:
        sd = ckpt.get("model_state_dict", ckpt)
        cur = runner.alg.policy.state_dict()
        keep = {k: v for k, v in sd.items() if k in cur and cur[k].shape == v.shape}
        dropped = sorted(set(sd) - set(keep))
        actor_dropped = [k for k in dropped if not k.startswith("critic")]
        actor_missing = sorted(
            key for key in cur
            if not key.startswith("critic") and key not in keep
        )
        if actor_dropped or actor_missing:
            # The mismatch is NOT confined to the critic — a partial load would silently corrupt the
            # actor. Re-raise the original strict error instead.
            raise RuntimeError(
                "checkpoint/actor key or shape mismatch (not just the critic): "
                f"dropped_from_checkpoint={actor_dropped}, missing_from_checkpoint={actor_missing}"
            ) from e
        runner.alg.policy.load_state_dict(keep, strict=False)
        _validate_and_restore_actor_normalizer(runner, ckpt, restore=True)
        print(
            f"[compat] strict checkpoint load failed (pre-2026-07-03 critic layout: base_target_pos_b "
            f"was removed from the DeployParity critic). Loaded {len(keep)}/{len(sd)} tensors, dropped "
            f"{dropped}. ACTOR intact — do NOT resume training from this load.",
            flush=True,
        )
        return trained_with_obs_norm
