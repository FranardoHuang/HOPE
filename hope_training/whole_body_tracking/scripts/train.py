"""Hydra training entry for HOPE Agibot A3 WBC (106B-Final-Project style).

Pick the task/algo YAML on the command line and override any field:

    python scripts/train.py task=HOPEPingPongDeployParity algo=ppo headless=true \
        registry_name=<entity>/wandb-registry-motions/hope_forehand

    python scripts/train.py task=TrackingFlat algo=ppo num_envs=2048 max_iterations=20000 \
        registry_name=<org>/wandb-registry-motions/hope_forehand

Tune by editing cfg/task/*.yaml (env / reward / racket / DR) and cfg/algo/ppo.yaml (PPO). This
script reuses BeyondMimic's training mechanics (Isaac Lab + rsl_rl). Local video-generated `.npz`
motions are first-class inputs; the WandB motion registry is an optional sharing/publishing layer.
The legacy `scripts/rsl_rl/train.py --task=... --registry_name=...` still works too.
"""

import pathlib
import sys

import hydra
from omegaconf import ListConfig, OmegaConf


def dump_pickle(filename: str, data):
    """Compatibility helper for IsaacLab builds that no longer expose dump_pickle."""
    import os
    import pickle

    if not filename.endswith("pkl"):
        filename += ".pkl"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(data, f)


# --------------------------------------------------------------------------- #
# Task YAML -> Isaac Lab env cfg overrides (only keys present in the YAML are applied).
# --------------------------------------------------------------------------- #
def _get(node, key, default=None):
    try:
        return node.get(key, default)
    except Exception:
        return default


def _as_bool(x):
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in ("true", "1", "yes")


def _is_noneish(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in ("", "none", "null")
    return False


def _configured_items(primary, secondary=None) -> list:
    items = []
    if not _is_noneish(primary):
        if isinstance(primary, (list, tuple, ListConfig)):
            items.extend(primary)
        else:
            items.append(primary)
    if not _is_noneish(secondary):
        if isinstance(secondary, (list, tuple, ListConfig)):
            items.extend(secondary)
        else:
            items.append(secondary)
    return [item for item in items if not _is_noneish(item)]


def _normalize_registry_name(name) -> str:
    reg = str(name)
    if ":" not in reg:
        reg += ":latest"
    return reg


def _motion_clip_name_from_path(value) -> str | None:
    items = _configured_items(value)
    if not items:
        return None
    parts = [p for p in str(items[0]).replace("\\", "/").split("/") if p]
    if not parts:
        return None
    if parts[-1] == "motion.npz" and len(parts) >= 2:
        return parts[-2].split(":")[0] or None
    return pathlib.PurePath(parts[-1]).stem or None


def _resolve_local_motion_files(primary, secondary=None, cwd: pathlib.Path | None = None) -> list[str]:
    files = []
    base = pathlib.Path.cwd() if cwd is None else pathlib.Path(cwd)
    for value in _configured_items(primary, secondary):
        path = pathlib.Path(str(value)).expanduser()
        if not path.is_absolute():
            path = base / path
        if not path.is_file():
            raise FileNotFoundError(f"[train.py] motion_file not found: {path}")
        files.append(str(path))
    return files


def _download_registry_motion_files(primary, secondary=None) -> tuple[list[str], list[str]]:
    registries = [_normalize_registry_name(value) for value in _configured_items(primary, secondary)]
    if not registries:
        raise RuntimeError(
            "[train.py] No reference motion configured. Pass motion_file=/path/to/motion.npz "
            "(and optional motion_file_2=/path/to/backhand.npz) for the local path, or pass "
            "registry_name=<org>/wandb-registry-motions/<name>."
        )
    # Import lazily and AFTER the guard: the no-WandB local path must never require wandb, and a
    # missing-motion misconfiguration should raise the guidance error above, not ModuleNotFoundError.
    import wandb

    api = wandb.Api()
    motion_files = []
    for reg in registries:
        art = api.artifact(reg)
        # Provenance: record exactly which artifact version/digest the run trains on (the registry
        # alias is mutable, e.g. ':latest' can move between runs).
        print(f"[train.py] motion clip: {reg} -> {art.source_qualified_name} (digest {art.digest[:12]})", flush=True)
        motion_files.append(str(pathlib.Path(art.download()) / "motion.npz"))
    return motion_files, registries


def resolve_motion_sources(cfg, *, cwd: pathlib.Path | None = None) -> tuple[list[str], list[str]]:
    """Resolve local or registry motion sources for train/play.

    Local files are intentionally all-or-nothing: once ``motion_file`` is set, training does not touch the
    registry. Use ``motion_file_2`` (or a list-valued ``motion_file``) for the unified forehand/backhand
    local workflow.
    """
    local_files = _resolve_local_motion_files(
        _get(cfg, "motion_file"),
        _get(cfg, "motion_file_2"),
        cwd=cwd,
    )
    if local_files:
        return local_files, []

    task = _get(cfg, "task")
    registry_name = _get(cfg, "registry_name") if _get(cfg, "registry_name") is not None else _get(task, "registry_name")
    reg2 = _get(cfg, "registry_name_2") if _get(cfg, "registry_name_2") is not None else _get(task, "registry_name_2")
    return _download_registry_motion_files(registry_name, reg2)


class _OverrideError(AttributeError):
    """Raised when the task YAML asks to override an attribute the composed env cfg does not have."""


def _require(cond, target):
    # The YAML explicitly set a value, but the target attribute is missing on the composed env cfg.
    # That is NEVER a benign no-op: either a STALE/shadowed whole_body_tracking was imported (so the
    # cfg classes differ from the working tree) or the Hydra base groups failed to compose. Fail loud
    # instead of silently dropping the override (the old behaviour that hid the std/curriculum edits).
    if not cond:
        raise _OverrideError(
            f"[train.py] task YAML overrides '{target}' but the composed env cfg has no such attribute. "
            f"Check the '[train.py] env cfg source:' line above — if it points into site-packages rather "
            f"than your working tree, a stale install is shadowing the source (fix PYTHONPATH ordering / "
            f"reinstall editable). Otherwise the Hydra base-group composition for this task failed."
        )


def _set_attr(obj, attr, val, cast, applied, where):
    if val is None:
        return  # key absent from YAML -> keep the code default (documented contract)
    _require(hasattr(obj, attr), f"{where}.{attr}")
    setattr(obj, attr, cast(val))
    applied.append(f"{where}.{attr}={cast(val)!r}")


def _set_range(obj, attr, val, applied, where):
    if val is None:
        return
    _require(hasattr(obj, attr), f"{where}.{attr}")
    rng = (float(val[0]), float(val[1]))
    setattr(obj, attr, rng)
    applied.append(f"{where}.{attr}={rng}")


def _set_vec3(obj, attr, val, applied, where):
    if val is None:
        return
    _require(hasattr(obj, attr), f"{where}.{attr}")
    vec = (float(val[0]), float(val[1]), float(val[2]))
    setattr(obj, attr, vec)
    applied.append(f"{where}.{attr}={vec}")


def _set_reward(rewards, name, weight, std, applied):
    if weight is None and std is None:
        return  # this reward term is not overridden by the YAML -> keep code defaults
    _require(hasattr(rewards, name), f"rewards.{name}")
    term = getattr(rewards, name)
    if weight is not None:
        term.weight = float(weight)
        applied.append(f"rewards.{name}.weight={float(weight)}")
    if std is not None:
        _require("std" in term.params, f"rewards.{name}.params['std']")
        term.params["std"] = float(std)
        applied.append(f"rewards.{name}.params.std={float(std)}")


def _check_unknown_keys(node, known, where):
    # _require guards one direction (YAML sets a key, env cfg lacks the attribute); this guards the
    # other: a key present under the node that no _set_attr/_set_range call below ever reads would be
    # a SILENT no-op (this is exactly how r3_P2_product's task.racket.target_noise_white=0.0019 /
    # target_noise_ar1_sigma=0.0052 / vb_spin_mode=minimize CLI overrides got dropped on 2026-07-03).
    if node is None:
        return
    try:
        present = list(node.keys())
    except Exception:
        return
    unknown = sorted(str(k) for k in present if str(k) not in known)
    if unknown:
        raise _OverrideError(
            f"[train.py] {where} sets key(s) {unknown} that the override translation layer does not "
            f"consume — they would be silently ignored. Add each to the whitelist AND a "
            f"_set_attr/_set_range call in _apply_task_overrides, or remove it from the YAML/CLI."
        )


# YAML keys under `racket:` that target the RacketTargetCommandCfg (used to decide whether the task
# actually requested racket overrides before requiring the command to exist).
_RACKET_KEYS = (
    "strike_phase", "strike_phase_by_motion", "strike_window_s", "strike_success_pos_thresh",
    "pos_x_range", "pos_y_range", "pos_z_range", "racket_pos_y_abs_range", "pos_range_per_clip",
    "vel_x_range", "vel_y_range", "vel_z_range", "vel_range_per_clip",
    "base_target_x_range", "base_target_y_range",
    "normal_mode", "forehand_on_negative_y", "mount_normal_axis", "mount_normal_sign",
    "target_mode", "ref_perturb_pos", "ref_perturb_vel", "ref_perturb_normal",
    "ref_perturb_curriculum_steps", "ref_perturb_curriculum_start", "ref_perturb_success_gated",
    "ref_perturb_advance_threshold", "ref_perturb_advance_rate", "ref_vel_scale", "ref_vel_scale_by_motion",
    "debug_reward_logging",
    "clean_reference_strike_velocity", "clean_strike_vel_window",
    "adaptive_sigma", "sigma_update_every", "sigma_ema_scale",
    "sigma_pos_min", "sigma_pos_max", "sigma_vel_min", "sigma_vel_max",
    # HER-style achieved-target replay (mixture sampling from previously-achieved strike states).
    "achieved_target_mix_prob", "achieved_buffer_size", "achieved_min_fill",
    "achieved_jitter_pos", "achieved_jitter_vel", "achieved_clamp_inflate",
    # A1 target latency & time-variance (actor-visible delay, SMASH tts-decaying jitter,
    # mid-swing target refinement). Defaults OFF; byte-identical baseline.
    "target_delay_steps", "target_jitter_pos_per_s", "target_jitter_vel_per_s",
    "midswing_resample_prob", "midswing_resample_tts_floor",
    # A1v2 calibrated mocap-degradation channels (white/AR1 noise, frame dropout, per-swing bias).
    "target_noise_white", "target_noise_ar1_sigma", "target_noise_ar1_rho",
    "target_dropout_prob", "target_post_strike_dropout_s", "target_bias_per_swing",
    # Tier-1 virtual ball: incoming-ball sampling boxes + outgoing-spin objective.
    "vb_spin_mode", "vb_spin_min_sigma", "vb_spin_abs_max",
    "vb_vel_x_range", "vb_vel_y_range", "vb_vel_z_range",
    # translated below but previously missing from this whitelist
    "strike_phase_per_clip", "base_couple_blend", "base_couple_max_offset",
    # Stage-1 question bank (fixed contact point, inverse-solved face+velocity targets) + the
    # face-command reward re-anchor / +4 actor obs channel (normal + rho placeholder, 175->179).
    "question_bank", "face_command", "face_command_obs",
    # SHADOW physical ball + table (flag-gated, METRICS-ONLY engine-vs-analytic landing
    # cross-check; requires the virtual-ball task variant). shadow_ball.py.
    "shadow_ball", "shadow_table",
)

# YAML keys under `motion:` that target the MotionCommandCfg swing-entry structure
# (Phase-A multi-swing machinery: no-teleport wrap, stand-entry resets, pre-swing hold,
# A8 post-swing initial-state buffer).
_MOTION_KEYS = (
    "wrap_teleport", "stand_start_prob", "hold_steps_range", "stand_start_min_hold",
    "post_swing_start_prob", "post_swing_buffer_size", "post_swing_min_fill", "post_swing_min_hold",
    # deploy-parity mid-swing clip switch (018467a added the yaml key + MotionCommandCfg field but not
    # this whitelist/translation, so every run of the task yaml raised in _check_unknown_keys).
    "clip_switch_prob",
    # P2.4/R14 per-swing reference playback speed range (retiming).
    "speed_scale_range",
)


def _registry_clip_name(cfg):
    """Motion clip name used to key per-motion settings (e.g. ``strike_phase_by_motion``).

    Resolution order (most explicit wins): CLI ``registry_name`` -> explicit ``motion_file`` (its parent
    dir, the artifact folder such as ``hope_backhand:v0``) -> the task default ``registry_name``. Any
    registry path prefix and ``:version`` suffix are stripped, so the result is e.g. ``hope_forehand`` /
    ``hope_backhand``. Returns ``None`` when nothing is set. Shared by train/play/probe so all three pick
    the same per-clip strike phase.
    """
    reg = _get(cfg, "registry_name")  # CLI override (train/play): wins over the forehand default
    if not _is_noneish(reg):
        return str(reg).split("/")[-1].split(":")[0] or None
    mf = _motion_clip_name_from_path(_get(cfg, "motion_file"))
    if mf is not None:
        return mf
    task = _get(cfg, "task")
    reg = _get(task, "registry_name") if task is not None else None  # task default (forehand): last
    if not _is_noneish(reg):
        return str(reg).split("/")[-1].split(":")[0] or None
    return None


def _resolve_strike_phase(rk, clip_name):
    """Select the strike phase for the trained clip (paddle-contact frame is PER-CLIP).

    A single global ``strike_phase`` cannot serve both swings (the racket-tip speed peak lands at a
    different fraction in each clip, e.g. forehand ~0.46 vs backhand ~0.59), so ``strike_phase_by_motion``
    maps a motion-name substring (the registry clip name) to its contact phase; the most-specific
    (longest) matching key wins. Falls back to the scalar ``strike_phase`` when nothing matches or the
    clip is unknown. Returns ``(phase_or_None, note_or_None)``; ``note`` records which mapping fired.
    """
    by_motion = _get(rk, "strike_phase_by_motion")
    if by_motion is not None and clip_name:
        cn = str(clip_name).lower()
        matches = [(str(k).lower(), v) for k, v in by_motion.items()
                   if str(k).lower() in cn or cn in str(k).lower()]
        if matches:
            matches.sort(key=lambda kv: len(kv[0]), reverse=True)  # longest key = most specific
            k, v = matches[0]
            return float(v), f"racket_target.strike_phase<-by_motion[{k}]={float(v)} (clip={clip_name})"
    sp = _get(rk, "strike_phase")
    return (None if sp is None else float(sp)), None


# YAML clip-name -> clip_id index. MUST match RacketTargetCommand._clip_names (0=forehand, 1=backhand).
_CLIP_NAME_TO_ID = {"forehand": 0, "backhand": 1}


def _resolve_vel_range_per_clip(rk):
    """Build the optional PER-CLIP racket target-velocity tuple from the YAML ``vel_range_per_clip`` block.

    YAML (readable, keyed by swing name; each axis a [lo, hi] list)::

        vel_range_per_clip:
          forehand: {x: [1.5, 3.5], y: [-1.0, 1.0], z: [0.0, 1.5]}
          backhand: {x: [1.2, 2.4], y: [-1.0, 1.0], z: [0.0, 1.2]}

    Returns a tuple indexed by clip_id (0=forehand, 1=backhand) of ``((xlo,xhi),(ylo,yhi),(zlo,zhi))``,
    or ``None`` when the key is absent (-> keep the shared ``vel_*_range`` box; backward compatible).
    Forehand and backhand reference clips have different natural strike speeds, so a shared box overshoots
    the slower backhand. Mirrors the per-clip ``strike_phase_per_clip`` / ``ref_vel_scale_by_motion`` style.
    """
    block = _get(rk, "vel_range_per_clip")
    if block is None:
        return None
    by_id = {}
    for name in block:
        cid = _CLIP_NAME_TO_ID.get(str(name).lower())
        if cid is None:
            raise _OverrideError(
                f"racket.vel_range_per_clip: unknown clip name {name!r} (expected forehand/backhand)")
        axes = _get(block, name)

        def _r(ax):
            v = _get(axes, ax)
            if v is None:
                raise _OverrideError(f"racket.vel_range_per_clip[{name}]: missing '{ax}' [lo,hi] range")
            return (float(v[0]), float(v[1]))

        by_id[cid] = (_r("x"), _r("y"), _r("z"))
    return tuple(by_id[i] for i in range(len(by_id)))


def _resolve_pos_range_per_clip(rk):
    """Build the optional PER-CLIP racket target-POSITION tuple from the YAML ``pos_range_per_clip`` block.

    YAML (readable, keyed by swing name; each axis a [lo, hi] list, ADDED to the env origin; y is SIGNED)::

        pos_range_per_clip:
          forehand: {x: [0.50, 0.62], y: [-0.45, -0.20], z: [0.72, 0.98]}
          backhand: {x: [0.50, 0.62], y: [ 0.20,  0.45], z: [1.05, 1.30]}

    Returns a tuple indexed by clip_id (0=forehand, 1=backhand) of ``((xlo,xhi),(ylo,yhi),(zlo,zhi))``,
    or ``None`` when the key is absent (-> keep the shared ``pos_*_range`` box; backward compatible).
    Mirrors ``vel_range_per_clip``: lets each clip's target track its own reference strike point (e.g. the
    backhand sits higher/forward at strike_phase 0.50, so a shared z<=1.05 box makes it unreachable).
    """
    block = _get(rk, "pos_range_per_clip")
    if block is None:
        return None
    by_id = {}
    for name in block:
        cid = _CLIP_NAME_TO_ID.get(str(name).lower())
        if cid is None:
            raise _OverrideError(
                f"racket.pos_range_per_clip: unknown clip name {name!r} (expected forehand/backhand)")
        axes = _get(block, name)

        def _r(ax):
            v = _get(axes, ax)
            if v is None:
                raise _OverrideError(f"racket.pos_range_per_clip[{name}]: missing '{ax}' [lo,hi] range")
            return (float(v[0]), float(v[1]))

        by_id[cid] = (_r("x"), _r("y"), _r("z"))
    return tuple(by_id[i] for i in range(len(by_id)))


def _resolve_ref_vel_scale(rk, clip_name):
    """Select the reference racket-velocity scale for the trained clip (PER-CLIP, like strike_phase).

    ``ref_vel_scale`` <1.0 trains a slower-than-reference hit. It was tuned to TAME the violent forehand
    (~6 m/s tip) — but the backhand is already a gentle swing (~3.3 m/s tip / ~1.8 m/s at the mount), so
    down-scaling it shrinks the velocity TARGET into the body-jitter floor AND pits the imitation prior
    (wants full speed) against the velocity goal (wants 0.6x). So the scale must be per-clip:
    ``ref_vel_scale_by_motion`` maps a motion-name substring to its scale (longest match wins); falls back
    to the scalar ``ref_vel_scale``. Returns ``(scale_or_None, note_or_None)``.
    """
    by_motion = _get(rk, "ref_vel_scale_by_motion")
    if by_motion is not None and clip_name:
        cn = str(clip_name).lower()
        matches = [(str(k).lower(), v) for k, v in by_motion.items()
                   if str(k).lower() in cn or cn in str(k).lower()]
        if matches:
            matches.sort(key=lambda kv: len(kv[0]), reverse=True)  # longest key = most specific
            k, v = matches[0]
            return float(v), f"racket_target.ref_vel_scale<-by_motion[{k}]={float(v)} (clip={clip_name})"
    rv = _get(rk, "ref_vel_scale")
    return (None if rv is None else float(rv)), None


def _apply_task_overrides(env_cfg, task, clip_name=None):
    """Apply cfg/task/<name>.yaml overrides (incl. the composed base/ groups) onto the env cfg.

    Returns the list of applied "attr=value" strings (logged by the caller). Keys absent from the
    YAML are left at the code default; keys present whose target attribute is missing RAISE (so a
    stale/shadowed cfg or a broken Hydra composition can never silently swallow an override).
    """
    applied = []

    # env base (num_envs is applied earlier via parse_env_cfg). Read every value through _get so the
    # logic works on both OmegaConf nodes (runtime) and plain dicts (unit tests).
    env = _get(task, "env")
    if env is not None:
        es = _get(env, "env_spacing")
        if es is not None:
            env_cfg.scene.env_spacing = float(es)
            applied.append(f"scene.env_spacing={float(es)}")
        els = _get(env, "episode_length_s")
        if els is not None:
            env_cfg.episode_length_s = float(els)
            applied.append(f"episode_length_s={float(els)}")

    # sim base (control frequency = 1 / (dt * decimation))
    sim = _get(task, "sim")
    if sim is not None:
        dt = _get(sim, "dt")
        if dt is not None:
            env_cfg.sim.dt = float(dt)
            applied.append(f"sim.dt={float(dt)}")
        dec = _get(sim, "decimation")
        if dec is not None:
            env_cfg.decimation = int(dec)
            env_cfg.sim.render_interval = env_cfg.decimation  # keep render in step with decimation
            applied.append(f"decimation={int(dec)}")

    # motion command (swing-entry structure): no-teleport wrap / stand-entry resets / pre-swing hold
    mt = _get(task, "motion")
    _check_unknown_keys(mt, _MOTION_KEYS, "task.motion")
    if mt is not None:
        provided = [k for k in _MOTION_KEYS if _get(mt, k) is not None]
        if provided:
            _require(hasattr(env_cfg.commands, "motion"),
                     f"commands.motion (task YAML sets motion keys {provided})")
            M = env_cfg.commands.motion
            _set_attr(M, "wrap_teleport", _get(mt, "wrap_teleport"), _as_bool, applied, "commands.motion")
            _set_attr(M, "stand_start_prob", _get(mt, "stand_start_prob"), float, applied, "commands.motion")
            _set_attr(M, "hold_steps_range", _get(mt, "hold_steps_range"),
                      lambda v: tuple(int(x) for x in v), applied, "commands.motion")
            _set_attr(M, "stand_start_min_hold", _get(mt, "stand_start_min_hold"), int, applied, "commands.motion")
            _set_attr(M, "post_swing_start_prob", _get(mt, "post_swing_start_prob"), float, applied, "commands.motion")
            _set_attr(M, "post_swing_buffer_size", _get(mt, "post_swing_buffer_size"), int, applied, "commands.motion")
            _set_attr(M, "post_swing_min_fill", _get(mt, "post_swing_min_fill"), int, applied, "commands.motion")
            _set_attr(M, "post_swing_min_hold", _get(mt, "post_swing_min_hold"), int, applied, "commands.motion")
            _set_attr(M, "clip_switch_prob", _get(mt, "clip_switch_prob"), float, applied, "commands.motion")
            _set_attr(M, "speed_scale_range", _get(mt, "speed_scale_range"),
                      lambda v: tuple(float(x) for x in v), applied, "commands.motion")

    rw = _get(task, "rewards")
    if rw is not None:
        R = env_cfg.rewards
        _set_reward(R, "racket_position", _get(rw, "racket_position_weight"), _get(rw, "racket_position_std"), applied)
        # Ablation B: swap the racket-position term to the no-swing-through (static strike-point) variant.
        if _as_bool(_get(rw, "racket_position_static", False)) and _get(rw, "racket_position_static") is not None:
            from whole_body_tracking.tasks.tracking import mdp as _mdp
            _require(hasattr(R, "racket_position"), "rewards.racket_position")
            R.racket_position.func = _mdp.racket_position_tracking_static_exp
            applied.append("rewards.racket_position.func=racket_position_tracking_static_exp")
        _set_reward(R, "racket_velocity", _get(rw, "racket_velocity_weight"), _get(rw, "racket_velocity_std"), applied)
        _set_reward(R, "racket_normal", _get(rw, "racket_normal_weight"), _get(rw, "racket_normal_std"), applied)
        _set_reward(R, "base_position", _get(rw, "base_position_weight"), _get(rw, "base_position_std"), applied)
        # Between-swing recovery: positive ready-stance reward during the pre-swing hold (deploy-parity).
        _set_reward(R, "hold_ready", _get(rw, "hold_ready_weight"), _get(rw, "hold_ready_std"), applied)
        _hr_reach = _get(rw, "hold_ready_reach")
        if _hr_reach is not None:
            _require(hasattr(R, "hold_ready"), "rewards.hold_ready")
            R.hold_ready.params["reach"] = float(_hr_reach)
            applied.append(f"rewards.hold_ready.params.reach={float(_hr_reach)}")
        # P2.4 PACE-style smooth deceleration (flag-gated, default weight 0.0 = OFF): pseudo base-speed
        # command proportional to the remaining planar racket->target error. REWARD-side only (the
        # frozen 175-D actor obs contract is untouched).
        _set_reward(R, "base_decel", _get(rw, "base_decel_weight"), _get(rw, "base_decel_std"), applied)
        for _pk, _yk in (("v_gain", "base_decel_v_gain"), ("v_max", "base_decel_v_max")):
            _bd = _get(rw, _yk)
            if _bd is not None:
                _require(hasattr(R, "base_decel"), "rewards.base_decel")
                R.base_decel.params[_pk] = float(_bd)
                applied.append(f"rewards.base_decel.params.{_pk}={float(_bd)}")
        # R16 (franco 2026-07-04): free the racket wrist from ORIENTATION mimic. Config-level only —
        # drop the racket-mount link from the body lists of the two orientation-imitation terms;
        # position / linear-velocity mimic keep the swing path, and the face orientation is then
        # shaped by the racket_normal reward alone (commanded normal at contract v3).
        if _get(rw, "free_wrist_ori_mimic") is not None and _as_bool(_get(rw, "free_wrist_ori_mimic")):
            _WRIST = "right_wrist_yaw_Link"
            for _tn in ("motion_body_ori", "motion_body_ang_vel"):
                _require(hasattr(R, _tn), f"rewards.{_tn}")
                _term = getattr(R, _tn)
                _names = [b for b in _term.params["body_names"] if b != _WRIST]
                _require(len(_names) < len(_term.params["body_names"]),
                         f"rewards.{_tn}.params.body_names contains {_WRIST}")
                _term.params["body_names"] = _names
                applied.append(f"rewards.{_tn}.body_names-={_WRIST}")
        jt = _get(rw, "joint_torques_weight")
        if jt is not None:
            _require(hasattr(R, "joint_torques"), "rewards.joint_torques")
            R.joint_torques.weight = float(jt)
            applied.append(f"rewards.joint_torques.weight={float(jt)}")

        # --- motion imitation prior (the 6 motion_* terms; base weights sum ~5.0) ---------------
        # `motion_scale` multiplies all six at once — the main lever to demote imitation to a soft
        # prior so the racket goal can dominate. Per-term weight/std overrides are also accepted
        # (e.g. motion_body_pos_weight / motion_body_pos_std) and are applied BEFORE the scale.
        _MOTION_TERMS = (
            "motion_global_anchor_pos", "motion_global_anchor_ori",
            "motion_body_pos", "motion_body_ori",
            "motion_body_lin_vel", "motion_body_ang_vel",
        )
        for _t in _MOTION_TERMS:
            _set_reward(R, _t, _get(rw, f"{_t}_weight"), _get(rw, f"{_t}_std"), applied)
        ms = _get(rw, "motion_scale")
        if ms is not None:
            ms = float(ms)
            for _t in _MOTION_TERMS:
                _require(hasattr(R, _t), f"rewards.{_t}")
                getattr(R, _t).weight *= ms
            applied.append(f"rewards.motion_scale={ms} (x{len(_MOTION_TERMS)} motion weights)")

        # --- penalties / regularization (negative weights: energy + smoothness + safety) --------
        for _name, _key in (
            ("action_rate_l2", "action_rate_weight"),
            ("joint_limit", "joint_limit_weight"),
            ("undesired_contacts", "undesired_contacts_weight"),
            ("pre_strike_foot_slip", "pre_strike_foot_slip_weight"),
            ("prestrike_waist_twist", "prestrike_waist_twist_weight"),
            # sim2real fine-tune (explicit-PD): torque-saturation penalty + pre-strike upright shaping.
            ("arm_torque_saturation", "arm_torque_saturation_weight"),
            ("prestrike_upright", "prestrike_upright_weight"),
        ):
            _w = _get(rw, _key)
            if _w is not None:
                _require(hasattr(R, _name), f"rewards.{_name}")
                getattr(R, _name).weight = float(_w)
                applied.append(f"rewards.{_name}.weight={float(_w)}")

    rk = _get(task, "racket")
    _check_unknown_keys(rk, _RACKET_KEYS, "task.racket")
    if rk is not None:
        # Only require the racket_target command when the YAML actually sets racket keys, so tasks
        # without a racket objective (e.g. TrackingFlat, which has no `racket:` block) never trip this.
        provided = [k for k in _RACKET_KEYS if _get(rk, k) is not None]
        if provided:
            _require(hasattr(env_cfg.commands, "racket_target"),
                     f"commands.racket_target (task YAML sets racket keys {provided})")
            C = env_cfg.commands.racket_target
            # strike_phase is PER-MOTION: the racket-tip contact frame differs per clip, so a single
            # global value is wrong when the trained motion changes (forehand 0.46 vs backhand 0.59).
            # `strike_phase_by_motion` (clip-name substring -> phase) wins when it matches `clip_name`;
            # `strike_phase` is the fallback. See _resolve_strike_phase / _registry_clip_name.
            _sp_val, _sp_note = _resolve_strike_phase(rk, clip_name)
            _set_attr(C, "strike_phase", _sp_val, float, applied, "racket_target")
            if _sp_note is not None:
                applied.append(_sp_note)
            _set_attr(C, "strike_window_s", _get(rk, "strike_window_s"), float, applied, "racket_target")
            _set_attr(C, "strike_success_pos_thresh", _get(rk, "strike_success_pos_thresh"), float, applied, "racket_target")
            # P2.3 adaptive tracking sigma (coarse-to-fine reward kernel widths)
            _set_attr(C, "adaptive_sigma", _get(rk, "adaptive_sigma"), _as_bool, applied, "racket_target")
            _set_attr(C, "sigma_update_every", _get(rk, "sigma_update_every"), int, applied, "racket_target")
            _set_attr(C, "sigma_ema_scale", _get(rk, "sigma_ema_scale"), float, applied, "racket_target")
            _set_attr(C, "sigma_pos_min", _get(rk, "sigma_pos_min"), float, applied, "racket_target")
            _set_attr(C, "sigma_pos_max", _get(rk, "sigma_pos_max"), float, applied, "racket_target")
            _set_attr(C, "sigma_vel_min", _get(rk, "sigma_vel_min"), float, applied, "racket_target")
            _set_attr(C, "sigma_vel_max", _get(rk, "sigma_vel_max"), float, applied, "racket_target")
            _set_range(C, "racket_pos_x_range", _get(rk, "pos_x_range"), applied, "racket_target")
            _set_range(C, "racket_pos_y_range", _get(rk, "pos_y_range"), applied, "racket_target")
            _set_range(C, "racket_pos_z_range", _get(rk, "pos_z_range"), applied, "racket_target")
            # Unified multi-clip: per-clip strike phase (aligned with the clip order) + per-clip |y| region.
            _spc = _get(rk, "strike_phase_per_clip")
            if _spc is not None:
                C.strike_phase_per_clip = tuple(float(x) for x in _spc)
                applied.append(f"racket_target.strike_phase_per_clip={C.strike_phase_per_clip}")
            _set_range(C, "racket_pos_y_abs_range", _get(rk, "racket_pos_y_abs_range"), applied, "racket_target")
            _set_range(C, "racket_vel_x_range", _get(rk, "vel_x_range"), applied, "racket_target")
            _set_range(C, "racket_vel_y_range", _get(rk, "vel_y_range"), applied, "racket_target")
            _set_range(C, "racket_vel_z_range", _get(rk, "vel_z_range"), applied, "racket_target")
            # Optional PER-CLIP velocity boxes (unified policy): forehand=clip 0, backhand=clip 1. Absent ->
            # keep the shared vel_*_range above (backward compatible). The slower backhand needs a lower box.
            _vpc = _resolve_vel_range_per_clip(rk)
            if _vpc is not None:
                _require(hasattr(C, "racket_vel_range_per_clip"), "racket_target.racket_vel_range_per_clip")
                C.racket_vel_range_per_clip = _vpc
                applied.append(f"racket_target.racket_vel_range_per_clip={_vpc}")
            # Optional PER-CLIP position boxes (unified policy): forehand=clip 0, backhand=clip 1. Absent ->
            # keep the shared pos_*_range + |y|-sign box above (backward compatible). Lets each clip's target
            # track its own reference strike point (e.g. backhand z~1.2 at strike_phase 0.50).
            _ppc = _resolve_pos_range_per_clip(rk)
            if _ppc is not None:
                _require(hasattr(C, "racket_pos_range_per_clip"), "racket_target.racket_pos_range_per_clip")
                C.racket_pos_range_per_clip = _ppc
                applied.append(f"racket_target.racket_pos_range_per_clip={_ppc}")
            _set_range(C, "base_target_x_range", _get(rk, "base_target_x_range"), applied, "racket_target")
            _set_range(C, "base_target_y_range", _get(rk, "base_target_y_range"), applied, "racket_target")
            # weak base->racket coupling (uniform mode): fraction of the racket Y offset + clamp (meters)
            _set_attr(C, "base_couple_blend", _get(rk, "base_couple_blend"), float, applied, "racket_target")
            _set_attr(C, "base_couple_max_offset", _get(rk, "base_couple_max_offset"), float, applied, "racket_target")
            _set_attr(C, "normal_mode", _get(rk, "normal_mode"), str, applied, "racket_target")
            _set_attr(C, "forehand_on_negative_y", _get(rk, "forehand_on_negative_y"), _as_bool, applied, "racket_target")
            _set_attr(C, "mount_normal_axis", _get(rk, "mount_normal_axis"), int, applied, "racket_target")
            _set_attr(C, "mount_normal_sign", _get(rk, "mount_normal_sign"), float, applied, "racket_target")
            # reference_perturbed target sampling (rank 5): couple targets to the reference swing.
            _set_attr(C, "target_mode", _get(rk, "target_mode"), str, applied, "racket_target")
            _set_vec3(C, "ref_perturb_pos", _get(rk, "ref_perturb_pos"), applied, "racket_target")
            _set_vec3(C, "ref_perturb_vel", _get(rk, "ref_perturb_vel"), applied, "racket_target")
            _set_attr(C, "ref_perturb_normal", _get(rk, "ref_perturb_normal"), float, applied, "racket_target")
            _set_attr(C, "ref_perturb_curriculum_steps", _get(rk, "ref_perturb_curriculum_steps"), int, applied, "racket_target")
            _set_attr(C, "ref_perturb_curriculum_start", _get(rk, "ref_perturb_curriculum_start"), float, applied, "racket_target")
            _set_attr(C, "ref_perturb_success_gated", _get(rk, "ref_perturb_success_gated"), _as_bool, applied, "racket_target")
            _set_attr(C, "ref_perturb_advance_threshold", _get(rk, "ref_perturb_advance_threshold"), float, applied, "racket_target")
            _set_attr(C, "ref_perturb_advance_rate", _get(rk, "ref_perturb_advance_rate"), float, applied, "racket_target")
            # Stage slow->fast hitting: scale the reference racket-velocity target (<1.0 trains slower hits).
            # PER-CLIP: ref_vel_scale_by_motion wins for the trained clip, else the scalar ref_vel_scale.
            _rv_val, _rv_note = _resolve_ref_vel_scale(rk, clip_name)
            _set_attr(C, "ref_vel_scale", _rv_val, float, applied, "racket_target")
            if _rv_note is not None:
                applied.append(_rv_note)
            # Debug logging (sign verification + raw/gated reward kernels). Off for production runs.
            _set_attr(C, "debug_reward_logging", _get(rk, "debug_reward_logging"), _as_bool, applied, "racket_target")
            # Clean reference strike velocity (denoise the FD'd target velocity at the racket tip).
            _set_attr(C, "clean_reference_strike_velocity", _get(rk, "clean_reference_strike_velocity"),
                      _as_bool, applied, "racket_target")
            _set_attr(C, "clean_strike_vel_window", _get(rk, "clean_strike_vel_window"), int, applied, "racket_target")
            # HER-style achieved-target replay: with prob achieved_target_mix_prob the next swing's target
            # is a jittered previously-ACHIEVED strike state (per-clip ring buffer) instead of a box sample.
            _set_attr(C, "achieved_target_mix_prob", _get(rk, "achieved_target_mix_prob"), float, applied, "racket_target")
            _set_attr(C, "achieved_buffer_size", _get(rk, "achieved_buffer_size"), int, applied, "racket_target")
            _set_attr(C, "achieved_min_fill", _get(rk, "achieved_min_fill"), int, applied, "racket_target")
            _set_attr(C, "achieved_jitter_pos", _get(rk, "achieved_jitter_pos"), float, applied, "racket_target")
            _set_attr(C, "achieved_jitter_vel", _get(rk, "achieved_jitter_vel"), float, applied, "racket_target")
            _set_attr(C, "achieved_clamp_inflate", _get(rk, "achieved_clamp_inflate"), float, applied, "racket_target")
            # A1 target latency & time-variance: the ACTOR-visible target arrives late
            # (target_delay_steps), noisy (SMASH-style tts-decaying jitter), and is refined
            # mid-swing (midswing_resample_*), matching the real mocap->planner->runner loop.
            # Rewards/critic keep the live target. All default OFF (byte-identical baseline).
            _set_attr(C, "target_delay_steps", _get(rk, "target_delay_steps"), int, applied, "racket_target")
            _set_attr(C, "target_jitter_pos_per_s", _get(rk, "target_jitter_pos_per_s"), float, applied, "racket_target")
            _set_attr(C, "target_jitter_vel_per_s", _get(rk, "target_jitter_vel_per_s"), float, applied, "racket_target")
            _set_attr(C, "midswing_resample_prob", _get(rk, "midswing_resample_prob"), float, applied, "racket_target")
            _set_attr(C, "midswing_resample_tts_floor", _get(rk, "midswing_resample_tts_floor"), float, applied, "racket_target")
            # A1v2 calibrated mocap-degradation channels — same actor-only scope as the delay/jitter
            # group above (venue fits documented in the task YAML: white 0.0019, ar1 0.0052).
            _set_attr(C, "target_noise_white", _get(rk, "target_noise_white"), float, applied, "racket_target")
            _set_attr(C, "target_noise_ar1_sigma", _get(rk, "target_noise_ar1_sigma"), float, applied, "racket_target")
            _set_attr(C, "target_noise_ar1_rho", _get(rk, "target_noise_ar1_rho"), float, applied, "racket_target")
            _set_attr(C, "target_dropout_prob", _get(rk, "target_dropout_prob"), float, applied, "racket_target")
            _set_attr(C, "target_post_strike_dropout_s", _get(rk, "target_post_strike_dropout_s"), float, applied, "racket_target")
            _set_attr(C, "target_bias_per_swing", _get(rk, "target_bias_per_swing"), float, applied, "racket_target")
            # Tier-1 virtual ball: incoming-ball sampling boxes + outgoing-spin objective. The reward
            # side reads vb_spin_mode with a default-else branch, so an unknown mode would silently
            # train topspin — validate the value here instead.
            _set_attr(C, "vb_spin_mode", _get(rk, "vb_spin_mode"), str, applied, "racket_target")
            if getattr(C, "vb_spin_mode", "topspin") not in ("topspin", "minimize"):
                raise _OverrideError(
                    f"[train.py] racket.vb_spin_mode must be 'topspin' or 'minimize', "
                    f"got {C.vb_spin_mode!r}")
            _set_attr(C, "vb_spin_min_sigma", _get(rk, "vb_spin_min_sigma"), float, applied, "racket_target")
            _set_attr(C, "vb_spin_abs_max", _get(rk, "vb_spin_abs_max"), float, applied, "racket_target")
            _set_range(C, "vb_vel_x_range", _get(rk, "vb_vel_x_range"), applied, "racket_target")
            _set_range(C, "vb_vel_y_range", _get(rk, "vb_vel_y_range"), applied, "racket_target")
            _set_range(C, "vb_vel_z_range", _get(rk, "vb_vel_z_range"), applied, "racket_target")
            # Stage-1 question bank + face-command channel (defaults OFF). question_bank = bank npz
            # path (gen_stage1_questions.py); face_command re-anchors the racket_normal reward onto
            # the demanded normal (target_normal_cmd).
            _set_attr(C, "question_bank", _get(rk, "question_bank"), str, applied, "racket_target")
            _set_attr(C, "face_command", _get(rk, "face_command"), _as_bool, applied, "racket_target")
            # Bank vs retiming: bank demanded velocities are ABSOLUTE physics answers (inverse-
            # solved racket velocity for a real incoming ball) — a swing replayed at speed s cannot
            # have its answer rescaled by s (the ball does not slow down). Same loud-fail pattern
            # as _check_unknown_keys: never let the combination start and silently train wrong.
            if str(getattr(C, "question_bank", "") or ""):
                _ssr = tuple(float(x) for x in getattr(
                    getattr(env_cfg.commands, "motion", None), "speed_scale_range", (1.0, 1.0)))
                if _ssr != (1.0, 1.0):
                    raise _OverrideError(
                        f"[train.py] racket.question_bank is set but motion.speed_scale_range="
                        f"{_ssr}: bank demanded velocities are absolute physics answers; retiming "
                        "cannot scale them. Set motion.speed_scale_range: [1.0, 1.0] or drop the bank.")
            # face_command_obs (+4 actor dims: demanded normal (3) + zero-filled rho placeholder (1),
            # the contract-day 175 -> 179 layout): the obs groups were finalized in __post_init__
            # BEFORE overrides run, so setting env_cfg.face_command_obs here would be a silent
            # no-op — attach the ObsTerm directly (same term/tail position as the cfg switch).
            # The enabling experiment must update/remove actor_obs_contract in its YAML:
            # validate_actor_observation_contract stays a loud error on the frozen 175-D value.
            _fc_obs = _get(rk, "face_command_obs")
            if _fc_obs is not None and _as_bool(_fc_obs):
                from isaaclab.managers import ObservationTermCfg as _ObsTerm

                from whole_body_tracking.tasks.tracking import mdp as _mdp

                env_cfg.observations.policy.racket_target_normal_cmd = _ObsTerm(
                    func=_mdp.racket_target_normal_cmd, params={"command_name": "racket_target"})
                if hasattr(env_cfg, "face_command_obs"):
                    env_cfg.face_command_obs = True  # keep the descriptive cfg field honest
                applied.append(
                    "observations.policy.racket_target_normal_cmd(+4D face-command obs, 175->179)")
            # SHADOW physical ball + table (METRICS-ONLY): a real PhysX ball flies each question
            # in, is struck via the same venue contact model, and lands under engine integration —
            # an online engine-vs-analytic cross-check of the vb landing prediction. The scene
            # entities must be attached HERE because __post_init__ already ran before overrides
            # (the exact face_command_obs timing problem above); attach_shadow_ball_scene is
            # idempotent so cfg-flag and YAML/CLI paths compose. Requires virtual_ball=True
            # (RacketTargetCommand.__init__ raises loudly otherwise).
            _set_attr(C, "shadow_ball", _get(rk, "shadow_ball"), _as_bool, applied, "racket_target")
            _set_attr(C, "shadow_table", _get(rk, "shadow_table"), _as_bool, applied, "racket_target")
            if getattr(C, "shadow_table", False) and not getattr(C, "shadow_ball", False):
                raise _OverrideError(
                    "[train.py] racket.shadow_table=true requires racket.shadow_ball=true "
                    "(the table exists only for the shadow ball to land on).")
            if getattr(C, "shadow_ball", False):
                from whole_body_tracking.tasks.tracking.config.agibot_a3.hope_env_cfg import (
                    attach_shadow_ball_scene as _attach_shadow,
                )

                _attach_shadow(env_cfg, shadow_table=bool(getattr(C, "shadow_table", False)))
                applied.append(
                    f"scene.shadow_ball attached (metrics-only; table={bool(C.shadow_table)})")

    # Domain randomization: behaviour preserved exactly (the pd_gain "absent/null -> disable" semantics
    # are intentional). Only logging is added; the hasattr guards stay so DR stays optional per task.
    dr = _get(task, "domain_rand")
    if dr is not None and hasattr(env_cfg, "events"):
        E = env_cfg.events
        mr = _get(dr, "link_mass_range")
        if mr is not None and hasattr(E, "randomize_link_mass"):
            E.randomize_link_mass.params["mass_distribution_params"] = (float(mr[0]), float(mr[1]))
            applied.append(f"events.randomize_link_mass.mass_distribution_params=({float(mr[0])}, {float(mr[1])})")
        if hasattr(E, "randomize_pd_gains"):
            pr = _get(dr, "pd_gain_range")
            if pr is None:
                E.randomize_pd_gains = None  # disable
                applied.append("events.randomize_pd_gains=None(disabled)")
            else:
                E.randomize_pd_gains.params["stiffness_distribution_params"] = (float(pr[0]), float(pr[1]))
                E.randomize_pd_gains.params["damping_distribution_params"] = (float(pr[0]), float(pr[1]))
                applied.append(f"events.randomize_pd_gains=({float(pr[0])}, {float(pr[1])})")

    return applied


# --------------------------------------------------------------------------- #
# Training (runs after the simulator is launched).
# --------------------------------------------------------------------------- #
def _run(cfg):
    import os
    from datetime import datetime

    import gymnasium as gym
    import torch

    from isaaclab.utils.io import dump_yaml
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg

    import whole_body_tracking  # noqa: F401
    import whole_body_tracking.tasks  # noqa: F401  -- registers the gym tasks
    from whole_body_tracking.tasks.tracking.actor_observation_contract import (
        validate_actor_observation_contract,
    )
    from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner
    from whole_body_tracking.utils.ppo_cfg import runner_kwargs

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Provenance: confirm we imported the WORKING TREE, not a stale install. If this path points into
    # site-packages instead of .../source/whole_body_tracking, a shadow copy is overriding your edits
    # (fix PYTHONPATH ordering in setup_train_env.sh / reinstall editable) and the YAML edits below are
    # being applied onto the wrong cfg classes.
    print(f"[train.py] whole_body_tracking imported from: {whole_body_tracking.__file__}", flush=True)

    task_id = str(cfg.task.gym_task)
    num_envs = int(cfg.num_envs) if cfg.num_envs is not None else int(cfg.task.env.num_envs)

    # 1) env cfg (gym registry) + task YAML overrides
    env_cfg = parse_env_cfg(task_id, device=str(cfg.device), num_envs=num_envs)
    _cfg_mod = sys.modules.get(type(env_cfg).__module__)
    print(f"[train.py] env cfg source: {type(env_cfg).__name__} <- {getattr(_cfg_mod, '__file__', '?')}", flush=True)
    applied = _apply_task_overrides(env_cfg, cfg.task, _registry_clip_name(cfg))
    print(f"[train.py] applied {len(applied)} task override(s) from cfg/task/{_get(cfg.task, 'name', task_id)}.yaml:", flush=True)
    for _a in applied:
        print(f"[train.py]     {_a}", flush=True)
    if not applied:
        print("[train.py] WARNING: 0 task overrides applied -> the run is using CODE DEFAULTS, not the "
              "YAML (the rewards/racket/env blocks did not compose, or all keys were absent).", flush=True)
    # Human-readable confirmation of the strike-training knobs, straight from the post-override cfg, so
    # you can read the actual runtime values off the launch log without opening logs/.../params/env.yaml.
    R = env_cfg.rewards
    if hasattr(R, "racket_position"):
        print("[train.py] racket reward std (post-override): "
              f"pos={R.racket_position.params.get('std')} vel={R.racket_velocity.params.get('std')} "
              f"normal={R.racket_normal.params.get('std')}", flush=True)
    if hasattr(env_cfg.commands, "racket_target"):
        _C = env_cfg.commands.racket_target
        print("[train.py] racket target (post-override): "
              f"target_mode={_C.target_mode} ref_perturb_curriculum_start={_C.ref_perturb_curriculum_start} "
              f"strike_window_s={_C.strike_window_s}", flush=True)
    env_cfg.seed = int(cfg.seed)
    env_cfg.sim.device = str(cfg.device)

    # 2) PPO runner cfg from cfg.algo
    algo = OmegaConf.to_container(cfg.algo, resolve=True)
    agent_cfg = RslRlOnPolicyRunnerCfg(**runner_kwargs(algo, str(cfg.task.experiment_name)))
    agent_cfg.seed = int(cfg.seed)
    agent_cfg.device = str(cfg.device)
    if cfg.max_iterations is not None:
        agent_cfg.max_iterations = int(cfg.max_iterations)
    if cfg.run_name is not None:
        agent_cfg.run_name = str(cfg.run_name)
    if cfg.logger is not None:
        agent_cfg.logger = str(cfg.logger)
    if agent_cfg.logger in {"wandb", "neptune"} and cfg.log_project_name:
        agent_cfg.wandb_project = str(cfg.log_project_name)
        agent_cfg.neptune_project = str(cfg.log_project_name)

    # 3) reference motion clip(s), LOCAL-FIRST: motion_file=/motion_file_2= (or a local .npz path passed
    #    as registry_name/registry_name_2) skips WandB entirely (the documented no-WandB path — see
    #    run_training.md); otherwise the WandB registry is used.
    #    ONE clip = single-swing-type policy. TWO clips (forehand + backhand) = unified HITTER policy:
    #    MotionLoader concatenates them and clip_id selects which swing each env imitates. Order matters:
    #    clip 0 = forehand, clip 1 = backhand; it must match racket.strike_phase_per_clip.
    def _local_motion(name):
        """If ``name`` is a local motion.npz (or a dir containing one), return that path, else None."""
        p = pathlib.Path(str(name).split(":")[0])  # tolerate a :version suffix
        if p.is_file() and p.suffix == ".npz":
            return str(p)
        if (p / "motion.npz").is_file():
            return str(p / "motion.npz")
        return None

    if not _configured_items(_get(cfg, "motion_file"), _get(cfg, "motion_file_2")):
        # Back-compat: local paths passed as registry_name/registry_name_2 become motion_file, so
        # resolve_motion_sources below stays the single source of truth for local-vs-registry.
        _reg_candidates = _configured_items(
            _get(cfg, "registry_name") if _get(cfg, "registry_name") is not None else _get(cfg.task, "registry_name"),
            _get(cfg, "registry_name_2")
            if _get(cfg, "registry_name_2") is not None
            else _get(cfg.task, "registry_name_2"),
        )
        _local_hits = [_local_motion(r) for r in _reg_candidates]
        if _local_hits and all(h is not None for h in _local_hits):
            cfg.motion_file = _local_hits
        elif any(h is not None for h in _local_hits):
            # Local clips are all-or-nothing (see resolve_motion_sources): fail loud instead of
            # letting wandb.Api().artifact(<local path>) throw a cryptic HTTP error below.
            raise RuntimeError(
                f"[train.py] Mixed motion sources in registry_name/registry_name_2: {_reg_candidates}. "
                "Some values are local .npz paths and some are registry refs. Pass ALL clips locally "
                "via motion_file=/motion_file_2= (or make every registry_name a local path), or "
                "publish the local clip to the registry."
            )
    motion_files, motion_registries = resolve_motion_sources(cfg)
    for i, mf in enumerate(motion_files):
        src = motion_registries[i] if i < len(motion_registries) else "LOCAL (no registry)"
        print(f"[train.py] motion clip {i}: {mf}  [{src}]", flush=True)
    if len(motion_files) > 1:
        print(f"[train.py] UNIFIED multi-clip policy: clip0=forehand  clip1=backhand", flush=True)
    env_cfg.commands.motion.motion_file = motion_files if len(motion_files) > 1 else motion_files[0]

    # 4) logging dir (same layout as scripts/rsl_rl/train.py so export/eval are unchanged)
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    print(f"[INFO] Task: {task_id} | experiment: {agent_cfg.experiment_name} | log: {log_dir}")

    # 5) build env, wrap, run
    render_mode = "rgb_array" if cfg.video else None
    env = gym.make(task_id, cfg=env_cfg, render_mode=render_mode)
    expected_contract = _get(cfg.task, "actor_obs_contract")
    if expected_contract is not None:
        contract = validate_actor_observation_contract(env.unwrapped, str(expected_contract))
        print(
            "[train.py] actor observation contract validated: "
            f"{contract.name} ({contract.total_dim}D, obs_mode={contract.obs_mode})",
            flush=True,
        )
    if cfg.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=os.path.join(log_dir, "videos", "train"),
            step_trigger=lambda step: step % int(cfg.video_interval) == 0,
            video_length=int(cfg.video_length),
            disable_logger=True,
        )
    env = RslRlVecEnvWrapper(env)

    # Only hand the runner registry refs for wandb lineage (use_artifact) when the clips actually came
    # from the registry; local runs pass None (a local motion path would crash wandb.run.use_artifact).
    # resolve_motion_sources already returned normalized 'collection:alias' refs (a bare collection name
    # is an HTTP 400). List-valued: the runner records ALL used clips, not just clip 0.
    runner_registry_name = motion_registries if motion_registries else None
    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device, registry_name=runner_registry_name
    )
    runner.add_git_repo_to_log(__file__)

    # Resume / curriculum hand-off: load weights+optimizer from a prior checkpoint and CONTINUE (the
    # iteration counter resumes from the checkpoint). Config changes in the task YAML (e.g. a tighter
    # racket_velocity_std) take effect immediately on the loaded policy — no fresh restart needed.
    ckpt = getattr(cfg, "checkpoint_path", None)
    if ckpt is not None:
        ckpt = os.path.abspath(str(ckpt))
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"[train.py] checkpoint_path does not exist: {ckpt}")
        if bool(getattr(cfg, "checkpoint_tolerant", False)):
            # Warm-start ACROSS critic-layout changes (e.g. the 318-D pre-merge lineage into the
            # 316-D merged model, or deploy-parity ckpts into VirtualBall's critic): actor + std
            # (+ obs normalizer if shapes agree) load strictly by name; the critic re-initializes
            # and re-learns — PPO tolerates this warm-start (fresh value function, ~hundreds of
            # iterations of value lag). Deliberate resume stays STRICT without this flag.
            from whole_body_tracking.utils.ckpt_compat import load_actor_tolerant

            load_actor_tolerant(runner, ckpt)
            print(f"[train.py] TOLERANT warm-start from {ckpt} (actor loaded; critic fresh if "
                  f"layout changed — deliberate warm-start semantics)", flush=True)
        else:
            runner.load(ckpt)
            print(f"[train.py] RESUMED from checkpoint: {ckpt} (continuing at iteration "
                  f"{getattr(runner, 'current_learning_iteration', '?')})", flush=True)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


@hydra.main(version_base=None, config_path="../cfg", config_name="train")
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    # Launch Isaac Sim BEFORE importing isaaclab modules. Clear argv so the kit app does not try to
    # parse Hydra's `task=...`/`algo=...` overrides.
    sys.argv = sys.argv[:1]
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(
        headless=bool(cfg.headless), device=str(cfg.device), enable_cameras=bool(cfg.video)
    )
    simulation_app = app_launcher.app
    # Print the traceback BEFORE closing the app: Isaac's simulation_app.close() hard-exits the
    # process (os._exit), which otherwise swallows any exception from _run and makes a real failure
    # look like a clean "exit 0" with the log truncated at startup.
    failed = False
    try:
        _run(cfg)
    except Exception:
        import traceback
        print("\n[train.py] ERROR during run:", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        failed = True
    finally:
        try:
            import wandb

            if getattr(wandb, "run", None) is not None:
                wandb.finish()
        except Exception as exc:
            print(f"[train.py] WARNING: wandb.finish() failed: {exc}", flush=True)
        simulation_app.close()
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
