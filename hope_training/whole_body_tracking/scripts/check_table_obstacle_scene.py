#!/usr/bin/env python3
"""Verify the TABLE OBSTACLE in a really-constructed Isaac env, and price it.

人话:真开一个 Isaac 环境,量三件事——桌子在不在、在不在该在的位置、加了它每步慢多少。

Three checks, in increasing cost:

1. ``--cfg-only`` — no simulator.  The env CFG carries a table collider, a wrist-vs-table filtered
   sensor, a ``robot_hit_table`` termination and a ``table_hit_penalty`` reward, and the collider's
   pose/extent equal the shared ``table_tennis.table_frame`` derivation.  Cheap enough to run
   anywhere with the Isaac imports available.
2. default — construct the env.  Read the SPAWNED prim's world transform back out of USD and
   compare it against the same derivation, so what is asserted is the thing PhysX actually has,
   not the thing the config asked for.  Also confirms the termination manager lists
   ``robot_hit_table`` as an active term (that is what makes it a named metrics channel:
   ``Live/Termination/robot_hit_table`` and ``termination_reason_robot_hit_table_count``).
3. ``--bench N`` — step-time with the table against step-time without it, same seed, same env
   count.  This is the runtime-cost number.

Usage (pod, inside the Isaac venv)::

    python hope_training/whole_body_tracking/scripts/check_table_obstacle_scene.py \
        --task Tracking-Flat-AgibotA3-Hope-VirtualBall-v0 --num-envs 64 --bench 200
"""

from __future__ import annotations

import argparse
import json
import sys
import time


def _parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="Tracking-Flat-AgibotA3-Hope-VirtualBall-v0")
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--cfg-only", action="store_true",
                    help="stop after the cfg checks (the Kit app still launches — isaaclab "
                         "cannot be imported without omni.kit)")
    ap.add_argument("--bench", type=int, default=0, help="steps to time (one arm per process)")
    ap.add_argument("--table-obstacle", choices=("on", "off"), default="on",
                    help="the arm this process measures; run twice and subtract")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--motion-file", default=None,
                    help="reference clip npz. Required to CONSTRUCT the env (train.py normally "
                         "pulls it from the registry); any canonical clip will do — this script "
                         "never steps a policy, it only needs the scene to exist.")
    return ap.parse_args(argv)


ARGS = _parse()

# The Kit app must exist before ``import isaaclab`` — ``isaaclab.managers`` imports omni.kit at
# module scope — so this is unconditional even for --cfg-only.
from isaaclab.app import AppLauncher  # noqa: E402

_app = AppLauncher({"headless": True, "device": ARGS.device}).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402  (registers the task ids)
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import whole_body_tracking.tasks  # noqa: F401,E402
from whole_body_tracking.tasks.table_tennis import table_frame as tt_frame  # noqa: E402

TOL = 1e-6
_results: dict = {}


def _fail(msg):
    print(f"FAIL {msg}", file=sys.stderr)
    raise SystemExit(1)


def _close(got, want, label):
    if len(got) != len(want) or any(abs(float(a) - float(b)) > TOL for a, b in zip(got, want)):
        _fail(f"{label}: {list(map(float, got))} != {list(map(float, want))}")


def check_cfg(env_cfg):
    """The cfg carries collider, filtered sensor, termination and penalty for the same table."""
    rt = env_cfg.commands.racket_target
    near_x, surface_z = float(rt.vb_table_near_x), float(rt.vb_table_surface_z)

    prim = getattr(env_cfg, "table_obstacle_prim", "")
    if not prim:
        _fail("env_cfg.table_obstacle_prim is empty — no table collider was attached")
    attr = {"{ENV_REGEX_NS}/TableObstacle": "table_obstacle",
            "{ENV_REGEX_NS}/ShadowTable": "shadow_table",
            "{ENV_REGEX_NS}/PhysicalTable": "pb_table"}.get(prim)
    if attr is None:
        _fail(f"unknown table_obstacle_prim {prim!r}")
    slab = getattr(env_cfg.scene, attr, None)
    if slab is None:
        _fail(f"scene.{attr} is missing although table_obstacle_prim={prim!r}")
    if not bool(slab.spawn.collision_props.collision_enabled):
        _fail(f"scene.{attr} has collision DISABLED — it is not an obstacle")

    _close(slab.init_state.pos, tt_frame.table_top_center_env(near_x, surface_z),
           f"scene.{attr}.init_state.pos")
    _close(slab.spawn.size, tt_frame.table_top_size(), f"scene.{attr}.spawn.size")
    top = float(slab.init_state.pos[2]) + float(slab.spawn.size[2]) / 2.0
    if abs(top - surface_z) > TOL:
        _fail(f"table TOP face {top} != vb_table_surface_z {surface_z}")

    done = getattr(env_cfg.terminations, "robot_hit_table", None)
    if done is None:
        _fail("terminations.robot_hit_table is missing — the table is a decoration")
    for key, want in (("near_x", near_x), ("surface_z", surface_z)):
        if abs(float(done.params[key]) - want) > TOL:
            _fail(f"terminations.robot_hit_table.params.{key} "
                  f"{done.params[key]} != {want} (box would not match the collider)")
    filtered_cfg = getattr(env_cfg.scene, "racket_table_contact", None)
    if filtered_cfg is None:
        _fail("scene.racket_table_contact is missing — offset racket contacts can be missed")
    if filtered_cfg.prim_path != "{ENV_REGEX_NS}/Robot/right_wrist_yaw_Link":
        _fail(f"scene.racket_table_contact.prim_path {filtered_cfg.prim_path!r} is not the A3 wrist")
    if list(filtered_cfg.filter_prim_paths_expr) != [prim]:
        _fail("scene.racket_table_contact filter does not match table_obstacle_prim: "
              f"{list(filtered_cfg.filter_prim_paths_expr)!r} != {[prim]!r}")
    done_filtered = done.params.get("filtered_sensor_cfg")
    if done_filtered is None or done_filtered.name != "racket_table_contact":
        _fail("terminations.robot_hit_table does not consume scene.racket_table_contact")

    rew = getattr(env_cfg.rewards, "table_hit_penalty", None)
    if rew is None:
        _fail("rewards.table_hit_penalty is missing")
    if rew.params.get("term_name") != "robot_hit_table":
        _fail(f"rewards.table_hit_penalty points at {rew.params.get('term_name')!r}")

    _results["cfg"] = {
        "table_obstacle_prim": prim,
        "scene_attr": attr,
        "pos": [float(v) for v in slab.init_state.pos],
        "size": [float(v) for v in slab.spawn.size],
        "surface_z": surface_z,
        "near_x": near_x,
        "termination_params": {k: (float(v) if isinstance(v, (int, float)) else str(v))
                               for k, v in done.params.items()},
        "filtered_contact_sensor": {
            "name": "racket_table_contact",
            "prim_path": filtered_cfg.prim_path,
            "filter_prim_paths_expr": list(filtered_cfg.filter_prim_paths_expr),
        },
        "table_hit_penalty_weight": float(rew.weight),
    }
    print("ok cfg: collider + filtered sensor + termination + penalty all mutually consistent")


def check_spawned(env, env_cfg):
    """Read the pose PhysX actually has, not the pose the config asked for."""
    from pxr import Usd, UsdGeom
    import isaacsim.core.utils.stage as stage_utils

    rt = env_cfg.commands.racket_target
    near_x, surface_z = float(rt.vb_table_near_x), float(rt.vb_table_surface_z)
    prim_path = _results["cfg"]["table_obstacle_prim"].replace("{ENV_REGEX_NS}", "/World/envs/env_0")
    stage = stage_utils.get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        _fail(f"{prim_path} does not exist on the stage — the collider was never spawned")

    xform = UsdGeom.Xformable(prim)
    m = xform.ComputeLocalToWorldTransform(0.0)
    t = m.ExtractTranslation()
    origin = env.unwrapped.scene.env_origins[0].tolist()
    local = (float(t[0]) - origin[0], float(t[1]) - origin[1], float(t[2]) - origin[2])
    _close(local, tt_frame.table_top_center_env(near_x, surface_z), f"{prim_path} world transform")

    # ``CuboidCfg`` spawns an Xform at ``prim_path`` with the actual Cube geometry underneath, and
    # the collision API lands on the geometry prim, not on the Xform. Walk the subtree.
    from pxr import UsdPhysics
    collider_paths = [str(d.GetPath()) for d in Usd.PrimRange(prim)
                      if d.HasAPI(UsdPhysics.CollisionAPI)]
    if not collider_paths:
        _fail(f"{prim_path} subtree carries no UsdPhysics.CollisionAPI — PhysX will ignore it")
    enabled = []
    for cp in collider_paths:
        api = UsdPhysics.CollisionAPI(stage.GetPrimAtPath(cp))
        attr = api.GetCollisionEnabledAttr()
        enabled.append(bool(attr.Get()) if attr and attr.HasAuthoredValue() else True)
    if not any(enabled):
        _fail(f"{prim_path}: every collider in the subtree has collisionEnabled=False")

    active = tuple(env.unwrapped.termination_manager.active_terms)
    if "robot_hit_table" not in active:
        _fail(f"robot_hit_table is not an active termination; active={active}")
    filtered_sensor = env.unwrapped.scene.sensors.get("racket_table_contact")
    if filtered_sensor is None:
        _fail("spawned scene has no racket_table_contact sensor")
    force_matrix = getattr(filtered_sensor.data, "force_matrix_w", None)
    if force_matrix is None or force_matrix.ndim != 4 or force_matrix.shape[-1] != 3:
        _fail("spawned racket_table_contact has no [env, body, filter, 3] force_matrix_w; got "
              f"{None if force_matrix is None else tuple(force_matrix.shape)}")
    rew_active = tuple(env.unwrapped.reward_manager.active_terms)
    _results["spawned"] = {
        "prim_path": prim_path,
        "env_local_translation": list(local),
        "collider_prims": collider_paths,
        "collision_enabled": enabled,
        "filtered_contact_force_matrix_shape": list(force_matrix.shape),
        "active_terminations": list(active),
        "table_hit_penalty_active": "table_hit_penalty" in rew_active,
        # These two names are the metrics channels the termination produces for free:
        # my_on_policy_runner logs Live/Termination/<term>, and the behavior ledger books
        # termination_reason_<term>_count from termination_manager.active_terms.
        "metric_channels": [
            "Live/Termination/robot_hit_table",
            "Live/racket_target/termination_reason_robot_hit_table_count",
        ],
    }
    print(f"ok spawned: {prim_path} at env-local {local}, collision API present, "
          f"robot_hit_table active")


def bench(env, steps):
    """Step time for THIS arm.  One arm per process, on purpose.

    Isaac Sim does not reliably build a second ``ManagerBasedRLEnv`` in one process — the second
    ``gym.make`` hangs after "Parsing configuration" — so this measures the env that is already
    up and the CALLER runs the script twice, once with ``--table-obstacle on`` and once with
    ``off``, and subtracts.  Trying to do both arms in one process is what the first version did
    and it deadlocked for 30 minutes.
    """
    act = torch.zeros(env.unwrapped.num_envs,
                      env.unwrapped.action_manager.total_action_dim,
                      device=env.unwrapped.device)
    for _ in range(20):          # warm-up: PhysX broadphase + CUDA graphs settle
        env.step(act)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        env.step(act)
    torch.cuda.synchronize()
    per_step = (time.perf_counter() - t0) / steps
    _results["bench"] = {
        "table_obstacle": bool(ARGS.table_obstacle == "on"),
        "num_envs": int(env.unwrapped.num_envs),
        "steps": int(steps),
        "seconds_per_step": per_step,
        "ms_per_step": per_step * 1e3,
    }
    print(f"ok bench (table={ARGS.table_obstacle}, {env.unwrapped.num_envs} envs, {steps} steps): "
          f"{per_step*1e3:.3f} ms/step")


def _cfg():
    cfg = parse_env_cfg(ARGS.task, device=ARGS.device, num_envs=ARGS.num_envs)
    if ARGS.motion_file:
        cfg.commands.motion.motion_file = ARGS.motion_file
    # This script constructs a scene and reads geometry back; it never trains and never reads a
    # reward. The virtual-ball command refuses to build without a solved question bank because an
    # UNBANKED landing reward is anti-correlated with returning the ball — a training concern that
    # does not apply here. Opting out explicitly (rather than silently picking a task variant that
    # dodges the check) keeps the thing under test the LIVE lineage's env class.
    rt = getattr(cfg.commands, "racket_target", None)
    if rt is not None and hasattr(rt, "allow_unbanked_landing_rewards"):
        rt.allow_unbanked_landing_rewards = True
    if ARGS.table_obstacle == "off":
        cfg.table_obstacle = False
        from whole_body_tracking.tasks.tracking.config.agibot_a3.hope_env_cfg import (
            apply_table_obstacle,
        )

        apply_table_obstacle(cfg)   # removes collider + termination + penalty together
    return cfg


def main():
    env_cfg = _cfg()
    if ARGS.table_obstacle == "off":
        # The no-table control arm: assert the removal is COMPLETE, not partial.
        for attr, where in (("table_obstacle", env_cfg.scene),
                            ("racket_table_contact", env_cfg.scene),
                            ("robot_hit_table", env_cfg.terminations),
                            ("table_hit_penalty", env_cfg.rewards)):
            if getattr(where, attr, None) is not None:
                _fail(f"--table-obstacle off left {attr} behind")
        print("ok cfg: no-table control arm — collider, sensor, termination and penalty all removed")
        _results["cfg"] = {"table_obstacle": False}
    else:
        check_cfg(env_cfg)
    if not ARGS.cfg_only:
        env = gym.make(ARGS.task, cfg=env_cfg)
        env.reset()
        if ARGS.table_obstacle != "off":
            check_spawned(env, env_cfg)
        if ARGS.bench:
            bench(env, ARGS.bench)
    print("HOPE_TABLE_OBSTACLE_CHECK_JSON=" + json.dumps(_results, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
