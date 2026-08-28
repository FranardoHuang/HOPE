#!/usr/bin/env python3
"""Two-stage mjlab environment for the AgiBot A3 ping-pong plant.

WHY TWO STAGES (plain language)
-------------------------------
`MjSpec.attach()` silently throws away the child spec's `<option>` block, and
mjlab's `MujocoCfg.apply()` then unconditionally rewrites twelve more option
fields on the compiled model.  So neither the MJCF nor mjlab alone can be
trusted to carry the vendor's physics settings.  We therefore split ownership:

  stage 1 (MJCF owns it):  bodies, geoms, joints, actuators, contact excludes,
                           sensors, keyframe.  Loaded verbatim through
                           `mujoco.MjSpec.from_file`.
  stage 2 (SimulationCfg owns it): every `<option>` field, written out
                           explicitly so the value survives attach + apply.

Everything mjlab offers as a *convenience rewriter* is deliberately switched
off: no `CollisionCfg`, no `BuiltinPositionActuator` (the vendor runs 31 pure
torque motors and computes PD outside the plant), no terrain, no default
keyframe overwrite (`init_state.joint_pos=None` keeps the MJCF's own "stand"
key).

Named deviations are printed by `--verify`; the only structural one is
`noslip_iterations` / `noslip_tolerance`, which MuJoCo Warp has no counterpart
for.

Usage
-----
  python a3_plant_env.py --verify                      # field-by-field audit
  python a3_plant_env.py --nworld 4096 --steps 500     # throughput smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Plant ground truth, transcribed from the vendor MJCF <option> block.
#
#   <option timestep="0.001" gravity="0 0 -9.81"
#           noslip_iterations="3" noslip_tolerance="1e-6" />
#
# Everything else below is the *MuJoCo compiler default* that the vendor
# inherited by not writing it down.  We write it down anyway, because mjlab's
# defaults differ (integrator=implicitfast, ccd_iterations=50, timestep=0.002)
# and `apply()` would otherwise stamp mjlab's values onto our model.
# ---------------------------------------------------------------------------

AGIBOT_OPTION = dict(
    timestep=0.001,          # vendor explicit
    integrator="euler",      # MuJoCo default; mjlab default is implicitfast
    solver="newton",         # MuJoCo default
    iterations=100,          # MuJoCo default
    ls_iterations=50,        # MuJoCo default
    tolerance=1e-8,          # MuJoCo default
    ls_tolerance=0.01,       # MuJoCo default
    ccd_iterations=35,       # MuJoCo default; mjlab explicitly sets 50
    impratio=1.0,            # MuJoCo default
    cone="pyramidal",        # MuJoCo default
    jacobian="auto",         # MuJoCo default
    gravity=(0.0, 0.0, -9.81),  # vendor explicit
    disableflags=(),
    enableflags=(),
)

# 1000 Hz physics / 50 Hz policy.  mjlab's own examples use 4; the vendor stack
# is 20 and that is what the deployed controller assumes.
DECIMATION = 20

# ---------------------------------------------------------------------------
# Static array capacities.  MuJoCo Warp sizes contacts and constraints once, up
# front, and both of its heuristics are far too small for this plant.
#
# RE-MEASURED 2026-08-06 with the convergence-gated `contact_census.py`.  The
# numbers that used to sit here (136 rows / 26 contacts / 242,714 candidates)
# came from a limp zero-torque drop over a FIXED 3000-step window, and both
# halves of that were wrong:
#
#   * zero torque is not the worst case.  Torque that never leaves the model's
#     own ctrlrange costs ~25% more rows, and a random configuration inside
#     jnt_range costs about twice as much.
#   * 3000 steps is not converged.  The zero-torque sprawl was still setting
#     new records at step 25,590 of 30,000, so the old peaks were lower bounds.
#
# Per-world demand at nworld=4096, pyramidal (this plant's default cone), each
# scenario run until it stopped setting records -- except `zero`, which never
# does (see below).  `ncollision` is all worlds, and it, not the contact count,
# is what naconmax really has to hold.
#
#   scenario   rows/world   contacts/world   broadphase candidates
#   zero          159*           31*              268,846*   (* NOT converged)
#   flail         170            32                77,615
#   bang          167            31                71,838
#   randpose      276            57               108,666
#   slam          244            44               120,825
#   warp heuristic  64           48               196,608 (= 48 * 4096)
#
# Two independent failures follow from trusting the heuristics:
#
#  1. njmax=64 drops the surplus constraint rows.  Not silently -- the engine
#     printfs a line and sets a bit in d.overflow -- but nothing downstream
#     reads either, so it may as well be.  At nworld=4096 that put ~96% of
#     worlds non-finite by step 2000, while the identical drop on CPU MuJoCo --
#     which grows its arena on demand and peaked at nefc=92 -- stayed finite
#     for the full run.
#  2. nconmax=48/world also sizes the broadphase candidate array
#     (naconmax = nconmax * nworld = 196,608).  The sprawl generates 268,846
#     candidates, a 37% overflow, and MuJoCo Warp does not fail safe: it writes
#     out of bounds and the process dies with a CUDA illegal memory access
#     around step ~2,300.  Raising njmax alone converts the silent NaN into
#     this hard crash, which is why both caps have to move together.
#
# Headroom the defaults below actually carry, against the converged numbers:
# suggest_njmax returns 508 for this plant, so rows are 508/276 = 1.84x;
# nconmax=128 gives 128/57 = 2.25x on contacts and 524,288/268,846 = 1.95x on
# broadphase candidates -- and that last one is a LOWER BOUND, because the
# zero-torque sprawl had not converged at 30,000 steps.  This is the tightest
# number anywhere in the lane.  It is a diagnostic scene, not a trained one; if
# you plan a long 4096-world plant run, raise nconmax to 192 first.
# The trained scene is the court, which sizes itself (a3_court_env: njmax=572)
# and has more room -- see EXP-ACTION-BALL-MUJOCO-NATIVE-READINESS-20260802
# section 9.2.7 for the full per-scenario, per-cone table.
# Never compare a peak measured under one friction cone with one measured under
# another: pyramidal spends 2*(condim-1) rows per contact, elliptic spends
# condim, so the same physics costs 4/3 as many rows.  `contact_census.py`
# refuses such a comparison outright.
NJMAX_SAFETY = 2.0
DEFAULT_NCON_PER_WORLD = 48   # contact-row term in suggest_njmax (peak was 26)
DEFAULT_NCONMAX_PER_WORLD = 128  # backs the broadphase array (peak was 59/world)

# Vendor A3 nominal PD gains (N m / rad, N m s / rad).  These live OUTSIDE the
# plant on purpose -- the MJCF actuators are pure torque motors and the real
# controller computes `ctrl = kp*(q_des-q) + kd*(qd_des-qd)` itself.  Used only
# by the `--ctrl pd` smoke run so the robot holds a pose instead of collapsing.
VENDOR_KP = {
    "waist_yaw_joint": 85.0, "waist_roll_joint": 50.0, "waist_pitch_joint": 50.0,
    "head_yaw_joint": 40.0, "head_pitch_joint": 40.0,
    "shoulder_pitch": 40.0, "shoulder_roll": 40.0, "shoulder_yaw": 30.0,
    "elbow": 30.0, "wrist_roll": 30.0, "wrist_pitch": 20.0, "wrist_yaw": 20.0,
    "hip_yaw": 80.0, "hip_roll": 120.0, "hip_pitch": 80.0, "knee": 250.0,
    "ankle_pitch": 50.0, "ankle_roll": 50.0,
}
VENDOR_KD = {
    "waist_yaw_joint": 3.0, "waist_roll_joint": 2.0, "waist_pitch_joint": 2.0,
    "head_yaw_joint": 2.0, "head_pitch_joint": 2.0,
    "shoulder_pitch": 3.0, "shoulder_roll": 3.0, "shoulder_yaw": 2.0,
    "elbow": 2.0, "wrist_roll": 2.0, "wrist_pitch": 2.0, "wrist_yaw": 2.0,
    "hip_yaw": 3.0, "hip_roll": 4.0, "hip_pitch": 3.0, "knee": 8.0,
    "ankle_pitch": 2.0, "ankle_roll": 2.0,
}

_REPO_XML = (
    "agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/"
    "a3p_pingpong_0807/a3p_pingpong_0807.xml"
)


def default_xml() -> Path:
    """Locate the A3P-P1 0807 MJCF: explicit binding, then repo checkout."""
    env = os.environ.get("A3_PINGPONG_XML")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / _REPO_XML
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        "a3p_pingpong_0807.xml not found; set A3_PINGPONG_XML"
    )


# ---------------------------------------------------------------------------
# Stage 1 + stage 2: build the env.
# ---------------------------------------------------------------------------


@dataclass
class A3PlantEnv:
    """The pieces a manager-based env would hold, minus the managers."""

    scene: Any
    sim: Any
    decimation: int
    xml_path: Path
    entity_prefix: str

    @property
    def mj_model(self):
        return self.sim.mj_model

    @property
    def physics_dt(self) -> float:
        return float(self.sim.mj_model.opt.timestep)

    @property
    def step_dt(self) -> float:
        return self.physics_dt * self.decimation


def build_sim_cfg(**overrides):
    from mjlab.sim.sim import MujocoCfg, SimulationCfg

    opt = dict(AGIBOT_OPTION)
    nconmax = overrides.pop("nconmax", None)
    njmax = overrides.pop("njmax", None)
    opt.update(overrides)
    # Explicit, field by field.  No dataclass-default inheritance.
    mujoco_cfg = MujocoCfg(
        timestep=opt["timestep"],
        integrator=opt["integrator"],
        impratio=opt["impratio"],
        cone=opt["cone"],
        jacobian=opt["jacobian"],
        solver=opt["solver"],
        iterations=opt["iterations"],
        tolerance=opt["tolerance"],
        ls_iterations=opt["ls_iterations"],
        ls_tolerance=opt["ls_tolerance"],
        ccd_iterations=opt["ccd_iterations"],
        gravity=opt["gravity"],
        disableflags=opt["disableflags"],
        enableflags=opt["enableflags"],
    )
    return SimulationCfg(mujoco=mujoco_cfg, nconmax=nconmax, njmax=njmax)


def build_scene_cfg(xml_path: Path, num_envs: int, entity_name: str = "robot"):
    import mujoco
    from mjlab.entity import EntityCfg
    from mjlab.scene import SceneCfg

    xml = str(xml_path)

    def spec_fn() -> "mujoco.MjSpec":
        # Verbatim.  No spec editing whatsoever.
        return mujoco.MjSpec.from_file(xml)

    robot = EntityCfg(
        spec_fn=spec_fn,
        # joint_pos=None => keep the MJCF's own "stand" keyframe.  Any dict here
        # would make mjlab synthesize a keyframe and overwrite the vendor pose.
        init_state=EntityCfg.InitialStateCfg(joint_pos=None),
        # articulation=None => mjlab adds no actuators, so the 31 MJCF pure
        # torque motors survive untouched (no BuiltinPositionActuator).
        articulation=None,
        # No collisions=/meshes=/materials= editors: no CollisionCfg rewrite.
    )
    return SceneCfg(
        num_envs=num_envs,
        terrain=None,          # no mjlab terrain generator
        entities={entity_name: robot},
        sensors=(),            # MJCF <sensor> block carries the real sensors
    )


def suggest_njmax(model, ncon_per_world: int = DEFAULT_NCON_PER_WORLD) -> int:
    """Worst-case constraint rows per world, from the compiled model.

    nefc = ne (equality) + nf (dof/tendon frictionloss) + nl (limits) + contacts.
    A pyramidal cone spends ``2*(condim-1)`` rows per contact (4 for condim=3);
    an elliptic cone spends ``condim``.
    """
    import mujoco

    ne = int(model.neq) * 6  # generous: a weld equality is 6 rows
    nf = int(np.count_nonzero(model.dof_frictionloss))
    nf += int(np.count_nonzero(model.tendon_frictionloss))
    nl = int(np.count_nonzero(model.jnt_limited))
    nl += int(np.count_nonzero(model.tendon_limited))
    dim = int(model.geom_condim.max()) if model.ngeom else 3
    if int(model.opt.cone) == int(mujoco.mjtCone.mjCONE_PYRAMIDAL):
        rows = 1 if dim == 1 else 2 * (dim - 1)
    else:
        rows = dim
    bound = ne + nf + nl + rows * ncon_per_world
    return int(np.ceil(bound * NJMAX_SAFETY))


def build_env(
    xml_path: Path,
    num_envs: int,
    device: str = "cuda:0",
    entity_name: str = "robot",
    auto_sizes: bool = True,
    **sim_overrides,
) -> A3PlantEnv:
    import mujoco
    from mjlab.scene import Scene
    from mjlab.sim.sim import Simulation

    if auto_sizes:
        if sim_overrides.get("njmax") is None:
            ref = mujoco.MjModel.from_xml_path(str(xml_path))
            sim_overrides["njmax"] = suggest_njmax(ref)
            print(f"[a3_plant_env] njmax auto-sized to {sim_overrides['njmax']} "
                  f"(warp heuristic returns 64 and silently drops rows)")
        if sim_overrides.get("nconmax") is None:
            sim_overrides["nconmax"] = DEFAULT_NCONMAX_PER_WORLD
            print(f"[a3_plant_env] nconmax set to "
                  f"{sim_overrides['nconmax']}/world (warp heuristic returns 48, "
                  f"which overflows the broadphase array and hard-crashes)")

    scene_cfg = build_scene_cfg(xml_path, num_envs, entity_name)
    sim_cfg = build_sim_cfg(**sim_overrides)

    scene = Scene(scene_cfg, device=device)
    sim = Simulation(
        num_envs=scene.num_envs,
        cfg=sim_cfg,
        spec=scene.spec,
        variant_info=scene.collect_variant_info(),
        device=device,
    )
    scene.initialize(mj_model=sim.mj_model, model=sim.model, data=sim.data)
    return A3PlantEnv(
        scene=scene,
        sim=sim,
        decimation=DECIMATION,
        xml_path=xml_path,
        entity_prefix=f"{entity_name}/",
    )


# ---------------------------------------------------------------------------
# Field-by-field verification against a raw MjModel compiled from the MJCF.
# ---------------------------------------------------------------------------


def _scalarize(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


def _opt_fields(opt) -> dict:
    out = {}
    for name in dir(opt):
        if name.startswith("_"):
            continue
        try:
            v = getattr(opt, name)
        except Exception:
            continue
        if callable(v):
            continue
        out[name] = np.array(v).copy() if isinstance(v, np.ndarray) else v
    return out


def _names(model, obj_enum, count) -> list[str]:
    import mujoco

    return [mujoco.mj_id2name(model, obj_enum, i) or "" for i in range(count)]


class Report:
    def __init__(self) -> None:
        self.match: list[str] = []
        self.mismatch: list[dict] = []
        self.notes: list[str] = []

    def check(self, label: str, got, ref, *, atol: float = 0.0, kind: str = "") -> bool:
        g, r = np.asarray(got), np.asarray(ref)
        if g.shape == r.shape and np.allclose(g, r, rtol=0.0, atol=atol, equal_nan=True):
            self.match.append(label)
            return True
        entry = {
            "field": label,
            "mjlab": _scalarize(g),
            "mjcf_reference": _scalarize(r),
        }
        if kind:
            entry["kind"] = kind
        self.mismatch.append(entry)
        return False

    def check_indexed(
        self, label: str, got: np.ndarray, ref: np.ndarray, names: list[str],
        *, atol: float = 0.0, kind: str = "",
    ) -> bool:
        if got.shape != ref.shape:
            self.mismatch.append({
                "field": label, "kind": kind or "shape",
                "mjlab": list(got.shape), "mjcf_reference": list(ref.shape),
            })
            return False
        bad = ~np.all(
            np.isclose(got, ref, rtol=0.0, atol=atol, equal_nan=True).reshape(
                got.shape[0], -1
            ),
            axis=1,
        )
        idx = np.nonzero(bad)[0]
        if idx.size == 0:
            self.match.append(f"{label} [{got.shape[0]} rows]")
            return True
        rows = [
            {
                "index": int(i),
                "name": names[i] if i < len(names) else "",
                "mjlab": _scalarize(got[i]),
                "mjcf_reference": _scalarize(ref[i]),
            }
            for i in idx[:20]
        ]
        self.mismatch.append({
            "field": label,
            "kind": kind or "per-element",
            "n_bad": int(idx.size),
            "rows": rows,
        })
        return False


def verify(env: A3PlantEnv, verbose: bool = True) -> dict:
    """Compile the env's model and compare it, field by field, to the MJCF."""
    import mujoco

    got = env.mj_model
    ref = mujoco.MjModel.from_xml_path(str(env.xml_path))
    prefix = env.entity_prefix
    rep = Report()

    # ---- counts -----------------------------------------------------------
    for f in ("nq", "nv", "nu", "nbody", "njnt", "ngeom", "nsite", "nmesh",
              "nsensor", "nexclude", "nkey"):
        rep.check(f"count.{f}", getattr(got, f), getattr(ref, f), kind="count")

    # ---- opt.* ------------------------------------------------------------
    g_opt, r_opt = _opt_fields(got.opt), _opt_fields(ref.opt)
    for name in sorted(set(g_opt) | set(r_opt)):
        if name not in g_opt or name not in r_opt:
            rep.mismatch.append({"field": f"opt.{name}", "kind": "missing-attr"})
            continue
        rep.check(f"opt.{name}", g_opt[name], r_opt[name], kind="option")

    # ---- name alignment (proves index-for-index comparison is legitimate) --
    def aligned(obj_enum, n_got, n_ref, what) -> list[str]:
        gn = _names(got, obj_enum, n_got)
        rn = _names(ref, obj_enum, n_ref)
        if len(gn) != len(rn):
            rep.mismatch.append({"field": f"{what}.count", "kind": "count",
                                 "mjlab": len(gn), "mjcf_reference": len(rn)})
            return rn
        bad = [
            {"index": i, "mjlab": a, "mjcf_reference": b}
            for i, (a, b) in enumerate(zip(gn, rn))
            if a.removeprefix(prefix) != b
        ]
        if bad:
            rep.mismatch.append({"field": f"{what}.names", "kind": "ordering",
                                 "n_bad": len(bad), "rows": bad[:20]})
        else:
            rep.match.append(f"{what}.names [{len(rn)} aligned under '{prefix}']")
        return rn

    OBJ = mujoco.mjtObj
    geom_names = aligned(OBJ.mjOBJ_GEOM, got.ngeom, ref.ngeom, "geom")
    jnt_names = aligned(OBJ.mjOBJ_JOINT, got.njnt, ref.njnt, "joint")
    act_names = aligned(OBJ.mjOBJ_ACTUATOR, got.nu, ref.nu, "actuator")
    body_names = aligned(OBJ.mjOBJ_BODY, got.nbody, ref.nbody, "body")

    # ---- per-geom contact fields -----------------------------------------
    if got.ngeom == ref.ngeom:
        for f in ("geom_solref", "geom_solimp", "geom_friction", "geom_condim",
                  "geom_contype", "geom_conaffinity", "geom_priority",
                  "geom_margin", "geom_gap"):
            rep.check_indexed(f, np.atleast_2d(getattr(got, f).reshape(got.ngeom, -1)),
                              np.atleast_2d(getattr(ref, f).reshape(ref.ngeom, -1)),
                              geom_names, kind="geom")

    # ---- per-actuator fields ---------------------------------------------
    if got.nu == ref.nu:
        for f in ("actuator_gear", "actuator_ctrlrange", "actuator_gainprm",
                  "actuator_biasprm", "actuator_gaintype", "actuator_biastype",
                  "actuator_ctrllimited", "actuator_forcerange",
                  "actuator_forcelimited", "actuator_trntype", "actuator_trnid",
                  "actuator_dyntype", "actuator_actlimited"):
            rep.check_indexed(f, getattr(got, f).reshape(got.nu, -1),
                              getattr(ref, f).reshape(ref.nu, -1),
                              act_names, kind="actuator")

    # ---- per-joint fields -------------------------------------------------
    if got.njnt == ref.njnt:
        for f in ("jnt_range", "jnt_limited", "jnt_type", "jnt_axis", "jnt_pos",
                  "jnt_stiffness", "jnt_margin", "jnt_solref", "jnt_solimp",
                  "jnt_actfrcrange", "jnt_actfrclimited"):
            if not hasattr(got, f):
                continue
            rep.check_indexed(f, getattr(got, f).reshape(got.njnt, -1),
                              getattr(ref, f).reshape(ref.njnt, -1),
                              jnt_names, kind="joint")

        # dof-indexed physics, gathered back onto joints so a mismatch names
        # the joint rather than a bare dof index.
        g_adr, r_adr = got.jnt_dofadr, ref.jnt_dofadr
        rep.check_indexed("jnt_dofadr", g_adr.reshape(-1, 1), r_adr.reshape(-1, 1),
                          jnt_names, kind="joint")
        if got.nv == ref.nv:
            for f in ("dof_armature", "dof_damping", "dof_frictionloss",
                      "dof_solref", "dof_solimp"):
                rep.check_indexed(
                    f, getattr(got, f).reshape(got.nv, -1),
                    getattr(ref, f).reshape(ref.nv, -1),
                    [f"dof{i}" for i in range(ref.nv)], kind="dof",
                )

    # ---- body inertials (attach must not perturb the tree) ----------------
    if got.nbody == ref.nbody:
        for f in ("body_mass", "body_inertia", "body_ipos", "body_iquat",
                  "body_pos", "body_quat", "body_parentid", "body_jntnum",
                  "body_dofnum"):
            rep.check_indexed(f, getattr(got, f).reshape(got.nbody, -1),
                              getattr(ref, f).reshape(ref.nbody, -1),
                              body_names, kind="body")

    # ---- contact excludes -------------------------------------------------
    if got.nexclude == ref.nexclude and got.nexclude > 0:
        rep.check("exclude_signature.count", got.nexclude, ref.nexclude,
                  kind="exclude")

    # ---- the two physics terms both engines had been zeroing ---------------
    damping_live = {
        jnt_names[i]: float(ref.dof_damping[ref.jnt_dofadr[i]])
        for i in range(ref.njnt)
        if ref.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE
        and ref.dof_damping[ref.jnt_dofadr[i]] != 0.0
    }
    friction_live = {
        jnt_names[i]: float(ref.dof_frictionloss[ref.jnt_dofadr[i]])
        for i in range(ref.njnt)
        if ref.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE
        and ref.dof_frictionloss[ref.jnt_dofadr[i]] != 0.0
    }
    got_damping_nonzero = int(np.count_nonzero(got.dof_damping))
    got_friction_nonzero = int(np.count_nonzero(got.dof_frictionloss))
    physics_terms = {
        "mjcf_nonzero_dof_damping": len(damping_live),
        "mjlab_nonzero_dof_damping": got_damping_nonzero,
        "mjcf_nonzero_dof_frictionloss": len(friction_live),
        "mjlab_nonzero_dof_frictionloss": got_friction_nonzero,
        "dof_damping_sum_mjlab": float(np.sum(got.dof_damping)),
        "dof_damping_sum_mjcf": float(np.sum(ref.dof_damping)),
        "dof_frictionloss_sum_mjlab": float(np.sum(got.dof_frictionloss)),
        "dof_frictionloss_sum_mjcf": float(np.sum(ref.dof_frictionloss)),
        "damping_by_joint": damping_live,
        "frictionloss_by_joint": friction_live,
    }
    if got_damping_nonzero == 0 or got_friction_nonzero == 0:
        rep.mismatch.append({
            "field": "dof_damping/dof_frictionloss",
            "kind": "ZEROED",
            "detail": "mjlab model has all-zero damping and/or frictionloss",
        })

    # ---- named deviations -------------------------------------------------
    deviations = []
    if float(got.opt.noslip_tolerance) != float(ref.opt.noslip_tolerance) or \
       int(got.opt.noslip_iterations) != int(ref.opt.noslip_iterations):
        deviations.append({
            "name": "noslip",
            "mjcf": {"noslip_iterations": int(ref.opt.noslip_iterations),
                     "noslip_tolerance": float(ref.opt.noslip_tolerance)},
            "mjlab": {"noslip_iterations": int(got.opt.noslip_iterations),
                      "noslip_tolerance": float(got.opt.noslip_tolerance)},
            "cause": "MjSpec.attach() drops the child <option>; MujocoCfg has no "
                     "noslip field; MuJoCo Warp implements no noslip pass at all.",
            "verdict": "irreducible -- registered deviation, not a fixable bug",
        })

    # A mismatch that a named deviation already accounts for is expected, not a
    # regression.  Tag it, and let the exit code speak only for the rest.
    registered_fields = set()
    for d in deviations:
        if d["name"] == "noslip":
            registered_fields |= {"opt.noslip_iterations", "opt.noslip_tolerance"}
    n_unregistered = 0
    for mm in rep.mismatch:
        if mm.get("field") in registered_fields:
            mm["registered_deviation"] = True
        else:
            n_unregistered += 1

    result = {
        "xml": str(env.xml_path),
        "entity_prefix": prefix,
        "decimation": env.decimation,
        "physics_dt": env.physics_dt,
        "step_dt": env.step_dt,
        "n_match": len(rep.match),
        "n_mismatch": len(rep.mismatch),
        "n_unregistered_mismatch": n_unregistered,
        "match": rep.match,
        "mismatch": rep.mismatch,
        "physics_terms": physics_terms,
        "named_deviations": deviations,
    }

    if verbose:
        print("=" * 78)
        print("FIELD-BY-FIELD VERIFICATION  (mjlab env model  vs  raw MJCF model)")
        print("=" * 78)
        print(f"  xml            : {env.xml_path}")
        print(f"  entity prefix  : {prefix}")
        print(f"  decimation     : {env.decimation}  "
              f"(physics {env.physics_dt*1e3:.3f} ms -> policy {1/env.step_dt:.1f} Hz)")
        print(f"  matched groups : {len(rep.match)}")
        print(f"  mismatches     : {len(rep.mismatch)} "
              f"({n_unregistered} not covered by a named deviation)")
        print("-" * 78)
        if rep.mismatch:
            print("MISMATCHES")
            for m in rep.mismatch:
                print(json.dumps(m, indent=2, default=str))
        else:
            print("MISMATCHES: none")
        print("-" * 78)
        print("dof_damping / dof_frictionloss (the two terms both engines zeroed)")
        for k in ("mjcf_nonzero_dof_damping", "mjlab_nonzero_dof_damping",
                  "mjcf_nonzero_dof_frictionloss", "mjlab_nonzero_dof_frictionloss",
                  "dof_damping_sum_mjcf", "dof_damping_sum_mjlab",
                  "dof_frictionloss_sum_mjcf", "dof_frictionloss_sum_mjlab"):
            print(f"  {k:34s} = {physics_terms[k]}")
        print("-" * 78)
        print("NAMED DEVIATIONS")
        if deviations:
            for d in deviations:
                print(json.dumps(d, indent=2, default=str))
        else:
            print("  none")
        print("=" * 78)
    return result


# ---------------------------------------------------------------------------
# Throughput smoke.
# ---------------------------------------------------------------------------


def vendor_pd_for_joint_names(joint_names):
    """Resolve ``(kp, kd)`` for an ordered list of *unprefixed* joint names.

    Pure and model-free on purpose.  ``_pd_wiring`` below is the runtime caller
    and this is the one place the substring rule lives, so a host-side audit
    (``isaac_alignment.py``, which cannot compile a MuJoCo model) reads the SAME
    table through the SAME matcher instead of writing a second copy of the rule.
    A second copy is precisely how ``VENDOR_KP`` would drift away from the
    vendor Kp table it was hand-copied from without anybody noticing.

    Fails closed on an unmatched joint rather than leaving a silent ``0`` gain.
    """
    kp = np.zeros(len(joint_names), dtype=np.float64)
    kd = np.zeros(len(joint_names), dtype=np.float64)
    unmatched = []
    for i, jname in enumerate(joint_names):
        hit = False
        for key, val in VENDOR_KP.items():
            if key in jname:
                kp[i], hit = val, True
                break
        for key, val in VENDOR_KD.items():
            if key in jname:
                kd[i] = val
                break
        if not hit:
            unmatched.append(jname)
    if unmatched:
        raise RuntimeError(f"no vendor PD gain for joints: {unmatched}")
    return kp, kd


def _pd_wiring(env: A3PlantEnv):
    """Per-actuator kp/kd plus the explicit actuator -> qpos/qvel address map.

    Nothing here assumes the actuator list happens to be in joint-tree order:
    each actuator is followed through ``actuator_trnid`` to its joint, and from
    there to ``jnt_qposadr`` / ``jnt_dofadr``.
    """
    import mujoco

    m = env.mj_model
    prefix = env.entity_prefix
    q_adr = np.zeros(m.nu, dtype=np.int64)
    v_adr = np.zeros(m.nu, dtype=np.int64)
    names = []
    for a in range(m.nu):
        assert m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT
        jid = int(m.actuator_trnid[a, 0])
        q_adr[a] = m.jnt_qposadr[jid]
        v_adr[a] = m.jnt_dofadr[jid]
        jname = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid) or "")
        names.append(jname.removeprefix(prefix))
    kp, kd = vendor_pd_for_joint_names(names)
    return kp, kd, q_adr, v_adr


def _nvidia_smi() -> dict:
    """Snapshot nvidia-smi from inside the process, so the numbers provably
    belong to this run and to the GPU this run is pinned to."""
    import subprocess

    out = {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "")}
    try:
        out["gpus"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,"
             "utilization.gpu,power.draw", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip().splitlines()
        out["compute_procs"] = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip().splitlines()
        out["pid"] = os.getpid()
    except Exception as exc:  # pragma: no cover
        out["error"] = repr(exc)
    return out


def benchmark(env: A3PlantEnv, steps: int, warmup: int, ctrl_mode: str,
              trace_at: tuple[int, ...] = ()) -> dict:
    import torch

    sim = env.sim
    m = env.mj_model
    device = sim.device

    sim.reset()
    # Seed every world from the MJCF "stand" keyframe (nq = 7 + 31).
    assert m.nkey > 0, "expected the MJCF 'stand' keyframe to survive into the scene"
    key_qpos = torch.as_tensor(m.key_qpos[0], dtype=torch.float32, device=device)
    sim.data.qpos[:] = key_qpos.unsqueeze(0).expand(sim.num_envs, -1)
    sim.data.qvel[:] = 0.0
    sim.forward()
    z0 = float(sim.data.qpos[:, 2].mean())

    kp_np, kd_np, q_adr, v_adr = _pd_wiring(env)
    if ctrl_mode == "pd":
        kp = torch.as_tensor(kp_np, dtype=torch.float32, device=device)
        kd = torch.as_tensor(kd_np, dtype=torch.float32, device=device)
        qi = torch.as_tensor(q_adr, dtype=torch.long, device=device)
        vi = torch.as_tensor(v_adr, dtype=torch.long, device=device)
        q_des = torch.as_tensor(
            m.key_qpos[0][q_adr], dtype=torch.float32, device=device
        )
        lo = torch.as_tensor(m.actuator_ctrlrange[:, 0], dtype=torch.float32,
                             device=device)
        hi = torch.as_tensor(m.actuator_ctrlrange[:, 1], dtype=torch.float32,
                             device=device)

        def apply_ctrl():
            q = sim.data.qpos[:, qi]
            qd = sim.data.qvel[:, vi]
            tau = kp * (q_des.unsqueeze(0) - q) - kd * qd
            sim.data.ctrl[:] = torch.clamp(tau, lo, hi)
    else:
        sim.data.ctrl[:] = 0.0

        def apply_ctrl():
            return None

    def health(tag: str, step_idx: int) -> dict:
        qpos = sim.data.qpos
        qvel = sim.data.qvel
        row = {
            "tag": tag,
            "step": step_idx,
            "sim_t_s": step_idx * env.physics_dt,
            "pelvis_z_mean": float(qpos[:, 2].mean()),
            "pelvis_z_min": float(qpos[:, 2].min()),
            "pelvis_z_max": float(qpos[:, 2].max()),
            "qvel_absmax": float(qvel.abs().max()),
            "worlds_nonfinite": int((~torch.isfinite(qpos).all(dim=1)).sum()),
        }
        for name in ("nacon", "nefc"):
            arr = getattr(sim.data, name, None)
            if arr is not None:
                try:
                    row[f"{name}_max"] = int(torch.as_tensor(arr[:]).max())
                except Exception:
                    pass
        return row

    trace = [health("init", 0)]
    seen = 0
    for target in sorted(set(trace_at)):
        while seen < target:
            apply_ctrl()
            sim.step()
            seen += 1
        torch.cuda.synchronize()
        trace.append(health("trace", seen))
    if trace_at:
        # Restart from the keyframe so the timed loop is not contaminated by the
        # synchronizing trace above.
        sim.data.qpos[:] = key_qpos.unsqueeze(0).expand(sim.num_envs, -1)
        sim.data.qvel[:] = 0.0
        sim.forward()

    for _ in range(warmup):
        apply_ctrl()
        sim.step()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(steps):
        apply_ctrl()
        sim.step()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    # nvidia-smi taken while the whole simulation is still resident on the GPU,
    # sampled again mid-flight in a short unsynchronized burst so utilization is
    # not read at an idle instant.
    smi_after = _nvidia_smi()
    smi_busy = None
    burst = max(steps // 4, 50)
    t_busy = time.perf_counter()
    for _ in range(burst):
        apply_ctrl()
        sim.step()
        if smi_busy is None and time.perf_counter() - t_busy > 0.15:
            smi_busy = _nvidia_smi()
    torch.cuda.synchronize()

    # Always close the trace with a post-run row, so nefc_max_observed means
    # something even when --trace-at is empty.
    trace.append(health("final", warmup + steps + burst))

    qpos = sim.data.qpos.clone()
    qvel = sim.data.qvel.clone()
    pelvis_z = qpos[:, 2]
    n_nan = int(torch.isnan(qpos).any(dim=1).sum().item()) + \
        int(torch.isnan(qvel).any(dim=1).sum().item())
    n_inf = int(torch.isinf(qpos).any(dim=1).sum().item())

    total = steps * sim.num_envs
    out = {
        "nworld": sim.num_envs,
        "steps": steps,
        "warmup": warmup,
        "ctrl_mode": ctrl_mode,
        "cuda_graph": bool(sim.use_cuda_graph),
        "wall_s": dt,
        "steps_per_s": total / dt,
        "sim_seconds_per_wall_second": total * env.physics_dt / dt,
        "env_steps_per_s_at_decimation": total / dt / env.decimation,
        "naconmax": int(getattr(sim.wp_data, "naconmax", -1)),
        "njmax": int(getattr(sim.wp_data, "njmax", -1)),
        "pelvis_z_at_keyframe": z0,
        "nefc_max_observed": max(
            (r.get("nefc_max", 0) for r in trace), default=0
        ),
        "constraint_headroom_ok": max(
            (r.get("nefc_max", 0) for r in trace), default=0
        ) <= int(getattr(sim.wp_data, "njmax", 0)),
        "health_trace": trace,
        "pelvis_z_min": float(pelvis_z.min()),
        "pelvis_z_max": float(pelvis_z.max()),
        "pelvis_z_mean": float(pelvis_z.mean()),
        "qvel_absmax": float(qvel.abs().max()),
        "worlds_with_nan": n_nan,
        "worlds_with_inf": n_inf,
        "torch_cuda_mem_alloc_MiB": torch.cuda.memory_allocated() / 2**20,
        "torch_cuda_mem_reserved_MiB": torch.cuda.memory_reserved() / 2**20,
        "nvidia_smi_after_timed_loop": smi_after,
        "nvidia_smi_mid_flight": smi_busy,
    }
    print("=" * 78)
    print("THROUGHPUT SMOKE")
    print("=" * 78)
    for k, v in out.items():
        if k.startswith("nvidia_smi"):
            print(f"  {k}:")
            print(json.dumps(v, indent=4, default=str))
        elif k == "health_trace":
            print("  health_trace (fixed-pose PD hold; a pose PD is NOT a balance")
            print("               controller, so a slow topple is physics, not a bug):")
            for row in v:
                print("    " + json.dumps(row, default=str))
        else:
            print(f"  {k:34s} = {v}")
    print("=" * 78)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--xml", type=Path, default=None)
    p.add_argument("--nworld", type=int, default=4096)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--ctrl", choices=("zero", "pd"), default="pd")
    p.add_argument("--verify", action="store_true",
                   help="run the field-by-field audit")
    p.add_argument("--no-bench", action="store_true")
    p.add_argument("--trace-at", type=int, nargs="*",
                   default=[10, 50, 100, 200, 500, 1000, 2000],
                   help="physics-step indices at which to log a health row")
    p.add_argument("--nconmax", type=int, default=None)
    p.add_argument("--njmax", type=int, default=None,
                   help="per-world constraint rows; omit to auto-size from the "
                        "model (MuJoCo Warp's own heuristic is too small here)")
    p.add_argument("--raw-sizes", "--raw-njmax", dest="raw_sizes",
                   action="store_true",
                   help="use MuJoCo Warp's njmax/nconmax heuristics verbatim -- "
                        "reproduces the blow-up, for diagnosis only")
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args(argv)

    xml = args.xml or default_xml()
    print(f"[a3_plant_env] xml    = {xml}")
    print(f"[a3_plant_env] nworld = {args.nworld}  device = {args.device}")

    overrides = {}
    if args.nconmax is not None:
        overrides["nconmax"] = args.nconmax
    if args.njmax is not None:
        overrides["njmax"] = args.njmax

    env = build_env(xml, args.nworld, device=args.device,
                    auto_sizes=not args.raw_sizes, **overrides)

    payload = {"xml": str(xml), "nworld": args.nworld}
    if args.verify:
        payload["verification"] = verify(env)
    if not args.no_bench:
        payload["benchmark"] = benchmark(
            env, args.steps, args.warmup, args.ctrl, tuple(args.trace_at or ())
        )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, default=str))
        print(f"[a3_plant_env] wrote {args.json_out}")

    v = payload.get("verification")
    if v is not None and v["n_unregistered_mismatch"] > 0:
        return 1
    b = payload.get("benchmark")
    if b is not None and (b["worlds_with_nan"] or b["worlds_with_inf"]
                          or not b["constraint_headroom_ok"]):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
