#!/usr/bin/env python3
"""A3 plant + regulation table + net + ball, as one batched mjlab/MuJoCo-Warp scene.

WHAT THIS ADDS TO ``a3_plant_env.py`` (plain language)
------------------------------------------------------
The plant script proved we can carry the vendor's robot physics into
MuJoCo Warp field by field.  This script puts a court around it: an ITTF table,
a rendered regulation net, and a 40 mm ball, plus the **contact parameters** -- which
have no vendor authority at all and had to be derived from our own venue
measurements.

Three rules are kept from the plant work and one is added.

* stage 1 (MJCF owns it): robot bodies/geoms/joints/actuators/excludes, loaded
  verbatim.  The court is appended to the *scene* spec, never to the robot spec,
  so the robot model is bit-identical to the plant run.
* stage 2 (``SimulationCfg`` owns it): every ``<option>`` field, written out
  explicitly.
* stage 3 (this file owns it): the court geometry -- taken from
  ``tasks/table_tennis/geometry.py``, the single source of truth -- and the ball
  contact parameters, taken from ``calibrate_restitution.py``.
* new rule: *nothing about the robot may change*.  ``--verify`` re-checks every
  robot geom / joint / actuator / body / dof **by name** against the raw MJCF,
  not by index, because the model now has extra bodies in it.

FRAMES
------
The vendor MJCF ships its own floor plane at its own ``z = 0``, so the scene's
world frame *is* the robot's local ground frame.  The HOPE frame (table surface
at ``z = 0``, floor at ``z = -0.76``, origin at the near-left table corner) is
therefore a pure translation away::

    p_scene = p_hope + (0.5, 0.7625, 0.76)      # = -(robot ground origin in HOPE)

That offset is exactly ``-(P1_STAND_X, P1_STAND_Y, FLOOR_Z)`` from
``geometry.py``, so it is derived, not typed in twice.

THE BOUNCE RECEIPT GRADES ITSELF
--------------------------------
The bounce block carries a ``restitution_acceptance`` verdict from
``calibrate_restitution.restitution_verdict`` -- the same gates the calibration
script uses, so the two receipts cannot drift apart.  Two things to know:

* a run at a **single drop height** reports ``NOT_MEASURED`` for the two gates
  that can actually fail (impact-speed spread, e-vs-v_n slope), and the overall
  verdict is then ``NOT_MEASURED``.  **That is not a pass.**  The slope field is
  ``null``, never ``0.0`` -- a written-in zero reads downstream as "measured,
  and flat", which is a claim a single-height run did not make.
* ``FAIL`` exits 3.  ``NOT_MEASURED`` does not block, because a capacity census
  never claimed to measure the bounce; it just may not be cited as evidence.

Usage
-----
  python a3_court_env.py --verify --no-bench
  python a3_court_env.py --nworld 4096 --steps 500 --bounce-steps 900

  # measure e across the validated impact-speed envelope (this is the run that
  # turns the slope from NOT_MEASURED into a number)
  python a3_court_env.py --nworld 4096 --ctrl zero --height-sweep 1.0 4.5 \
      --bounce-steps 1400 --steps 200
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent


def _load(name: str, *candidates: Path):
    for cand in candidates:
        if cand and cand.is_file():
            spec = importlib.util.spec_from_file_location(name, cand)
            mod = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            # dataclasses resolves annotations through sys.modules, so the
            # module has to be registered *before* it executes.
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            mod.__source_path__ = str(cand)  # type: ignore[attr-defined]
            return mod
    raise FileNotFoundError(f"cannot locate {name}: tried {candidates}")


def _repo_root() -> Path | None:
    for parent in _HERE.parents:
        if (parent / ".git").exists():
            return parent
    return None


_ROOT = _repo_root()
_GEOM_REL = ("hope_training/whole_body_tracking/source/whole_body_tracking/"
             "whole_body_tracking/tasks/table_tennis/geometry.py")

plant = _load("a3_plant_env", _HERE / "a3_plant_env.py")
cal = _load("calibrate_restitution", _HERE / "calibrate_restitution.py")
geom = _load(
    "hope_geometry",
    Path(os.environ["HOPE_GEOMETRY_PY"]) if os.environ.get("HOPE_GEOMETRY_PY") else None,
    _HERE / "geometry.py",
    (_ROOT / _GEOM_REL) if _ROOT else None,
)

# --------------------------------------------------------------------------
# HOPE frame <-> scene (robot local ground) frame.
# --------------------------------------------------------------------------

ROBOT_GROUND_ORIGIN_HOPE = (geom.P1_STAND_X, geom.P1_STAND_Y, geom.FLOOR_Z)
HOPE_TO_SCENE = tuple(-v for v in ROBOT_GROUND_ORIGIN_HOPE)  # (0.5, 0.7625, 0.76)


def hope_to_scene(p) -> tuple[float, float, float]:
    return (p[0] + HOPE_TO_SCENE[0], p[1] + HOPE_TO_SCENE[1],
            p[2] + HOPE_TO_SCENE[2])


# --------------------------------------------------------------------------
# Ball contact parameters.  Derived, not guessed -- see calibrate_restitution.
# --------------------------------------------------------------------------

BALL_K = cal.DEFAULT_K                     # 1e3, fixed by the *resting* ball
BALL_MU = cal.BALL_FRICTION[0]             # 1.0
BALL_SOLIMP = cal.BALL_SOLIMP              # constant impedance, dmin == dmax
BALL_B = cal.calibrated_b()                # analytic seed + measured correction
BALL_SOLREF = cal.kb_to_solref(BALL_K, BALL_B)   # (-902.5, -1921.42)

# Friction-row reference for the elliptic cone.  Solved so the *sticking*
# tangential correction lands on the measured grip gain a_t = 0.369 instead of
# the rigid ceiling 0.4:  b_t = (a_t/0.4)/(ieff_t*dt), timeconst = 2/(dmax*b_t).
BALL_SOLREFFRICTION = (0.002236, 1.0)

# Historical diagnostic only: the net has NO measurement behind it, so this
# assumed response must not alter ball physics.  FullMDP keeps the net geom for
# rendering/robot geometry and adjudicates ball clearance from the observed
# centre crossing.  NET_SOLREF remains available to old alignment receipts but
# is not wired to a geom or explicit pair.
NET_E_ASSUMED = 0.10
NET_SOLREF = cal.kb_to_solref(
    BALL_K, cal.analytic_seed_b(NET_E_ASSUMED, BALL_K, mu=BALL_MU))

# Ball-racket.  NOT calibrated this round -- registered as a named gap -- but the
# recipe is measured: a direct-form solreffriction lets the tangential channel
# over-correct past the rigid stick ceiling 0.4, which is what a rubber sheet
# physically does.  (0, -1261.1) returns a_t = 0.522-0.535 against the measured
# paddle 0.52.  The paddle's *normal* e is velocity dependent
# (e = 0.759*exp(-0.0441 u_n)) and a static solref cannot express that.
RACKET_SOLREFFRICTION_RECIPE = (0.0, -1261.1)
RACKET_E_CONSTANT = 0.654

COURT_FRICTION = (BALL_MU, 0.005, 0.0001)

# --------------------------------------------------------------------------
# NAMED GAP -- AERODYNAMICS.  Deliberately NOT implemented this round.
#
# The fitted flight law is  a = g - k_d|v|v + k_m (omega x v)  with omega held
# constant (measured: fractional spin decay per 0.5 s is +2.4%, CI [-8.1,+10.7],
# i.e. consistent with zero, tau > 5 s at 95%).  Two ways to land it, both
# priced; the recommendation is A.
#
# ROUTE A -- external force injection (RECOMMENDED).  Write the force straight
#   into `data.xfrc_applied[:, ball_body, :3]` once per physics step.  Exact
#   functional match to the fit, ~4 lines, batched, and ZERO plant impact.
#   `aero_force_torch` below is that function, ready but unused.
#
# ROUTE B -- MuJoCo's native ellipsoid fluid model (REJECTED, and the reason is
#   NOT that the engine lacks it).  mujoco_warp DOES implement it, Magnus
#   included: io.py::_fluid_force computes
#       magnus_force = cross(omega, v) * (magnus_coef * density * volume)
#       drag         = density * |v| * A_proj * blunt_drag_coef * v
#   so our constants map exactly onto
#       blunt_drag = m*k_d/(rho*pi*R^2)          = 0.284317
#       magnus     = m*k_m/(rho*(4/3)*pi*R^3)    = 0.375407
#       slender_drag = 0   (A_max == A_proj for a sphere; the term vanishes)
#       ang_drag     = 0   (measured spin decay is nil inside the envelope)
#       kutta        = 0   (degenerate for a sphere)
#   REJECTED because `opt.density` is a GLOBAL option.  Setting it to 1.20 flips
#   `m.has_fluid` on and puts all 33 robot bodies into the inertia-box fluid
#   model (io.py: body_fluid_box covers every body with mass > 0 that is not an
#   ellipsoid), silently adding unmeasured drag to a plant we just verified
#   field by field.  That is a plant change; route A is not.
# --------------------------------------------------------------------------
AERO_K_D = 0.1261          # 1/m   venue fit, 100 low-spin arcs, C_d = 0.569
AERO_K_M = 0.00444         # 1/m   sidespin channel, 66 arcs
AIR_RHO = 1.20
AERO_NATIVE_FLUIDCOEF = (0.284317, 0.0, 0.0, 0.0, 0.375407)
AERO_ENVELOPE = {"ball_speed_m_s": (1.0, 7.0), "spin_rev_s": (0.0, 15.0),
                 "spin_ratio_covered": 0.5, "spin_ratio_empty_above": 1.6}


def aero_force_torch(vel, omega, mass: float = geom.BALL_MASS):
    """``F = m(-k_d|v|v + k_m (omega x v))`` -- route A, batched over worlds.

    Not wired into :func:`run`; this exists so the gap is closable in one line
    (`sim.data.xfrc_applied[:, ball_body, :3] = aero_force_torch(v, w)`) rather
    than re-derived later.  Applied once per *physics* step, whereas the fit was
    integrated with RK4 at h = 5e-4 -- that integration-order difference is a
    separate, smaller, registered gap.
    """
    import torch

    speed = torch.linalg.norm(vel, dim=-1, keepdim=True)
    return mass * (-AERO_K_D * speed * vel + AERO_K_M * torch.cross(omega, vel,
                                                                   dim=-1))

# Static-surface geoms keep the vendor's own contact stiffness so that
# *robot*-vs-table contacts behave like every other robot contact.  The ball's
# negative solref wins on ball contacts regardless: mujoco_warp mixes solref
# with elementwise min() whenever either side is non-positive.
SURFACE_SOLREF = (0.005, 1.0)

COURT_GEOM_NAMES = ("court_table_top", "court_net")
BALL_BODY = "ball"
BALL_GEOM = "ball_geom"
BALL_JOINT = "ball_free"
FLOOR_GEOM = "robot/floor"
RACKET_GEOMS = ("robot/right_racket_collision", "robot/right_racket_handle_collision")


# --------------------------------------------------------------------------
# Scene construction.
# --------------------------------------------------------------------------


def make_court_spec_fn(ball_pos_hope, cone: str, add_pairs: bool):
    """Return the ``SceneCfg.spec_fn`` that appends table, net and ball."""
    import mujoco

    ball_pos_scene = hope_to_scene(ball_pos_hope)

    def spec_fn(spec: "mujoco.MjSpec") -> None:
        wb = spec.worldbody

        def add_static(name, size, pos, rgba, solref, *, contact_type=1):
            g = wb.add_geom()
            g.name = name
            g.type = mujoco.mjtGeom.mjGEOM_BOX
            g.size = list(size)
            g.pos = list(pos)
            g.contype, g.conaffinity = contact_type, 7
            g.condim = 3
            g.friction = list(COURT_FRICTION)
            g.solref = list(solref)
            g.priority = 0
            g.rgba = list(rgba)
            return g

        tw, tl, tt = geom.TABLE_WIDTH, geom.TABLE_LENGTH, geom.TABLE_THICKNESS
        add_static(
            "court_table_top",
            (tl / 2.0, tw / 2.0, tt / 2.0),
            hope_to_scene(geom.table_top_center()),
            (0.05, 0.25, 0.45, 1.0),
            SURFACE_SOLREF,
        )
        nsx, nsy, nsz = geom.net_size()
        add_static(
            "court_net",
            (nsx / 2.0, nsy / 2.0, nsz / 2.0),
            hope_to_scene(geom.net_center()),
            (0.85, 0.85, 0.9, 0.6),
            SURFACE_SOLREF,
            # The ball uses source bit 8 and the net uses 16.  Both still
            # accept the vendor/court 1|2|4 categories, preserving every
            # robot-net and ball-table/racket/floor contact while filtering
            # only ball-net.
            contact_type=16,
        )

        body = wb.add_body()
        body.name = BALL_BODY
        body.pos = list(ball_pos_scene)
        # Explicit inertial: a *hollow* sphere is I = (2/3) m R^2.  Letting the
        # compiler infer it from the sphere geom would give the SOLID 2/5 and
        # silently break every spin-coupled bounce.
        body.explicitinertial = True
        body.mass = geom.BALL_MASS
        inertia = geom.BALL_INERTIA_COEFF * geom.BALL_MASS * geom.BALL_RADIUS**2
        body.inertia = [inertia, inertia, inertia]
        body.ipos = [0.0, 0.0, 0.0]
        body.iquat = [1.0, 0.0, 0.0, 0.0]

        j = body.add_joint()
        j.name = BALL_JOINT
        j.type = mujoco.mjtJoint.mjJNT_FREE
        # build_1 (PhysX) precedent: linear_damping = angular_damping = 0 so the
        # aerodynamic model is never double-counted by a viscous joint term.
        # MjSpec types these per-dof-group for free joints, so set both shapes.
        for attr in ("damping", "frictionloss", "armature", "stiffness"):
            for value in ([0.0, 0.0, 0.0], 0.0):
                try:
                    setattr(j, attr, value)
                    break
                except TypeError:
                    continue

        g = body.add_geom()
        g.name = BALL_GEOM
        g.type = mujoco.mjtGeom.mjGEOM_SPHERE
        g.size = [geom.BALL_RADIUS, 0.0, 0.0]
        g.contype, g.conaffinity = 8, 7
        g.condim = 3
        # priority 1 => the ball dictates condim/friction/solimp on every contact
        # it takes part in, so the calibration cannot be diluted by whatever the
        # other geom happens to carry.
        g.priority = 1
        g.friction = [BALL_MU, 0.005, 0.0001]
        g.solref = list(BALL_SOLREF)
        g.solimp = list(BALL_SOLIMP)
        g.mass = 0.0        # inertia comes from <inertial>, not from density
        g.density = 0.0
        g.rgba = [1.0, 0.55, 0.0, 1.0]

        if add_pairs:
            # solreffriction exists ONLY on <pair>, and only the elliptic cone
            # reads it (mujoco_warp constraint.py: the pyramidal branch never
            # does).  These pairs are what separate "how bouncy" from "how grippy".
            def add_pair(g2, solref, solreffriction):
                pr = spec.add_pair()
                pr.geomname1 = BALL_GEOM
                pr.geomname2 = g2
                pr.condim = 3
                pr.friction = [BALL_MU, BALL_MU, 0.005, 0.0001, 0.0001]
                pr.solref = list(solref)
                pr.solreffriction = list(solreffriction)
                pr.solimp = list(BALL_SOLIMP)

            add_pair("court_table_top", BALL_SOLREF, BALL_SOLREFFRICTION)
            add_pair(FLOOR_GEOM, BALL_SOLREF, BALL_SOLREFFRICTION)
            for rg in RACKET_GEOMS:
                add_pair(rg, BALL_SOLREF, BALL_SOLREFFRICTION)

        # mjlab already merged the robot's "stand" keyframe into a scene key;
        # the ball's 7 free-joint values have to be appended or the compiler
        # rejects the keyframe length.
        ball_key = list(ball_pos_scene) + [1.0, 0.0, 0.0, 0.0]
        for key in spec.keys:
            key.qpos = np.concatenate([np.asarray(key.qpos, dtype=float),
                                       np.asarray(ball_key, dtype=float)])

    return spec_fn


def build_court_env(xml_path: Path, num_envs: int, device: str,
                    ball_pos_hope, cone: str, add_pairs: bool,
                    njmax: int | None, nconmax: int | None,
                    ncon_per_world: int = 56):
    import mujoco
    from mjlab.scene import Scene
    from mjlab.sim.sim import Simulation

    scene_cfg = plant.build_scene_cfg(xml_path, num_envs, "robot")
    scene_cfg.spec_fn = make_court_spec_fn(ball_pos_hope, cone, add_pairs)
    scene_cfg.env_spacing = 6.0     # a court is 2.74 m long; 2.0 would overlap

    if njmax is None or nconmax is None:
        # Size from the *court* model, not the bare robot.
        probe = Scene(scene_cfg, device="cpu")
        ref = probe.compile()
        if njmax is None:
            njmax = plant.suggest_njmax(ref, ncon_per_world=ncon_per_world)
            print(f"[a3_court_env] njmax auto-sized to {njmax} from the court "
                  f"model (warp heuristic returns 64)")
        if nconmax is None:
            nconmax = plant.DEFAULT_NCONMAX_PER_WORLD
            print(f"[a3_court_env] nconmax = {nconmax}/world")
        del probe, ref

    overrides: dict[str, Any] = {"njmax": njmax, "nconmax": nconmax}
    if cone != "pyramidal":
        overrides["cone"] = cone
    sim_cfg = plant.build_sim_cfg(**overrides)

    scene = Scene(scene_cfg, device=device)
    sim = Simulation(
        num_envs=scene.num_envs, cfg=sim_cfg, spec=scene.spec,
        variant_info=scene.collect_variant_info(), device=device,
    )
    scene.initialize(mj_model=sim.mj_model, model=sim.model, data=sim.data)
    return plant.A3PlantEnv(scene=scene, sim=sim, decimation=plant.DECIMATION,
                            xml_path=xml_path, entity_prefix="robot/")


# --------------------------------------------------------------------------
# Verification.
# --------------------------------------------------------------------------


def verify_court(env, cone: str, add_pairs: bool, verbose: bool = True) -> dict:
    """Two audits in one: the robot is untouched, and the court is what we said.

    The robot half compares **by name**, so extra court bodies cannot mask a
    shifted index the way a positional compare would.
    """
    import mujoco

    got = env.mj_model
    ref = mujoco.MjModel.from_xml_path(str(env.xml_path))
    prefix = env.entity_prefix
    rep = plant.Report()
    OBJ = mujoco.mjtObj

    def idmap(obj_enum, n_ref):
        """raw-MJCF index -> court-model index.

        Names first.  The vendor MJCF leaves ~20 *visual* geoms unnamed, and the
        world body is by definition not part of the entity, so those two cases
        get resolved structurally instead of being dropped from the audit:
        an unnamed geom is located by (owning body, ordinal within that body),
        which is exactly as precise as a name.
        """
        out, missing = [], []
        for i in range(n_ref):
            nm = mujoco.mj_id2name(ref, obj_enum, i) or ""
            gid = mujoco.mj_name2id(got, obj_enum, prefix + nm) if nm else -1
            if gid < 0 and obj_enum == mujoco.mjtObj.mjOBJ_BODY and i == 0:
                gid = 0                      # world <-> world
            if gid < 0 and obj_enum == mujoco.mjtObj.mjOBJ_GEOM:
                bref = int(ref.geom_bodyid[i])
                bname = mujoco.mj_id2name(ref, mujoco.mjtObj.mjOBJ_BODY, bref) or ""
                bgot = (0 if bref == 0
                        else mujoco.mj_name2id(got, mujoco.mjtObj.mjOBJ_BODY,
                                               prefix + bname))
                if bgot >= 0:
                    ordinal = i - int(ref.body_geomadr[bref])
                    if 0 <= ordinal < int(got.body_geomnum[bgot]):
                        gid = int(got.body_geomadr[bgot]) + ordinal
            if gid < 0:
                missing.append(f"{obj_enum}[{i}]='{nm or '<unnamed>'}'")
            out.append(gid)
        return np.array(out, dtype=int), missing

    maps = {}
    for label, obj_enum, n in (("geom", OBJ.mjOBJ_GEOM, ref.ngeom),
                               ("joint", OBJ.mjOBJ_JOINT, ref.njnt),
                               ("actuator", OBJ.mjOBJ_ACTUATOR, ref.nu),
                               ("body", OBJ.mjOBJ_BODY, ref.nbody)):
        m, missing = idmap(obj_enum, n)
        maps[label] = m
        if missing:
            rep.mismatch.append({"field": f"{label}.name_resolution",
                                 "kind": "missing", "rows": missing[:20]})
        else:
            rep.match.append(f"{label}.names [{n} resolved under '{prefix}']")

    names = {
        k: [mujoco.mj_id2name(ref, e, i) or ""
            for i in range(n)]
        for k, e, n in (("geom", OBJ.mjOBJ_GEOM, ref.ngeom),
                        ("joint", OBJ.mjOBJ_JOINT, ref.njnt),
                        ("actuator", OBJ.mjOBJ_ACTUATOR, ref.nu),
                        ("body", OBJ.mjOBJ_BODY, ref.nbody))
    }

    def cmp(label, field, group):
        gm, rm = maps[group], None
        if np.any(gm < 0):
            return
        g_arr = getattr(got, field)
        r_arr = getattr(ref, field)
        n = r_arr.shape[0]
        rep.check_indexed(field, g_arr[gm].reshape(n, -1),
                          r_arr.reshape(n, -1), names[group], kind=group)

    for f in ("geom_solref", "geom_solimp", "geom_friction", "geom_condim",
              "geom_contype", "geom_conaffinity", "geom_priority",
              "geom_margin", "geom_gap"):
        cmp(f, f, "geom")
    for f in ("actuator_gear", "actuator_ctrlrange", "actuator_gainprm",
              "actuator_biasprm", "actuator_gaintype", "actuator_biastype",
              "actuator_ctrllimited", "actuator_forcerange",
              "actuator_forcelimited", "actuator_trntype", "actuator_dyntype",
              "actuator_actlimited"):
        cmp(f, f, "actuator")
    for f in ("jnt_range", "jnt_limited", "jnt_type", "jnt_axis", "jnt_pos",
              "jnt_stiffness", "jnt_margin", "jnt_solref", "jnt_solimp",
              "jnt_actfrcrange", "jnt_actfrclimited"):
        cmp(f, f, "joint")
    for f in ("body_mass", "body_inertia", "body_ipos", "body_iquat",
              "body_pos", "body_quat", "body_jntnum", "body_dofnum"):
        cmp(f, f, "body")

    # dof-indexed physics, gathered through the joint map so a failure names a
    # joint.  This is the pair both engines had been silently zeroing.
    jm = maps["joint"]
    dof_report = {}
    if not np.any(jm < 0):
        for f in ("dof_armature", "dof_damping", "dof_frictionloss",
                  "dof_solref", "dof_solimp"):
            g_rows, r_rows, lbl = [], [], []
            for i in range(ref.njnt):
                nd = {mujoco.mjtJoint.mjJNT_FREE: 6,
                      mujoco.mjtJoint.mjJNT_BALL: 3}.get(int(ref.jnt_type[i]), 1)
                for d in range(nd):
                    g_rows.append(np.atleast_1d(
                        getattr(got, f)[got.jnt_dofadr[jm[i]] + d]).ravel())
                    r_rows.append(np.atleast_1d(
                        getattr(ref, f)[ref.jnt_dofadr[i] + d]).ravel())
                    lbl.append(f"{names['joint'][i]}[{d}]")
            rep.check_indexed(f, np.array(g_rows), np.array(r_rows), lbl,
                              kind="dof")
        dof_report = {
            "robot_dof_damping_sum_court": float(sum(
                float(got.dof_damping[got.jnt_dofadr[jm[i]] + d])
                for i in range(ref.njnt)
                for d in range({mujoco.mjtJoint.mjJNT_FREE: 6,
                                mujoco.mjtJoint.mjJNT_BALL: 3}.get(
                                    int(ref.jnt_type[i]), 1)))),
            "robot_dof_damping_sum_mjcf": float(np.sum(ref.dof_damping)),
            "robot_dof_frictionloss_sum_court": float(sum(
                float(got.dof_frictionloss[got.jnt_dofadr[jm[i]] + d])
                for i in range(ref.njnt)
                for d in range({mujoco.mjtJoint.mjJNT_FREE: 6,
                                mujoco.mjtJoint.mjJNT_BALL: 3}.get(
                                    int(ref.jnt_type[i]), 1)))),
            "robot_dof_frictionloss_sum_mjcf": float(np.sum(ref.dof_frictionloss)),
        }

    # ---- opt.* ------------------------------------------------------------
    g_opt, r_opt = plant._opt_fields(got.opt), plant._opt_fields(ref.opt)
    for name in sorted(set(g_opt) | set(r_opt)):
        if name not in g_opt or name not in r_opt:
            rep.mismatch.append({"field": f"opt.{name}", "kind": "missing-attr"})
            continue
        rep.check(f"opt.{name}", g_opt[name], r_opt[name], kind="option")

    # ---- court audit ------------------------------------------------------
    def gid(n):
        return mujoco.mj_name2id(got, OBJ.mjOBJ_GEOM, n)

    court: dict[str, Any] = {}
    tt, nt = gid("court_table_top"), gid("court_net")
    bg = gid(BALL_GEOM)
    bb = mujoco.mj_name2id(got, OBJ.mjOBJ_BODY, BALL_BODY)
    bj = mujoco.mj_name2id(got, OBJ.mjOBJ_JOINT, BALL_JOINT)
    court["ids"] = {"table_top": tt, "net": nt, "ball_geom": bg,
                    "ball_body": bb, "ball_joint": bj}

    def in_hope(p):
        return [float(p[i] - HOPE_TO_SCENE[i]) for i in range(3)]

    court["geometry_hope_frame"] = {
        "hope_to_scene_offset": list(HOPE_TO_SCENE),
        "table_top_center_expected": list(geom.table_top_center()),
        "table_top_center_built": in_hope(got.geom_pos[tt]),
        "table_top_halfsize_expected": [geom.TABLE_LENGTH / 2,
                                        geom.TABLE_WIDTH / 2,
                                        geom.TABLE_THICKNESS / 2],
        "table_top_halfsize_built": got.geom_size[tt].tolist(),
        "table_surface_z_hope": float(got.geom_pos[tt][2] + got.geom_size[tt][2]
                                      - HOPE_TO_SCENE[2]),
        "net_center_expected": list(geom.net_center()),
        "net_center_built": in_hope(got.geom_pos[nt]),
        "net_halfsize_expected": [s / 2 for s in geom.net_size()],
        "net_halfsize_built": got.geom_size[nt].tolist(),
        "floor_z_hope": float(-HOPE_TO_SCENE[2]),
    }
    for label, exp, act in (
        ("table_top_center", geom.table_top_center(), in_hope(got.geom_pos[tt])),
        ("table_top_halfsize",
         [geom.TABLE_LENGTH / 2, geom.TABLE_WIDTH / 2, geom.TABLE_THICKNESS / 2],
         got.geom_size[tt].tolist()),
        ("net_center", geom.net_center(), in_hope(got.geom_pos[nt])),
        ("net_halfsize", [s / 2 for s in geom.net_size()],
         got.geom_size[nt].tolist()),
    ):
        rep.check(f"court.{label}", act, exp, atol=1e-9, kind="court")
    rep.check("court.table_surface_z_hope",
              float(got.geom_pos[tt][2] + got.geom_size[tt][2] - HOPE_TO_SCENE[2]),
              0.0, atol=1e-12, kind="court")

    inertia = geom.BALL_INERTIA_COEFF * geom.BALL_MASS * geom.BALL_RADIUS**2
    court["ball"] = {
        "mass_built": float(got.body_mass[bb]),
        "mass_expected": geom.BALL_MASS,
        "diaginertia_built": got.body_inertia[bb].tolist(),
        "diaginertia_expected": [inertia] * 3,
        "inertia_coeff_recovered": float(
            got.body_inertia[bb][0] / (got.body_mass[bb] * geom.BALL_RADIUS**2)),
        "radius_built": float(got.geom_size[bg][0]),
        "solref_built": got.geom_solref[bg].tolist(),
        "solimp_built": got.geom_solimp[bg].tolist(),
        "friction_built": got.geom_friction[bg].tolist(),
        "condim_built": int(got.geom_condim[bg]),
        "priority_built": int(got.geom_priority[bg]),
        "free_joint_dof_damping": [float(got.dof_damping[got.jnt_dofadr[bj] + d])
                                   for d in range(6)],
        "free_joint_dof_frictionloss": [
            float(got.dof_frictionloss[got.jnt_dofadr[bj] + d]) for d in range(6)],
        "free_joint_dof_armature": [float(got.dof_armature[got.jnt_dofadr[bj] + d])
                                    for d in range(6)],
        "qpos_adr": int(got.jnt_qposadr[bj]),
        "qvel_adr": int(got.jnt_dofadr[bj]),
    }
    rep.check("ball.mass", got.body_mass[bb], geom.BALL_MASS, atol=1e-12,
              kind="ball")
    rep.check("ball.diaginertia", got.body_inertia[bb], [inertia] * 3,
              atol=1e-15, kind="ball")
    rep.check("ball.radius", got.geom_size[bg][0], geom.BALL_RADIUS, atol=1e-12,
              kind="ball")
    rep.check("ball.solref", got.geom_solref[bg], list(BALL_SOLREF), atol=1e-9,
              kind="ball")
    rep.check("ball.solimp", got.geom_solimp[bg], list(BALL_SOLIMP), atol=1e-12,
              kind="ball")
    rep.check("ball.friction", got.geom_friction[bg],
              [BALL_MU, 0.005, 0.0001], atol=1e-12, kind="ball")
    rep.check("ball.condim", got.geom_condim[bg], 3, kind="ball")
    rep.check("ball.priority", got.geom_priority[bg], 1, kind="ball")
    rep.check("ball.free_joint_damping_is_zero",
              [float(got.dof_damping[got.jnt_dofadr[bj] + d]) for d in range(6)],
              [0.0] * 6, atol=0.0, kind="ball")
    rep.check("ball.free_joint_frictionloss_is_zero",
              [float(got.dof_frictionloss[got.jnt_dofadr[bj] + d])
               for d in range(6)], [0.0] * 6, atol=0.0, kind="ball")
    rep.check("ball.free_joint_armature_is_zero",
              [float(got.dof_armature[got.jnt_dofadr[bj] + d]) for d in range(6)],
              [0.0] * 6, atol=0.0, kind="ball")

    court["pairs"] = []
    for i in range(got.npair):
        court["pairs"].append({
            "geom1": mujoco.mj_id2name(got, OBJ.mjOBJ_GEOM,
                                       int(got.pair_geom1[i])),
            "geom2": mujoco.mj_id2name(got, OBJ.mjOBJ_GEOM,
                                       int(got.pair_geom2[i])),
            "dim": int(got.pair_dim[i]),
            "solref": got.pair_solref[i].tolist(),
            "solreffriction": got.pair_solreffriction[i].tolist(),
            "solimp": got.pair_solimp[i].tolist(),
            "friction": got.pair_friction[i].tolist(),
        })
    court["npair"] = int(got.npair)
    court["cone"] = ("elliptic" if int(got.opt.cone) ==
                     int(mujoco.mjtCone.mjCONE_ELLIPTIC) else "pyramidal")
    court["counts"] = {
        "nq": int(got.nq), "nv": int(got.nv), "nu": int(got.nu),
        "nbody": int(got.nbody), "ngeom": int(got.ngeom),
        "njnt": int(got.njnt),
        "nq_robot_reference": int(ref.nq), "nu_robot_reference": int(ref.nu),
        "ngeom_robot_reference": int(ref.ngeom),
    }
    rep.check("counts.nu_unchanged", got.nu, ref.nu, kind="count")
    rep.check("counts.nq_is_robot_plus_ball", got.nq, ref.nq + 7, kind="count")
    rep.check("counts.nv_is_robot_plus_ball", got.nv, ref.nv + 6, kind="count")
    rep.check("counts.ngeom_is_robot_plus_court", got.ngeom, ref.ngeom + 3,
              kind="count")

    # ---- named deviations -------------------------------------------------
    deviations = []
    if int(got.opt.noslip_iterations) != int(ref.opt.noslip_iterations):
        deviations.append({
            "name": "noslip",
            "mjcf": int(ref.opt.noslip_iterations),
            "built": int(got.opt.noslip_iterations),
            "cause": "MjSpec.attach() drops the child <option>; MujocoCfg has no "
                     "noslip field; MuJoCo Warp implements no noslip pass.",
            "verdict": "irreducible (inherited from the plant step)",
        })
    if court["cone"] != "pyramidal":
        deviations.append({
            "name": "cone",
            "mjcf": "pyramidal (MuJoCo default, vendor never wrote it down)",
            "built": court["cone"],
            "cause": "solreffriction -- the only way to stop a velocity-level "
                     "restitution from also multiplying the tangential impulse "
                     "-- is read ONLY by the elliptic branch of "
                     "mujoco_warp constraint.py::_efc_contact_update.",
            "effect_measured": "pyramidal gives a_t = 0.78 (2.1x the measured "
                               "0.369); elliptic + pair gives 0.370-0.380.",
            "verdict": "deliberate, registered",
        })

    registered = {"opt.noslip_iterations", "opt.noslip_tolerance"}
    if court["cone"] != "pyramidal":
        registered.add("opt.cone")
    n_unregistered = 0
    for mm in rep.mismatch:
        if mm.get("field") in registered:
            mm["registered_deviation"] = True
        else:
            n_unregistered += 1

    result = {
        "xml": str(env.xml_path),
        "geometry_module": getattr(geom, "__source_path__", "?"),
        "n_match": len(rep.match), "n_mismatch": len(rep.mismatch),
        "n_unregistered_mismatch": n_unregistered,
        "match": rep.match, "mismatch": rep.mismatch,
        "robot_dof_physics": dof_report,
        "court": court,
        "named_deviations": deviations,
    }
    if verbose:
        print("=" * 78)
        print("COURT VERIFICATION  (robot by name vs raw MJCF; court vs geometry.py)")
        print("=" * 78)
        print(f"  matched groups : {len(rep.match)}")
        print(f"  mismatches     : {len(rep.mismatch)} "
              f"({n_unregistered} not covered by a named deviation)")
        for m in rep.mismatch:
            print(json.dumps(m, indent=2, default=str))
        print("-" * 78)
        print("ROBOT dof_damping / dof_frictionloss carried into the court model")
        print(json.dumps(dof_report, indent=2, default=str))
        print("-" * 78)
        print("COURT")
        print(json.dumps(court, indent=2, default=str))
        print("-" * 78)
        print("NAMED DEVIATIONS")
        print(json.dumps(deviations, indent=2, default=str) if deviations
              else "  none")
        print("=" * 78)
    return result


# --------------------------------------------------------------------------
# Ready pose.
# --------------------------------------------------------------------------


def load_ready_pose_bytes(payload: bytes, source: str) -> dict:
    doc = json.loads(payload.decode("utf-8"))
    return {
        "joint_names": list(doc["robot"]["joint_names"]),
        "joint_pos": np.asarray(doc["physical_ready"]["joint_pos_rad"], float),
        "joint_vel": np.asarray(doc["physical_ready"]["joint_vel_radps"], float),
        # NOTE: this root pose is expressed in the ROBOT LOCAL GROUND frame
        # (floor at z = 0), which is exactly the scene frame -- not HOPE.
        "root_pos_local": np.asarray(doc["physical_ready"]["root_pos_w_m"], float),
        "root_quat_wxyz": np.asarray(doc["physical_ready"]["root_quat_wxyz"], float),
        "mujoco_row_for_runtime_joint": list(
            doc["hold_candidate"]["mujoco_row_for_runtime_joint"]),
        "source": source,
    }


def load_ready_pose(path: Path) -> dict:
    path = Path(path)
    return load_ready_pose_bytes(path.read_bytes(), str(path))


def ready_qpos(env, pose: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Scatter the 31 ready-pose angles onto the court model's qpos, by name."""
    import mujoco

    m = env.mj_model
    ref_jnt = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "robot/floating_base_joint")
    q_adr = np.zeros(31, dtype=np.int64)
    v_adr = np.zeros(31, dtype=np.int64)
    for i, nm in enumerate(pose["joint_names"]):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "robot/" + nm)
        if jid < 0:
            raise RuntimeError(f"ready-pose joint '{nm}' not in the court model")
        q_adr[i] = m.jnt_qposadr[jid]
        v_adr[i] = m.jnt_dofadr[jid]

    # Cross-check the JSON's own runtime->MuJoCo row table against the model.
    root_jid = int(np.argmin(np.where(m.jnt_type == mujoco.mjtJoint.mjJNT_FREE,
                                      np.arange(m.njnt), m.njnt)))
    root_qadr = int(m.jnt_qposadr[root_jid])
    rows_from_model = ((q_adr - root_qadr - 7)).tolist()
    consistency = {
        "root_free_joint_qpos_adr": root_qadr,
        "rows_from_model": rows_from_model,
        "rows_from_json": pose["mujoco_row_for_runtime_joint"],
        "agree": rows_from_model == list(pose["mujoco_row_for_runtime_joint"]),
    }

    qpos = np.array(m.key_qpos[0], dtype=np.float64) if m.nkey else np.zeros(m.nq)
    qpos[root_qadr:root_qadr + 3] = pose["root_pos_local"]
    qpos[root_qadr + 3:root_qadr + 7] = pose["root_quat_wxyz"]
    qpos[q_adr] = pose["joint_pos"]
    qvel = np.zeros(m.nv)
    qvel[v_adr] = pose["joint_vel"]
    return qpos, qvel, {"q_adr": q_adr, "v_adr": v_adr,
                        "root_qadr": root_qadr, "consistency": consistency}


# --------------------------------------------------------------------------
# Run: bounce probe + ready-pose hold + throughput.
# --------------------------------------------------------------------------


def run(env, pose: dict, ball_pos_hope, steps: int, warmup: int,
        bounce_steps: int, ctrl_mode: str,
        height_sweep: tuple[float, float] | None = None) -> dict:
    import mujoco
    import torch

    sim, m = env.sim, env.mj_model
    dev = sim.device
    bj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, BALL_JOINT)
    b_q, b_v = int(m.jnt_qposadr[bj]), int(m.jnt_dofadr[bj])
    # Read the cone off the BUILT model, not off a CLI string -- the effective
    # impedance (and therefore the predicted spread) depends on it.
    env_cone = ("elliptic" if int(m.opt.cone)
                == int(mujoco.mjtCone.mjCONE_ELLIPTIC) else "pyramidal")

    qpos0, qvel0, idx = ready_qpos(env, pose)
    ball_scene = hope_to_scene(ball_pos_hope)
    qpos0[b_q:b_q + 3] = ball_scene
    qpos0[b_q + 3:b_q + 7] = [1.0, 0.0, 0.0, 0.0]

    q0 = torch.as_tensor(qpos0, dtype=torch.float32, device=dev)
    v0 = torch.as_tensor(qvel0, dtype=torch.float32, device=dev)

    # Optionally give every world its own drop height, so one batched run
    # measures e across the whole validated impact-speed envelope at once.
    rest_z = HOPE_TO_SCENE[2] + geom.BALL_RADIUS
    if height_sweep is not None:
        v_lo, v_hi = height_sweep
        v_n = np.linspace(v_lo, v_hi, sim.num_envs)
        drop_h = v_n**2 / (2.0 * 9.81)
    else:
        drop_h = np.full(sim.num_envs, float(ball_scene[2] - rest_z))
    ball_z0 = torch.as_tensor(rest_z + drop_h, dtype=torch.float32, device=dev)

    # Fall + rebound has to fit inside the traced window or the apex is never
    # bracketed and `e` reads low.  Say so up front; the per-world bracketing
    # check below is the fail-closed half.
    _t_needed = (math.sqrt(2.0 * float(drop_h.max()) / 9.81)
                 * (1.0 + cal.E_TABLE_MEASURED) + 0.05)
    _steps_needed = int(math.ceil(_t_needed / env.physics_dt))
    if bounce_steps < _steps_needed:
        print(f"[a3_court_env] WARNING: --bounce-steps {bounce_steps} is short "
              f"for a {float(drop_h.max()):.3f} m drop; need >= "
              f"{_steps_needed}. Worlds whose apex is not bracketed will be "
              f"DROPPED from the restitution statistics, not silently biased.")

    def reset():
        sim.data.qpos[:] = q0.unsqueeze(0).expand(sim.num_envs, -1)
        sim.data.qvel[:] = v0.unsqueeze(0).expand(sim.num_envs, -1)
        sim.data.qpos[:, b_q + 2] = ball_z0
        sim.forward()

    sim.reset()
    reset()

    kp_np, kd_np, q_adr_act, v_adr_act = plant._pd_wiring(env)
    if ctrl_mode == "pd":
        kp = torch.as_tensor(kp_np, dtype=torch.float32, device=dev)
        kd = torch.as_tensor(kd_np, dtype=torch.float32, device=dev)
        qi = torch.as_tensor(q_adr_act, dtype=torch.long, device=dev)
        vi = torch.as_tensor(v_adr_act, dtype=torch.long, device=dev)
        q_des = torch.as_tensor(qpos0[q_adr_act], dtype=torch.float32, device=dev)
        lo = torch.as_tensor(m.actuator_ctrlrange[:, 0], dtype=torch.float32,
                             device=dev)
        hi = torch.as_tensor(m.actuator_ctrlrange[:, 1], dtype=torch.float32,
                             device=dev)

        def apply_ctrl():
            tau = kp * (q_des.unsqueeze(0) - sim.data.qpos[:, qi]) \
                - kd * sim.data.qvel[:, vi]
            sim.data.ctrl[:] = torch.clamp(tau, lo, hi)
    else:
        sim.data.ctrl[:] = 0.0

        def apply_ctrl():
            return None

    # ---- phase 1: bounce probe + ready-pose hold, traced ------------------
    root_qadr = idx["root_qadr"]
    ball_z = torch.zeros((bounce_steps + 1, sim.num_envs), dtype=torch.float32,
                         device=dev)
    pelvis_z = torch.zeros_like(ball_z)
    ball_z[0] = sim.data.qpos[:, b_q + 2]
    pelvis_z[0] = sim.data.qpos[:, root_qadr + 2]
    # Capacity peaks have to be sampled DURING this phase.  Reading them after
    # the timed loop (which starts from a fresh reset) would miss the sprawl,
    # which is the case that sized njmax/nconmax in the first place.
    nefc_arr = getattr(sim.data, "nefc", None)
    nacon_arr = getattr(sim.data, "nacon", None)
    peak_nefc = torch.zeros((), dtype=torch.int64, device=dev)
    peak_nacon = torch.zeros((), dtype=torch.int64, device=dev)
    for i in range(bounce_steps):
        apply_ctrl()
        sim.step()
        ball_z[i + 1] = sim.data.qpos[:, b_q + 2]
        pelvis_z[i + 1] = sim.data.qpos[:, root_qadr + 2]
        if nefc_arr is not None:
            torch.maximum(peak_nefc,
                          torch.as_tensor(nefc_arr[:]).max().to(torch.int64),
                          out=peak_nefc)
        if nacon_arr is not None:
            torch.maximum(peak_nacon,
                          torch.as_tensor(nacon_arr[:]).max().to(torch.int64),
                          out=peak_nacon)
    torch.cuda.synchronize()
    phase1_peaks = {"nefc_peak_per_world": int(peak_nefc),
                    "nacon_peak_all_worlds": int(peak_nacon),
                    "steps": bounce_steps}

    bz = ball_z.cpu().numpy().astype(np.float64)
    pz = pelvis_z.cpu().numpy().astype(np.float64)
    e_per_world, apex_per_world, pen_per_world, vn_per_world = [], [], [], []
    n_never_touched = n_never_left = n_unbracketed = 0
    for w in range(bz.shape[1]):
        z = bz[:, w]
        inside = z < rest_z
        ii = np.nonzero(inside)[0]
        if ii.size == 0:
            n_never_touched += 1
            continue
        i0 = int(ii[0])
        aft = np.nonzero(~inside[i0:])[0]
        if aft.size == 0:
            n_never_left += 1
            continue
        i1 = i0 + int(aft[0])
        pen_per_world.append(float(rest_z - z[i0:i1].min()))
        nxt = np.nonzero(inside[i1:])[0]
        i2 = i1 + int(nxt[0]) if nxt.size else len(z)
        seg = z[i1:i2]
        # The apex has to be BRACKETED.  If the trace stops while the ball is
        # still rising, max(seg) is a lower bound and `e` silently reads low --
        # exactly the kind of "measured" number that is not a measurement.
        if seg.size == 0 or (i2 >= len(z) and int(seg.argmax()) == seg.size - 1):
            n_unbracketed += 1
            continue
        apex = float(seg.max())
        apex_per_world.append(apex)
        e_per_world.append(math.sqrt(max(apex - rest_z, 0.0) / drop_h[w]))
        vn_per_world.append(math.sqrt(2 * 9.81 * drop_h[w]))
    e_arr = np.asarray(e_per_world)
    vn_arr = np.asarray(vn_per_world)

    distinct_vn = int(np.unique(np.round(vn_arr, 9)).size) if vn_arr.size else 0
    slope_measurable = (
        vn_arr.size > 2
        and distinct_vn >= cal.E_MIN_DISTINCT_IMPACT_SPEEDS
        and float(vn_arr.max() - vn_arr.min())
        >= cal.E_COVERAGE_FRACTION * (cal.V_N_ENVELOPE[1] - cal.V_N_ENVELOPE[0])
    )
    if slope_measurable:
        slope_val: float | None = float(np.polyfit(vn_arr, e_arr, 1)[0])
        slope_status = "MEASURED"
        slope_why = None
    else:
        # NEVER emit 0.0 here.  A written-in zero is read downstream as
        # "measured, and the slope is flat", which is a claim this run did not
        # make.  null + NOT_MEASURED is the honest pair.
        slope_val = None
        slope_status = "NOT_MEASURED"
        slope_why = (
            f"the run used {distinct_vn} distinct drop height(s) spanning "
            f"{float(vn_arr.max() - vn_arr.min()) if vn_arr.size else 0.0:.3f}"
            f" m/s of impact speed. A slope over the validated envelope "
            f"{cal.V_N_ENVELOPE[0]}-{cal.V_N_ENVELOPE[1]} m/s needs at least "
            f"{cal.E_MIN_DISTINCT_IMPACT_SPEEDS} distinct speeds covering "
            f"{cal.E_COVERAGE_FRACTION:.0%} of it. Re-run with --height-sweep "
            f"{cal.V_N_ENVELOPE[0]} {cal.V_N_ENVELOPE[1]}.")

    acceptance = cal.restitution_verdict(
        # dt comes from the built sim, not from the calibration constant: the
        # integrator artefact scales with dt^2, so a changed timestep must move
        # the gate with it.
        e_arr, vn_arr, k=BALL_K, mu=BALL_MU, cone=env_cone,
        dt=float(env.physics_dt), n_worlds=int(bz.shape[1]),
        context=f"a3_court_env bounce probe, {bz.shape[1]} worlds, "
                f"cone={env_cone}, "
                + ("height-sweep" if slope_measurable else "single drop height"))

    pelvis_drop = pz[-1] - pz[0]
    bounce = {
        "drop_height_m": [float(drop_h.min()), float(drop_h.max())],
        "impact_v_n_m_s": [float(vn_arr.min()), float(vn_arr.max())]
        if vn_arr.size else None,
        "e_vs_v_n_slope_per_m_s": slope_val,
        "e_vs_v_n_slope_status": slope_status,
        "e_vs_v_n_slope_not_measured_reason": slope_why,
        "n_distinct_impact_speeds": distinct_vn,
        "worlds_with_a_bounce": int(e_arr.size),
        "worlds_total": int(bz.shape[1]),
        "worlds_never_touched_table": n_never_touched,
        "worlds_never_separated": n_never_left,
        "worlds_apex_not_bracketed": n_unbracketed,
        "e_mean": float(e_arr.mean()) if e_arr.size else float("nan"),
        "e_std": float(e_arr.std()) if e_arr.size else float("nan"),
        "e_std_meaning": (
            "spread across DISTINCT impact speeds -- a response curve, not a "
            "sampling error" if distinct_vn > 1 else
            "mujoco-warp scheduling non-determinism between identical worlds; "
            "it is NOT a measurement uncertainty and must not be used to size "
            "an acceptance band"),
        "e_min": float(e_arr.min()) if e_arr.size else float("nan"),
        "e_max": float(e_arr.max()) if e_arr.size else float("nan"),
        "e_in_accept_band_all_worlds": bool(
            e_arr.size and np.all((e_arr >= cal.E_ACCEPT[0])
                                  & (e_arr <= cal.E_ACCEPT[1]))),
        "accept_band": list(cal.E_ACCEPT),
        "measured_authority_e": cal.E_TABLE_MEASURED,
        "restitution_acceptance": acceptance,
        "penetration_max_mm": float(1e3 * max(pen_per_world))
        if pen_per_world else float("nan"),
        "ball_z_final_hope": float(bz[-1].mean() - HOPE_TO_SCENE[2]),
    }
    ready = {
        "ready_pose_source": pose["source"],
        "row_map_agrees_with_json": idx["consistency"]["agree"],
        "pelvis_z_initial_m": float(pz[0].mean()),
        "pelvis_z_final_m": float(pz[-1].mean()),
        "pelvis_z_drop_mean_mm": float(1e3 * pelvis_drop.mean()),
        "pelvis_z_drop_max_mm": float(1e3 * pelvis_drop.min()),
        "pelvis_z_final_min_m": float(pz[-1].min()),
        "pelvis_z_final_max_m": float(pz[-1].max()),
        "worlds_fallen_below_0p8m": int((pz[-1] < 0.8).sum()),
        "hold_seconds": bounce_steps * env.physics_dt,
    }

    # ---- phase 2: throughput ---------------------------------------------
    reset()
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
    smi = plant._nvidia_smi()

    qpos, qvel = sim.data.qpos, sim.data.qvel
    total = steps * sim.num_envs
    nefc = 0
    nacon = 0
    for nm, dst in (("nefc", "nefc"), ("nacon", "nacon")):
        arr = getattr(sim.data, nm, None)
        if arr is not None:
            try:
                v = int(torch.as_tensor(arr[:]).max())
                if nm == "nefc":
                    nefc = v
                else:
                    nacon = v
            except Exception:
                pass

    bench = {
        "nworld": sim.num_envs, "steps": steps, "warmup": warmup,
        "ctrl_mode": ctrl_mode, "cuda_graph": bool(sim.use_cuda_graph),
        "wall_s": dt,
        "steps_per_s": total / dt,
        "sim_seconds_per_wall_second": total * env.physics_dt / dt,
        "env_steps_per_s_at_decimation": total / dt / env.decimation,
        "njmax": int(getattr(sim.wp_data, "njmax", -1)),
        "naconmax": int(getattr(sim.wp_data, "naconmax", -1)),
        "nefc_max_observed": nefc, "nacon_max_observed": nacon,
        "phase1_peaks": phase1_peaks,
        "nefc_peak_overall": max(nefc, phase1_peaks["nefc_peak_per_world"]),
        "nacon_peak_overall": max(nacon, phase1_peaks["nacon_peak_all_worlds"]),
        "njmax_headroom_x": int(getattr(sim.wp_data, "njmax", 0))
        / max(1, max(nefc, phase1_peaks["nefc_peak_per_world"])),
        "naconmax_headroom_x": int(getattr(sim.wp_data, "naconmax", 0))
        / max(1, max(nacon, phase1_peaks["nacon_peak_all_worlds"])),
        "constraint_headroom_ok": max(
            nefc, phase1_peaks["nefc_peak_per_world"]
        ) <= int(getattr(sim.wp_data, "njmax", 0)),
        "worlds_with_nan": int(torch.isnan(qpos).any(dim=1).sum().item()),
        "worlds_with_inf": int(torch.isinf(qpos).any(dim=1).sum().item()),
        "qvel_absmax": float(qvel.abs().max()),
        "torch_cuda_mem_reserved_MiB": torch.cuda.memory_reserved() / 2**20,
        "nvidia_smi": smi,
    }
    return {"bounce": bounce, "ready_hold": ready, "benchmark": bench,
            "row_map_consistency": idx["consistency"]}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--xml", type=Path, default=None)
    p.add_argument("--nworld", type=int, default=4096)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--bounce-steps", type=int, default=900)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--ctrl", choices=("zero", "pd"), default="pd")
    p.add_argument("--cone", choices=("pyramidal", "elliptic"), default="elliptic")
    p.add_argument("--no-pairs", action="store_true")
    p.add_argument("--verify", action="store_true")
    p.add_argument("--no-bench", action="store_true")
    p.add_argument("--njmax", type=int, default=None)
    p.add_argument("--nconmax", type=int, default=None)
    p.add_argument("--ready-pose", type=Path, default=None)
    p.add_argument("--ball-hope", type=float, nargs=3, default=None,
                   help="ball spawn in the HOPE frame; default = P2 half "
                        "centre, 0.35 m above the table")
    p.add_argument("--height-sweep", type=float, nargs=2, default=None,
                   metavar=("V_N_LO", "V_N_HI"),
                   help="give every world its own drop height so one batched "
                        "run measures e across the impact-speed envelope. "
                        "Without this the e-vs-v_n slope is reported as null / "
                        "NOT_MEASURED, never as 0.0. Use "
                        "`--height-sweep 1.0 4.5 --bounce-steps 1400` to cover "
                        "the validated envelope; the 1.032 m drop at v_n=4.5 "
                        "needs ~930 steps just to reach its apex")
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args(argv)

    xml = args.xml or plant.default_xml()
    ball_hope = tuple(args.ball_hope) if args.ball_hope else (
        geom.P2_HALF_CENTER[0], geom.P2_HALF_CENTER[1], 0.35)

    ready_path = args.ready_pose
    if ready_path is None:
        rel = ("configs/action_ball_n1_measured_20260803/"
               "evidence_holdpass_robust20n_20260803/"
               "take061.measured_teacher.yaw_aligned_full_seed.robust20n."
               "dynamic_ready.v2.json")
        for cand in ([_HERE / "ready_pose.json"]
                     + ([_ROOT / rel] if _ROOT else [])):
            if cand.is_file():
                ready_path = cand
                break
    if ready_path is None:
        raise SystemExit("no ready-pose JSON found; pass --ready-pose")

    print(f"[a3_court_env] xml        = {xml}")
    print(f"[a3_court_env] geometry   = {getattr(geom, '__source_path__', '?')}")
    print(f"[a3_court_env] ready pose = {ready_path}")
    print(f"[a3_court_env] nworld={args.nworld} cone={args.cone} "
          f"pairs={not args.no_pairs} device={args.device}")
    print(f"[a3_court_env] ball spawn HOPE={ball_hope} "
          f"scene={hope_to_scene(ball_hope)}")

    env = build_court_env(xml, args.nworld, args.device, ball_hope, args.cone,
                          not args.no_pairs, args.njmax, args.nconmax)

    payload: dict[str, Any] = {
        "xml": str(xml), "nworld": args.nworld, "cone": args.cone,
        "pairs": not args.no_pairs,
        "ball_spawn_hope": list(ball_hope),
        "ball_contact_parameters": {
            "solref": list(BALL_SOLREF),
            "solimp": list(BALL_SOLIMP),
            "friction": [BALL_MU, 0.005, 0.0001],
            "solreffriction_pair": list(BALL_SOLREFFRICTION),
            "k_physical": BALL_K, "b_physical": BALL_B,
            "net_solref_assumed": list(NET_SOLREF),
            "net_e_assumed": NET_E_ASSUMED,
        },
    }
    if args.verify:
        payload["verification"] = verify_court(env, args.cone, not args.no_pairs)
    if not args.no_bench:
        pose = load_ready_pose(ready_path)
        out = run(env, pose, ball_hope, args.steps, args.warmup,
                  args.bounce_steps, args.ctrl,
                  height_sweep=tuple(args.height_sweep)
                  if args.height_sweep else None)
        payload.update(out)
        print("=" * 78)
        print("BOUNCE / READY-HOLD / THROUGHPUT")
        print("=" * 78)
        for section in ("bounce", "ready_hold", "benchmark"):
            print(f"-- {section}")
            print(json.dumps(payload[section], indent=2, default=str))

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, default=str))
        print(f"[a3_court_env] wrote {args.json_out}")

    v = payload.get("verification")
    if v is not None and v["n_unregistered_mismatch"] > 0:
        return 1
    b = payload.get("benchmark")
    if b is not None and (b["worlds_with_nan"] or b["worlds_with_inf"]
                          or not b["constraint_headroom_ok"]):
        return 2
    # Restitution.  FAIL blocks; NOT_MEASURED does not block a capacity run
    # that never claimed to measure the bounce, but it is never PASS and the
    # receipt says so out loud.
    bo = payload.get("bounce")
    if bo is not None:
        acc = bo["restitution_acceptance"]
        print(f"[a3_court_env] restitution acceptance = {acc['verdict']}: "
              f"{acc['verdict_plain']}")
        for g in acc["gates"]:
            if g["verdict"] != "PASS":
                print(f"[a3_court_env]   [{g['verdict']}] {g['gate']}: "
                      f"stat={g['statistic']} limit={g['limit']}")
        if acc["verdict"] == "FAIL":
            return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
