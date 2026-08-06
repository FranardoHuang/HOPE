#!/usr/bin/env python3
"""How many contacts and constraint rows does one world actually need?

Plain English, one line per thing this script does:

  * it drives the robot instead of only dropping it.  ``ctrl=0`` sprawl is
    **not** the worst case: random torque that never leaves the model's own
    ``ctrlrange`` costs ~25% more constraint rows, and a random joint
    configuration inside ``jnt_range`` costs about twice as much.
  * it stops when the answer stops moving, not after a fixed 3000 steps.  A
    fixed window reports a *lower bound* and mislabels it a peak -- the
    2026-08-06 ``ctrl=0`` run was still setting new records at step 11,640
    of 12,000, so every "headroom Nx" derived from it was a lower bound.
  * it writes the friction cone into the receipt and refuses to diff two
    receipts taken under different cones.  Pyramidal spends ``2*(condim-1)``
    rows per contact, elliptic spends ``condim``; comparing across cones is
    what produced the fake "the table catches the robot, so rows go down"
    causality in the first write-up.
  * it reads the engine's own ``d.overflow`` sticky bitmask, so all nine
    overflow types are covered, including the four that never printf.

Fail-closed.  A census that did not converge, that saw any engine overflow
bit, that recorded no samples, or whose peak reached the reference
allocation exits non-zero, and its headroom numbers are emitted under
``*_lower_bound`` keys with ``usable_as_evidence: false`` so they cannot be
quoted as a margin.

mujoco-warp is non-deterministic: two identical invocations will not return
the same peak or the same step index.  Nothing here should be copied into a
constant.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

SCHEMA = "contact_census/2"

# ---------------------------------------------------------------------------
# Convergence rule.  Both halves must hold, per signal, per scenario.
#   1. no new record in the last K steps ("stall")
#   2. the peak landed in the first CONVERGE_FRACTION of the run
# (2) is the T9 rule from the readiness doc: peak_at_step > 0.7 * steps means
# the curve was still climbing when we stopped looking.
# ---------------------------------------------------------------------------
CONVERGE_FRACTION = 0.7
DEFAULT_STALL_STEPS = 3000
DEFAULT_MAX_STEPS = 20000
DEFAULT_MIN_STEPS = 3000

# The allocation the shipped trainer actually runs with.  Measurement runs use
# a deliberately larger allocation (so nothing is clipped) and then score the
# measured demand against these.
REF_NJMAX = 572
REF_NCONMAX_PER_WORLD = 128
REF_SOURCE = ("a3_court_env.py suggest_njmax(ncon_per_world=56) -> 572; "
              "a3_plant_env.DEFAULT_NCONMAX_PER_WORLD -> 128/world")

# mujoco_warp/_src/types.py OverflowType, bit index == tuple index.
OVERFLOW_FLAG_NAMES = (
    "NEFC",           # nefc > njmax                       (printf)
    "NJMAX_NNZ",      # sparse Jacobian nnz overflow       (printf)
    "BROADPHASE",     # ncollision > naconmax              (printf)
    "NARROWPHASE",    # contact count > naconmax           (printf)
    "CCD",            # convex-collision overflow          (printf)
    "HFIELD",         # height-field collision overflow    (silent)
    "CONTACT_MATCH",  # contact-match sensor overflow      (silent)
    "NVMAX",          # island solver overflow             (silent)
    "EPA_HORIZON",    # EPA horizon buffer overflow        (silent)
)

SCENARIOS_DEFAULT = "zero,flail,bang,randpose"

SCENARIO_HUMAN = {
    "zero": "limp robot: torque held at zero from the start pose, so it "
            "sprawls onto the floor.  This is what the old census called "
            "the worst case; it is not.",
    "flail": "random torque, redrawn every few steps, never leaving the "
             "model's own ctrlrange -- a completely legal policy output.",
    "bang": "bang-bang torque: every actuator pinned to one end of its "
            "ctrlrange, flipping sign periodically.  Also legal.",
    "randpose": "random joint angles inside jnt_range plus a random root "
                "orientation and height, released and re-drawn periodically.",
    "slam": "randpose, plus a downward root velocity so the robot is thrown "
            "at the table instead of dropped near it.",
}

SIGNAL_HUMAN = {
    "nefc_rows_per_world": "constraint rows one world needs; njmax is the cap",
    "nacon_contacts_all_worlds": "narrowphase contacts summed over all worlds; "
                                 "naconmax is the cap",
    "ncollision_broadphase_all_worlds": "broadphase candidate pairs summed "
                                        "over all worlds; naconmax is the cap "
                                        "for these too, and this is the one "
                                        "that actually binds",
    "contacts_per_world": "contacts owned by the busiest single world; "
                          "nconmax/world is the cap",
}


class CensusComparisonError(RuntimeError):
    """Raised when two censuses must not be compared (e.g. different cones)."""


# ---------------------------------------------------------------------------
# Pure logic.  No torch, no mujoco -- importable and unit-testable anywhere.
# ---------------------------------------------------------------------------


def decode_overflow_mask(mask: int) -> list:
    """Turn the engine's sticky bitmask into the flag names it stands for."""
    mask = int(mask)
    return [name for i, name in enumerate(OVERFLOW_FLAG_NAMES) if mask & (1 << i)]


def peak_and_step(values, first_step: int = 1, step_gap: int = 1):
    """First-attained maximum and the 1-based step it was attained at.

    Returns ``(None, 0)`` for an empty series -- "no samples", which is a
    distinct outcome from "measured, and the answer is zero".
    """
    best = None
    at = 0
    for i, v in enumerate(values):
        v = int(v)
        if best is None or v > best:
            best, at = v, first_step + i * step_gap
    return best, at


def running_max_series(values, stride: int, first_step: int = 1,
                       step_gap: int = 1) -> list:
    """``[[step, running max up to that step], ...]``, downsampled by stride.

    The final sample is always included, so the last entry is the peak.
    """
    out = []
    best = None
    n = len(values)
    for i, v in enumerate(values):
        v = int(v)
        best = v if best is None else max(best, v)
        if (i + 1) % stride == 0 or i == n - 1:
            out.append([first_step + i * step_gap, best])
    return out


def signal_convergence(peak_at_step: int, steps: int, stall_steps: int,
                       converge_fraction: float = CONVERGE_FRACTION) -> dict:
    """Did this one signal stop moving, or did we just stop watching?"""
    reasons = []
    stalled_for = max(0, steps - peak_at_step)
    if peak_at_step <= 0:
        reasons.append("no samples recorded")
    else:
        if stalled_for < stall_steps:
            reasons.append(
                "still climbing: only {0} steps since the last new record, "
                "need {1}".format(stalled_for, stall_steps))
        if peak_at_step > converge_fraction * steps:
            reasons.append(
                "peak landed at step {0} = {1:.0f}% of the run, past the "
                "{2:.0f}% mark".format(peak_at_step,
                                       100.0 * peak_at_step / max(1, steps),
                                       100.0 * converge_fraction))
    return {
        "converged": not reasons,
        "steps_since_last_new_record": stalled_for,
        "not_converged_because": reasons,
    }


def headroom_block(peak, capacity: int, converged: bool, denominator: str
                   ) -> dict:
    """Headroom, named so an unconverged number cannot be quoted as a margin.

    When the run converged the key is ``headroom_x``.  When it did not, the
    key is ``headroom_x_lower_bound`` and ``usable_as_evidence`` is false --
    the true demand is somewhere above what we measured.
    """
    out = {
        "capacity": int(capacity),
        "peak_measured": None if peak is None else int(peak),
        "denominator": denominator,
        "usable_as_evidence": bool(converged and peak),
    }
    if peak is None:
        out["headroom_x_unknown"] = None
        out["note"] = ("no samples recorded; a census that measured nothing "
                       "never signs a headroom claim")
        return out
    if int(peak) == 0:
        out["headroom_x_unknown"] = None
        out["note"] = ("measured peak is zero: this signal was never "
                       "exercised, so there is no ratio to report")
        return out
    key = "headroom_x" if converged else "headroom_x_lower_bound"
    out[key] = round(float(capacity) / float(peak), 3)
    return out


def scenario_verdict(signals: dict, overflow_flags, ref_hits: dict) -> tuple:
    """Fold one scenario's signals into a verdict plus a human sentence.

    Severity order, worst first.  "no samples" outranks everything because a
    census that measured nothing must never be able to sign a PASS.
    """
    missing = [k for k, s in signals.items()
               if s.get("required", True) and s.get("peak") is None]
    if missing:
        return "NO_SAMPLES", "nothing was measured for: " + ",".join(missing)
    if overflow_flags:
        return ("ENGINE_OVERFLOW",
                "the engine set overflow bits: " + ",".join(overflow_flags))
    over = [k for k, v in ref_hits.items() if v == "over"]
    if over:
        return ("OVER_REFERENCE_ALLOCATION",
                "measured demand exceeds the shipped allocation on: "
                + ",".join(over))
    at = [k for k, v in ref_hits.items() if v == "at"]
    if at:
        return ("AT_REFERENCE_ALLOCATION",
                "measured demand exactly fills the shipped allocation (zero "
                "headroom) on: " + ",".join(at))
    bad = [k for k, v in signals.items() if not v["convergence"]["converged"]]
    if bad:
        return ("NOT_CONVERGED",
                "still climbing when the run stopped: " + ",".join(bad))
    return "PASS_CONVERGED", "converged, and inside the shipped allocation"


_VERDICT_RANK = {
    "NO_SAMPLES": 0,
    "ENGINE_OVERFLOW": 1,
    "OVER_REFERENCE_ALLOCATION": 2,
    "AT_REFERENCE_ALLOCATION": 3,
    "NOT_CONVERGED": 4,
    "PASS_CONVERGED": 5,
}


def worst_verdict(verdicts) -> str:
    """Run-level verdict = the worst scenario's verdict.  Never the best."""
    verdicts = list(verdicts)
    if not verdicts:
        return "NO_SAMPLES"
    return min(verdicts, key=lambda v: _VERDICT_RANK.get(v, -1))


def assert_single_cone(receipt: dict) -> str:
    """Every scenario in one receipt must share the run's cone.

    Cheap, but this is the guard whose absence produced the "adding the table
    lowers the row count" claim: that diff was plant-pyramidal against
    court-elliptic, i.e. 4 rows per contact against 3.
    """
    run_cone = (receipt.get("cone") or {}).get("built")
    if not run_cone:
        raise CensusComparisonError("receipt does not record its friction cone")
    for name, scen in (receipt.get("scenarios") or {}).items():
        if scen.get("cone") != run_cone:
            raise CensusComparisonError(
                "scenario {0!r} ran under cone {1!r} but the receipt claims "
                "{2!r}".format(name, scen.get("cone"), run_cone))
    return run_cone


def compare_receipts(a: dict, b: dict, a_name: str = "A", b_name: str = "B"
                     ) -> dict:
    """Diff two censuses -- refusing, cell by cell, when the diff would be junk.

    Two different refusals, on purpose:

    * **Cone mismatch aborts the whole comparison.**  A pyramidal-vs-elliptic
      row delta is 4/3 of pure bookkeeping and says nothing about the scene.
      No cell of such a diff is salvageable.
    * **A signal that did not converge on either side drops that one cell.**
      Its peak is a lower bound, so the delta is unsigned.  Sibling signals
      that did converge still compare fine, which is why this is per-cell and
      not per-receipt.

    If nothing survives, the comparison itself is refused.
    """
    cone_a = assert_single_cone(a)
    cone_b = assert_single_cone(b)
    if cone_a != cone_b:
        raise CensusComparisonError(
            "refusing to compare {0} (cone={1}, {2} rows/contact) with {3} "
            "(cone={4}, {5} rows/contact): the row delta would be a cone "
            "conversion, not physics".format(
                a_name, cone_a, (a.get("cone") or {}).get("rows_per_contact"),
                b_name, cone_b, (b.get("cone") or {}).get("rows_per_contact")))
    shared = sorted(set(a["scenarios"]) & set(b["scenarios"]))
    rows = {}
    refused = []
    for scen in shared:
        cells = {}
        for sig in sorted(set(a["scenarios"][scen]["signals"])
                          & set(b["scenarios"][scen]["signals"])):
            sa = a["scenarios"][scen]["signals"][sig]
            sb = b["scenarios"][scen]["signals"][sig]
            ca = bool((sa.get("convergence") or {}).get("converged"))
            cb = bool((sb.get("convergence") or {}).get("converged"))
            if not (ca and cb):
                side = ([] if ca else [a_name]) + ([] if cb else [b_name])
                refused.append({
                    "scenario": scen, "signal": sig,
                    "reason": "peak is a lower bound on " + ",".join(side)
                              + "; an unsigned delta is not a comparison",
                })
                continue
            pa, pb = sa["peak"], sb["peak"]
            cells[sig] = {
                a_name: pa, b_name: pb,
                "delta": None if (pa is None or pb is None) else pb - pa,
                "ratio": None if not pa or pb is None else round(pb / float(pa), 3),
            }
        if cells:
            rows[scen] = cells
    if not rows:
        raise CensusComparisonError(
            "refusing to compare {0} with {1}: no signal converged on both "
            "sides, so every delta would be unsigned ({2} cells dropped)"
            .format(a_name, b_name, len(refused)))
    return {
        "cone": cone_a,
        "same_cone_enforced": True,
        "a": {"name": a_name, "scene": a.get("scene"), "nworld": a.get("nworld"),
              "verdict": a.get("verdict")},
        "b": {"name": b_name, "scene": b.get("scene"), "nworld": b.get("nworld"),
              "verdict": b.get("verdict")},
        "scenarios_compared": sorted(rows),
        "only_in_a": sorted(set(a["scenarios"]) - set(b["scenarios"])),
        "only_in_b": sorted(set(b["scenarios"]) - set(a["scenarios"])),
        "refused_cells": refused,
        "peaks": rows,
    }


def cone_name_from_int(cone_int: int) -> str:
    """MuJoCo mjtCone: 0 == pyramidal, 1 == elliptic."""
    return "pyramidal" if int(cone_int) == 0 else "elliptic"


def rows_per_contact(cone: str, condim: int) -> int:
    """What one contact costs in constraint rows under this cone."""
    condim = int(condim)
    if cone == "pyramidal":
        return 1 if condim == 1 else 2 * (condim - 1)
    return condim


# ---------------------------------------------------------------------------
# Measurement.  Everything below needs torch + mujoco + mjlab.
# ---------------------------------------------------------------------------


def _import_lane():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import a3_plant_env as plant  # noqa: E402

    return plant


def _find_ready_pose(explicit):
    if explicit is not None:
        return Path(explicit)
    here = Path(__file__).resolve().parent
    cand = here / "ready_pose.json"
    if cand.is_file():
        return cand
    for parent in here.parents:
        if (parent / ".git").exists():
            rel = (parent / "configs/action_ball_n1_measured_20260803/"
                            "evidence_holdpass_robust20n_20260803/"
                            "take061.measured_teacher.yaw_aligned_full_seed."
                            "robust20n.dynamic_ready.v2.json")
            if rel.is_file():
                return rel
    raise SystemExit("no ready-pose JSON found; pass --ready-pose")


class _Harness(object):
    """Builds the scene and holds the addresses the scenarios need."""

    def __init__(self, args):
        import numpy as np
        import mujoco
        import torch

        plant = _import_lane()
        self.np, self.mujoco, self.torch, self.plant = np, mujoco, torch, plant

        xml = args.xml or plant.default_xml()
        self.xml = xml
        if args.scene == "court":
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "a3_court_env",
                str(Path(__file__).resolve().parent / "a3_court_env.py"))
            court = importlib.util.module_from_spec(spec)
            sys.modules["a3_court_env"] = court
            assert spec.loader is not None
            spec.loader.exec_module(court)
            self.court = court
            ball_hope = (tuple(args.ball_hope) if args.ball_hope else
                         (court.geom.P2_HALF_CENTER[0],
                          court.geom.P2_HALF_CENTER[1], 0.35))
            self.env = court.build_court_env(
                xml, args.nworld, args.device, ball_hope, args.cone,
                not args.no_pairs, args.njmax, args.nconmax)
        else:
            self.court = None
            self.env = plant.build_env(xml, args.nworld, device=args.device,
                                       njmax=args.njmax, nconmax=args.nconmax,
                                       cone=args.cone)

        self.args = args
        self.sim = self.env.sim
        self.m = self.env.mj_model
        self.dev = self.sim.device
        self.N = args.nworld

        # --- the cone the engine actually built, not the one we asked for ---
        self.cone_built = cone_name_from_int(self.m.opt.cone)
        self.condim = int(self.m.geom_condim.max()) if self.m.ngeom else 3
        if self.cone_built != args.cone:
            raise SystemExit(
                "cone mismatch: asked for {0!r}, model reports {1!r}".format(
                    args.cone, self.cone_built))

        # --- start state -------------------------------------------------
        if self.court is not None:
            pose = self.court.load_ready_pose(_find_ready_pose(args.ready_pose))
            qpos0, qvel0, idx = self.court.ready_qpos(self.env, pose)
            bj = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT,
                                   self.court.BALL_JOINT)
            self.b_q = int(self.m.jnt_qposadr[bj])
            self.b_v = int(self.m.jnt_dofadr[bj])
            ball_scene = self.court.hope_to_scene(ball_hope)
            qpos0[self.b_q:self.b_q + 3] = ball_scene
            qpos0[self.b_q + 3:self.b_q + 7] = [1.0, 0.0, 0.0, 0.0]
            self.root_q = int(idx["root_qadr"])
            self.start_source = pose["source"]
        else:
            if not self.m.nkey:
                raise SystemExit("plant model has no keyframe to start from")
            qpos0 = np.array(self.m.key_qpos[0], dtype=np.float64)
            qvel0 = np.zeros(self.m.nv)
            self.b_q = self.b_v = None
            root_jid = int(np.argmin(np.where(
                self.m.jnt_type == mujoco.mjtJoint.mjJNT_FREE,
                np.arange(self.m.njnt), self.m.njnt)))
            self.root_q = int(self.m.jnt_qposadr[root_jid])
            self.start_source = "key_qpos[0]"

        root_jid = int(np.argmin(np.where(
            self.m.jnt_type == mujoco.mjtJoint.mjJNT_FREE,
            np.arange(self.m.njnt), self.m.njnt)))
        self.root_v = int(self.m.jnt_dofadr[root_jid])

        self.q0 = torch.as_tensor(qpos0, dtype=torch.float32, device=self.dev)
        self.v0 = torch.as_tensor(qvel0, dtype=torch.float32, device=self.dev)

        # --- actuator wiring, ctrlrange and jnt_range --------------------
        _kp, _kd, q_adr_act, v_adr_act = plant._pd_wiring(self.env)
        self.qi = torch.as_tensor(q_adr_act, dtype=torch.long, device=self.dev)
        self.vi = torch.as_tensor(v_adr_act, dtype=torch.long, device=self.dev)
        self.ctrl_lo = torch.as_tensor(self.m.actuator_ctrlrange[:, 0],
                                       dtype=torch.float32, device=self.dev)
        self.ctrl_hi = torch.as_tensor(self.m.actuator_ctrlrange[:, 1],
                                       dtype=torch.float32, device=self.dev)
        jr = np.zeros((self.m.nu, 2))
        unlimited = 0
        for k in range(self.m.nu):
            jid = int(self.m.actuator_trnid[k, 0])
            if self.m.jnt_limited[jid]:
                jr[k] = self.m.jnt_range[jid]
            else:
                jr[k] = (-np.pi, np.pi)
                unlimited += 1
        self.unlimited_joints = unlimited
        self.jlo = torch.as_tensor(jr[:, 0], dtype=torch.float32, device=self.dev)
        self.jhi = torch.as_tensor(jr[:, 1], dtype=torch.float32, device=self.dev)

        # --- capacity handles --------------------------------------------
        self.nefc_arr = self.sim.data.nefc
        self.nacon_arr = self.sim.data.nacon
        self.ncol_arr = getattr(self.sim.data, "ncollision", None)
        self.ovf_arr = getattr(self.sim.data, "overflow", None)
        if self.ovf_arr is None:
            raw = getattr(self.sim.wp_data, "overflow", None)
            if raw is not None:
                import warp as wp
                self.ovf_arr = wp.to_torch(raw)
        self.con_world = self.sim.data.contact.worldid
        self.naconmax = int(self.sim.wp_data.naconmax)
        self.njmax_alloc = int(self.sim.wp_data.njmax)
        self.con_idx = torch.arange(self.naconmax, device=self.dev)
        self.bit_shifts = torch.as_tensor(
            list(range(len(OVERFLOW_FLAG_NAMES))), dtype=torch.int64,
            device=self.dev)

        self.g = torch.Generator(device=self.dev)
        self.g.manual_seed(args.seed)

    # -- state setters ----------------------------------------------------

    def _rand(self, *shape, **kw):
        lo = kw.get("lo", 0.0)
        hi = kw.get("hi", 1.0)
        return (self.torch.rand(*shape, generator=self.g, device=self.dev)
                * (hi - lo) + lo)

    def set_start(self):
        self.sim.data.qpos[:] = self.q0.unsqueeze(0).expand(self.N, -1)
        self.sim.data.qvel[:] = self.v0.unsqueeze(0).expand(self.N, -1)
        self.sim.forward()

    def set_randpose(self, slam: bool):
        torch, np, m, N = self.torch, self.np, self.m, self.N
        self.sim.data.qpos[:] = self.q0.unsqueeze(0).expand(N, -1)
        self.sim.data.qvel[:] = 0.0
        u = self._rand(N, 3)
        quat = torch.stack([
            torch.sqrt(1 - u[:, 0]) * torch.sin(2 * np.pi * u[:, 1]),
            torch.sqrt(1 - u[:, 0]) * torch.cos(2 * np.pi * u[:, 1]),
            torch.sqrt(u[:, 0]) * torch.sin(2 * np.pi * u[:, 2]),
            torch.sqrt(u[:, 0]) * torch.cos(2 * np.pi * u[:, 2]),
        ], dim=1)
        rq = self.root_q
        self.sim.data.qpos[:, rq + 3:rq + 7] = quat
        if slam:
            self.sim.data.qpos[:, rq + 0] = self._rand(N, lo=-0.6, hi=1.4)
            self.sim.data.qpos[:, rq + 1] = self._rand(N, lo=-1.0, hi=1.0)
            self.sim.data.qpos[:, rq + 2] = self._rand(N, lo=0.85, hi=1.60)
            self.sim.data.qvel[:, self.root_v + 0] = self._rand(N, lo=-3.0, hi=3.0)
            self.sim.data.qvel[:, self.root_v + 1] = self._rand(N, lo=-3.0, hi=3.0)
            self.sim.data.qvel[:, self.root_v + 2] = self._rand(N, lo=-6.0, hi=-1.0)
        else:
            self.sim.data.qpos[:, rq + 0] = self._rand(N, lo=-0.5, hi=0.5)
            self.sim.data.qpos[:, rq + 1] = self._rand(N, lo=-0.5, hi=0.5)
            self.sim.data.qpos[:, rq + 2] = self._rand(N, lo=0.25, hi=1.10)
        rj = self.jlo.unsqueeze(0) + (self.jhi - self.jlo).unsqueeze(0) * \
            torch.rand((N, m.nu), generator=self.g, device=self.dev)
        self.sim.data.qpos[:, self.qi] = rj
        self.sim.data.qvel[:, self.vi] = self._rand(N, m.nu, lo=-4.0, hi=4.0)
        if self.b_q is not None:
            bq, bv = self.b_q, self.b_v
            self.sim.data.qpos[:, bq + 0] = self._rand(N, lo=-1.4, hi=1.4)
            self.sim.data.qpos[:, bq + 1] = self._rand(N, lo=-0.8, hi=0.8)
            self.sim.data.qpos[:, bq + 2] = self._rand(N, lo=0.78, hi=1.4)
            self.sim.data.qvel[:, bv + 0] = self._rand(N, lo=-12.0, hi=12.0)
            self.sim.data.qvel[:, bv + 1] = self._rand(N, lo=-6.0, hi=6.0)
            self.sim.data.qvel[:, bv + 2] = self._rand(N, lo=-15.0, hi=2.0)
        self.sim.forward()

    def overflow_bits(self):
        """OR of the engine's sticky per-world overflow mask over all worlds."""
        if self.ovf_arr is None:
            return None, 0
        torch = self.torch
        ovf = torch.as_tensor(self.ovf_arr[:]).to(torch.int64)
        bits = ((ovf.unsqueeze(1) >> self.bit_shifts) & 1).sum(0)
        nonzero = int((ovf != 0).sum())
        mask = 0
        bits_cpu = bits.detach().cpu().tolist()
        for i, c in enumerate(bits_cpu):
            if c:
                mask |= (1 << i)
        return mask, nonzero


def run_scenario(h, scen: str) -> dict:
    """Drive one scenario until its peaks stall, or until --max-steps."""
    torch = h.torch
    a = h.args
    N, m, dev = h.N, h.m, h.dev
    S_MAX = a.max_steps
    pw_every = a.pw_every

    h.sim.reset()
    if scen in ("randpose", "slam"):
        h.set_randpose(scen == "slam")
    else:
        h.set_start()
    h.sim.data.ctrl[:] = 0.0

    efc_ts = torch.zeros(S_MAX, dtype=torch.int32, device=dev)
    acon_ts = torch.zeros(S_MAX, dtype=torch.int32, device=dev)
    col_ts = torch.zeros(S_MAX, dtype=torch.int32, device=dev)
    pw_ts = torch.zeros(S_MAX // pw_every + 2, dtype=torch.int32, device=dev)
    over_ref_njmax = torch.zeros((), dtype=torch.int64, device=dev)
    over_ref_nconmax = torch.zeros((), dtype=torch.int64, device=dev)
    ref_njmax = a.ref_njmax
    ref_nconmax = a.ref_nconmax

    # incremental host-side peak trackers: (peak, first step it was attained)
    track = {k: [None, 0] for k in ("efc", "acon", "col", "pw")}
    ovf_mask, ovf_worlds = 0, 0

    def _update(key, host_vals, first_step, gap):
        p, at = peak_and_step(host_vals, first_step=first_step, step_gap=gap)
        if p is not None and (track[key][0] is None or p > track[key][0]):
            track[key][0], track[key][1] = p, at

    steps_done = 0
    j = 0
    stopped = "hit_max_steps"
    while steps_done < S_MAX:
        chunk_end = min(steps_done + a.check_every, S_MAX)
        for i in range(steps_done, chunk_end):
            if scen == "flail" and i % a.flail_resample == 0:
                h.sim.data.ctrl[:] = h.ctrl_lo + (h.ctrl_hi - h.ctrl_lo) * \
                    torch.rand((N, m.nu), generator=h.g, device=dev)
            elif scen == "bang" and i % a.bang_flip == 0:
                sign = torch.where(
                    torch.rand((N, m.nu), generator=h.g, device=dev) < 0.5,
                    -1.0, 1.0)
                h.sim.data.ctrl[:] = torch.where(sign > 0, h.ctrl_hi, h.ctrl_lo)
            elif scen in ("randpose", "slam") and i and i % a.reseed_every == 0:
                h.set_randpose(scen == "slam")
            h.sim.step()
            e = torch.as_tensor(h.nefc_arr[:])
            efc_ts[i] = e.max()
            over_ref_njmax += (e > ref_njmax).sum()
            acon_ts[i] = torch.as_tensor(h.nacon_arr[:]).max()
            if h.ncol_arr is not None:
                col_ts[i] = torch.as_tensor(h.ncol_arr[:]).max()
            if i % pw_every == 0:
                nc = torch.as_tensor(h.nacon_arr[:])[0]
                mask = (h.con_idx < nc).float()
                w = torch.as_tensor(h.con_world[:]).long().clamp_(0, N - 1)
                cnt = torch.zeros(N, device=dev).scatter_add_(0, w, mask)
                pw_ts[j] = cnt.max()
                over_ref_nconmax += (cnt > ref_nconmax).sum()
                j += 1
        torch.cuda.synchronize()
        new_lo, new_hi = steps_done, chunk_end
        _update("efc", efc_ts[new_lo:new_hi].cpu().tolist(), new_lo + 1, 1)
        _update("acon", acon_ts[new_lo:new_hi].cpu().tolist(), new_lo + 1, 1)
        _update("col", col_ts[new_lo:new_hi].cpu().tolist(), new_lo + 1, 1)
        j_lo = (new_lo + pw_every - 1) // pw_every
        if j > j_lo:
            _update("pw", pw_ts[j_lo:j].cpu().tolist(),
                    j_lo * pw_every + 1, pw_every)
        steps_done = chunk_end

        bits, worlds = h.overflow_bits()
        if bits is not None:
            ovf_mask |= bits
            ovf_worlds = max(ovf_worlds, worlds)

        print("  [{0}] step {1:6d}  peaks rows/world={2}  contacts/world={3}  "
              "nacon={4}  ncollision={5}  last-new @{6}".format(
                  scen, steps_done, track["efc"][0], track["pw"][0],
                  track["acon"][0], track["col"][0], track["efc"][1]),
              flush=True)

        if steps_done >= a.min_steps:
            # Stopping must satisfy BOTH halves of the convergence rule, or we
            # would stop into a NOT_CONVERGED verdict of our own making: the
            # 0.7 rule needs the last record to sit in the first 70% of the
            # run, which is a stall of at least 30% of the steps so far.
            need = max(a.stall_steps,
                       int(math.ceil((1.0 - CONVERGE_FRACTION) * steps_done)))
            stalls = [steps_done - track[k][1] for k in track
                      if track[k][0] is not None]
            if stalls and min(stalls) >= need:
                stopped = "stalled"
                break

    steps = steps_done
    series = {}
    stride = a.ts_stride
    series["nefc_rows_per_world"] = running_max_series(
        efc_ts[:steps].cpu().tolist(), stride)
    series["nacon_contacts_all_worlds"] = running_max_series(
        acon_ts[:steps].cpu().tolist(), stride)
    series["ncollision_broadphase_all_worlds"] = running_max_series(
        col_ts[:steps].cpu().tolist(), stride)
    pw_stride = max(1, stride // pw_every)
    series["contacts_per_world"] = running_max_series(
        pw_ts[:j].cpu().tolist(), pw_stride, first_step=1, step_gap=pw_every)

    caps = {
        "nefc_rows_per_world": ("njmax", ref_njmax),
        "nacon_contacts_all_worlds": ("naconmax", ref_nconmax * N),
        "ncollision_broadphase_all_worlds": ("naconmax", ref_nconmax * N),
        "contacts_per_world": ("nconmax_per_world", ref_nconmax),
    }
    keymap = {"nefc_rows_per_world": "efc",
              "nacon_contacts_all_worlds": "acon",
              "ncollision_broadphase_all_worlds": "col",
              "contacts_per_world": "pw"}

    signals = {}
    ref_hits = {}
    for name, (cap_name, cap) in caps.items():
        peak, at = track[keymap[name]]
        if name == "ncollision_broadphase_all_worlds" and h.ncol_arr is None:
            # never measured -- do not let a buffer of zeros masquerade as a
            # measurement that happened to come out zero.
            peak, at = None, 0
        conv = signal_convergence(at, steps, a.stall_steps)
        sample_gap = pw_every if name == "contacts_per_world" else 1
        signals[name] = {
            "human": SIGNAL_HUMAN[name],
            "peak": peak,
            "peak_at_step": at,
            "sample_every_steps": sample_gap,
            "bounded_by": cap_name,
            "reference_capacity": int(cap),
            "convergence": conv,
            "running_max_series": series[name],
            "series_stride_steps": (pw_every * pw_stride
                                    if name == "contacts_per_world" else stride),
            # non-required signals: ncollision is absent on engines that do not
            # expose it, and it is the only one allowed to be missing.
            "required": name != "ncollision_broadphase_all_worlds"
                        or h.ncol_arr is not None,
        }
        if peak:
            if peak > cap:
                ref_hits[name] = "over"
            elif peak == cap:
                ref_hits[name] = "at"

    conv_all = all(s["convergence"]["converged"] for s in signals.values()
                   if s["required"])
    flags = decode_overflow_mask(ovf_mask)
    verdict, why = scenario_verdict(signals, flags, ref_hits)

    naconmax_denom = max(signals["nacon_contacts_all_worlds"]["peak"] or 0,
                         signals["ncollision_broadphase_all_worlds"]["peak"] or 0)
    binding = ("ncollision"
               if (signals["ncollision_broadphase_all_worlds"]["peak"] or 0) >=
                  (signals["nacon_contacts_all_worlds"]["peak"] or 0)
               else "nacon")

    return {
        "scenario": scen,
        "human": SCENARIO_HUMAN.get(scen, "?"),
        "cone": h.cone_built,
        "steps_run": steps,
        "steps_max": S_MAX,
        "stopped_because": stopped,
        "stall_steps_required": a.stall_steps,
        "converge_fraction": CONVERGE_FRACTION,
        "signals": signals,
        "engine_overflow": {
            "read_from": ("d.overflow sticky per-world bitmask"
                          if h.ovf_arr is not None else "UNAVAILABLE"),
            "any": bool(flags),
            "flags": flags,
            "worlds_with_any_flag": ovf_worlds,
            "covers_all_nine_types": h.ovf_arr is not None,
        },
        "world_steps_over_reference_njmax": int(over_ref_njmax),
        "world_samples_over_reference_nconmax": int(over_ref_nconmax),
        "converged": conv_all,
        "verdict": verdict,
        "verdict_human": why,
        "headroom_vs_reference": {
            "njmax": headroom_block(signals["nefc_rows_per_world"]["peak"],
                                    ref_njmax, conv_all, "nefc rows/world"),
            "naconmax": headroom_block(
                naconmax_denom, ref_nconmax * N, conv_all,
                "max(nacon, ncollision) -- ncollision is >= contacts and is "
                "sized by the same naconmax, so it is the binding one; "
                "binding here = " + binding),
            "nconmax_per_world": headroom_block(
                signals["contacts_per_world"]["peak"], ref_nconmax, conv_all,
                "contacts owned by the busiest world"),
        },
    }


def _device_self_report(torch) -> dict:
    """Which card did this process actually get?  Not which one we asked for.

    ``CUDA_VISIBLE_DEVICES`` is an intention string; the uuid below is the
    card the process is provably running on, so the receipt can be checked
    against nvidia-smi after the fact.
    """
    import os

    out = {
        "cuda_visible_devices_requested": os.environ.get(
            "CUDA_VISIBLE_DEVICES", "<unset>"),
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER", "<unset>"),
    }
    try:
        out["visible_device_count"] = int(torch.cuda.device_count())
        props = torch.cuda.get_device_properties(0)
        out["device_name"] = props.name
        out["device_uuid"] = str(getattr(props, "uuid", "<unavailable>"))
        out["pci_bus_id"] = int(getattr(props, "pci_bus_id", -1))
    except Exception as exc:  # pragma: no cover - depends on the driver
        out["error"] = repr(exc)
    return out


def measure(args) -> dict:
    h = _Harness(args)
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    unknown = [s for s in scenarios if s not in SCENARIO_HUMAN]
    if unknown:
        raise SystemExit("unknown scenario(s): {0}".format(unknown))

    out = {
        "schema": SCHEMA,
        "scene": args.scene,
        "xml": str(h.xml),
        "nworld": args.nworld,
        "seed": args.seed,
        "start_state": h.start_source,
        "device": args.device,
        "gpu": _device_self_report(h.torch),
        "cone": {
            "requested": args.cone,
            "built": h.cone_built,
            "condim_max": h.condim,
            "rows_per_contact": rows_per_contact(h.cone_built, h.condim),
            "why_it_matters": "pyramidal spends 2*(condim-1) rows per contact "
                              "and elliptic spends condim, so the same physics "
                              "costs 4/3 as many rows under pyramidal.  Never "
                              "compare peaks across cones.",
        },
        "allocation_for_measurement": {
            "njmax": h.njmax_alloc,
            "nconmax_per_world": args.nconmax,
            "naconmax_all_worlds": h.naconmax,
            "naccdmax_all_worlds": int(getattr(h.sim.wp_data, "naccdmax", -1)),
            "note": "deliberately larger than the shipped allocation so the "
                    "engine never clips what we are trying to measure",
        },
        "reference_allocation": {
            "njmax": args.ref_njmax,
            "nconmax_per_world": args.ref_nconmax,
            "naconmax_all_worlds": args.ref_nconmax * args.nworld,
            "source": REF_SOURCE,
        },
        "convergence_rule": {
            "human": "a peak counts only if the curve stopped setting records "
                     "for {0} steps AND the record landed in the first {1:.0f}% "
                     "of the run".format(args.stall_steps,
                                         100 * CONVERGE_FRACTION),
            "stall_steps": args.stall_steps,
            "converge_fraction": CONVERGE_FRACTION,
            "min_steps": args.min_steps,
            "max_steps": args.max_steps,
        },
        "nondeterminism_warning": "mujoco-warp is non-deterministic; peaks and "
                                  "step indices below will not reproduce "
                                  "exactly.  Do not hard-code them.",
        "unlimited_actuated_joints_clamped_to_pi": h.unlimited_joints,
        "scenarios": {},
    }

    for scen in scenarios:
        print("== scenario {0}: {1}".format(scen, SCENARIO_HUMAN[scen]),
              flush=True)
        out["scenarios"][scen] = run_scenario(h, scen)
        rec = out["scenarios"][scen]
        print("   -> {0}: {1}".format(rec["verdict"], rec["verdict_human"]),
              flush=True)

    assert_single_cone(out)
    out["verdict"] = worst_verdict(s["verdict"] for s in out["scenarios"].values())
    out["verdict_human"] = {
        "PASS_CONVERGED": "every scenario converged and stayed inside the "
                          "shipped allocation",
        "NOT_CONVERGED": "at least one peak was still climbing when the run "
                         "stopped; its headroom is a LOWER BOUND and must not "
                         "be quoted as a margin",
        "AT_REFERENCE_ALLOCATION": "a peak exactly filled the shipped "
                                   "allocation: zero headroom",
        "OVER_REFERENCE_ALLOCATION": "a peak exceeded the shipped allocation: "
                                     "production would have dropped rows",
        "ENGINE_OVERFLOW": "the engine itself set an overflow bit",
        "NO_SAMPLES": "nothing was measured",
    }.get(out["verdict"], "?")

    if out["scenarios"]:
        worst = max(out["scenarios"].values(),
                    key=lambda s: s["signals"]["nefc_rows_per_world"]["peak"] or 0)
        out["worst_scenario"] = {
            "same_cone_enforced": True,
            "cone": h.cone_built,
            "by": "nefc rows per world",
            "name": worst["scenario"],
            "peak_rows_per_world": worst["signals"]["nefc_rows_per_world"]["peak"],
            "converged": worst["converged"],
            "headroom_vs_reference": worst["headroom_vs_reference"],
        }
    return out


def _compare_mode(paths) -> int:
    a = json.loads(Path(paths[0]).read_text())
    b = json.loads(Path(paths[1]).read_text())
    try:
        diff = compare_receipts(a, b, Path(paths[0]).name, Path(paths[1]).name)
    except CensusComparisonError as exc:
        print("REFUSED: {0}".format(exc), file=sys.stderr)
        return 2
    print(json.dumps(diff, indent=2))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"),
                   help="diff two receipts; refuses across cones or on any "
                        "receipt that did not converge")
    p.add_argument("--scene", choices=("plant", "court"), default="plant",
                   help="plant = bare robot; court = robot + table + net + ball")
    p.add_argument("--xml", type=Path, default=None)
    p.add_argument("--nworld", type=int, default=1024)
    p.add_argument("--cone", choices=("pyramidal", "elliptic"), default=None,
                   help="REQUIRED for a measurement run.  Peaks are only "
                        "comparable within one cone.")
    p.add_argument("--scenarios", default=SCENARIOS_DEFAULT,
                   help="comma list from: " + ",".join(sorted(SCENARIO_HUMAN)))
    p.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    p.add_argument("--min-steps", type=int, default=DEFAULT_MIN_STEPS)
    p.add_argument("--stall-steps", type=int, default=DEFAULT_STALL_STEPS,
                   help="stop once no signal has set a new record for this "
                        "many steps")
    p.add_argument("--check-every", type=int, default=500)
    p.add_argument("--ts-stride", type=int, default=100,
                   help="running-max time series is emitted every N steps")
    p.add_argument("--pw-every", type=int, default=20,
                   help="per-world contact census stride (the only sampled "
                        "signal; the rest are every step)")
    p.add_argument("--flail-resample", type=int, default=20)
    p.add_argument("--bang-flip", type=int, default=10)
    p.add_argument("--reseed-every", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--njmax", type=int, default=1024,
                   help="measurement allocation, deliberately generous")
    p.add_argument("--nconmax", type=int, default=192,
                   help="measurement allocation, deliberately generous")
    p.add_argument("--ref-njmax", type=int, default=REF_NJMAX,
                   help="the allocation the shipped trainer runs with")
    p.add_argument("--ref-nconmax", type=int, default=REF_NCONMAX_PER_WORLD)
    p.add_argument("--ready-pose", type=Path, default=None)
    p.add_argument("--ball-hope", type=float, nargs=3, default=None)
    p.add_argument("--no-pairs", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--json-out", type=Path, default=None)
    a = p.parse_args(argv)

    if a.compare:
        return _compare_mode(a.compare)
    if a.cone is None:
        p.error("--cone is required: a census without a declared friction "
                "cone cannot be compared with anything")
    if a.stall_steps > a.max_steps:
        p.error("--stall-steps cannot exceed --max-steps; the run could never "
                "satisfy its own convergence rule")

    out = {"schema": SCHEMA, "status": "crashed_before_finishing",
           "argv": sys.argv[1:]}
    try:
        out = measure(a)
        out["status"] = "complete"
        return 0 if out["verdict"] == "PASS_CONVERGED" else 1
    finally:
        # Receipts on the failure path too: the run that trips the gate is
        # exactly the run whose evidence people go looking for.
        print(json.dumps(out, indent=2, default=str))
        if a.json_out:
            a.json_out.write_text(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    sys.exit(main())
