#!/usr/bin/env python3
"""Where does the limp-robot (ctrl=0) free fall go non-finite?

Runs the identical drop three ways so the finger can be pointed at one of them:

  cpu      : plain single-threaded MuJoCo, `mj_step`, the MJCF as authored.
  warp1    : MuJoCo Warp through the mjlab env, nworld=1.
  warpN    : MuJoCo Warp through the mjlab env, nworld=N.

Same initial state everywhere: the MJCF "stand" keyframe, zero ctrl.
Reports the first physics step at which qpos/qvel stops being finite, plus the
state just before it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a3_plant_env import build_env, default_xml  # noqa: E402


def run_cpu(xml: Path, steps: int, opt_overrides: dict | None = None) -> dict:
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(xml))
    for k, v in (opt_overrides or {}).items():
        setattr(m.opt, k, v)
    d = mujoco.MjData(m)
    d.qpos[:] = m.key_qpos[0]
    d.qvel[:] = 0.0
    d.ctrl[:] = 0.0
    mujoco.mj_forward(m, d)

    prev = None
    trace = []
    for i in range(steps):
        mujoco.mj_step(m, d)
        ok = np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel))
        if i % 100 == 0 or not ok:
            trace.append({
                "step": i + 1,
                "pelvis_z": float(d.qpos[2]),
                "qvel_absmax": float(np.abs(d.qvel).max()) if ok else None,
                "ncon": int(d.ncon),
                "nefc": int(d.nefc),
                "finite": bool(ok),
            })
        if not ok:
            return {"backend": "cpu", "first_nonfinite_step": i + 1,
                    "state_before": prev, "trace": trace}
        prev = {"step": i + 1, "pelvis_z": float(d.qpos[2]),
                "qvel_absmax": float(np.abs(d.qvel).max()),
                "ncon": int(d.ncon), "nefc": int(d.nefc)}
    return {"backend": "cpu", "first_nonfinite_step": None,
            "final": prev, "trace": trace}


def run_warp(xml: Path, steps: int, nworld: int, device: str,
             auto_sizes: bool = True, **sim_overrides) -> dict:
    import torch

    env = build_env(xml, nworld, device=device, auto_sizes=auto_sizes,
                    **sim_overrides)
    sim = env.sim
    m = env.mj_model
    sim.reset()
    key = torch.as_tensor(m.key_qpos[0], dtype=torch.float32, device=sim.device)
    sim.data.qpos[:] = key.unsqueeze(0).expand(nworld, -1)
    sim.data.qvel[:] = 0.0
    sim.data.ctrl[:] = 0.0
    sim.forward()

    prev = None
    trace = []
    for i in range(steps):
        sim.step()
        qpos = sim.data.qpos
        qvel = sim.data.qvel
        finite = torch.isfinite(qpos).all(dim=1) & torch.isfinite(qvel).all(dim=1)
        n_bad = int((~finite).sum())
        if i % 100 == 0 or n_bad:
            good = finite.nonzero().squeeze(-1)
            print(f"    step {i+1:5d}  nonfinite={n_bad}  "
                  f"nacon_max={int(torch.as_tensor(sim.data.nacon[:]).max())}  "
                  f"nefc_max={int(torch.as_tensor(sim.data.nefc[:]).max())}",
                  flush=True)
            trace.append({
                "step": i + 1,
                "worlds_nonfinite": n_bad,
                "pelvis_z_mean_finite": float(qpos[good, 2].mean()) if good.numel() else None,
                "qvel_absmax_finite": float(qvel[good].abs().max()) if good.numel() else None,
                "nacon_max": int(torch.as_tensor(sim.data.nacon[:]).max()),
                "nefc_max": int(torch.as_tensor(sim.data.nefc[:]).max()),
            })
        if n_bad:
            return {"backend": f"warp{nworld}", "first_nonfinite_step": i + 1,
                    "worlds_nonfinite": n_bad, "state_before": prev,
                    "trace": trace, "sim_overrides": sim_overrides,
                    "njmax": int(sim.wp_data.njmax)}
        prev = {
            "step": i + 1,
            "pelvis_z_mean": float(qpos[:, 2].mean()),
            "qvel_absmax": float(qvel.abs().max()),
            "nacon_max": int(torch.as_tensor(sim.data.nacon[:]).max()),
            "nefc_max": int(torch.as_tensor(sim.data.nefc[:]).max()),
        }
    return {"backend": f"warp{nworld}", "first_nonfinite_step": None,
            "final": prev, "trace": trace, "sim_overrides": sim_overrides,
            "njmax": int(sim.wp_data.njmax)}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--xml", type=Path, default=None)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--nworld", type=int, default=4096)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--mode", choices=("cpu", "warp1", "warpN", "all"),
                   default="all")
    p.add_argument("--njmax", type=int, default=None)
    p.add_argument("--nconmax", type=int, default=None)
    p.add_argument("--raw-njmax", action="store_true",
                   help="use MuJoCo Warp's njmax heuristic verbatim")
    p.add_argument("--json-out", type=Path, default=None)
    a = p.parse_args()

    xml = a.xml or default_xml()
    over = {}
    if a.njmax is not None:
        over["njmax"] = a.njmax
    if a.nconmax is not None:
        over["nconmax"] = a.nconmax
    auto = not a.raw_njmax

    res = {"xml": str(xml), "steps": a.steps, "raw_njmax": a.raw_njmax}
    if a.mode in ("cpu", "all"):
        res["cpu"] = run_cpu(xml, a.steps)
        print(json.dumps({"cpu": res["cpu"]}, indent=2)[:4000])
    if a.mode in ("warp1", "all"):
        res["warp1"] = run_warp(xml, a.steps, 1, a.device, auto, **over)
        print(json.dumps({"warp1": res["warp1"]}, indent=2)[:4000])
    if a.mode in ("warpN", "all"):
        res["warpN"] = run_warp(xml, a.steps, a.nworld, a.device, auto, **over)
        print(json.dumps({"warpN": res["warpN"]}, indent=2)[:4000])

    if a.json_out:
        a.json_out.write_text(json.dumps(res, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
