#!/usr/bin/env python3
"""How many contacts and constraint rows does one world actually need?

MuJoCo Warp allocates the contact and constraint arrays statically, and its
built-in heuristics for this plant return nconmax=48 contacts/world and
njmax=64 rows/world regardless of nworld.  This script measures the real
per-world demand during the worst case we can produce cheaply -- a limp
(zero-torque) robot dropped from the vendor "stand" keyframe -- so the caps can
be sized from data instead of from a guess.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a3_plant_env import build_env, default_xml  # noqa: E402


def main() -> int:
    import torch

    p = argparse.ArgumentParser()
    p.add_argument("--xml", type=Path, default=None)
    p.add_argument("--nworld", type=int, default=1024)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--njmax", type=int, default=1024)
    p.add_argument("--nconmax", type=int, default=512)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--json-out", type=Path, default=None)
    a = p.parse_args()

    xml = a.xml or default_xml()
    env = build_env(xml, a.nworld, device=a.device,
                    njmax=a.njmax, nconmax=a.nconmax)
    sim = env.sim
    m = env.mj_model

    sim.reset()
    key = torch.as_tensor(m.key_qpos[0], dtype=torch.float32, device=sim.device)
    sim.data.qpos[:] = key.unsqueeze(0).expand(a.nworld, -1)
    sim.data.qvel[:] = 0.0
    sim.data.ctrl[:] = 0.0
    sim.forward()

    peak_con = 0
    peak_efc = 0
    peak_global_con = 0
    peak_collision = 0
    peak_at = {"contacts": 0, "rows": 0}
    for i in range(a.steps):
        sim.step()
        nacon = int(torch.as_tensor(sim.data.nacon[:]).max())
        peak_global_con = max(peak_global_con, nacon)
        ncol = getattr(sim.data, "ncollision", None)
        if ncol is not None:
            peak_collision = max(peak_collision,
                                 int(torch.as_tensor(ncol[:]).max()))
        # contact.worldid holds the owning world for each of the nacon live
        # contacts; bincount gives the per-world census.
        wid = torch.as_tensor(sim.data.contact.worldid[:nacon])
        per_world = torch.bincount(wid.long(), minlength=a.nworld)
        c = int(per_world.max())
        e = int(torch.as_tensor(sim.data.nefc[:]).max())
        if c > peak_con:
            peak_con, peak_at["contacts"] = c, i + 1
        if e > peak_efc:
            peak_efc, peak_at["rows"] = e, i + 1
        if (i + 1) % 500 == 0:
            print(f"  step {i+1:5d}  max contacts/world={c:4d}  "
                  f"max rows/world={e:4d}  (peaks {peak_con}/{peak_efc})",
                  flush=True)

    out = {
        "nworld": a.nworld,
        "steps": a.steps,
        "allocated_njmax": int(sim.wp_data.njmax),
        "allocated_nconmax_per_world": a.nconmax,
        "peak_contacts_per_world": peak_con,
        "peak_constraint_rows_per_world": peak_efc,
        "peak_contacts_all_worlds": peak_global_con,
        "allocated_naconmax_all_worlds": int(sim.wp_data.naconmax),
        "allocated_naccdmax_all_worlds": int(sim.wp_data.naccdmax),
        "peak_broadphase_candidates": peak_collision,
        "peak_at_step": peak_at,
        "mujoco_warp_heuristic_njmax": 64,
        "mujoco_warp_heuristic_nconmax_per_world": 48,
    }
    print(json.dumps(out, indent=2))
    if a.json_out:
        a.json_out.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
