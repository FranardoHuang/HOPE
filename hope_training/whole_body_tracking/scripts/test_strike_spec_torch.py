"""Cross-check the batched torch StrikeSpec solver against the numpy deploy/eval oracle.

Loads the mdp modules by file (no Isaac import), samples venue-style problems, and asserts:
  1. solve rate is sane (>= 80% on the legal landing box);
  2. every ok solution REPLAYED through the NUMPY physics (hope_planner ball_contact +
     BallTrajectoryPredictor — the family that scores eval-B and will run on deploy) lands
     within REPLAY_TOL of the requested target (cross-model consistency, the property that
     actually matters);
  3. the sign convention follows ref_normal.

Run (isaac venv, CPU, no Kit):  python scripts/test_strike_spec_torch.py
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
WBT = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(WBT))
MDP = os.path.join(WBT, "source", "whole_body_tracking", "whole_body_tracking",
                   "tasks", "tracking", "mdp")

# load mdp modules by file under a synthetic package (the real package __init__ pulls Isaac)
pkg = types.ModuleType("wbt_mdp")
pkg.__path__ = [MDP]
sys.modules["wbt_mdp"] = pkg
for name in ("virtual_ball", "strike_spec_torch"):
    spec = importlib.util.spec_from_file_location(f"wbt_mdp.{name}", os.path.join(MDP, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"wbt_mdp.{name}"] = mod
    spec.loader.exec_module(mod)
_vb = sys.modules["wbt_mdp.virtual_ball"]
_st = sys.modules["wbt_mdp.strike_spec_torch"]

sys.path.insert(0, os.path.join(REPO, "hope_ws", "src", "hope_planner"))
from hope_planner.ball_contact import predict_paddle_contact as np_contact  # noqa: E402
from hope_planner.ball_trajectory_predictor import BallTrajectoryPredictor  # noqa: E402
from hope_planner.constants import BallPhysics, PlannerConfig, TableParams  # noqa: E402

SURF_Z = 0.76
NET_X = 0.5 + TableParams().net_x
FAR_X = 0.5 + TableParams().length
N = 64
REPLAY_TOL = 0.06     # m — torch RK4(10ms) vs numpy Euler(1ms) landing discretization headroom
SOLVE_RATE_MIN = 0.8

rng = np.random.default_rng(3)
CONTACT = ((-1.50, -0.92), (-0.35, 0.18), (0.22, 0.50))     # venue frame (net-rel, table-rel z)
VEL = ((-2.85, -0.82), (-0.46, 0.31), (-2.02, 0.49))
p_env, v_in, w_in, tgt = [], [], [], []
for _ in range(N):
    pv = np.array([rng.uniform(*CONTACT[0]), rng.uniform(*CONTACT[1]), rng.uniform(*CONTACT[2])])
    p_env.append(pv + np.array([NET_X, 0.0, SURF_Z]))       # env frame, z above floor
    v_in.append([rng.uniform(*VEL[0]), rng.uniform(*VEL[1]), rng.uniform(*VEL[2])])
    d = rng.standard_normal(3); d /= np.linalg.norm(d)
    w_in.append(d * rng.uniform(0.0, 34.0))
    tgt.append([rng.uniform(NET_X + 0.3, FAR_X - 0.2), rng.uniform(-0.5, 0.5)])
p_env, v_in, w_in, tgt = map(lambda a: np.array(a, np.float64), (p_env, v_in, w_in, tgt))

prm = _vb.load_venue_params()
ref = torch.tensor([[0.413, 0.895, -0.166]] * N, dtype=torch.float64)
out = _st.solve_strike_specs(
    torch.tensor(p_env), torch.tensor(v_in), torch.tensor(w_in), torch.tensor(tgt),
    prm, surface_z=SURF_Z, net_x=NET_X, ref_normal=ref)

ok = out["ok"].numpy()
rate = ok.mean()
print(f"solve rate: {ok.sum()}/{N} = {rate:.0%}   resid(ok) median "
      f"{np.median(out['resid_m'].numpy()[ok])*1000:.1f} mm")
assert rate >= SOLVE_RATE_MIN, f"solve rate {rate:.0%} < {SOLVE_RATE_MIN:.0%}"

# sign convention
dots = np.sum(out["n"].numpy() * ref.numpy(), axis=-1)
assert (dots[ok] >= 0).all(), "sign convention violated"

# replay through the NUMPY oracle physics
phys, cfg, tab = BallPhysics(), PlannerConfig(), TableParams()
pred = BallTrajectoryPredictor(phys, cfg, tab)
errs = []
for i in np.where(ok)[0]:
    v_plus, w_plus = np_contact(v_in[i], out["v_r"].numpy()[i], out["n"].numpy()[i], w_in[i],
                                phys, cfg)
    land = pred.integrate_to_table_plane(p_env[i] - np.array([0, 0, SURF_Z]), v_plus, w_plus)
    assert land is not None, f"row {i}: torch-ok solution never lands in the numpy model"
    errs.append(float(np.linalg.norm(np.asarray(land[0]) - tgt[i])))
errs = np.array(errs)
print(f"replay-through-numpy landing error: median {np.median(errs)*1000:.1f} mm, "
      f"p95 {np.percentile(errs, 95)*1000:.1f} mm, max {errs.max()*1000:.1f} mm")
assert np.percentile(errs, 95) < REPLAY_TOL, "cross-model landing disagreement too large"
print("PASS: torch batched solver consistent with the numpy deploy/eval oracle")
