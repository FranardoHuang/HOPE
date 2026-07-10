#!/usr/bin/env python3
"""Isaac in-loop ball FLIGHT validation: PhysX integration vs the venue-fitted RK4 model.

The long-pending check behind "sim 里的球物理属性才有意义": the training/eval stack
integrates ball flight analytically (tracking/mdp/virtual_ball.py, venue constants), while
a physical in-engine ball would be integrated by PhysX (semi-implicit Euler at the training
physics dt) with our aero force injected per step. This script quantifies that gap on real
question-bank answers (or a synthetic envelope sweep) WITHOUT a robot:

  spawn K collision-free balls -> set (p, v_plus, omega_plus) -> each physics step apply
  F = m * (-k_d |v| v + k_m (omega x v)) read from the SIM state -> record trajectories ->
  first descending crossing of z = surface + R (linear interp) -> compare landing xy and
  flight time against the RK4 reference (h = 1 ms) from identical initial states.

Honest boundaries: FLIGHT ONLY. Table bounce and racket contact stay code-driven in every
consumer and are already cross-verified analytically (numpy oracle / torch / C++ to
sub-micron); the in-engine bounce validation is a separate follow-up. Angular velocity is
set once at spawn (angular damping 0, no torques) — the constant-omega assumption of the
fit; the script also reports |omega_end - omega_0| to prove PhysX kept it.

PASS bar (--tol-mm, default p90 <= 20 mm): PhysX Euler @ 5 ms vs RK4 @ 1 ms on a ~0.5 s
arc is expected at the few-mm level; 20 mm is far below both the sigma = 0.3 m landing
reward kernel and the 0.10 m exam bar, so a PASS means the engine-integrated ball is
interchangeable with the analytic one for training/exam purposes.

Run (pod, Isaac venv):
    python scripts/isaac_ball_inloop_check.py --bank /workspace/yikang/s1_v5cal_train.npz --n 128
    python scripts/isaac_ball_inloop_check.py --synthetic --n 125
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WBT = os.path.dirname(HERE)
_MDP = os.path.join(WBT, "source", "whole_body_tracking", "whole_body_tracking",
                    "tasks", "tracking", "mdp")


def _load_mod(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_virtual_ball():
    return _load_mod("ibc_virtual_ball", os.path.join(_MDP, "virtual_ball.py"))


# --------------------------------------------------------------------------- #
# pure helpers (importable without Isaac; unit-tested)
# --------------------------------------------------------------------------- #
def states_from_bank(bank_path: str, vb, prm, n_cap: int):
    """Post-contact ball states for every bank question (same math as the generator self-test).

    Returns (p0, v0, w0) float64 torch tensors of shape (N, 3), N <= n_cap.
    """
    import torch

    z = np.load(bank_path, allow_pickle=True)
    clips = sorted({k.split("/")[0] for k in z.files if "/" in k})
    ps, vs, ws = [], [], []
    for c in clips:
        vin = torch.tensor(np.asarray(z[f"{c}/incoming_vel"]), dtype=torch.float64)
        if not len(vin):
            continue
        spin = torch.tensor(np.asarray(z[f"{c}/incoming_spin"]), dtype=torch.float64)
        dvel = torch.tensor(np.asarray(z[f"{c}/demanded_vel"]), dtype=torch.float64)
        dnrm = torch.tensor(np.asarray(z[f"{c}/demanded_normal"]), dtype=torch.float64)
        v_plus, w_plus = vb.predict_paddle_contact(vin, dvel, dnrm, spin, prm)
        p = torch.tensor(np.asarray(z[f"{c}/contact_pos_env"]), dtype=torch.float64)
        ps.append(p.expand(len(vin), 3).clone())
        vs.append(v_plus)
        ws.append(w_plus)
    p0 = torch.cat(ps)[:n_cap]
    v0 = torch.cat(vs)[:n_cap]
    w0 = torch.cat(ws)[:n_cap]
    return p0, v0, w0


def states_synthetic(n_cap: int):
    """Deterministic envelope sweep: speeds 1-7 m/s x directions x spins 0-95 rad/s."""
    import torch

    speeds = np.linspace(1.0, 7.0, 5)
    yaws = np.radians(np.linspace(-25.0, 25.0, 5))
    pitches = np.radians(np.linspace(5.0, 45.0, 5))
    spins = np.linspace(0.0, 95.0, 5)
    states = []
    for s in speeds:
        for yaw in yaws:
            for pitch in pitches:
                spin = spins[len(states) % len(spins)]
                v = [s * np.cos(pitch) * np.cos(yaw), s * np.cos(pitch) * np.sin(yaw),
                     s * np.sin(pitch)]
                # topspin about the horizontal axis perpendicular to travel
                w = [-spin * np.sin(yaw), spin * np.cos(yaw), 0.0]
                states.append((v, w))
    states = states[:n_cap]
    p0 = torch.tensor([[0.5, 0.0, 0.95]], dtype=torch.float64).expand(len(states), 3).clone()
    v0 = torch.tensor([s[0] for s in states], dtype=torch.float64)
    w0 = torch.tensor([s[1] for s in states], dtype=torch.float64)
    return p0, v0, w0


def reference_landing(p0, v0, w0, vb, prm, contact_z: float, h: float = 1.0e-3,
                      max_time: float = 2.0):
    """RK4 (h = 1 ms) first descending crossing of z = contact_z, linear interp.

    Returns (xy (N,2), t (N,), valid (N,)) float64 torch tensors.
    """
    import torch

    p, v = p0.clone(), v0.clone()
    n = p.shape[0]
    landed = torch.zeros(n, dtype=torch.bool)
    xy = torch.zeros(n, 2, dtype=torch.float64)
    t_land = torch.zeros(n, dtype=torch.float64)
    steps = int(max_time / h)
    for k in range(steps):
        p_new, v_new = vb.rk4_step(p, v, w0, h, prm)
        cross = (~landed) & (p[:, 2] > contact_z) & (p_new[:, 2] <= contact_z)
        if cross.any():
            f = ((p[:, 2] - contact_z) / (p[:, 2] - p_new[:, 2]).clamp(min=1e-12)).clamp(0, 1)
            xy[cross] = (p[:, :2] + (p_new[:, :2] - p[:, :2]) * f.unsqueeze(-1))[cross]
            t_land[cross] = (k + f[cross]) * h
            landed |= cross
        p, v = p_new, v_new
        if bool(landed.all()):
            break
    return xy, t_land, landed


def crossing_from_history(zs, xys, ts, contact_z: float):
    """First descending crossing from a recorded (T,) z / (T,2) xy / (T,) t history (numpy)."""
    for k in range(1, len(zs)):
        if zs[k - 1] > contact_z >= zs[k]:
            f = (zs[k - 1] - contact_z) / max(zs[k - 1] - zs[k], 1e-12)
            xy = xys[k - 1] + (xys[k] - xys[k - 1]) * f
            t = ts[k - 1] + (ts[k] - ts[k - 1]) * f
            return xy, t, True
    return xys[-1], ts[-1], False


def summarize(err_mm: np.ndarray, dt_ms: np.ndarray, tol_mm: float):
    med = float(np.median(err_mm))
    p90 = float(np.percentile(err_mm, 90))
    mx = float(err_mm.max())
    ok = p90 <= tol_mm
    line = (f"landing err mm: med={med:.2f} p90={p90:.2f} max={mx:.2f} | "
            f"flight-time diff ms: med={float(np.median(np.abs(dt_ms))):.2f} "
            f"max={float(np.max(np.abs(dt_ms))):.2f} | {'PASS' if ok else 'FAIL'} "
            f"(p90 tol {tol_mm} mm)")
    return ok, line


# --------------------------------------------------------------------------- #
# Isaac main
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", default=None, help="stage-1 bank npz (post-contact states)")
    parser.add_argument("--synthetic", action="store_true", help="envelope sweep instead of bank")
    parser.add_argument("--n", type=int, default=128, help="max ball count")
    parser.add_argument("--physics-dt", type=float, default=0.005,
                        help="PhysX step (training uses 1/200 s; decimation handles the 50 Hz policy)")
    parser.add_argument("--surface-z", type=float, default=0.76)
    parser.add_argument("--max-time", type=float, default=2.0)
    parser.add_argument("--tol-mm", type=float, default=20.0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.bank is None and not args.synthetic:
        parser.error("need --bank <npz> or --synthetic")

    # Kit parses sys.argv itself and hangs on unknown args — strip ours BEFORE launching,
    # exactly like train.py:942. Launch with kwargs (headless always: no display on the pod).
    sys.argv = sys.argv[:1]
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True, device=str(args.device))
    simulation_app = app_launcher.app

    import torch  # noqa: E402
    import isaaclab.sim as sim_utils  # noqa: E402
    from isaaclab.assets import RigidObject, RigidObjectCfg  # noqa: E402
    from isaaclab.utils.math import quat_rotate_inverse  # noqa: E402

    vb = load_virtual_ball()
    prm = vb.load_venue_params()
    contact_z = args.surface_z + prm.ball_radius

    if args.synthetic:
        p0, v0, w0 = states_synthetic(args.n)
        tag = "synthetic"
    else:
        p0, v0, w0 = states_from_bank(args.bank, vb, prm, args.n)
        tag = os.path.basename(args.bank)
    K = p0.shape[0]
    print(f"[in-loop] {tag}: {K} balls, physics_dt={args.physics_dt}s, "
          f"contact plane z={contact_z:.4f} (surface {args.surface_z} + R {prm.ball_radius})")

    # --- scene: K collision-free spheres via ONE regex view (128 separate RigidObjects
    # would build 128 physics views and take forever; spawn prims manually, view them once) ---
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=args.physics_dt, device=args.device))
    # NOTE: inertia is irrelevant here — zero applied torque, angular damping 0, gyroscopic
    # forces disabled => omega constant by construction (the fit's assumption); only mass matters.
    grid = int(np.ceil(np.sqrt(K)))
    sphere_cfg = sim_utils.SphereCfg(
        radius=prm.ball_radius,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            linear_damping=0.0, angular_damping=0.0, disable_gravity=False,
            enable_gyroscopic_forces=False,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.0034),
        collision_props=None,  # no collider: pure flight, nothing to touch
    )
    for i in range(K):
        sphere_cfg.func(f"/World/Ball_{i:04d}", sphere_cfg,
                        translation=(30.0 * (i % grid), 30.0 * (i // grid), 5.0))
    balls = RigidObject(RigidObjectCfg(prim_path="/World/Ball_.*", spawn=None))
    sim.reset()
    print(f"[in-loop] scene ready: {balls.num_instances} instances in one view", flush=True)

    offsets = torch.tensor([[30.0 * (i % grid), 30.0 * (i // grid), 0.0] for i in range(K)],
                           dtype=torch.float32, device=sim.device)
    p0f = p0.to(torch.float32).to(sim.device) + offsets
    v0f = v0.to(torch.float32).to(sim.device)
    w0f = w0.to(torch.float32).to(sim.device)
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=sim.device).expand(K, 4)
    balls.write_root_pose_to_sim(torch.cat([p0f, quat], dim=-1))
    balls.write_root_velocity_to_sim(torch.cat([v0f, w0f], dim=-1))

    # --- step: inject aero force from SIM state each physics step (batched) --------------
    steps = int(args.max_time / args.physics_dt)
    zs = np.zeros((steps + 1, K))
    xys = np.zeros((steps + 1, K, 2))
    ts = np.arange(steps + 1) * args.physics_dt
    kd, km = prm.k_d, prm.k_m
    mass = 0.0034

    def record(row):
        pos = (balls.data.root_pos_w - offsets).cpu().numpy()
        zs[row] = pos[:, 2]
        xys[row] = pos[:, :2]

    record(0)
    for k in range(steps):
        v = balls.data.root_lin_vel_w
        w = balls.data.root_ang_vel_w
        speed = torch.linalg.norm(v, dim=-1, keepdim=True)
        a = -kd * speed * v + km * torch.cross(w, v, dim=-1)
        # Isaac Lab 2.1 applies external wrenches in the BODY frame at COM (is_global only in
        # >=2.2); rotate world->body — same convention as table_tennis_env.py. The ball spins
        # fast (up to ~95 rad/s): skipping this rotates the force ~0.5 rad per 5 ms step.
        f_b = quat_rotate_inverse(balls.data.root_quat_w, mass * a).unsqueeze(1)
        balls.set_external_force_and_torque(f_b, torch.zeros_like(f_b))
        balls.write_data_to_sim()
        sim.step()
        balls.update(args.physics_dt)
        record(k + 1)
        if (zs[k + 1] <= contact_z - 0.3).all():
            zs, xys, ts = zs[: k + 2], xys[: k + 2], ts[: k + 2]
            break
    print(f"[in-loop] stepping done at t={ts[-1]:.3f}s", flush=True)

    # omega drift check (constant-omega assumption under PhysX)
    w_end = balls.data.root_ang_vel_w.cpu().to(torch.float64)
    w_drift = torch.linalg.norm(w_end - w0.to(torch.float64), dim=-1)

    # --- compare -------------------------------------------------------------------------
    ref_xy, ref_t, ref_ok = reference_landing(p0, v0, w0, vb, prm, contact_z)
    err, dts, used = [], [], 0
    for i in range(K):
        xy_i, t_i, ok_i = crossing_from_history(zs[:, i], xys[:, i], ts, contact_z)
        if not (ok_i and bool(ref_ok[i])):
            continue
        used += 1
        err.append(1e3 * float(np.linalg.norm(xy_i - ref_xy[i].numpy())))
        dts.append(1e3 * (t_i - float(ref_t[i])))
    err = np.asarray(err)
    dts = np.asarray(dts)
    ok, line = summarize(err, dts, args.tol_mm)
    print(f"[in-loop] compared {used}/{K} balls (both crossed the plane)", flush=True)
    print(f"[in-loop] {line}", flush=True)
    print(f"[in-loop] |omega drift| rad/s: med={float(w_drift.median()):.4f} "
          f"max={float(w_drift.max()):.4f} (constant-omega check)")

    simulation_app.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
