"""Regression: the torch ball-physics port must match the Record numpy oracle.

This is the load-bearing correctness gate for Task A/B4. It checks that
``tasks/table_tennis/physics`` (torch, vectorized, runs inside the Isaac env) reproduces
``Record/analysis/{contact_model/spin_equation,flight_model/simulator}`` (the fitted, validated numpy
reference) to tight tolerance, for both the table-bounce and paddle-hit contact params, the flight
integrator, and the first-bounce landing point.

Pure CPU, no Isaac Sim. Skips gracefully if torch / PyYAML / numpy / the Record folder are missing.

Run:  python tests/test_ball_physics_vs_record.py
"""

from __future__ import annotations

import importlib.util
import os
import sys

_HERE = os.path.dirname(__file__)
_PHYS = os.path.join(
    os.path.dirname(_HERE),  # whole_body_tracking/
    "source", "whole_body_tracking", "whole_body_tracking", "tasks", "table_tennis", "physics",
)
_RECORD = os.environ.get("RECORD_DIR", "/Users/yyk956614/Desktop/Hope/Record")
_FLIGHT_DIR = os.path.join(_RECORD, "analysis", "flight_model")
_CONTACT_DIR = os.path.join(_RECORD, "analysis", "contact_model")


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _deps_ok() -> str | None:
    for m in ("torch", "numpy", "yaml"):
        try:
            importlib.import_module(m)
        except Exception:
            return f"{m} unavailable"
    if not os.path.isdir(_CONTACT_DIR) or not os.path.isfile(os.path.join(_CONTACT_DIR, "spin_equation.py")):
        return f"Record oracle not found at {_RECORD} (set $RECORD_DIR)"
    return None


def run() -> int:
    skip = _deps_ok()
    if skip:
        print(f"[skip] ball-physics-vs-Record tests: {skip}")
        return 0

    import numpy as np
    import torch

    torch.manual_seed(0)
    dt64 = torch.double

    # --- load both sides -----------------------------------------------------
    sys.path.insert(0, _CONTACT_DIR)  # simulator.py imports spin_equation from here
    oracle_contact = _load("rec_spin_equation", os.path.join(_CONTACT_DIR, "spin_equation.py"))
    oracle_sim = _load("rec_simulator", os.path.join(_FLIGHT_DIR, "simulator.py"))

    params_mod = _load("phys_params", os.path.join(_PHYS, "params.py"))
    # spin_contact/flight/landing do `from .params import ...`; build a fake package so the relative
    # imports resolve to the module we just loaded.
    import types
    pkg = types.ModuleType("ttphys")
    pkg.__path__ = [_PHYS]
    sys.modules["ttphys"] = pkg
    sys.modules["ttphys.params"] = params_mod
    spin_contact = _load("ttphys.spin_contact", os.path.join(_PHYS, "spin_contact.py"))
    flight_mod = _load("ttphys.flight", os.path.join(_PHYS, "flight.py"))
    landing_mod = _load("ttphys.landing", os.path.join(_PHYS, "landing.py"))

    cfg = params_mod.load_ball_physics()

    failures = []

    # --- 1. contact model: torch vs numpy, table + paddle --------------------
    def oracle_params(kind):
        return oracle_sim.TABLE_PARAMS if kind == "table" else oracle_contact.FULL_REFLECT_PARAMS

    N = 256
    for kind, cp in (("table", cfg.table), ("paddle", cfg.paddle)):
        v_minus = (torch.rand(N, 3, dtype=dt64) - 0.5) * 12.0
        v_r = (torch.rand(N, 3, dtype=dt64) - 0.5) * (0.0 if kind == "table" else 6.0)
        n = torch.rand(N, 3, dtype=dt64) - 0.5
        omega = (torch.rand(N, 3, dtype=dt64) - 0.5) * 120.0

        vp_t, wp_t = spin_contact.predict_contact(v_minus, v_r, n, omega, cp)

        op = oracle_params(kind)
        max_dv = 0.0
        max_dw = 0.0
        for i in range(N):
            out = oracle_contact.predict_contact(
                v_minus=v_minus[i].numpy(), v_r=v_r[i].numpy(), n=n[i].numpy(),
                omega_minus=omega[i].numpy(), params=op,
            )
            max_dv = max(max_dv, float(np.max(np.abs(out["v_plus"] - vp_t[i].numpy()))))
            max_dw = max(max_dw, float(np.max(np.abs(out["omega_plus"] - wp_t[i].numpy()))))
        ok = max_dv < 1e-6 and max_dw < 1e-6
        print(f"[{'ok' if ok else 'FAIL'}] contact[{kind}]: max|dv|={max_dv:.2e} max|dw|={max_dw:.2e}")
        if not ok:
            failures.append(f"contact[{kind}]")

    # --- 2. flight integrator: torch RK4 vs numpy RK4 ------------------------
    fp = cfg.flight
    p = torch.tensor([[0.0, 0.0, 1.0]], dtype=dt64)
    v = torch.tensor([[4.0, 1.0, 1.5]], dtype=dt64)
    w = torch.tensor([[0.0, 50.0, 0.0]], dtype=dt64)
    pp, vv = p.clone(), v.clone()
    pn, vn = p[0].numpy().copy(), v[0].numpy().copy()
    h = fp.rk4_h
    max_pos_err = 0.0
    for _ in range(400):  # 0.2 s
        pp, vv = flight_mod.rk4_step(pp, vv, w, h, fp)
        pn, vn = oracle_sim.rk4(pn, vn, w[0].numpy(), fp.k_d, h, k_m=fp.k_m)
        max_pos_err = max(max_pos_err, float(np.max(np.abs(pp[0].numpy() - pn))))
    ok = max_pos_err < 1e-6
    print(f"[{'ok' if ok else 'FAIL'}] flight RK4 vs oracle: max|dp|={max_pos_err:.2e} m over 0.2 s")
    if not ok:
        failures.append("flight")

    # --- 3. landing point: torch predict_landing vs oracle first table bounce -
    R = cfg.ball.radius
    surf = 0.0
    table = oracle_sim.Table(center_m=[0, 0, surf], normal=[0, 0, 1], surface_z_m=surf)
    p0 = torch.tensor([[0.0, 0.0, 0.9]], dtype=dt64)
    v0 = torch.tensor([[6.0, 0.5, 0.5]], dtype=dt64)
    w0 = torch.tensor([[0.0, 40.0, 0.0]], dtype=dt64)
    res = landing_mod.predict_landing(
        p0, v0, w0, fp, contact_z=surf + R, table_x=(-5.0, 5.0), table_y=(-5.0, 5.0),
        net_x=-100.0, max_time=2.0, dt=1e-3,
    )
    times, P, contacts = oracle_sim.simulate(
        p0[0].numpy(), v0[0].numpy(), w0[0].numpy(), fp.k_d, table, t_total=2.0, h=fp.rk4_h, k_m=fp.k_m
    )
    assert contacts, "oracle produced no table bounce"
    land_oracle = contacts[0]["p"][:2]
    land_torch = res.xy[0].numpy()
    land_err_mm = float(np.linalg.norm(land_oracle - land_torch)) * 1000.0
    # 1 ms coarse step + bisection vs 0.5 ms oracle -> sub-mm agreement expected.
    ok = bool(res.valid[0]) and land_err_mm < 2.0
    print(f"[{'ok' if ok else 'FAIL'}] landing vs oracle: err={land_err_mm:.3f} mm "
          f"(torch {land_torch}, oracle {land_oracle})")
    if not ok:
        failures.append("landing")

    print(f"\n{'ALL PASSED' if not failures else 'FAILURES: ' + ', '.join(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(run())
