"""Guard against drift between configs/ball_physics_venue.yaml (single source of truth) and
every ball-physics implementation that mirrors or loads it.

Pattern follows hope_ws/src/hope_planner/test/test_venue_param_consistency.py (franco 2026-07-04):
if any assertion fails, someone re-fitted the venue yaml without updating a mirror (or vice
versa). Update BOTH, in one commit.

Covered implementations:
  1. tasks/table_tennis/physics/params.py       (Isaac torch port; loads the yaml)
  2. tasks/tracking/mdp/virtual_ball.py         (Tier-1 reward virtual ball; loads the yaml)
  3. tasks/table_tennis/geometry.py             (scene constants; hand-mirrored)
  4. tasks/table_tennis/ball.py                 (aero-wrench fallback defaults; hand-mirrored)
  5. mujoco_sim_module/ball/ball_dynamics.h     (C++ defaults; regex-extracted, hand-mirrored)
  6. MuJoCo model-dir yaml copy                 (key-for-key equal to the repo-root yaml)
Plus a torch cross-parity check: virtual_ball vs physics/ must produce identical flight steps and
paddle contacts (they are two ports of the same model).

Runs standalone (python tests/test_ball_params_consistency.py) or under pytest.
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_WBT = os.path.dirname(_HERE)  # whole_body_tracking/
_PKG = os.path.join(_WBT, "source", "whole_body_tracking", "whole_body_tracking", "tasks")
_PHYS = os.path.join(_PKG, "table_tennis", "physics")


def _repo_root() -> str:
    d = _HERE
    while True:
        if os.path.exists(os.path.join(d, "configs", "ball_physics_venue.yaml")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            raise FileNotFoundError("configs/ball_physics_venue.yaml not found above tests/")
        d = parent


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _venue_yaml() -> dict:
    import yaml

    with open(os.path.join(_repo_root(), "configs", "ball_physics_venue.yaml")) as fh:
        return yaml.safe_load(fh)


def _params_cfg():
    params_mod = _load("bpc_params", os.path.join(_PHYS, "params.py"))
    return params_mod.load_ball_physics(
        os.path.join(_repo_root(), "configs", "ball_physics_venue.yaml")
    )


def _approx(a: float, b: float, rel: float = 1e-9) -> bool:
    return abs(a - b) <= rel * max(1.0, abs(a), abs(b))


# --------------------------------------------------------------------------- #
# 1+2. the two YAML loaders agree with the yaml and with each other
# --------------------------------------------------------------------------- #
def test_physics_params_match_venue_yaml():
    raw, cfg = _venue_yaml(), _params_cfg()
    assert _approx(cfg.ball.mass, float(raw["ball"]["mass"]))
    assert _approx(cfg.ball.radius, float(raw["ball"]["radius"]))
    assert _approx(cfg.flight.k_d, float(raw["flight"]["k_d"]))
    assert _approx(cfg.flight.k_m, float(raw["flight"]["k_m"]))
    assert _approx(
        cfg.flight.magnus_sr_sat,
        float(raw["flight"]["cl_max"]) / float(raw["flight"]["cl_slope"]),
    )
    assert not cfg.flight.use_saturating_magnus  # behavior switch defaults OFF
    tab, pad = raw["contact"]["table"], raw["contact"]["paddle"]
    assert _approx(cfg.table.e_eff, float(tab["e_eff"])) and not cfg.table.use_e_exp
    assert _approx(cfg.paddle.a_t, float(pad["a_t"]))
    assert cfg.paddle.use_e_exp
    assert _approx(cfg.paddle.e_exp_g1, float(pad["e_exp_g1"]))
    assert _approx(cfg.paddle.e_exp_g2, float(pad["e_exp_g2"]))


def test_virtual_ball_matches_venue_yaml():
    raw = _venue_yaml()
    vb = _load("bpc_virtual_ball", os.path.join(_PKG, "tracking", "mdp", "virtual_ball.py"))
    prm = vb.load_venue_params(os.path.join(_repo_root(), "configs", "ball_physics_venue.yaml"))
    pad = raw["contact"]["paddle"]
    assert _approx(prm.k_d, float(raw["flight"]["k_d"]))
    assert _approx(prm.k_m, float(raw["flight"]["k_m"]))
    assert _approx(prm.ball_radius, float(raw["ball"]["radius"]))
    assert _approx(prm.paddle_a_t, float(pad["a_t"]))
    assert _approx(prm.paddle_e_g1, float(pad["e_exp_g1"]))
    assert _approx(prm.paddle_e_g2, float(pad["e_exp_g2"]))
    assert _approx(prm.paddle_mu, float(pad["mu_safety"]))


# --------------------------------------------------------------------------- #
# 3+4. hand-mirrored Python constants
# --------------------------------------------------------------------------- #
def test_geometry_constants_match_venue_yaml():
    raw = _venue_yaml()
    geom = _load("bpc_geometry", os.path.join(_PKG, "table_tennis", "geometry.py"))
    assert _approx(geom.BALL_RADIUS, float(raw["ball"]["radius"]))
    assert _approx(geom.BALL_MASS, float(raw["ball"]["mass"]))
    assert _approx(geom.BALL_INERTIA_COEFF, float(raw["ball"]["inertia_coeff"]))


def test_ball_aero_fallbacks_match_venue_yaml():
    raw = _venue_yaml()
    ball = _load("bpc_ball", os.path.join(_PKG, "table_tennis", "ball.py"))
    cfg = ball.BallAerodynamicsCfg()
    assert _approx(cfg.drag_coefficient, float(raw["flight"]["k_d"]))
    assert _approx(cfg.magnus_coefficient, float(raw["flight"]["k_m"]))


# --------------------------------------------------------------------------- #
# 5. C++ header defaults (regex-extracted drift guard)
# --------------------------------------------------------------------------- #
def test_cpp_header_defaults_match_venue_yaml():
    raw = _venue_yaml()
    hdr_path = os.path.join(
        _repo_root(), "agi", "A3_MuJoCo_Sim", "aimrt_mujoco_sim", "src", "module",
        "mujoco_sim_module", "ball", "ball_dynamics.h",
    )
    src = open(hdr_path).read()

    def grab(pattern: str) -> float:
        m = re.search(pattern, src)
        assert m, f"pattern not found in ball_dynamics.h: {pattern}"
        return float(m.group(1))

    assert _approx(grab(r"double mass = ([0-9.e-]+)"), float(raw["ball"]["mass"]))
    assert _approx(grab(r"double k_d = ([0-9.e-]+)"), float(raw["flight"]["k_d"]))
    assert _approx(grab(r"double k_m = ([0-9.e-]+)"), float(raw["flight"]["k_m"]))
    assert _approx(grab(r"double e_eff = ([0-9.e-]+)"), float(raw["contact"]["table"]["e_eff"]))
    # paddle aggregate-initializer: {e_eff, a_t, b_t, mu, R, c, e_exp_g1, e_exp_g2, use_e_exp}
    m = re.search(r"ContactParams paddle\{([^}]+)\}", src)
    assert m, "paddle initializer not found in ball_dynamics.h"
    vals = [v.strip() for v in m.group(1).split(",")]
    pad = raw["contact"]["paddle"]
    assert _approx(float(vals[0]), float(pad["e_eff"]))
    assert _approx(float(vals[1]), float(pad["a_t"]))
    assert _approx(float(vals[3]), float(pad["mu_safety"]))
    assert _approx(float(vals[6]), float(pad["e_exp_g1"]))
    assert _approx(float(vals[7]), float(pad["e_exp_g2"]))
    assert vals[8] == "true"


# --------------------------------------------------------------------------- #
# 6. MuJoCo model-dir yaml copy == repo-root yaml (key-for-key)
# --------------------------------------------------------------------------- #
def test_mujoco_model_yaml_copy_in_sync():
    import yaml

    root = _repo_root()
    with open(os.path.join(root, "configs", "ball_physics_venue.yaml")) as fh:
        canonical = yaml.safe_load(fh)
    copy_path = os.path.join(
        root, "agi", "A3_MuJoCo_Sim", "aimrt_mujoco_sim", "src", "models", "bin", "cfg",
        "model", "a3_pingpong", "ball_physics.yaml",
    )
    with open(copy_path) as fh:
        copy = yaml.safe_load(fh)
    assert copy == canonical, (
        "MuJoCo model-dir ball_physics.yaml drifted from configs/ball_physics_venue.yaml — "
        "regenerate the copy (cat configs/ball_physics_venue.yaml with the DO-NOT-EDIT header)."
    )


# --------------------------------------------------------------------------- #
# 7. torch cross-parity: virtual_ball vs physics/ (two ports of one model)
# --------------------------------------------------------------------------- #
def test_virtual_ball_vs_physics_port_parity():
    try:
        import torch
    except Exception:  # pragma: no cover - torch-less checkout
        import pytest

        pytest.skip("torch unavailable")

    torch.manual_seed(0)
    dt64 = torch.double
    root = _repo_root()
    yaml_path = os.path.join(root, "configs", "ball_physics_venue.yaml")

    vb = _load("bpc_vb2", os.path.join(_PKG, "tracking", "mdp", "virtual_ball.py"))
    prm = vb.load_venue_params(yaml_path)

    params_mod = sys.modules.get("bpc_params") or _load("bpc_params", os.path.join(_PHYS, "params.py"))
    import types

    pkg = types.ModuleType("bpc_ttphys")
    pkg.__path__ = [_PHYS]
    sys.modules["bpc_ttphys"] = pkg
    sys.modules["bpc_ttphys.params"] = params_mod
    spin_contact = _load("bpc_ttphys.spin_contact", os.path.join(_PHYS, "spin_contact.py"))
    flight_mod = _load("bpc_ttphys.flight", os.path.join(_PHYS, "flight.py"))
    cfg = params_mod.load_ball_physics(yaml_path)

    N = 256
    v_minus = (torch.rand(N, 3, dtype=dt64) - 0.5) * 12.0
    v_r = (torch.rand(N, 3, dtype=dt64) - 0.5) * 6.0
    n = torch.rand(N, 3, dtype=dt64) - 0.5
    omega = (torch.rand(N, 3, dtype=dt64) - 0.5) * 120.0

    vp_a, wp_a = vb.predict_paddle_contact(v_minus, v_r, n, omega, prm)
    vp_b, wp_b = spin_contact.predict_contact(v_minus, v_r, n, omega, cfg.paddle)
    assert float((vp_a - vp_b).abs().max()) < 1e-9
    assert float((wp_a - wp_b).abs().max()) < 1e-9

    p = (torch.rand(N, 3, dtype=dt64) - 0.5) * 2.0
    v = (torch.rand(N, 3, dtype=dt64) - 0.5) * 10.0
    pa, va = vb.rk4_step(p, v, omega, 1.0e-3, prm)
    pb, vv = flight_mod.rk4_step(p, v, omega, 1.0e-3, cfg.flight)
    assert float((pa - pb).abs().max()) < 1e-12
    assert float((va - vv).abs().max()) < 1e-12


def _main() -> int:
    checks = [
        test_physics_params_match_venue_yaml,
        test_virtual_ball_matches_venue_yaml,
        test_geometry_constants_match_venue_yaml,
        test_ball_aero_fallbacks_match_venue_yaml,
        test_cpp_header_defaults_match_venue_yaml,
        test_mujoco_model_yaml_copy_in_sync,
        test_virtual_ball_vs_physics_port_parity,
    ]
    failed = []
    for fn in checks:
        try:
            fn()
            print(f"[ok] {fn.__name__}")
        except Exception as exc:  # noqa: BLE001 - standalone reporter
            print(f"[FAIL] {fn.__name__}: {exc}")
            failed.append(fn.__name__)
    print("ALL PASSED" if not failed else f"FAILURES: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
