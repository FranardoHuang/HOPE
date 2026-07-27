"""Pin the coarse-landing physics across the fused (one-launch) and eager rollouts.

人话:``coarse_landing`` 现在把 100 步 RK4 塞进一个 GPU kernel(原来每步发 86 个 kernel、一次调用
8600 次发射、45 ms)。速度快了 ~700 倍,但物理必须**一个 bit 都不许变**——题库、physics contract、
所有历史测量都绑在这套数上。这个文件就是那道闸:

* ``test_fused_matches_eager_bitwise`` — 5 个分布(常规 / 出界乱数 / 擦网擦线 / 地板以下 /
  NaN-Inf)逐 bit 对比融合 kernel 与 eager 参考,差一个 bit 就红。需要 CUDA + Triton。
* ``test_solver_output_is_bit_identical`` — 整条 Gauss-Newton 逆解(12 迭代)在两条路径下逐 bit
  相同,证明差异不会在迭代里被放大。需要 CUDA + Triton。
* ``test_reference_physics_golden`` — 冻结的输入->落点/过网数值表(含不落地、地板以下、落点在网前
  三种边界),任何设备任何后端都要对得上(容差 1e-5 m,远低于 sigma = 0.3 m 落点核,也低于
  h = 10 ms 插值本身的 ~0.5 mm 误差)。CPU 也跑,所以误改物理常数在没有 GPU 的机器上也会被挡。
* ``test_disabled_env_var_forces_the_reference`` — ``HOPE_VIRTUAL_BALL_FAST=0`` 真的能关掉快路径。
* ``test_fused_rollout_cost_regression`` — 成本护栏:融合路径每次调用必须 < 5 ms 且比 eager 快
  至少 20 倍。将来谁把循环拆回逐步发射,这里会红。需要 CUDA + Triton。

Run (pod, Isaac venv; the pod checkout is a COPY of the repo, never the live training tree):
    /workspace/hope_isaac_venv/bin/python -m pytest -q \
        hope_training/whole_body_tracking/tests/test_virtual_ball_fused_rollout.py
The local macOS host has no torch, so this file is a pod-only suite (same as the other torch
tests in this directory).
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
import types

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
MDP_DIR = os.path.abspath(os.path.join(
    HERE, "..", "source", "whole_body_tracking", "whole_body_tracking", "tasks", "tracking", "mdp"))
VENUE_YAML = os.path.join(REPO, "configs", "ball_physics_venue.yaml")

# Load the mdp modules under a private package name: the mdp ``__init__`` pulls isaaclab, and
# ``strike_spec_torch`` uses a relative import of ``virtual_ball`` (same trick as the other
# simulator-free tests in this directory).
_PKG = "wbt_fused_rollout_test_mdp"
if _PKG not in sys.modules:
    sys.modules[_PKG] = types.ModuleType(_PKG)
sys.modules[_PKG].__path__ = [MDP_DIR]


def _load(name):
    dotted = f"{_PKG}.{name}"
    if dotted in sys.modules:
        return sys.modules[dotted]
    spec = importlib.util.spec_from_file_location(dotted, os.path.join(MDP_DIR, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = mod
    spec.loader.exec_module(mod)
    return mod


vb = _load("virtual_ball")
PRM = vb.load_venue_params(VENUE_YAML)
SURFACE_Z = 0.76 + PRM.ball_radius          # the plane the ball CENTER crosses
NET_X = 1.87

_HAVE_TRITON = importlib.util.find_spec("triton") is not None
_KILLED = os.environ.get(vb._FAST_ENV, "1").strip().lower() in ("0", "off", "false", "no", "eager")
cuda_fused = pytest.mark.skipif(
    _KILLED or not (torch.cuda.is_available() and _HAVE_TRITON),
    reason=("fast path disabled by $" + vb._FAST_ENV) if _KILLED
    else "the fused rollout needs CUDA + Triton",
)


def _regime(mode, n, seed, device):
    """Deterministic draw for one regime (mirrors the in-module parity probe)."""
    g = torch.Generator(device="cpu").manual_seed(seed)

    def r(*shape):
        return torch.rand(*shape, generator=g)

    if mode == "nominal":                      # inside the venue-fitted envelope
        p = torch.stack([r(n) * 0.3 + 0.45, r(n) * 0.9 - 0.45, r(n) * 0.4 + 0.80], -1)
        v = torch.stack([r(n) * 3.0 + 1.5, r(n) * 1.6 - 0.8, r(n) * 2.5 + 0.2], -1)
        w = (r(n, 3) * 2 - 1) * 60.0
    elif mode == "wild":                       # solver probes that leave the landing manifold
        p, v, w = r(n, 3) * 8 - 3, r(n, 3) * 60 - 30, (r(n, 3) * 2 - 1) * 600
    elif mode == "grazing":                    # graze the net plane / sit on the landing plane
        gz = SURFACE_Z + (r(n) - 0.5) * 1e-4
        gz[::5] = SURFACE_Z                    # exactly ON the landing plane (line-call case)
        gx = NET_X - r(n) * 0.02
        gx[::7] = NET_X                        # exactly ON the net plane
        p = torch.stack([gx, r(n) * 1.5 - 0.75, gz], -1)
        v = torch.stack([r(n) * 2 + 0.05, r(n) * 0.4 - 0.2, (r(n) - 0.5) * 0.02], -1)
        w = (r(n, 3) * 2 - 1) * 100
    elif mode == "subfloor":                   # start below the plane -> must never latch a landing
        p = torch.stack([r(n) * 2, r(n) * 1.5 - 0.75, r(n) * 0.7], -1)
        v, w = r(n, 3) * 6 - 3, (r(n, 3) * 2 - 1) * 80
    elif mode == "nonfinite":                  # NaN / +-Inf / exact zeros must propagate the same
        p, v, w = r(n, 3), r(n, 3) * 4, r(n, 3) * 50
        p[::7] = float("nan")
        v[1::11] = float("inf")
        v[2::13] = -float("inf")
        w[3::17] = float("nan")
        v[4::19] = 0.0
        p[5::23] = 0.0
    else:
        raise AssertionError(mode)
    return tuple(x.to(device=device, dtype=torch.float32).contiguous() for x in (p, v, w))


def _bit_diff(a, b):
    """Number of elements whose float32 bit pattern (or bool value) differs."""
    if a.dtype == torch.bool:
        return int((a != b).sum())
    return int((a.view(torch.int32) != b.view(torch.int32)).sum())


def _force_eager():
    """Context-manager-free helper: returns a restore callable."""
    saved_cache = dict(vb._parity_cache)
    saved_env = os.environ.get(vb._FAST_ENV)
    vb._parity_cache.clear()
    os.environ[vb._FAST_ENV] = "0"

    def restore():
        vb._parity_cache.clear()
        vb._parity_cache.update(saved_cache)
        if saved_env is None:
            os.environ.pop(vb._FAST_ENV, None)
        else:
            os.environ[vb._FAST_ENV] = saved_env

    return restore


@cuda_fused
@pytest.mark.parametrize("mode", ["nominal", "wild", "grazing", "subfloor", "nonfinite"])
def test_fused_matches_eager_bitwise(mode):
    """Every output field, every bit, on 4 x 8192 balls per regime."""
    dev = torch.device("cuda")
    for seed in range(4):
        p, v, w = _regime(mode, 8192, 1000 + seed, dev)
        assert vb._fast_path_admitted(p, PRM, SURFACE_Z, NET_X, 0.01, 100), (
            "the fused path was not admitted, so this test would compare eager with eager"
        )
        ref = vb._coarse_landing_eager(p, v, w, PRM, SURFACE_Z, NET_X)
        got = vb.coarse_landing(p, v, w, PRM, surface_z=SURFACE_Z, net_x=NET_X)
        for field in ("land_xy", "land_valid", "net_z", "net_valid"):
            assert _bit_diff(ref[field], got[field]) == 0, f"{mode}/{field} differs"
        # NaN has to land in the SAME places, not merely compare unequal everywhere
        assert bool((torch.isnan(ref["land_xy"]) == torch.isnan(got["land_xy"])).all())


@cuda_fused
def test_solver_output_is_bit_identical():
    """A 12-iteration Gauss-Newton inversion cannot amplify a difference that does not exist."""
    sst = _load("strike_spec_torch")
    dev = torch.device("cuda")
    g = torch.Generator(device="cpu").manual_seed(99)
    m = 2048
    ps = torch.stack([torch.rand(m, generator=g) * 0.12 + 0.50,
                      torch.rand(m, generator=g) * 0.9 - 0.45,
                      torch.rand(m, generator=g) * 0.4 + 0.80], -1).to(dev)
    vin = torch.stack([-(torch.rand(m, generator=g) * 2.5 + 2.0),
                       torch.rand(m, generator=g) * 1.2 - 0.6,
                       torch.rand(m, generator=g) * 1.5 - 1.0], -1).to(dev)
    win = ((torch.rand(m, 3, generator=g) * 2 - 1) * 50.0).to(dev)
    aim = torch.stack([torch.rand(m, generator=g) * 0.7 + 2.2,
                       torch.rand(m, generator=g) * 1.1 - 0.55], -1).to(dev)
    kwargs = dict(prm=PRM, surface_z=SURFACE_Z, net_x=NET_X, speed_budget=3.4, n_iters=12,
                  tol_m=0.02)

    fast = sst.solve_strike_specs(ps, vin, win, aim, **kwargs)
    restore = _force_eager()
    try:
        slow = sst.solve_strike_specs(ps, vin, win, aim, **kwargs)
    finally:
        restore()

    for field in ("v_r", "n", "landing_xy", "resid_m", "ok"):
        assert _bit_diff(fast[field], slow[field]) == 0, f"solver {field} differs"
    assert float(fast["ok"].float().mean()) > 0.5, "degenerate problem set - the test proves nothing"


# Frozen 2026-07-27 from the eager reference; identical on CPU eager, CUDA eager and CUDA fused.
# Rows: 0-2 clear the net and land on the far half; 3 lands BEFORE the net (so the net event never
# latches — the net check is gated on ~landed); 4 is thrown up and never comes back inside the
# 1.0 s horizon; 5 starts BELOW the crossing plane so no landing may ever latch, while still
# crossing the net plane at a sub-table height.
GOLDEN_P = [[0.55, 0.10, 1.00], [0.50, -0.30, 0.90], [0.60, 0.25, 1.15],
            [0.52, 0.00, 0.85], [0.55, 0.00, 1.00], [0.55, 0.00, 0.70]]
GOLDEN_V = [[5.50, 0.20, 0.80], [6.20, -0.55, 1.10], [4.80, 0.35, 1.60],
            [6.50, 0.10, 0.40], [0.50, 0.00, 6.00], [3.00, 0.00, 0.50]]
GOLDEN_W = [[0.0, 0.0, 0.0], [0.0, 60.0, 0.0], [0.0, -60.0, 0.0],
            [30.0, 0.0, -30.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
GOLDEN_LAND_XY = [[2.0984311, 0.1563066], [2.0194163, -0.4347587], [2.7931612, 0.4099107],
                  [1.5445259, 0.0048041], [0.0, 0.0], [0.0, 0.0]]
GOLDEN_NET_Z = [0.8750682, 0.8304833, 1.2344464, 0.0, 0.0, -0.1532846]
GOLDEN_LAND_VALID = [True, True, True, True, False, False]
GOLDEN_NET_VALID = [True, True, True, False, False, True]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_reference_physics_golden(device):
    """Frozen input -> landing/net table. A physics pin, not a backend pin.

    Tolerance 1e-5 m: torch's own float32 3-vector norm reduction already differs in the last bits
    between its CPU and CUDA backends, so a BITWISE golden would pin the backend rather than the
    physics. 1e-5 m is 30000x below the sigma = 0.3 m landing kernel and 50x below the h = 10 ms
    interpolation error this rollout was accepted with, while still catching any real change to the
    constants, the integrator or the crossing extraction.
    """
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA")
    kw = dict(dtype=torch.float32, device=device)
    out = vb.coarse_landing(torch.tensor(GOLDEN_P, **kw), torch.tensor(GOLDEN_V, **kw),
                            torch.tensor(GOLDEN_W, **kw), PRM, surface_z=SURFACE_Z, net_x=NET_X)
    assert out["land_valid"].tolist() == GOLDEN_LAND_VALID
    assert out["net_valid"].tolist() == GOLDEN_NET_VALID
    assert torch.allclose(out["land_xy"].double().cpu(),
                          torch.tensor(GOLDEN_LAND_XY, dtype=torch.float64), atol=1e-5, rtol=0.0)
    assert torch.allclose(out["net_z"].double().cpu(),
                          torch.tensor(GOLDEN_NET_Z, dtype=torch.float64), atol=1e-5, rtol=0.0)


def test_disabled_env_var_forces_the_reference():
    """The kill switch must actually kill the fast path (and the answer must not move)."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p, v, w = _regime("nominal", 256, 7, dev)
    restore = _force_eager()
    try:
        assert vb._fast_path_admitted(p, PRM, SURFACE_Z, NET_X, 0.01, 100) is False
        off = vb.coarse_landing(p, v, w, PRM, surface_z=SURFACE_Z, net_x=NET_X)
    finally:
        restore()
    ref = vb._coarse_landing_eager(p, v, w, PRM, SURFACE_Z, NET_X)
    assert _bit_diff(off["land_xy"], ref["land_xy"]) == 0
    assert _bit_diff(off["net_z"], ref["net_z"]) == 0


@cuda_fused
def test_fused_rollout_cost_regression():
    """Cost guard: one rollout must stay ~one kernel launch, not ~8600 of them."""
    dev = torch.device("cuda")
    p, v, w = _regime("nominal", 4096, 3, dev)

    def _median_ms(fn, reps=7):
        fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
        ts.sort()
        return ts[len(ts) // 2]

    fast_ms = _median_ms(lambda: vb.coarse_landing(p, v, w, PRM, surface_z=SURFACE_Z, net_x=NET_X))
    slow_ms = _median_ms(lambda: vb._coarse_landing_eager(p, v, w, PRM, SURFACE_Z, NET_X), reps=3)
    assert fast_ms < 5.0, f"fused rollout regressed to {fast_ms:.2f} ms/call (measured ~0.06 ms)"
    assert slow_ms / fast_ms > 20.0, (
        f"the fused rollout is only {slow_ms / fast_ms:.1f}x the eager loop; the fusion is gone"
    )
