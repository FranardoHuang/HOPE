"""Tier-1 virtual-ball physics for the at-strike landing reward (rewardDesign.md).

Vectorized torch port of the VENUE-FITTED ball model (``configs/ball_physics_venue.yaml``,
2026-07-03 Avatar Pro fit on the 3.4 g coated MATCH ball):

* flight:  a = g - k_d |v| v + k_m (omega x v)          (k_d = 0.1261, k_m = 0.00444)
* paddle:  fitted spin-impulse contact (spin_contact.py form) with the F4-adopted
  VELOCITY-DEPENDENT restitution e(u_n) = g1 * exp(g2 * |u_n|) instead of constant e
  (report §4 F4: constant-e KILLED; exponential preferred for extrapolation).

The landing predictor here is the COARSE, fixed-length, branch-free rollout mandated by
verify_tier1-reward-soundness.md (b): RK4 at h = 10 ms with linear-interpolation crossing
extraction — NOT the shipped ``predict_landing`` (h = 1 ms + 24-iter bisection), which is
~15x too slow for the training step loop (+300 %/iter). Interp error at h = 10 ms is
~0.5 mm, far below the sigma = 0.3 m landing kernel.

Everything is per-env ``(N, 3)`` tensors and runs on the full batch every strike step.

COST (人话:这个 rollout 为什么曾经贵得不能内联).  The rollout was never serial over envs —
it was already one tensor op per env-batch — but the 100 RK4 steps issued ~8600 separate CUDA
kernels per call (86 per step), so one call cost ~45 ms of pure launch overhead on an RTX 5090
NO MATTER how many envs (measured 2026-07-27: n = 64 / 1024 / 4096 / 16384 all 44.7-45.6 ms,
total GPU kernel time only 8.5 ms of that).  ``coarse_landing`` now runs the whole 100-step
rollout inside ONE fused Triton kernel (one launch), which is 0.064 ms per call — 700x cheaper —
and BIT-IDENTICAL to the eager loop.  See ``_coarse_landing_eager`` (the reference implementation,
unchanged) and ``_fused_rollout``; the fast path is admitted only after a per-configuration
bitwise parity gate against the reference, and falls back to the reference on any doubt.
Kill switch: ``HOPE_VIRTUAL_BALL_FAST=0``.

Validity envelope of the constants (do not extrapolate silently): ball speed 1-7 m/s,
spin 0-15 rev/s, paddle u_n 1.4-7.2 m/s. The samplers in hope_commands stay inside it.
"""

from __future__ import annotations

import functools
import os
import struct
import warnings
from dataclasses import dataclass

import torch
import yaml

_EPS = 1e-12  # matches spin_contact.py / C++ kEps / the numpy oracle (was 1e-9; pure guard)


@dataclass(frozen=True)
class VirtualBallParams:
    """Flight + paddle-contact constants (venue fit)."""

    k_d: float
    k_m: float
    g: float
    ball_radius: float
    inertia_coeff: float
    paddle_a_t: float
    paddle_b_t: float
    paddle_mu: float
    paddle_e_g1: float  # e(u_n) = g1 * exp(g2 * |u_n|)
    paddle_e_g2: float
    source_path: str


def default_venue_yaml_path() -> str:
    """Resolve ``configs/ball_physics_venue.yaml`` (env override, then upward search)."""
    env = os.environ.get("HOPE_BALL_PHYSICS_YAML")
    if env:
        return env
    d = os.path.dirname(os.path.abspath(__file__))
    while True:
        cand = os.path.join(d, "configs", "ball_physics_venue.yaml")
        if os.path.exists(cand):
            return cand
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    raise FileNotFoundError(
        "ball_physics_venue.yaml not found; set $HOPE_BALL_PHYSICS_YAML or place it at "
        "<repo>/configs/ball_physics_venue.yaml"
    )


def load_venue_params(path: str | None = None) -> VirtualBallParams:
    path = path or default_venue_yaml_path()
    with open(path, "r") as fh:
        raw = yaml.safe_load(fh)
    pad = raw["contact"]["paddle"]
    return VirtualBallParams(
        k_d=float(raw["flight"]["k_d"]),
        k_m=float(raw["flight"]["k_m"]),
        g=float(raw["flight"]["g"]),
        ball_radius=float(raw["ball"]["radius"]),
        inertia_coeff=float(raw["ball"]["inertia_coeff"]),
        paddle_a_t=float(pad["a_t"]),
        paddle_b_t=float(pad["b_t"]),
        paddle_mu=float(pad["mu_safety"]),
        paddle_e_g1=float(pad["e_exp_g1"]),
        paddle_e_g2=float(pad["e_exp_g2"]),
        source_path=os.path.abspath(path),
    )


def _normalize(v: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    return v / (torch.linalg.norm(v, dim=-1, keepdim=True) + eps)


def orient_normal(n: torch.Tensor, v_minus: torch.Tensor, v_r: torch.Tensor) -> torch.Tensor:
    """Orient ``n`` per env so the incoming relative velocity approaches the face.

    Same as the v0 branch ``spin_contact.orient_normal``: flip where ``(v_minus - v_r) . n > 0``.
    The returned normal points from the face toward the incoming-ball side.
    """
    n = _normalize(n)
    approaching = torch.sum((v_minus - v_r) * n, dim=-1, keepdim=True) > 0.0
    return torch.where(approaching, -n, n)


def signed_face_hemisphere(
    achieved_normal: torch.Tensor,
    target_normal: torch.Tensor,
    *,
    achieved_physical_b: torch.Tensor | None = None,
    target_physical_b: torch.Tensor | None = None,
    physical_b_min_x: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Grade face identity before ``orient_normal`` can erase ``n`` versus ``-n``.

    The caller supplies a convention-matched pair: raw mount +Y/A for face-command training, or
    the selected physical striking face B for legacy clip-reference training.  Formal callers also
    supply the achieved and demanded physical-B normals.  Their strict opponent-facing ``x`` gate
    prevents a positive A-frame dot near the 90-degree boundary from accepting a face that points
    away from the opponent.  Non-finite or degenerate rows fail closed.  The strict ``dot > 0``
    boundary implements the registered 90-degree signed-face identity gate; exact angular tracking
    remains a separate metric.
    """

    achieved_norm = torch.linalg.norm(achieved_normal, dim=-1, keepdim=True)
    target_norm = torch.linalg.norm(target_normal, dim=-1, keepdim=True)
    finite = (
        torch.isfinite(achieved_normal).all(dim=-1)
        & torch.isfinite(target_normal).all(dim=-1)
        & torch.isfinite(achieved_norm.squeeze(-1))
        & torch.isfinite(target_norm.squeeze(-1))
        & (achieved_norm.squeeze(-1) > _EPS)
        & (target_norm.squeeze(-1) > _EPS)
    )
    achieved = achieved_normal / (achieved_norm + _EPS)
    target = target_normal / (target_norm + _EPS)
    dot = torch.sum(achieved * target, dim=-1).clamp(-1.0, 1.0)
    physical_ok = torch.ones_like(finite)
    for physical in (achieved_physical_b, target_physical_b):
        if physical is None:
            continue
        physical_norm = torch.linalg.norm(physical, dim=-1, keepdim=True)
        physical_finite = (
            torch.isfinite(physical).all(dim=-1)
            & torch.isfinite(physical_norm.squeeze(-1))
            & (physical_norm.squeeze(-1) > _EPS)
        )
        physical_unit = physical / (physical_norm + _EPS)
        physical_ok &= physical_finite & (physical_unit[:, 0] > float(physical_b_min_x))
    return finite & physical_ok & (dot > 0.0), dot


def predict_paddle_contact(
    v_minus: torch.Tensor,
    v_r: torch.Tensor,
    n: torch.Tensor,
    omega_minus: torch.Tensor,
    prm: VirtualBallParams,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Outgoing ``(v_plus, omega_plus)`` from the fitted spin equation with e(u_n).

    Port of the v0-branch ``spin_contact.predict_contact`` with the single F4 change:
    ``e_eff`` -> ``e(u_n) = g1 * exp(g2 * |u_n|)`` per env (clamped to [0.05, 0.95] so
    far-out-of-envelope contacts stay physical). All inputs ``(N, 3)``.
    """
    R = prm.ball_radius
    c = prm.inertia_coeff

    n = orient_normal(n, v_minus, v_r)
    r = -R * n

    u = v_minus + torch.cross(omega_minus, r, dim=-1) - v_r
    u_n_signed = torch.sum(u * n, dim=-1, keepdim=True)          # (N,1)
    u_t_vec = u - u_n_signed * n                                  # (N,3)
    u_t_mag = torch.linalg.norm(u_t_vec, dim=-1, keepdim=True)    # (N,1)
    u_n_abs = torch.abs(u_n_signed)                               # (N,1)

    # F4: velocity-dependent normal restitution (venue fit, valid u_n 1.4-7.2 m/s).
    e_eff = (prm.paddle_e_g1 * torch.exp(prm.paddle_e_g2 * u_n_abs)).clamp(0.05, 0.95)

    cos_theta = u_n_abs / (torch.hypot(u_t_mag, u_n_signed) + _EPS)
    raw = (prm.paddle_a_t + prm.paddle_b_t * cos_theta) * u_t_mag
    cap = prm.paddle_mu * (1.0 + e_eff) * u_n_abs
    s = torch.clamp(raw, min=0.0).minimum(cap)

    safe_dir = u_t_vec / (u_t_mag + _EPS)
    delta_v_t = torch.where(u_t_mag > _EPS, -s * safe_dir, torch.zeros_like(u_t_vec))

    delta_v_n = -(1.0 + e_eff) * u_n_signed * n
    delta_omega = -(1.0 / (c * R)) * torch.cross(n, delta_v_t, dim=-1)

    v_plus = v_minus + delta_v_n + delta_v_t
    omega_plus = omega_minus + delta_omega
    return v_plus, omega_plus


def flight_accel(v: torch.Tensor, omega: torch.Tensor, prm: VirtualBallParams) -> torch.Tensor:
    """a = g - k_d |v| v + k_m (omega x v). Inputs ``(N, 3)``."""
    speed = torch.linalg.norm(v, dim=-1, keepdim=True)
    a = -prm.k_d * speed * v + prm.k_m * torch.cross(omega, v, dim=-1)
    a[..., 2] -= prm.g
    return a


def rk4_step(
    p: torch.Tensor, v: torch.Tensor, omega: torch.Tensor, h: float, prm: VirtualBallParams
) -> tuple[torch.Tensor, torch.Tensor]:
    """One RK4 flight step (omega constant in flight, matching the fit)."""
    a1 = flight_accel(v, omega, prm)
    a2 = flight_accel(v + 0.5 * h * a1, omega, prm)
    a3 = flight_accel(v + 0.5 * h * a2, omega, prm)
    a4 = flight_accel(v + h * a3, omega, prm)
    v_new = v + (h / 6.0) * (a1 + 2 * a2 + 2 * a3 + a4)
    p_new = p + (h / 6.0) * (v + 2 * (v + 0.5 * h * a1) + 2 * (v + 0.5 * h * a2) + (v + h * a3))
    return p_new, v_new


def _coarse_landing_eager(
    p0: torch.Tensor,
    v0: torch.Tensor,
    omega0: torch.Tensor,
    prm: VirtualBallParams,
    surface_z: float,
    net_x: float,
    h: float = 0.01,
    n_steps: int = 100,
) -> dict[str, torch.Tensor]:
    """Fixed-length branch-free rollout: first descending table-plane crossing + net-plane crossing.

    ``surface_z`` is the plane the ball CENTER crosses. Physical table contact happens at
    ``table_surface + ball_radius`` (the oracle / C++ / landing.py convention — the venue landing
    points were extracted at that plane), so pass surface + R, not the bare surface.

    Returns (all first-crossing values; envs that never cross keep valid=False):
      ``land_xy`` (N,2)  interpolated table-plane crossing point,
      ``land_valid`` (N) crossed the plane downward within the horizon,
      ``net_z`` (N)      ball height when first crossing the net plane in +x,
      ``net_valid`` (N)  crossed the net plane (in +x) BEFORE landing.

    The net collider itself is NOT simulated (v0 parity); the reward gates on the
    predicted clearance instead, so net-hitting shots score zero rather than bounce.

    THIS IS THE REFERENCE IMPLEMENTATION — the definition of the physics. ``coarse_landing``
    dispatches to the fused Triton kernel only where that kernel is proven bit-identical to this
    function, and comes back here otherwise.
    """
    p, v = p0, v0
    landed = torch.zeros(p0.shape[0], dtype=torch.bool, device=p0.device)
    net_crossed = torch.zeros_like(landed)
    land_xy = torch.zeros(p0.shape[0], 2, device=p0.device)
    net_z = torch.zeros(p0.shape[0], device=p0.device)

    for _ in range(n_steps):
        p_new, v_new = rk4_step(p, v, omega0, h, prm)

        # net-plane crossing (+x through net_x), only counts before landing
        ncross = (~net_crossed) & (~landed) & (p[:, 0] < net_x) & (p_new[:, 0] >= net_x)
        fn = ((net_x - p[:, 0]) / (p_new[:, 0] - p[:, 0]).clamp(min=_EPS)).clamp(0.0, 1.0)
        z_at = p[:, 2] + (p_new[:, 2] - p[:, 2]) * fn
        net_z = torch.where(ncross, z_at, net_z)
        net_crossed = net_crossed | ncross

        # first downward table-plane crossing
        cross = (~landed) & (p[:, 2] > surface_z) & (p_new[:, 2] <= surface_z)
        fz = ((p[:, 2] - surface_z) / (p[:, 2] - p_new[:, 2]).clamp(min=_EPS)).clamp(0.0, 1.0)
        xy = p[:, :2] + (p_new[:, :2] - p[:, :2]) * fz.unsqueeze(-1)
        land_xy = torch.where(cross.unsqueeze(-1), xy, land_xy)
        landed = landed | cross

        p, v = p_new, v_new

    return {"land_xy": land_xy, "land_valid": landed, "net_z": net_z, "net_valid": net_crossed}


# ---------------------------------------------------------------------------------------------
# Fused single-launch rollout (CUDA / float32).  Bit-identical to ``_coarse_landing_eager``.
#
# 人话:上面那个循环每一步要发 86 个 GPU kernel,100 步就是 8600 次发射,光发射就 45 ms——跟球
# 的数量完全无关,所以逆解没法内联。下面把整个 100 步塞进一个 kernel,一次发射,0.064 ms。数值
# 必须一模一样(到最后一个 bit),否则题库和历史测量全作废,所以每个浮点操作的次序都对齐了 eager:
#
#   * torch.linalg.norm over the last dim of a (N,3) float32 CUDA tensor accumulates as
#     ``(x*x + z*z) + y*y`` then sqrt (verified exact for N = 7 ... 262144; N = 1,3 degenerate).
#   * torch.cross on CUDA contracts to ``fma(a1, b2, -(a2 * b1))`` (nvcc default -fmad=true).
#   * torch's float32 divide and sqrt are IEEE round-to-nearest, so the kernel must use Triton's
#     ``div_rn`` / ``sqrt_rn`` — plain ``/`` and ``tl.sqrt`` in Triton are the APPROX instructions
#     and disagree with torch on ~30 % of inputs.
#   * ``enable_fp_fusion=False`` stops Triton contracting anything we did not ask it to.
#   * torch's clamp propagates NaN (``isnan(v) ? v : min(max(v, lo), hi)``) — replicated.
#
# Every one of those was measured, not assumed, and the parity gate below re-measures the whole
# rollout per configuration at runtime before the fast path is ever used.
# ---------------------------------------------------------------------------------------------

_FAST_ENV = "HOPE_VIRTUAL_BALL_FAST"
_fused_kernel = None          # compiled triton kernel, or False once known unavailable
_parity_cache: dict = {}      # config key -> bool (fast path admitted)


def _build_fused_kernel():
    """Compile the fused rollout kernel once; return None when Triton is unusable."""
    global _fused_kernel
    if _fused_kernel is not None:
        return _fused_kernel or None
    try:
        import triton as _triton_mod
        import triton.language as _tl_mod
    except Exception:                                     # no triton (CPU wheels, py3.8 host)
        _fused_kernel = False
        return None

    # Triton resolves free names of a @jit body from the DEFINING MODULE's globals, so the lazy
    # import has to be published there before the kernels below are declared (a plain local
    # ``import triton.language as tl`` yields ``NameError: tl is not defined`` at compile time).
    globals()["triton"] = triton = _triton_mod
    globals()["tl"] = tl = _tl_mod

    @triton.jit
    def _vbk_clamp_min(x, lo):                             # torch clamp(min=) semantics incl. NaN
        return tl.where(x != x, x, tl.maximum(x, lo))

    @triton.jit
    def _vbk_clamp01(x):                                   # torch clamp(0, 1) semantics incl. NaN
        return tl.where(x != x, x, tl.minimum(tl.maximum(x, 0.0), 1.0))

    globals()["_vbk_clamp_min"] = _vbk_clamp_min           # same reason as ``tl`` above:
    globals()["_vbk_clamp01"] = _vbk_clamp01               # @jit callees are resolved as globals

    @triton.jit
    def _vbk_accel(vx, vy, vz, wx, wy, wz, kd, km, g):
        """``flight_accel`` for one env, in the eager op order (see the note above)."""
        speed = tl.math.sqrt_rn((vx * vx + vz * vz) + vy * vy)
        t = (-kd) * speed
        cx = tl.math.fma(wy, vz, -(wz * vy))
        cy = tl.math.fma(wz, vx, -(wx * vz))
        cz = tl.math.fma(wx, vy, -(wy * vx))
        return t * vx + km * cx, t * vy + km * cy, (t * vz + km * cz) - g

    globals()["_vbk_accel"] = _vbk_accel

    @triton.jit
    def _rollout(p_ptr, v_ptr, w_ptr, lxy_ptr, landed_ptr, netz_ptr, netc_ptr,
                 N, n_steps, kd, km, g, hh, hf, h6, surface_z, net_x, eps,
                 BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < N
        b = offs * 3
        px = tl.load(p_ptr + b + 0, mask=m, other=0.0)
        py = tl.load(p_ptr + b + 1, mask=m, other=0.0)
        pz = tl.load(p_ptr + b + 2, mask=m, other=0.0)
        vx = tl.load(v_ptr + b + 0, mask=m, other=0.0)
        vy = tl.load(v_ptr + b + 1, mask=m, other=0.0)
        vz = tl.load(v_ptr + b + 2, mask=m, other=0.0)
        wx = tl.load(w_ptr + b + 0, mask=m, other=0.0)
        wy = tl.load(w_ptr + b + 1, mask=m, other=0.0)
        wz = tl.load(w_ptr + b + 2, mask=m, other=0.0)

        landed = tl.zeros((BLOCK,), tl.int1)
        netc = tl.zeros((BLOCK,), tl.int1)
        lx = tl.zeros((BLOCK,), tl.float32)
        ly = tl.zeros((BLOCK,), tl.float32)
        nz = tl.zeros((BLOCK,), tl.float32)

        for _ in range(n_steps):
            a1x, a1y, a1z = _vbk_accel(vx, vy, vz, wx, wy, wz, kd, km, g)
            v2x = vx + hh * a1x
            v2y = vy + hh * a1y
            v2z = vz + hh * a1z
            a2x, a2y, a2z = _vbk_accel(v2x, v2y, v2z, wx, wy, wz, kd, km, g)
            v3x = vx + hh * a2x
            v3y = vy + hh * a2y
            v3z = vz + hh * a2z
            a3x, a3y, a3z = _vbk_accel(v3x, v3y, v3z, wx, wy, wz, kd, km, g)
            v4x = vx + hf * a3x
            v4y = vy + hf * a3y
            v4z = vz + hf * a3z
            a4x, a4y, a4z = _vbk_accel(v4x, v4y, v4z, wx, wy, wz, kd, km, g)

            nvx = vx + h6 * (((a1x + 2.0 * a2x) + 2.0 * a3x) + a4x)
            nvy = vy + h6 * (((a1y + 2.0 * a2y) + 2.0 * a3y) + a4y)
            nvz = vz + h6 * (((a1z + 2.0 * a2z) + 2.0 * a3z) + a4z)
            npx = px + h6 * (((vx + 2.0 * v2x) + 2.0 * v3x) + v4x)
            npy = py + h6 * (((vy + 2.0 * v2y) + 2.0 * v3y) + v4y)
            npz = pz + h6 * (((vz + 2.0 * v2z) + 2.0 * v3z) + v4z)

            ncross = (netc == 0) & (landed == 0) & (px < net_x) & (npx >= net_x)
            fn = _vbk_clamp01(tl.math.div_rn(net_x - px, _vbk_clamp_min(npx - px, eps)))
            nz = tl.where(ncross, pz + (npz - pz) * fn, nz)
            netc = netc | ncross

            cr = (landed == 0) & (pz > surface_z) & (npz <= surface_z)
            fz = _vbk_clamp01(tl.math.div_rn(pz - surface_z, _vbk_clamp_min(pz - npz, eps)))
            lx = tl.where(cr, px + (npx - px) * fz, lx)
            ly = tl.where(cr, py + (npy - py) * fz, ly)
            landed = landed | cr

            px, py, pz = npx, npy, npz
            vx, vy, vz = nvx, nvy, nvz

        tl.store(lxy_ptr + offs * 2 + 0, lx, mask=m)
        tl.store(lxy_ptr + offs * 2 + 1, ly, mask=m)
        tl.store(netz_ptr + offs, nz, mask=m)
        tl.store(landed_ptr + offs, landed.to(tl.uint8), mask=m)
        tl.store(netc_ptr + offs, netc.to(tl.uint8), mask=m)

    _fused_kernel = (triton, _rollout)
    return _fused_kernel


@functools.lru_cache(maxsize=None)
def _f32(x: float) -> float:
    """Round a python double to float32 the way a torch Scalar operand is rounded.

    ``a_float32_tensor * some_python_float`` casts the scalar to float32 before the op, so the
    kernel must receive the SAME rounded constant or the last bits diverge.
    """
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


_FUSED_BLOCK = 128


def _fused_rollout(p0, v0, omega0, prm, surface_z, net_x, h, n_steps):
    """One-launch rollout. Returns the same dict as ``_coarse_landing_eager``."""
    triton, kernel = _build_fused_kernel()
    n = int(p0.shape[0])
    dev = p0.device
    land_xy = torch.empty(n, 2, device=dev, dtype=torch.float32)
    net_z = torch.empty(n, device=dev, dtype=torch.float32)
    landed = torch.empty(n, device=dev, dtype=torch.bool)
    net_crossed = torch.empty(n, device=dev, dtype=torch.bool)
    kernel[(triton.cdiv(n, _FUSED_BLOCK),)](
        p0.contiguous(), v0.contiguous(), omega0.contiguous(),
        land_xy, landed.view(torch.uint8), net_z, net_crossed.view(torch.uint8),
        n, int(n_steps),
        _f32(prm.k_d), _f32(prm.k_m), _f32(prm.g),
        _f32(0.5 * h), _f32(h), _f32(h / 6.0),
        _f32(surface_z), _f32(net_x), _f32(_EPS),
        BLOCK=_FUSED_BLOCK, num_warps=4, enable_fp_fusion=False,
    )
    return {"land_xy": land_xy, "land_valid": landed, "net_z": net_z, "net_valid": net_crossed}


def _parity_probe(device, surface_z, net_x) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic probe batch: nominal + wild + net/table grazers + sub-floor + non-finite.

    The grazing block is built around the CALLER's planes (and includes rows sitting exactly on
    them), so the gate exercises the crossing tests it is actually about to admit.
    """
    g = torch.Generator(device="cpu").manual_seed(20260727)
    n = 512

    def r(*shape):
        return torch.rand(*shape, generator=g)

    blocks = []
    blocks.append((torch.stack([r(n) * 0.3 + 0.45, r(n) * 0.9 - 0.45, r(n) * 0.4 + 0.80], -1),
                   torch.stack([r(n) * 3.0 + 1.5, r(n) * 1.6 - 0.8, r(n) * 2.5 + 0.2], -1),
                   (r(n, 3) * 2 - 1) * 60.0))
    blocks.append((r(n, 3) * 8 - 3, r(n, 3) * 60 - 30, (r(n, 3) * 2 - 1) * 600))
    graze_z = surface_z + (r(n) - 0.5) * 1e-4
    graze_z[::5] = surface_z                                  # exactly ON the landing plane
    graze_x = net_x - r(n) * 0.02
    graze_x[::7] = net_x                                      # exactly ON the net plane
    blocks.append((torch.stack([graze_x, r(n) * 1.5 - 0.75, graze_z], -1),
                   torch.stack([r(n) * 2 + 0.05, r(n) * 0.4 - 0.2, (r(n) - 0.5) * 0.02], -1),
                   (r(n, 3) * 2 - 1) * 100))
    blocks.append((torch.stack([r(n) * 2, r(n) * 1.5 - 0.75, r(n) * 0.7], -1),
                   r(n, 3) * 6 - 3, (r(n, 3) * 2 - 1) * 80))
    p, v, w = r(n, 3), r(n, 3) * 4, r(n, 3) * 50
    p[::7] = float("nan")
    v[1::11] = float("inf")
    v[2::13] = -float("inf")
    w[3::17] = float("nan")
    v[4::19] = 0.0
    p[5::23] = 0.0
    blocks.append((p, v, w))
    cat = [torch.cat(x, 0).to(device=device, dtype=torch.float32).contiguous() for x in zip(*blocks)]
    return cat[0], cat[1], cat[2]


def _fast_path_admitted(p0, prm, surface_z, net_x, h, n_steps) -> bool:
    """Gate: the fused kernel is used only where it reproduces the reference BIT for BIT.

    Runs once per (device, params, plane, step) configuration, on a fixed probe batch that spans
    the venue envelope plus the crossing/non-finite edge cases. A mismatch (or any exception)
    permanently disables the fast path for that configuration and warns loudly — the run stays
    correct, only slow.
    """
    mode = os.environ.get(_FAST_ENV, "1").strip().lower()
    if mode in ("0", "off", "false", "no", "eager"):
        return False
    if p0.device.type != "cuda" or p0.dtype != torch.float32:
        return False
    if p0.requires_grad or _build_fused_kernel() is None:
        return False

    key = (p0.device.type, p0.device.index, prm, float(surface_z), float(net_x),
           float(h), int(n_steps))
    hit = _parity_cache.get(key)
    if hit is not None:
        return hit
    if mode in ("trust", "nogate"):
        _parity_cache[key] = True
        return True
    try:
        pp, vv, ww = _parity_probe(p0.device, float(surface_z), float(net_x))
        ref = _coarse_landing_eager(pp, vv, ww, prm, surface_z, net_x, h=h, n_steps=n_steps)
        got = _fused_rollout(pp, vv, ww, prm, surface_z, net_x, h, n_steps)
        ok = True
        for name in ("land_xy", "net_z"):
            ok &= bool((ref[name].view(torch.int32) == got[name].view(torch.int32)).all())
        for name in ("land_valid", "net_valid"):
            ok &= bool((ref[name] == got[name]).all())
    except Exception as exc:                                   # pragma: no cover - safety net
        warnings.warn(f"virtual_ball: fused rollout unusable ({exc!r}); using the eager reference",
                      RuntimeWarning, stacklevel=2)
        _parity_cache[key] = False
        return False
    if not ok:
        warnings.warn(
            "virtual_ball: fused rollout is NOT bit-identical to the eager reference on this "
            "build/device; falling back to the eager rollout (correct but ~700x slower). "
            "Investigate before trusting any speed number.", RuntimeWarning, stacklevel=2)
    _parity_cache[key] = ok
    return ok


def coarse_landing(
    p0: torch.Tensor,
    v0: torch.Tensor,
    omega0: torch.Tensor,
    prm: VirtualBallParams,
    surface_z: float,
    net_x: float,
    h: float = 0.01,
    n_steps: int = 100,
) -> dict[str, torch.Tensor]:
    """Fixed-length branch-free rollout (see ``_coarse_landing_eager`` for the physics).

    Dispatches to the one-launch fused Triton kernel when it has been proven bit-identical to the
    eager reference for this device/parameter/plane/step configuration, otherwise runs the eager
    reference. Both return the identical dict; the outputs are the same down to the last bit.
    """
    if int(p0.shape[0]) > 0 and _fast_path_admitted(p0, prm, surface_z, net_x, h, n_steps):
        return _fused_rollout(p0, v0, omega0, prm, surface_z, net_x, h, n_steps)
    return _coarse_landing_eager(p0, v0, omega0, prm, surface_z, net_x, h=h, n_steps=n_steps)
