"""Shadow-ball pure-helper unit tests — NO Isaac imports.

Loads ``tasks/tracking/mdp/shadow_ball.py`` (and ``virtual_ball.py`` for the aero consistency
check) directly by file path, the same standalone pattern as ``test_stage1_wiring.py`` (the mdp
package ``__init__`` pulls isaaclab). Covers the driver's pure math:

* prestrike_ball_state: linear backward extrapolation hits the contact point exactly at the
  strike frame (tts=0), carries the incoming velocity, clamps the horizon and negative tts.
* landing_crossing: descending-crossing detection + linear interpolation against an analytic
  segment; ascending / non-crossing / already-below cases never fire (matches the
  virtual_ball.coarse_landing crossing convention).
* bounce_detect: v_z sign flip near the plane (the table-collider case where the center never
  samples below surface+R); far-from-plane flips and pure descents never fire.
* venue_aero_force: force/mass + gravity == virtual_ball.flight_accel on the REAL venue yaml
  (the engine wrench and the analytic flight model are the same law), plus the |v| clip bound.
* quat_rotate_inverse_wxyz: matches the analytic inverse rotation (world->body) for known and
  random quaternions.
* shadow_vs_virtual_err: planar norm.

Run:  /opt/anaconda3/bin/python3 hope_training/whole_body_tracking/tests/test_shadow_ball_helpers.py
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MDP_DIR = os.path.join(
    HERE, "..", "source", "whole_body_tracking", "whole_body_tracking", "tasks", "tracking", "mdp"
)


def _load(fname, name):
    spec = importlib.util.spec_from_file_location(name, os.path.abspath(os.path.join(MDP_DIR, fname)))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclass resolution needs the module registered during exec
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------------------- #
# pytest fixtures — the same modules main() loads for the direct-run path, so the file works
# both ways: ``pytest test_shadow_ball_helpers.py`` and ``python3 test_shadow_ball_helpers.py``.
# ------------------------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def sb():
    return _load("shadow_ball.py", "shadow_ball_standalone")


@pytest.fixture(scope="module")
def vb():
    return _load("virtual_ball.py", "virtual_ball_standalone")


def test_prestrike_path_hits_contact(sb):
    g = torch.Generator().manual_seed(0)
    n = 64
    contact = torch.randn(n, 3, generator=g)
    v_in = torch.randn(n, 3, generator=g) * 3.0

    # tts = 0: ball is exactly at the contact point, moving at v_in.
    pos, vel = sb.prestrike_ball_state(contact, v_in, torch.zeros(n))
    assert torch.allclose(pos, contact, atol=1e-6), "tts=0 must place the ball AT the contact point"
    assert torch.allclose(vel, v_in), "incoming velocity must be carried verbatim"

    # Linear path: for any tts inside the horizon, integrating pos + v_in*tts recovers contact.
    for tts_val in (0.02, 0.1, 0.37, sb.PRESTRIKE_HORIZON_S):
        tts = torch.full((n,), tts_val)
        pos, vel = sb.prestrike_ball_state(contact, v_in, tts)
        assert torch.allclose(pos + v_in * tts_val, contact, atol=1e-5), (
            f"linear path broken at tts={tts_val}"
        )

    # Consecutive control steps approach the contact point monotonically along the line.
    d_prev = None
    for tts_val in (0.5, 0.4, 0.3, 0.2, 0.1, 0.0):
        pos, _ = sb.prestrike_ball_state(contact, v_in, torch.full((n,), tts_val))
        d = torch.norm(pos - contact, dim=-1)
        if d_prev is not None:
            assert bool((d <= d_prev + 1e-6).all()), "approach must be monotone in tts"
        d_prev = d

    # Horizon clamp: tts beyond the horizon parks the ball at the horizon point (bounded spawn).
    pos_h, _ = sb.prestrike_ball_state(contact, v_in, torch.full((n,), sb.PRESTRIKE_HORIZON_S))
    pos_far, _ = sb.prestrike_ball_state(contact, v_in, torch.full((n,), 100.0))
    assert torch.allclose(pos_far, pos_h, atol=1e-6), "tts > horizon must clamp to the horizon point"
    # Negative tts (post-strike stragglers) clamps to the contact point, never overshoots past it.
    pos_neg, _ = sb.prestrike_ball_state(contact, v_in, torch.full((n,), -0.3))
    assert torch.allclose(pos_neg, contact, atol=1e-6), "negative tts must clamp at the contact point"
    print("[ok] prestrike: linear path hits contact at strike, monotone approach, horizon/neg clamps")


def test_landing_crossing_interp(sb):
    z_thr = 0.78
    # Analytic segment: from (0, 0, 0.88) to (0.4, -0.2, 0.68) — crosses z_thr halfway.
    prev = torch.tensor([[0.0, 0.0, 0.88]])
    new = torch.tensor([[0.4, -0.2, 0.68]])
    crossed, xy = sb.landing_crossing(prev, new, z_thr)
    assert bool(crossed[0]), "descending segment through the plane must cross"
    assert torch.allclose(xy[0], torch.tensor([0.2, -0.1]), atol=1e-6), f"interp xy wrong: {xy[0]}"

    # General fractional crossing: f = (z_prev - thr) / (z_prev - z_new).
    prev = torch.tensor([[1.0, 2.0, 0.80]])
    new = torch.tensor([[1.6, 2.9, 0.72]])
    f = (0.80 - z_thr) / (0.80 - 0.72)
    crossed, xy = sb.landing_crossing(prev, new, z_thr)
    expect = prev[0, :2] + (new[0, :2] - prev[0, :2]) * f
    assert bool(crossed[0]) and torch.allclose(xy[0], expect, atol=1e-6), f"{xy[0]} vs {expect}"

    # Never fires: ascending through the plane, fully above, fully below, starting exactly at thr.
    cases = [
        ([[0.0, 0.0, 0.70]], [[0.1, 0.1, 0.90]]),   # ascending
        ([[0.0, 0.0, 0.90]], [[0.1, 0.1, 0.85]]),   # stays above
        ([[0.0, 0.0, 0.70]], [[0.1, 0.1, 0.60]]),   # stays below (already landed earlier)
        ([[0.0, 0.0, z_thr]], [[0.1, 0.1, 0.60]]),  # starts on the plane (strict > required)
    ]
    for p, q in cases:
        crossed, _ = sb.landing_crossing(torch.tensor(p), torch.tensor(q), z_thr)
        assert not bool(crossed[0]), f"false crossing for segment {p} -> {q}"
    print("[ok] landing_crossing: analytic interp exact, ascending/above/below/on-plane never fire")


def test_bounce_detect(sb):
    z_thr = 0.78
    z = torch.tensor([z_thr + 0.005, z_thr + 0.005, z_thr + 0.20, z_thr + 0.005, z_thr - 0.05])
    vz_prev = torch.tensor([-2.0, 1.0, -2.0, -2.0, -2.0])
    vz_new = torch.tensor([1.8, 2.0, 1.8, -1.8, 1.8])
    hit = sb.bounce_detect(z, vz_prev, vz_new, z_thr)
    # 0: real bounce (flip near plane). 1: already ascending. 2: flip far above the plane
    # (racket/net-like contact, not a landing). 3: still descending. 4: below the band.
    assert hit.tolist() == [True, False, False, False, False], f"bounce mask wrong: {hit.tolist()}"
    print("[ok] bounce_detect: v_z flip inside the plane band only")


def test_aero_matches_venue_flight_model(sb, vb):
    prm = vb.load_venue_params()
    g = torch.Generator().manual_seed(1)
    n = 256
    v = torch.randn(n, 3, generator=g) * 4.0    # within the 1-7 m/s envelope scale
    w = torch.randn(n, 3, generator=g) * 40.0   # rad/s
    mass = 0.0034
    force = sb.venue_aero_force(v, w, mass, prm.k_d, prm.k_m)
    # force/m + g_vec must equal the analytic flight acceleration: the per-substep engine wrench
    # and the RK4 reward rollout integrate the SAME venue law (gravity is PhysX's job).
    a_from_force = force / mass
    a_from_force[:, 2] -= prm.g
    a_ref = vb.flight_accel(v, w, prm)
    err = (a_from_force - a_ref).abs().max()
    assert float(err) < 1e-5, f"aero force disagrees with virtual_ball.flight_accel by {float(err)}"

    # |v| clip bounds the drag force under a numerical blowup (same guard as table_tennis ball.py).
    v_blow = torch.tensor([[1.0e4, 0.0, 0.0]])
    f_blow = sb.venue_aero_force(v_blow, torch.zeros(1, 3), mass, prm.k_d, prm.k_m, speed_clip=50.0)
    bound = mass * prm.k_d * 50.0**2 * (1.0 + 1e-5)  # float32 rounding headroom
    assert float(f_blow.norm()) <= bound, f"clipped drag force {float(f_blow.norm())} exceeds bound {bound}"
    print(f"[ok] venue_aero_force == mass*(flight_accel + g z_hat) on {os.path.basename(prm.source_path)} "
          f"(max err {float(err):.2e}); |v| clip bounds the force")


def test_quat_rotate_inverse(sb):
    # 90 deg about +Z: world +X maps to body -Y under the inverse rotation.
    q = torch.tensor([[math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]])
    v = torch.tensor([[1.0, 0.0, 0.0]])
    out = sb.quat_rotate_inverse_wxyz(q, v)
    assert torch.allclose(out, torch.tensor([[0.0, -1.0, 0.0]]), atol=1e-6), f"got {out}"

    # Random quaternions vs the transposed rotation matrix.
    g = torch.Generator().manual_seed(2)
    q = torch.randn(128, 4, generator=g)
    q = q / q.norm(dim=-1, keepdim=True)
    v = torch.randn(128, 3, generator=g)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        dim=1,
    )  # (N, 3, 3) body->world; inverse rotation = R^T v
    expect = torch.einsum("nij,ni->nj", R, v)
    out = sb.quat_rotate_inverse_wxyz(q, v)
    err = (out - expect).abs().max()
    assert float(err) < 1e-5, f"quat_rotate_inverse deviates from R^T v by {float(err)}"
    print(f"[ok] quat_rotate_inverse_wxyz == R^T v (max err {float(err):.2e})")


def test_shadow_vs_virtual_err(sb):
    s = torch.tensor([[2.0, 0.5], [1.0, -1.0], [0.0, 0.0]])
    p = torch.tensor([[2.3, 0.1], [1.0, -1.0], [3.0, 4.0]])
    err = sb.shadow_vs_virtual_err(s, p)
    expect = torch.tensor([0.5, 0.0, 5.0])
    assert torch.allclose(err, expect, atol=1e-6), f"{err} vs {expect}"
    print("[ok] shadow_vs_virtual_err: planar norm per env")


def main():
    sb = _load("shadow_ball.py", "shadow_ball_standalone")
    vb = _load("virtual_ball.py", "virtual_ball_standalone")
    test_prestrike_path_hits_contact(sb)
    test_landing_crossing_interp(sb)
    test_bounce_detect(sb)
    test_aero_matches_venue_flight_model(sb, vb)
    test_quat_rotate_inverse(sb)
    test_shadow_vs_virtual_err(sb)
    print("ALL SHADOW-BALL HELPER TESTS PASSED")


if __name__ == "__main__":
    main()
