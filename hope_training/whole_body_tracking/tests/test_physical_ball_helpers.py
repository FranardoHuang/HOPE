"""Physical-ball pure-helper unit tests — NO Isaac imports.

Loads ``tasks/tracking/mdp/physical_ball.py`` (and ``virtual_ball.py`` for the forward-flight
reference) directly by file path, the same standalone pattern as ``test_shadow_ball_helpers.py``
(the mdp package ``__init__`` pulls isaaclab; physical_ball's own sibling imports fall back to
file-path loading in this context). Covers the Phase A truth-instrument math:

* back_integrate_incoming: reverse-time venue-model integration ROUNDTRIPS — forward RK4 flight
  (virtual_ball.rk4_step) from the returned launch state for tts_effective seconds re-arrives at
  the question (contact_pos, incoming_vel) to < 1 mm / < 0.01 m/s, for tts in {0.3, 0.6, 1.0}
  (+ the 1.5 s support bound), with and without spin, and with MIXED per-env tts in one call
  (pure reverse math isolated by passing the truncation plane far below).
* TABLE-PLANE TRUNCATION (seed=1 pod-defect fix): rising-contact (vz=+0.2) and long-tts
  descending cases whose backward path dips below z = surface+R stop at the LAST state strictly
  above plane+margin, report tts_effective < tts, and the forward roundtrip over tts_effective
  still arrives < 1 mm / < 0.01 m/s (final-ballistic-segment serve; pre-bounce = future
  bounce-aware serve). Re-serving with tts = tts_effective (the manager's delayed-serve path)
  is un-truncated.
* predict_table_contact: the fitted table bounce with the venue ``contact.table`` params —
  analytic vertical-drop case (v_z+ = e |v_z-|, spin untouched), and a crafted descending
  crossing cross-checked NUMERICALLY against the canonical numpy contact model
  (hope_training/ball_physics_fit/contact_model.predict_contact) including a cap-binding case.
* load_venue_table_params: reads the REAL venue yaml (e_eff/a_t/b_t/mu + ball constants).
* schedule_serves / table_bounds_mask: parked-ball serve scheduling and the bounce footprint.

Run:  /opt/anaconda3/bin/python3 hope_training/whole_body_tracking/tests/test_physical_ball_helpers.py
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MDP_DIR = os.path.join(
    HERE, "..", "source", "whole_body_tracking", "whole_body_tracking", "tasks", "tracking", "mdp"
)
CONTACT_MODEL_PATH = os.path.join(HERE, "..", "..", "ball_physics_fit", "contact_model.py")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.abspath(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclass resolution needs the module registered during exec
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------------------- #
# pytest fixtures — the same modules main() loads for the direct-run path, so the file works
# both ways: ``pytest test_physical_ball_helpers.py`` and ``python3 test_physical_ball_helpers.py``.
# ------------------------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def pb():
    return _load(os.path.join(MDP_DIR, "physical_ball.py"), "physical_ball_standalone")


@pytest.fixture(scope="module")
def vb():
    return _load(os.path.join(MDP_DIR, "virtual_ball.py"), "virtual_ball_standalone")


@pytest.fixture(scope="module")
def sbmod():
    return _load(os.path.join(MDP_DIR, "shadow_ball.py"), "shadow_ball_standalone")


@pytest.fixture(scope="module")
def cm():
    return _load(CONTACT_MODEL_PATH, "contact_model_standalone")


@pytest.fixture(scope="module")
def pbmod(pb):
    # the manager test takes the SAME physical_ball module (main() passes pb to it too)
    return pb


# Pure reverse-flight math is ISOLATED from the plane truncation by pushing the truncation
# plane far below (surface_z = -100): with nothing to truncate, tts_effective == tts exactly and
# the full-tts roundtrip bound applies. The truncation behaviour itself (the real default plane)
# is covered by test_plane_truncation below.
_NO_PLANE = -100.0


def test_back_integration_roundtrip(pb, vb):
    prm = vb.load_venue_params()
    h = 1e-3
    contact = torch.tensor([[0.42, -0.15, 0.95]])
    cases = [
        # (tts, v_in, omega) — velocities inside the venue sampling box, spins inside 0-15 rev/s.
        (0.3, (-3.5, 0.3, -0.6), (0.0, 0.0, 0.0)),
        (0.3, (-2.4, -0.5, 0.4), (30.0, -20.0, 40.0)),
        (0.6, (-4.2, 0.1, -0.9), (0.0, 0.0, 0.0)),
        (0.6, (-3.0, 0.6, 0.2), (-25.0, 35.0, -15.0)),
        (0.6, (-4.5, -0.6, -1.0), (50.0, 50.0, 50.0)),  # venue-box worst corner at the serve horizon
        (1.0, (-2.8, -0.3, -0.4), (0.0, 0.0, 0.0)),
        (1.0, (-3.8, 0.4, 0.5), (45.0, 10.0, -30.0)),
    ]
    worst_pos, worst_vel = 0.0, 0.0
    for tts_val, v_in_t, omega_t in cases:
        v_in = torch.tensor([list(v_in_t)])
        omega = torch.tensor([list(omega_t)])
        tts = torch.full((1,), tts_val)
        launch_p, launch_v, t_eff = pb.back_integrate_incoming(
            contact, v_in, omega, tts, prm, h=h, surface_z=_NO_PLANE
        )
        # (1e-4 tolerance: t_eff is a float32 sum over up to 1000 per-env steps ~ 1e-5 noise,
        # far below half a step h = 1e-3.)
        assert abs(float(t_eff[0]) - tts_val) < 1e-4, (
            f"tts={tts_val}: un-truncatable case reported tts_effective={float(t_eff[0])}"
        )

        # forward flight from the launch state for exactly tts seconds via virtual_ball.rk4_step
        # (the SAME per-env-uniform step-count convention the helper uses).
        n = max(1, int(math.ceil(tts_val / h)))
        hf = tts_val / n
        p, v = launch_p, launch_v
        for _ in range(n):
            p, v = vb.rk4_step(p, v, omega, hf, prm)

        pos_err = float(torch.linalg.norm(p - contact))
        vel_err = float(torch.linalg.norm(v - v_in))
        worst_pos, worst_vel = max(worst_pos, pos_err), max(worst_vel, vel_err)
        assert pos_err < 1e-3, f"tts={tts_val} spin={omega_t}: roundtrip pos err {pos_err:.2e} m >= 1 mm"
        assert vel_err < 1e-2, f"tts={tts_val} spin={omega_t}: roundtrip vel err {vel_err:.2e} m/s >= 0.01"

    # MIXED per-env tts in ONE batched call (the manager's serve path): every row must roundtrip.
    tts_vec = torch.tensor([0.3, 0.6, 1.0, 0.0])
    n_env = tts_vec.shape[0]
    contact_b = torch.tensor([[0.42, -0.15, 0.95]]).expand(n_env, 3).contiguous()
    v_in_b = torch.tensor([[-3.5, 0.3, -0.6], [-3.0, 0.6, 0.2], [-2.8, -0.3, -0.4], [-4.0, 0.0, 0.0]])
    omega_b = torch.tensor([[0.0, 0.0, 0.0], [30.0, -20.0, 40.0], [-25.0, 35.0, -15.0], [10.0, 10.0, 10.0]])
    lp, lv, te = pb.back_integrate_incoming(
        contact_b, v_in_b, omega_b, tts_vec, prm, h=h, surface_z=_NO_PLANE
    )
    assert torch.allclose(te, tts_vec, atol=1e-4), f"batched tts_effective wrong: {te.tolist()}"
    # tts = 0 row: launch state IS the contact state.
    assert torch.allclose(lp[3], contact_b[3], atol=1e-9) and torch.allclose(lv[3], v_in_b[3], atol=1e-9)
    for i in range(3):
        tv = float(tts_vec[i])
        n = max(1, int(math.ceil(float(tts_vec.max()) / h)))  # helper used max-tts step count
        hf = tv / n
        p, v = lp[i : i + 1], lv[i : i + 1]
        for _ in range(n):
            p, v = vb.rk4_step(p, v, omega_b[i : i + 1], hf, prm)
        assert float(torch.linalg.norm(p - contact_b[i])) < 1e-3, f"batched row {i} pos roundtrip"
        assert float(torch.linalg.norm(v - v_in_b[i])) < 1e-2, f"batched row {i} vel roundtrip"

    # tts = 1.5 s support bound: reverse-time quadratic drag has a finite-time blowup at
    # t_back ~ 1.3 s for venue contact states (a slow arrival cannot have a longer pure-flight
    # history — the real one includes a table bounce), so the guarantee at 1.5 s is FINITENESS
    # via the documented BACKINT_SPEED_CAP stop (tts_effective reports the integrated span),
    # not exact arrival.
    tts15 = torch.full((1,), 1.5)
    lp15, lv15, te15 = pb.back_integrate_incoming(
        contact, torch.tensor([[-3.2, 0.2, -0.5]]), torch.tensor([[20.0, -40.0, 25.0]]),
        tts15, prm, h=h, surface_z=_NO_PLANE,
    )
    assert bool(torch.isfinite(lp15).all()) and bool(torch.isfinite(lv15).all()), "1.5 s must stay finite"
    sp15 = float(torch.linalg.norm(lv15, dim=-1))
    assert sp15 <= pb.BACKINT_SPEED_CAP * 1.1, f"capped speed {sp15:.1f} escaped the stop"
    assert float(te15[0]) < 1.5, "speed-capped row must report a shorter tts_effective"
    print(
        f"[ok] back_integrate_incoming roundtrip: pos err < 1 mm, vel err < 0.01 m/s "
        f"(worst {worst_pos:.2e} m / {worst_vel:.2e} m/s over tts 0.3/0.6/1.0 s, with/without "
        f"spin; tts_effective == tts un-truncated; mixed per-env tts batched call exact; 1.5 s "
        f"support = finite via speed-cap stop, |v|={sp15:.1f} m/s, t_eff={float(te15[0]):.2f} s)"
    )


def test_plane_truncation(pb, vb):
    """Seed=1 pod-defect fix: the backward path must STOP above the table plane, and the forward
    roundtrip over the returned tts_effective must still arrive exactly."""
    prm = vb.load_venue_params()
    h = 1e-3
    surface_z = 0.76
    z_min = surface_z + prm.ball_radius + pb.SERVE_PLANE_MARGIN
    contact = torch.tensor([[0.42, -0.15, 0.95]])
    cases = [
        # rising contact velocity (vz = +0.2, ~11% of the bank): the backward path dips below the
        # plane almost immediately (the real history bounced here).
        ("rising vz=+0.2", 0.6, (-3.0, 0.2, 0.2), (0.0, 0.0, 0.0)),
        ("rising vz=+0.2 spin", 0.6, (-3.0, 0.2, 0.2), (30.0, -20.0, 40.0)),
        # long-tts descending case: backward first rises above the contact, then its backward
        # apex passes below the plane further out (pre-bounce segment).
        ("long-tts descending", 1.0, (-3.5, 0.0, -0.6), (0.0, 0.0, 0.0)),
        ("long-tts descending spin", 1.0, (-3.5, 0.0, -0.6), (-25.0, 35.0, -15.0)),
    ]
    for name, tts_val, v_in_t, omega_t in cases:
        v_in = torch.tensor([list(v_in_t)])
        omega = torch.tensor([list(omega_t)])
        tts = torch.full((1,), tts_val)
        lp, lv, te = pb.back_integrate_incoming(
            contact, v_in, omega, tts, prm, h=h, surface_z=surface_z
        )
        t_eff = float(te[0])
        assert 0.0 < t_eff < tts_val, f"{name}: expected truncation, got tts_effective={t_eff}"
        assert float(lp[0, 2]) > z_min, (
            f"{name}: launch z={float(lp[0, 2]):.4f} not strictly above plane+margin {z_min:.4f}"
        )

        # forward roundtrip over the TRUNCATED span: k accepted steps of the helper's per-env
        # step h_i reproduce the contact state.
        n_total = max(1, int(math.ceil(tts_val / h)))
        h_i = tts_val / n_total
        k = int(round(t_eff / h_i))
        assert abs(k * h_i - t_eff) < 0.5 * h_i, f"{name}: tts_effective not a multiple of h_i"
        p, v = lp, lv
        for _ in range(k):
            p, v = vb.rk4_step(p, v, omega, h_i, prm)
        pos_err = float(torch.linalg.norm(p - contact))
        vel_err = float(torch.linalg.norm(v - v_in))
        assert pos_err < 1e-3, f"{name}: truncated roundtrip pos err {pos_err:.2e} m >= 1 mm"
        assert vel_err < 1e-2, f"{name}: truncated roundtrip vel err {vel_err:.2e} m/s >= 0.01"

        # Manager delayed-serve equivalence: re-asking for exactly tts_effective must be
        # UN-truncated (the whole final ballistic segment is above the plane) — this is the
        # per-env condition (t_eff >= t_back) the manager serves on.
        lp2, lv2, te2 = pb.back_integrate_incoming(
            contact, v_in, omega, torch.full((1,), t_eff), prm, h=h, surface_z=surface_z
        )
        assert abs(float(te2[0]) - t_eff) < 2e-3, (
            f"{name}: re-serve at tts_effective re-truncated ({float(te2[0])} vs {t_eff})"
        )
        print(f"[ok] plane truncation ({name}): tts {tts_val} -> tts_effective {t_eff:.3f}, "
              f"launch z {float(lp[0, 2]):.3f} > {z_min:.3f}, truncated roundtrip "
              f"{pos_err:.1e} m / {vel_err:.1e} m/s")


def test_table_params_loader(pb):
    tp = pb.load_venue_table_params()
    # The venue fit's table block (configs/ball_physics_venue.yaml, 2026-07-03).
    assert abs(tp.e_eff - 0.9215) < 1e-9, f"table e_eff {tp.e_eff}"
    assert abs(tp.a_t - 0.369) < 1e-9, f"table a_t {tp.a_t}"
    assert tp.b_t == 0.0, f"table b_t {tp.b_t}"
    assert abs(tp.mu - 2.0) < 1e-9, f"table mu_safety {tp.mu}"
    assert abs(tp.ball_radius - 0.020) < 1e-12 and abs(tp.inertia_coeff - 2.0 / 3.0) < 1e-9
    print(
        f"[ok] load_venue_table_params: e_eff={tp.e_eff} a_t={tp.a_t} b_t={tp.b_t} mu={tp.mu} "
        f"from {os.path.basename(tp.source_path)}"
    )


def test_table_bounce_math(pb, sbmod, cm):
    tp = pb.load_venue_table_params()
    # The numpy canonical model hard-codes the same ball constants — precondition for the mirror.
    assert abs(cm.R_BALL - tp.ball_radius) < 1e-12 and abs(cm.C_INERTIA - tp.inertia_coeff) < 1e-9

    # (a) analytic: pure vertical drop, no spin -> v+ = (0, 0, e|vz|), spin untouched (u_t = 0).
    v_minus = torch.tensor([[0.0, 0.0, -3.0]])
    w_minus = torch.zeros(1, 3)
    v_plus, w_plus = pb.predict_table_contact(v_minus, w_minus, tp)
    assert torch.allclose(v_plus, torch.tensor([[0.0, 0.0, tp.e_eff * 3.0]]), atol=1e-6), f"{v_plus}"
    assert torch.allclose(w_plus, w_minus, atol=1e-9), "no tangential slip -> no spin change"

    # (b) crafted descending crossing through surface+R inside the table footprint: extract the
    # crossing (shadow_ball.landing_crossing, the manager's detector), then bounce there and
    # cross-check the full (v+, w+) against the canonical numpy contact model with the SAME
    # venue table params (v_r = 0, n = +z).
    z_thr = 0.76 + tp.ball_radius
    prev = torch.tensor([[1.20, 0.10, z_thr + 0.02]])
    new = torch.tensor([[1.25, 0.12, z_thr - 0.02]])
    crossed, xy = sbmod.landing_crossing(prev, new, z_thr)
    assert bool(crossed[0]), "crafted segment must cross the bounce plane descending"
    assert bool(pb.table_bounds_mask(xy, 0.5, 2.74, 1.525 / 2.0)[0]), "crossing inside the footprint"

    v_minus = torch.tensor([[1.0, 0.5, -3.0]])
    w_minus = torch.tensor([[10.0, 5.0, -8.0]])
    v_plus, w_plus = pb.predict_table_contact(v_minus, w_minus, tp)
    ref = cm.predict_contact(
        v_minus.numpy(), np.zeros((1, 3)), np.array([[0.0, 0.0, 1.0]]), w_minus.numpy(),
        tp.e_eff, tp.a_t, tp.b_t, tp.mu,
    )
    dv = float(np.abs(v_plus.numpy() - ref["v_plus"]).max())
    dw = float(np.abs(w_plus.numpy() - ref["omega_plus"]).max())
    assert dv < 1e-5, f"v_plus deviates from canonical contact model by {dv}"
    assert dw < 1e-4, f"omega_plus deviates from canonical contact model by {dw}"
    assert float(v_plus[0, 2]) > 0.0, "bounce must send the ball back up"

    # (c) cap-binding regime: huge tangential slip vs a soft normal component -> the friction cap
    # s = mu (1+e) |u_n| binds (needs |u_t|/|u_n| > mu(1+e)/a_t ~ 10.4 at the venue table params;
    # the numpy model must agree AND report the cap as binding).
    v_minus = torch.tensor([[9.0, 0.0, -0.5]])
    w_minus = torch.zeros(1, 3)
    v_plus, w_plus = pb.predict_table_contact(v_minus, w_minus, tp)
    ref = cm.predict_contact(
        v_minus.numpy(), np.zeros((1, 3)), np.array([[0.0, 0.0, 1.0]]), w_minus.numpy(),
        tp.e_eff, tp.a_t, tp.b_t, tp.mu,
    )
    assert bool(ref["cap_binds"][0]), "crafted case must bind the friction cap"
    assert float(np.abs(v_plus.numpy() - ref["v_plus"]).max()) < 1e-5
    assert float(np.abs(w_plus.numpy() - ref["omega_plus"]).max()) < 1e-4
    print("[ok] predict_table_contact: vertical-drop analytic, crafted-crossing + cap-binding "
          "cases match the canonical numpy contact model on the venue table params")


def test_serve_scheduling(pb):
    horizon, min_tts = pb.SERVE_HORIZON_S, 0.02
    parked = torch.tensor([True, True, True, True, True, False])
    tts = torch.tensor([0.70, horizon, 0.30, min_tts, -0.10, 0.30])
    due = pb.schedule_serves(parked, tts, horizon, min_tts)
    # 0: beyond horizon -> keep waiting. 1: exactly at horizon -> serve (<=). 2: inside -> serve.
    # 3: exactly at min_tts -> too late, never serve (strict >). 4: past the strike -> never.
    # 5: not parked (already flying) -> never double-serve.
    assert due.tolist() == [False, True, True, False, False, False], f"serve mask wrong: {due.tolist()}"

    # A served env leaves PARKED, so the next step's mask cannot re-serve it (idempotence),
    # and an un-served env re-parks only via on_resample (new question) — modeled here by the
    # parked flag the manager owns.
    parked2 = parked & ~due
    due2 = pb.schedule_serves(parked2, tts - 0.02, horizon, min_tts)
    assert not bool(due2[1]) and not bool(due2[2]), "served envs must not serve twice"
    assert not bool(due2[0]), "0.68 s is still beyond the 0.6 s horizon"

    # Bounds mask sanity (the bounce footprint in env-local frame).
    xy = torch.tensor([[0.5, 0.0], [3.24, 0.762], [0.49, 0.0], [3.25, 0.0], [1.5, 0.77]])
    m = pb.table_bounds_mask(xy, 0.5, 2.74, 1.525 / 2.0)
    assert m.tolist() == [True, True, False, False, False], f"bounds mask wrong: {m.tolist()}"
    print("[ok] schedule_serves: horizon/min-tts/parked gating + no double serve; "
          "table_bounds_mask edges exact")


# ------------------------------------------------------------------------------------------- #
# Manager harness (Isaac-free mocks driving the REAL PhysicalBallManager.update path)
# ------------------------------------------------------------------------------------------- #
class _MockBall:
    def __init__(self, n):
        import types

        self.data = types.SimpleNamespace(
            root_pos_w=torch.zeros(n, 3),
            root_lin_vel_w=torch.zeros(n, 3),
            root_ang_vel_w=torch.zeros(n, 3),
            root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(n, 4).contiguous(),
        )

    def write_root_pose_to_sim(self, pose, env_ids=None):
        self.data.root_pos_w[env_ids] = pose[:, :3]

    def write_root_velocity_to_sim(self, vel6, env_ids=None):
        self.data.root_lin_vel_w[env_ids] = vel6[:, :3]
        self.data.root_ang_vel_w[env_ids] = vel6[:, 3:]


def _make_mock_world(n):
    """Minimal (env, cmd) pair for PhysicalBallManager: no Isaac, no physics callback (degraded
    control-rate path), flat env origins at zero (env-local == world)."""
    import types

    ball = _MockBall(n)

    class _Scene:
        env_origins = torch.zeros(n, 3)

        def __getitem__(self, key):
            assert key == "pb_ball", key
            return ball

    class _Sim:
        def add_physics_callback(self, name, fn):
            raise RuntimeError("no sim in the Isaac-free harness")

    env = types.SimpleNamespace(scene=_Scene(), sim=_Sim(), step_dt=0.02)
    cmd = types.SimpleNamespace(
        device="cpu",
        num_envs=n,
        metrics={},
        cfg=types.SimpleNamespace(
            vb_table_near_x=0.5,
            vb_table_surface_z=0.76,
            exact_success_decay=1.0,
            question_bank="",
            vb_target_x=2.555,
            vb_target_y=0.0,
        ),
        time_to_strike=torch.zeros(n),
        racket_target_pos_w=torch.zeros(n, 3),
        vb_vel_in_w=torch.zeros(n, 3),
        vb_spin_in_w=torch.zeros(n, 3),
    )
    return env, cmd, ball


def _run_swing(pb, mgr, cmd, tts0, resample_every_step=False):
    """Drive one swing: on_resample, then tts marches down by step_dt to the exact-strike frame.
    ``resample_every_step`` simulates the upstream pathology (motion.just_resampled staying
    latched across steps at low env counts -> _resample_command repeats within one wait)."""
    n = cmd.num_envs
    no_strike = torch.zeros(n, dtype=torch.bool)
    mgr.on_resample(torch.arange(n))
    serve_tts = None
    tts = tts0
    while tts > 0.01:
        if resample_every_step and bool((mgr._mode == 0).all()):
            mgr.on_resample(torch.arange(n))  # upstream repeat while everyone still waits
        cmd.time_to_strike[:] = tts
        mgr.update(no_strike)
        if serve_tts is None and float(cmd.metrics["pb_serve_count"][0]) > mgr._harness_served:
            serve_tts = tts
        tts = round(tts - 0.02, 10)
    cmd.time_to_strike[:] = 0.0
    mgr.update(torch.ones(n, dtype=torch.bool))  # exact-strike frame
    mgr._harness_served = float(cmd.metrics["pb_serve_count"][0])
    return serve_tts


def test_manager_truncation_counting(pbmod):
    """Seed=1 cosmetic-defect regression: pb_serve_truncated_count must advance EXACTLY once per
    truncation-delayed swing — counted at consumption (the serve), invariant to upstream
    _resample_command repeats within the wait — and waiting steps must NOT re-run the reverse
    integration (cached tts_effective; <= 2 integrations per quiet swing)."""
    n = 2
    env, cmd, ball = _make_mock_world(n)
    mgr = pbmod.PhysicalBallManager(cmd, env)
    mgr._harness_served = 0.0

    # Count the batched reverse-integration calls the manager actually makes.
    calls = {"n": 0}
    orig = pbmod.back_integrate_incoming

    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    pbmod.back_integrate_incoming = counting
    try:
        # A truncating question: rising contact velocity at bank-like height (t_eff ~ 0.16 s).
        cmd.racket_target_pos_w[:] = torch.tensor([0.42, -0.15, 0.95])
        cmd.vb_vel_in_w[:] = torch.tensor([-3.0, 0.2, 0.2])
        cmd.vb_spin_in_w[:] = 0.0

        # Swing 1 (quiet): one truncation count per env, serve fired inside the truncated
        # window, and exactly 2 integrations (discovery + serve) — NOT one per waiting step.
        calls["n"] = 0
        serve_tts = _run_swing(pbmod, mgr, cmd, tts0=0.60)
        m = cmd.metrics
        assert float(m["pb_serve_count"][0]) == n, f"swing 1 serves {float(m['pb_serve_count'][0])}"
        assert float(m["pb_serve_truncated_count"][0]) == n, (
            f"swing 1 truncation count {float(m['pb_serve_truncated_count'][0])} != {n}"
        )
        assert serve_tts is not None and 0.10 <= serve_tts <= 0.20, (
            f"truncated serve should fire near t_eff~0.16, got {serve_tts}"
        )
        assert calls["n"] == 2, f"quiet swing ran {calls['n']} integrations (cache broken)"

        # Swing 2 (PATHOLOGICAL: on_resample repeats every waiting step — the pod cadence that
        # produced 1945 counts for 110 serves): the count still advances by exactly n.
        serve_tts = _run_swing(pbmod, mgr, cmd, tts0=0.60, resample_every_step=True)
        assert float(m["pb_serve_truncated_count"][0]) == 2 * n, (
            f"pathological swing re-counted: {float(m['pb_serve_truncated_count'][0])} != {2 * n}"
        )
        assert float(m["pb_serve_count"][0]) == 2 * n and serve_tts is not None

        # Swing 3 (un-truncated question: high contact, descending): serves on the FIRST
        # candidate step, 1 integration, truncation count unchanged.
        cmd.racket_target_pos_w[:] = torch.tensor([0.42, -0.15, 2.50])
        cmd.vb_vel_in_w[:] = torch.tensor([-3.5, 0.0, -0.6])
        calls["n"] = 0
        serve_tts = _run_swing(pbmod, mgr, cmd, tts0=0.30)
        assert float(m["pb_serve_truncated_count"][0]) == 2 * n, "un-truncated swing must not count"
        assert float(m["pb_serve_count"][0]) == 3 * n and serve_tts == 0.30, (
            f"un-truncated swing must serve on the first candidate step (got {serve_tts})"
        )
        assert calls["n"] == 1, f"un-truncated swing ran {calls['n']} integrations"
        assert float(m["pb_strike_meas_count"][0]) == 3 * n, "every served swing must be measured"
    finally:
        pbmod.back_integrate_incoming = orig
    print("[ok] manager truncation counting: exactly one pb_serve_truncated_count per delayed "
          "swing (quiet AND under per-step upstream resample repeats); tts_effective cached — "
          "2 integrations per truncated swing, 1 per clean swing")


def main():
    pb = _load(os.path.join(MDP_DIR, "physical_ball.py"), "physical_ball_standalone")
    vb = _load(os.path.join(MDP_DIR, "virtual_ball.py"), "virtual_ball_standalone")
    sbmod = _load(os.path.join(MDP_DIR, "shadow_ball.py"), "shadow_ball_standalone")
    cm = _load(CONTACT_MODEL_PATH, "contact_model_standalone")
    test_back_integration_roundtrip(pb, vb)
    test_plane_truncation(pb, vb)
    test_table_params_loader(pb)
    test_table_bounce_math(pb, sbmod, cm)
    test_serve_scheduling(pb)
    test_manager_truncation_counting(pb)
    print("ALL PHYSICAL-BALL HELPER TESTS PASSED")


if __name__ == "__main__":
    main()
