"""Unit tests for scripts/synthesize_timing_v2.py (oracle-guided TOPP-lite time-law).

Pure CPU, NO mujoco/torch: the module is loaded by file path and the mj_inverse oracle
is replaced by a deterministic STUB that flags frames by their joint-speed (a faithful
proxy — slowing a frame lowers |q̇| ∝ ṡ, so the stub's "infeasibility" recedes exactly as
the real dose does). body_mode="interp" so no FK is needed.

Covered (task spec 2026-07-09, steps ①-⑤):
  * ρ(s) stretch field: ρ≡1 default; lock window keeps ρ=1 (contact speed preserved);
    bumps raise ρ only OUTSIDE the lock window
  * warp: ρ≡1 reproduces the v1 baseline law (contact bitwise, rest start/end, |v*| held);
    ρ>1 stretches the timeline (T_out grows, duration grows) while contact stays bitwise
  * TOPP loop: oracle-flagged OUT-OF-WINDOW frames get slowed -> CoP dose DROPS across
    iterations and beats the v5syn accept threshold; contact blade speed never lowered
  * convergence reasons: clean oracle -> immediate PASS; window-locked residual ->
    'no_out_of_window_flags' + residual_infeasibility_window_locked reported
  * monotone s(t); first output frame healthy

Run:  python3 -m pytest hope_training/whole_body_tracking/tests/test_synthesize_timing_v2.py -q
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
for _name in ("synthesize_timing", "synthesize_timing_v2"):
    _spec = importlib.util.spec_from_file_location(_name, _SCRIPTS / f"{_name}.py")
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_name] = _mod
    _spec.loader.exec_module(_mod)
v2 = sys.modules["synthesize_timing_v2"]
v1 = sys.modules["synthesize_timing"]

FPS = 50.0
J = 31
NB = 32


def make_clip(T=81, contact=48, blade_step=0.02, joint_amp=3.0):
    """One active joint (right elbow col 24) linear in s; wrist body translates blade_step
    m/frame along +x with identity orientation -> source clean blade speed = blade_step*FPS."""
    q = np.zeros((T, J), dtype=np.float32)
    q[:, 24] = np.linspace(0.0, joint_amp, T, dtype=np.float32)
    dq = np.gradient(q.astype(np.float64), 1.0 / FPS, axis=0).astype(np.float32)
    bp = np.zeros((T, NB, 3), dtype=np.float32)
    bp[:, v1.RACKET_BODY, 0] = np.arange(T, dtype=np.float32) * blade_step
    bq = np.zeros((T, NB, 4), dtype=np.float32)
    bq[..., 0] = 1.0
    bl = np.gradient(bp.astype(np.float64), 1.0 / FPS, axis=0).astype(np.float32)
    ba = np.zeros_like(bp)
    return {"fps": np.array([int(FPS)], dtype=np.int64), "joint_pos": q, "joint_vel": dq,
            "body_pos_w": bp, "body_quat_w": bq, "body_lin_vel_w": bl,
            "body_ang_vel_w": ba}, contact / (T - 1)


LOOSE_VLIM = np.full(J, 100.0)
LOOSE_BUDGET = np.full(J, 1000.0)


def stub_oracle(vel_thresh: float):
    """Flag frames where max|q̇| exceeds vel_thresh; CoP excess grows with the overshoot.
    Slowing a frame lowers |q̇| -> the flag clears, mirroring the real dose response."""
    def judge(out, stem, phase_out):
        jv = np.abs(np.asarray(out["joint_vel"], float)).max(axis=1)
        T = jv.shape[0]
        cop = np.where(jv > vel_thresh, (jv - vel_thresh) * 0.05, -0.01)
        cop[0] = cop[-1] = np.nan
        fric = np.full(T, np.nan)
        util = np.zeros(T)
        n_eval = max(T - 2, 1)
        cop_dose = float(np.sum(np.nan_to_num(cop, nan=-np.inf) > 0.0)) / n_eval
        verdict = "PASS" if cop_dose < 0.05 else ("WARN" if cop_dose < 0.20 else "FAIL")
        return v2.OracleReading(cop_excess=cop, fric_ratio=fric, util_max=util,
                                fz=np.full(T, 500.0),
                                doses={"cop": cop_dose, "friction": 0.0, "torque": 0.0},
                                verdict=verdict, contact_frame=None)
    return judge


def run(data, phase, judge, **kw):
    kw.setdefault("body_mode", "interp")
    return v2.topp_lite(data, phase, kw.pop("vlim", LOOSE_VLIM),
                        kw.pop("acc_budget", LOOSE_BUDGET), judge, "clip", **kw)


# ------------------------------------------------------------- stretch field ---------- #
def test_lock_window_keeps_rho_one():
    fld = v2.StretchField.build(s_end=80.0, c=48.0, half=5.0, taper=3.0)
    # a bump centered INSIDE the window must not move ρ there
    fld.bump(np.array([48.0, 30.0]), np.array([1.5, 1.5]))
    win = np.abs(fld.s_grid - 48.0) <= 5.0
    assert np.allclose(fld.rho[win], 1.0)                 # contact window locked
    assert fld.rho[np.argmin(np.abs(fld.s_grid - 30.0))] > 1.0   # out-of-window bumped
    assert (fld.rho >= 1.0).all()                          # never speeds up


def test_lock_weight_shape():
    fld = v2.StretchField.build(s_end=80.0, c=48.0, half=5.0, taper=3.0)
    assert fld.lock_weight(np.array([48.0]))[0] == 0.0     # dead center: fully locked
    assert fld.lock_weight(np.array([48.0 + 5.0 + 3.0]))[0] == pytest.approx(1.0)  # beyond taper
    assert 0.0 < fld.lock_weight(np.array([48.0 + 5.0 + 1.5]))[0] < 1.0            # in taper


# --------------------------------------------------- ρ≡1 reproduces v1 baseline ------- #
def test_identity_warp_reproduces_baseline():
    data, phase = make_clip()
    out, res, law, meta = run(data, phase, stub_oracle(vel_thresh=1e9))  # never flags
    rep = v2.build_report(data, out, res, law, meta, "interp")
    assert res.converged_reason in ("pass", "no_out_of_window_flags")
    assert res.iters == 0
    assert rep["fidelity"]["contact_row_bitwise"] is True
    assert rep["fidelity"]["blade_speed_dev_frac"] < 0.02      # |v*| held
    assert rep["fidelity"]["first_frame_max_joint_vel"] < 0.5  # rest start
    assert float(np.abs(out["joint_vel"][-1]).max()) < 0.5     # rest end
    # ρ everywhere 1 -> duration ~ the v1 baseline law duration
    assert rep["stretch"]["rho_peak"] == pytest.approx(1.0, abs=1e-6)


def test_contact_bitwise_and_phase_convention():
    data, phase = make_clip()
    out, res, law, meta = run(data, phase, stub_oracle(vel_thresh=1e9))
    k, T_out = res.warp.k_star, res.warp.T_out
    c = meta["c"]
    assert np.array_equal(out["joint_pos"][k], data["joint_pos"][c])
    rep = v2.build_report(data, out, res, law, meta, "interp")
    assert rep["output"]["phase_out"] == pytest.approx(k / (T_out - 1), abs=1e-6)


def test_monotone_s_and_grid_snap():
    data, phase = make_clip()
    out, res, law, meta = run(data, phase, stub_oracle(vel_thresh=1e9))
    s = res.warp.s_out
    assert (np.diff(s) >= -1e-9).all()                    # monotone path progression
    assert abs(s[res.warp.k_star] - meta["c"]) < 1e-9     # contact lands exactly on grid


# ------------------------------------------------- TOPP loop lowers the dose ---------- #
def test_topp_lowers_dose_and_beats_threshold():
    data, phase = make_clip(joint_amp=3.0)
    # threshold below the cruise speed so the accel/decel RAMPS (out of window) are flagged
    out0, res0, _, _ = run(data, phase, stub_oracle(vel_thresh=1e9))
    judge = stub_oracle(vel_thresh=1.2)
    out, res, law, meta = run(data, phase, judge, dose_accept=0.30, dose_target=0.02,
                              max_iters=30)
    d0 = res.reading0.doses["cop"]
    df = res.reading.doses["cop"]
    assert df < d0                                         # dose dropped
    assert res.iters >= 1
    # stretched OUTSIDE the window (ρ grew there), window still locked
    assert res.stretch_field.rho.max() > 1.0
    win = np.abs(res.stretch_field.s_grid - meta["c"]) <= meta["half"]
    assert np.allclose(res.stretch_field.rho[win], 1.0)
    # contact speed never lowered
    rep = v2.build_report(data, out, res, law, meta, "interp")
    assert rep["fidelity"]["blade_speed_dev_frac"] < 0.02
    assert rep["fidelity"]["contact_row_bitwise"] is True
    # timeline stretched
    assert rep["output"]["duration_change_x"] >= 1.0


def test_stretch_extends_duration_and_frames():
    data, phase = make_clip(joint_amp=4.0)
    _, res_id, _, _ = run(data, phase, stub_oracle(vel_thresh=1e9))
    _, res_st, _, _ = run(data, phase, stub_oracle(vel_thresh=1.0), max_iters=30)
    assert res_st.warp.T_out >= res_id.warp.T_out
    assert res_st.warp.duration_s >= res_id.warp.duration_s


# ----------------------------------------- window-locked residual is reported --------- #
def test_window_locked_residual_reported():
    data, phase = make_clip(joint_amp=3.0)
    # threshold so low that even the locked cruise window is over it and can't be fixed
    judge = stub_oracle(vel_thresh=0.05)
    out, res, law, meta = run(data, phase, judge, dose_target=0.001, max_iters=40)
    rep = v2.build_report(data, out, res, law, meta, "interp")
    # loop should stop because the only remaining flags are inside the lock window
    assert res.converged_reason in ("no_out_of_window_flags", "duration_guard")
    if res.converged_reason == "no_out_of_window_flags":
        # residual infeasibility localized to the geometry lock -> path-morph signal
        assert rep["acceptance"]["residual_infeasibility_window_locked"] is True


def test_blade_speed_never_lowered_under_tight_flags():
    data, phase = make_clip(joint_amp=5.0)
    out, res, law, meta = run(data, phase, stub_oracle(vel_thresh=0.8), max_iters=30)
    rep = v2.build_report(data, out, res, law, meta, "interp")
    assert rep["answer"]["v_star_mps"] == pytest.approx(rep["source"]["clean_blade_speed_mps"],
                                                        rel=1e-6)
    assert rep["fidelity"]["blade_speed_dev_frac"] < 0.02


def test_unknown_keys_refused():
    data, phase = make_clip()
    data["mystery"] = np.zeros((3, 2))
    with pytest.raises(SystemExit):
        run(data, phase, stub_oracle(1e9))
