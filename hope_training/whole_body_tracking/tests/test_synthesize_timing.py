"""Unit tests for scripts/synthesize_timing.py (robot-centric time-law synthesis).

Pure CPU, NO mujoco/torch — the module under test is loaded by file path (same
pattern as test_audit_motion_npz.py) and exercised through its API with
body_mode="interp" on synthetic clips where every expected number is computable
by hand.

Covered (task spec 2026-07-09):
  * uniform-acceleration profile assertions (s̈ constant = sdot*/T_a on the accel
    segment; velocity-continuous piecewise law; monotone s)
  * boundary conditions: rest at start, |v*| exactly at the contact time, rest at
    the end; first output frame healthy
  * contact-frame hit: clean-FD blade speed on the output grid within 2% of |v*|,
    both at the default (source speed) and at a CLI-style override (变速)
  * unreachable-speed handling: tighter budgets / higher |v*| EXTEND T_a (拉长
    run-up), never lower the contact speed; output may exceed the source length
  * irreducible violations (contact-frame speed above the cap) -> verdict
    budget_exceeded_irreducible, T_a falls back to the gentlest profile
  * contact row bitwise copy + face normal preserved; new phase registry 口径

Run:  python3 -m pytest hope_training/whole_body_tracking/tests/test_synthesize_timing.py -q
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "synthesize_timing.py"
_spec = importlib.util.spec_from_file_location("synthesize_timing", _SCRIPT)
syn = importlib.util.module_from_spec(_spec)
sys.modules["synthesize_timing"] = syn
_spec.loader.exec_module(syn)

FPS = 50.0
J = 31          # house joint dim
NB = 32         # house body count (col 31 = right_wrist_yaw_Link)


def make_clip(T=61, contact=40, blade_step=0.02, joint_amp=1.0):
    """Synthetic source clip: one active joint (right elbow, col 24) moving linearly
    in s; the wrist body translates blade_step metres per frame along +x with the
    identity orientation, so the blade path derivative is exactly blade_step m/frame
    and the source clean blade speed is blade_step * FPS m/s everywhere."""
    q = np.zeros((T, J), dtype=np.float32)
    q[:, 24] = np.linspace(0.0, joint_amp, T, dtype=np.float32)
    dq = np.gradient(q.astype(np.float64), 1.0 / FPS, axis=0).astype(np.float32)
    bp = np.zeros((T, NB, 3), dtype=np.float32)
    bp[:, syn.RACKET_BODY, 0] = np.arange(T, dtype=np.float32) * blade_step
    bq = np.zeros((T, NB, 4), dtype=np.float32)
    bq[..., 0] = 1.0  # identity wxyz
    bl = np.gradient(bp.astype(np.float64), 1.0 / FPS, axis=0).astype(np.float32)
    ba = np.zeros_like(bp)
    return {
        "fps": np.array([int(FPS)], dtype=np.int64),
        "joint_pos": q, "joint_vel": dq,
        "body_pos_w": bp, "body_quat_w": bq,
        "body_lin_vel_w": bl, "body_ang_vel_w": ba,
    }, contact / (T - 1)


LOOSE_VLIM = np.full(J, 100.0)
LOOSE_BUDGET = np.full(J, 1000.0)


def run(data, phase, **kw):
    kw.setdefault("body_mode", "interp")
    return syn.synthesize(data, phase, kw.pop("vlim", LOOSE_VLIM),
                          kw.pop("acc_budget", LOOSE_BUDGET), **kw)


# ------------------------------------------------------------------ time law ---- #
def test_uniform_acceleration_profile():
    law = syn.build_time_law(c=40.0, s_end=60.0, sdot_star=50.0, Ta=0.8,
                             fps_out=FPS, post_hold_s=0.04)
    t = np.linspace(law.tw + 1e-6, law.tw + law.Ta - 1e-6, 200)
    s, v, a = law.s_sdot_sddot(t)
    assert np.allclose(a, law.sdot_star / law.Ta)          # 匀加速: s̈ constant
    assert np.allclose(v, (t - law.tw) * law.A)            # ṡ linear ramp
    # velocity continuity across all segment joins
    for tj in (law.tw, law.tw + law.Ta, law.t_star, law.tD0, law.t_end):
        _, v_m, _ = law.s_sdot_sddot(np.array([tj - 1e-7]))
        _, v_p, _ = law.s_sdot_sddot(np.array([tj + 1e-7]))
        assert abs(v_m[0] - v_p[0]) < 1e-3
    # monotone s
    tt = np.linspace(0.0, law.t_end + 0.1, 500)
    ss, _, _ = law.s_sdot_sddot(tt)
    assert (np.diff(ss) >= -1e-12).all()


def test_boundary_conditions():
    law = syn.build_time_law(c=40.0, s_end=60.0, sdot_star=50.0, Ta=0.8,
                             fps_out=FPS, post_hold_s=0.04)
    _, v0, _ = law.s_sdot_sddot(np.array([0.0]))
    assert v0[0] == 0.0                                    # ready 静止起步
    s_star, v_star, _ = law.s_sdot_sddot(np.array([law.t_star]))
    assert abs(s_star[0] - 40.0) < 1e-9                    # contact at s = c
    assert abs(v_star[0] - 50.0) < 1e-9                    # 以答案速度通过
    s_e, v_e, _ = law.s_sdot_sddot(np.array([law.t_end + 1e-9]))
    assert v_e[0] == 0.0 and abs(s_e[0] - 60.0) < 1e-9     # 收拍减速回静止


# ------------------------------------------------------------- speed fidelity --- #
def test_default_speed_reproduces_source():
    data, phase = make_clip()
    out, rep = run(data, phase)
    v_src = rep["source"]["clean_blade_speed_mps"]
    assert rep["answer"]["v_star_mps"] == pytest.approx(v_src, rel=1e-6)
    assert rep["fidelity"]["blade_speed_dev_frac"] < 0.02  # 触球帧拍速偏差 <2%
    assert rep["answer"]["sdot_star_frames_per_s"] == pytest.approx(FPS, rel=1e-6)
    assert rep["verdict"] == "ok"


def test_override_speed_hits_target():
    data, phase = make_clip()
    v_target = 2.5  # blade_step 0.02 * 50 fps = 1.0 m/s source -> 2.5x speed-up
    out, rep = run(data, phase, v_star=v_target)
    assert rep["fidelity"]["blade_speed_clean_out_mps"] == pytest.approx(v_target, rel=0.02)
    assert rep["answer"]["sdot_star_frames_per_s"] == pytest.approx(2.5 * FPS, rel=1e-6)


def test_first_frame_healthy_and_end_rest():
    data, phase = make_clip()
    out, rep = run(data, phase)
    assert rep["fidelity"]["first_frame_max_joint_vel"] < 0.5   # 首帧健康 (audit FAIL @2.0)
    assert float(np.abs(out["joint_vel"][-1]).max()) < 0.5      # ends at rest


# ------------------------------------------------- unreachable -> T_a extension - #
def test_tight_acc_budget_extends_ta():
    data, phase = make_clip()
    _, rep_loose = run(data, phase)
    tight = np.full(J, 1000.0)
    tight[24] = 3.0     # clamp the active joint's acceleration budget [rad/s^2]
    _, rep_tight = run(data, phase, acc_budget=tight)
    assert rep_tight["time_law"]["Ta_s"] > rep_loose["time_law"]["Ta_s"]  # T_a 延长
    # speed is NEVER lowered (绝不降速)
    assert (rep_tight["answer"]["v_star_mps"]
            == pytest.approx(rep_loose["answer"]["v_star_mps"], rel=1e-9))
    assert rep_tight["fidelity"]["blade_speed_dev_frac"] < 0.02


def test_ta_exceeding_source_runup_stretches_timeline():
    data, phase = make_clip(T=61, contact=40)
    tight = np.full(J, 1000.0)
    tight[24] = 0.7     # q'*sdot*/Ta <= 0.7 -> T_a >= (1/60*50)/0.7 = 1.19 s > 0.8 s run-up
    out, rep = run(data, phase, acc_budget=tight)
    src_runup_s = rep["source"]["runup_s"]                # 0.8 s
    assert rep["time_law"]["Ta_s"] > src_runup_s          # T_a 超出源 run-up
    assert rep["output"]["runup_s"] > src_runup_s         # 拉长时间轴
    assert rep["output"]["duration_s"] > rep["source"]["duration_s"]  # 允许超过源片长
    # the forced post-contact decel (follow-through path 20 frames, q'*D = 1.16 rad/s^2)
    # exceeds the 0.7 budget — reported separately, NEVER traded against the accel phase
    assert rep["verdict"] == "ok"                         # pre-contact feasible
    assert rep["budgets"]["decel_over_budget"] is True


def test_vel_cap_extends_ta_via_cruise():
    """A velocity cap on the active joint below its cruise requirement is
    IRREDUCIBLE (contact speed is pinned) -> verdict says so, speed untouched."""
    data, phase = make_clip(joint_amp=2.0)                # q' = 2/60 rad/frame
    vlim = np.full(J, 100.0)
    vlim[24] = 1.5                                        # cap 0.85*1.5 < q'*sdot* = 1.667
    out, rep = run(data, phase, vlim=vlim)
    assert rep["verdict"] == "budget_exceeded_irreducible"
    assert rep["fidelity"]["blade_speed_dev_frac"] < 0.02  # 速度绝不降
    assert rep["budgets"]["worst_vel_joint"] == "right_elbow_joint"


def test_monotone_ta_in_vstar():
    """变速语义: higher |v*| -> larger minimal T_a under the same budget."""
    data, phase = make_clip()
    budget = np.full(J, 1000.0)
    budget[24] = 5.0
    tas = []
    for v in (1.0, 1.5, 2.0):
        _, rep = run(data, phase, v_star=v, acc_budget=budget)
        tas.append(rep["time_law"]["Ta_s"])
    assert tas[0] < tas[1] < tas[2]


# ------------------------------------------------------------ contact fidelity -- #
def test_contact_row_bitwise_and_face_preserved():
    data, phase = make_clip()
    out, rep = run(data, phase)
    assert rep["fidelity"]["contact_row_bitwise"] is True
    assert rep["fidelity"]["face_normal_diff_deg"] < 1e-6
    k = rep["output"]["contact_frame"]
    c = rep["source"]["contact_frame"]
    assert np.array_equal(out["joint_pos"][k], data["joint_pos"][c])


def test_phase_out_registry_convention():
    data, phase = make_clip()
    out, rep = run(data, phase)
    k, T_out = rep["output"]["contact_frame"], rep["output"]["frames"]
    assert rep["output"]["phase_out"] == pytest.approx(k / (T_out - 1), abs=1e-6)
    assert out["joint_pos"].shape[0] == T_out
    # contact time lands EXACTLY on the output grid (wait-based snap)
    assert abs(rep["time_law"]["t_star_s"] * FPS - k) < 1e-6


def test_joint_vel_rediff_consistency():
    data, phase = make_clip()
    out, rep = run(data, phase)
    grad = np.gradient(out["joint_pos"].astype(np.float64), 1.0 / FPS, axis=0)
    assert float(np.abs(out["joint_vel"] - grad).max()) < 1e-4  # csv_to_npz convention


def test_unknown_keys_refused():
    data, phase = make_clip()
    data["mystery_track"] = np.zeros((5, 3))
    with pytest.raises(SystemExit):
        run(data, phase)


def test_contact_too_close_to_edge_refused():
    data, _ = make_clip(T=61)
    with pytest.raises(SystemExit):
        run(data, 1.0)   # contact at the last frame: no follow-through path
