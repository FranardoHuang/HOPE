"""Unit tests for scripts/retime_motion_clip.py (non-uniform retiming, 非均匀重定时).

Pure CPU, NO mujoco/isaac/torch — the module under test is loaded by file path
(same pattern as test_audit_motion_npz.py) and exercised in --body-mode interp.
The synthetic clip has a violent high-acceleration burst OUTSIDE the strike window
and calm motion inside it, so every guarantee below is checkable by hand:

  * strike window rows (joint_pos AND body_pos_w/body_quat_w) are BITWISE copies
    of the source frames on the new uniform grid;
  * the time map is monotone and the total duration stays within the budget cap;
  * audit-style mean|acc| and the out-of-window peak |acc| both DROP;
  * the printed new phase round-trips to the bitwise-copied contact frame;
  * the re-differentiated contact-frame joint_vel reproduces the source values;
  * fail-closed: unknown time-axis npz keys and over/under-specified objectives
    refuse loudly.

Run:  python3 -m pytest hope_training/whole_body_tracking/tests/test_retime_motion_clip.py -q
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

# --- load the module under test by path (scripts/ is not a package) ----------- #
_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))  # so its own `import csv_to_npz_mujoco` resolves
_spec = importlib.util.spec_from_file_location("retime_motion_clip",
                                               _SCRIPTS / "retime_motion_clip.py")
rt = importlib.util.module_from_spec(_spec)
sys.modules["retime_motion_clip"] = rt
_spec.loader.exec_module(rt)

FPS = 50.0
T = 61
J = 3
CONTACT = 40                      # phase = 40/60
PHASE = CONTACT / (T - 1)
W0, W1 = CONTACT - 5, CONTACT + 5  # +-0.1 s @ 50 fps


def _synthetic_clip() -> dict:
    """Calm swing + a violent out-of-window burst at frames 5-15."""
    t = np.arange(T) / FPS
    q = np.zeros((T, J))
    q[:, 0] = 0.4 * np.sin(2 * np.pi * 0.8 * t)                # calm background
    burst = np.exp(-0.5 * ((np.arange(T) - 10) / 2.0) ** 2)     # sharp bump @ f10
    q[:, 1] = 0.8 * burst
    q[:, 2] = 0.2 * np.cos(2 * np.pi * 0.5 * t)
    q = q.astype(np.float32)
    dt = 1.0 / FPS
    dq = np.gradient(q.astype(np.float64), dt, axis=0).astype(np.float32)

    nb = 2
    body_pos = np.zeros((T, nb, 3), dtype=np.float32)
    body_pos[:, 0, 0] = 0.1 * t                                 # root drifts in x
    body_pos[:, 0, 2] = 0.8
    body_pos[:, 1] = body_pos[:, 0] + np.array([0.1, 0.0, 0.4], dtype=np.float32)
    ang = 0.01 * np.arange(T)
    body_quat = np.zeros((T, nb, 4), dtype=np.float32)
    body_quat[:, :, 0] = np.cos(ang / 2)[:, None]
    body_quat[:, :, 3] = np.sin(ang / 2)[:, None]
    body_lin = np.gradient(body_pos.astype(np.float64), dt, axis=0).astype(np.float32)
    body_ang = np.zeros((T, nb, 3), dtype=np.float32)
    return {
        "fps": np.array([50], dtype=np.int64),
        "joint_pos": q, "joint_vel": dq,
        "body_pos_w": body_pos, "body_quat_w": body_quat,
        "body_lin_vel_w": body_lin, "body_ang_vel_w": body_ang,
    }


def _retime(data, **kw):
    args = dict(phase=PHASE, window_s=0.1, s_min=0.9, s_max=3.0, smooth_frames=5,
                ramp_frames=4, body_mode="interp", mjcf=None, body_order=None,
                budget=None, target_mean_acc=None, target_peak_acc=None,
                stretch_max=1.3)
    args.update(kw)
    return rt.retime_clip(data, **args)


# ------------------------------------------------------------------ guarantees - #
def test_window_rows_bitwise_and_phase_roundtrip():
    data = _synthetic_clip()
    out, rep = _retime(data, budget=1.25)
    k0, k1 = rep["frames"]["window_out"]
    assert k1 - k0 == W1 - W0
    assert np.array_equal(out["joint_pos"][k0:k1 + 1], data["joint_pos"][W0:W1 + 1])
    assert np.array_equal(out["body_pos_w"][k0:k1 + 1], data["body_pos_w"][W0:W1 + 1])
    assert np.array_equal(out["body_quat_w"][k0:k1 + 1], data["body_quat_w"][W0:W1 + 1])
    c_out, T_out = rep["frames"]["contact_out"], rep["frames"]["T_out"]
    assert np.array_equal(out["joint_pos"][c_out], data["joint_pos"][CONTACT])
    assert rt.contact_frame(rep["phase"]["out"], T_out) == c_out


def test_monotone_map_and_budget_respected():
    data = _synthetic_clip()
    q = data["joint_pos"]
    res = rt.retime_joint_space(q, FPS, PHASE, 0.1, 1.25, 0.9, 3.0, 5, 4)
    assert np.all(np.diff(res["tau"]) > 0)
    assert abs(res["s"].mean() - 1.25) < 0.06          # taper/alignment slack
    assert res["s"].min() < 1.0                        # 略压 path exercised (s_min=0.9)
    assert res["s"][0] >= 1.0 and res["s"][-1] >= 1.0  # head/tail never compressed
    _, rep = _retime(data, budget=1.25)
    assert rep["duration_s"]["out"] <= rep["duration_s"]["in"] * 1.31
    assert rep["frames"]["T_out"] >= T                  # never net-shortens


def test_acceleration_drops_out_of_window():
    data = _synthetic_clip()
    _, rep = _retime(data, budget=1.25)
    a = rep["acc_rad_s2"]
    assert a["mean_out"] < a["mean_in"]
    assert a["out_of_window_peak_out"] < a["out_of_window_peak_in"]
    # the burst is out-of-window, so the GLOBAL peak must drop too
    assert a["peak_out"] < a["peak_in"]


def test_contact_velocity_reproduced():
    data = _synthetic_clip()
    _, rep = _retime(data, budget=1.25)
    # source joint_vel is np.gradient of joint_pos; contact neighbours are bitwise
    # in-window rows, so the re-diff reproduces it (float32 arithmetic floor)
    assert rep["fidelity"]["contact_joint_vel_max_abs_diff"] < 1e-4


def test_first_frame_health_no_regression():
    data = _synthetic_clip()
    _, rep = _retime(data, budget=1.25)
    ff = rep["first_frame"]
    # head protection: even with s_min=0.9 compression enabled, frame-0 velocity
    # may not regress (pre-window alignment rescale allows a <=2% wiggle)
    assert ff["max_joint_vel_out"] <= ff["max_joint_vel_in"] * 1.02 + 1e-6


def test_target_mean_acc_search_attains_or_reports():
    data = _synthetic_clip()
    _, rep_free = _retime(data, budget=1.3)
    reachable = rep_free["acc_rad_s2"]["mean_out"] * 1.05  # just above the 1.3x floor
    _, rep = _retime(data, target_mean_acc=reachable)
    assert rep["target_attained"]
    assert rep["acc_rad_s2"]["mean_out"] <= reachable + 1e-9
    assert rep["budget_used"] <= 1.3 + 1e-9
    # unreachable target: ships the cap, reports the needed budget honestly
    _, rep2 = _retime(data, target_mean_acc=rep_free["acc_rad_s2"]["mean_out"] * 0.5)
    assert not rep2["target_attained"]
    assert rep2["budget_used"] == pytest.approx(1.3)


def test_fail_closed():
    data = _synthetic_clip()
    data_bad = dict(data)
    data_bad["mystery"] = np.zeros((T, 2), dtype=np.float32)   # time-axis stranger
    with pytest.raises(SystemExit, match="unknown time-axis key"):
        _retime(data_bad, budget=1.2)
    with pytest.raises(SystemExit, match="exactly ONE"):
        _retime(data)                                          # no objective at all
    with pytest.raises(SystemExit, match="exactly ONE"):
        _retime(data, budget=1.2, target_mean_acc=4.5)         # two objectives
    with pytest.raises(SystemExit, match="requires --mjcf"):
        _retime(data, budget=1.2, body_mode="fk")


def test_window_at_clip_head_is_supported():
    data = _synthetic_clip()
    out, rep = _retime(data, phase=3 / (T - 1), budget=1.2)    # window clamps at 0
    k0, k1 = rep["frames"]["window_out"]
    assert k0 == 0 and rep["frames"]["K_align"] == 0
    assert np.array_equal(out["joint_pos"][k0:k1 + 1], data["joint_pos"][0:k1 + 1])
