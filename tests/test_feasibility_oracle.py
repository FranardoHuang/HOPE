"""CPU sanity tests for scripts/feasibility_oracle.py (A-layer mj_inverse oracle).

Two tiers:
  * pure-geometry tests (numpy only) — run anywhere;
  * mujoco tests — need `mujoco` + the vendor MJCF; skipped otherwise.
    MJCF resolution: $FEAS_ORACLE_MJCF, else the repo-relative vendor path.

Physics sanities encoded here (the calibration contract of the oracle):
  1. static stand      -> required ground fz == robot weight, CoP == CoM xy,
                          friction ~ 0, verdict PASS;
  2. yawed static pose -> CoP == CoM xy ONLY under the body-LOCAL rotational
                          free-joint convention (regression for the frame
                          pinning experiment, 2026-07-09);
  3. free fall         -> qfrc_inverse ~ 0 on every DOF;
  4. finite differences reproduce analytic sin-trajectory derivatives;
  5. violent joint oscillation -> sustained torque violation -> FAIL verdict.

Pod run (mjeval venv, CPU only):
    FEAS_ORACLE_MJCF=/workspace/franco/nohope/agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/a3_pingpong/a3_pingpong.xml \
        /workspace/hope_mjeval_venv/bin/python -m pytest tests/test_feasibility_oracle.py -q
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import feasibility_oracle as fo  # noqa: E402

MJCF = os.environ.get("FEAS_ORACLE_MJCF") or str(REPO / fo.DEFAULT_MJCF)
HAVE_MUJOCO = fo.mujoco is not None
HAVE_MJCF = Path(MJCF).is_file()
needs_model = pytest.mark.skipif(
    not (HAVE_MUJOCO and HAVE_MJCF),
    reason=f"mujoco installed={HAVE_MUJOCO}, mjcf found={HAVE_MJCF} ({MJCF})",
)


# ---------------------------------------------------------------------------
# tier 1: pure geometry
# ---------------------------------------------------------------------------

def test_convex_hull_square():
    pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5], [0.2, 0.7]])
    hull = fo.convex_hull_2d(pts)
    assert len(hull) == 4
    assert {tuple(v) for v in hull} == {(0, 0), (1, 0), (1, 1), (0, 1)}


def test_signed_dist_inside_outside_corner():
    hull = fo.convex_hull_2d(np.array([[0, 0], [2, 0], [2, 2], [0, 2]]))
    assert fo.signed_dist_to_hull(np.array([1.0, 1.0]), hull) == pytest.approx(-1.0)
    assert fo.signed_dist_to_hull(np.array([3.0, 1.0]), hull) == pytest.approx(1.0)
    # outside past a corner: exact Euclidean distance, not edge-line distance
    assert fo.signed_dist_to_hull(np.array([3.0, 3.0]), hull) == pytest.approx(np.sqrt(2))
    assert fo.signed_dist_to_hull(np.array([1.0, 0.0]), hull) == pytest.approx(0.0, abs=1e-12)


def test_cop_recovers_application_point():
    rng = np.random.default_rng(7)
    for _ in range(20):
        ground_z = float(rng.uniform(-0.1, 0.1))
        p_c = np.array([*rng.uniform(-0.5, 0.5, 2), ground_z])  # true CoP
        f = np.array([*rng.uniform(-100, 100, 2), float(rng.uniform(50, 800))])
        nz = float(rng.uniform(-30, 30))  # free normal torque
        p_root = rng.uniform(-1, 1, 3) + np.array([0, 0, 1])
        tau_about_root = np.cross(p_c - p_root, f) + np.array([0.0, 0.0, nz])
        px, py = fo.cop_from_wrench(f, tau_about_root, p_root, ground_z)
        assert np.allclose([px, py], p_c[:2], atol=1e-9)


def test_runs_of():
    m = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 1], dtype=bool)
    assert fo.runs_of(m) == [(1, 2), (4, 4), (7, 9)]
    assert fo.runs_of(np.zeros(5, dtype=bool)) == []


def test_mini_yaml_phase(tmp_path):
    y = tmp_path / "ann.yaml"
    y.write_text(
        "clips:\n"
        "  hope_backhand_v5hLt_cal:   # comment\n"
        "    phase: 0.3714            # contact f26 of 71\n"
        "    status: verified\n"
    )
    ann = fo.load_annotations(str(y))
    assert fo.contact_frame_of("hope_backhand_v5hLt_cal", ann, 71) == 26
    assert fo.contact_frame_of("unknown_clip", ann, 71) is None


# ---------------------------------------------------------------------------
# tier 2: mujoco + vendor MJCF
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def om():
    return fo.load_oracle_model(MJCF)


def _keyframe_qpos(om):
    m = om.model
    assert m.nkey >= 1, "vendor MJCF should ship the 'stand' keyframe"
    return np.array(m.key_qpos[0])


def _fk_bodies(om, qpos, bodies):
    m = om.model
    d = fo.mujoco.MjData(m)
    d.qpos[:] = qpos
    fo.mujoco.mj_forward(m, d)
    pos, quat = [], []
    for b in bodies:
        bid = fo.mujoco.mj_name2id(m, fo.mujoco.mjtObj.mjOBJ_BODY, b)
        pos.append(np.array(d.xpos[bid]))
        quat.append(np.array(d.xquat[bid]))
    return np.array(pos), np.array(quat), d


def _synth_npz(tmp_path, om, qpos_traj, fps=50.0, name="synth.npz"):
    """Pack a qpos trajectory into the oracle's npz contract."""
    bodies = [fo.ROOT_BODY, *fo.FOOT_BODIES]
    T = qpos_traj.shape[0]
    bp = np.zeros((T, len(bodies), 3))
    bq = np.zeros((T, len(bodies), 4))
    for t in range(T):
        p, q, _ = _fk_bodies(om, qpos_traj[t], bodies)
        bp[t], bq[t] = p, q
    jp = qpos_traj[:, om.joint_qposadr]
    jv = np.vstack([np.zeros((1, 31)), np.diff(jp, axis=0) * fps])
    path = tmp_path / name
    np.savez(
        path,
        fps=np.array([fps], dtype=np.int64),
        joint_pos=jp.astype(np.float32),
        joint_vel=jv.astype(np.float32),
        body_pos_w=bp.astype(np.float32),
        body_quat_w=bq.astype(np.float32),
        body_names=np.array(bodies),
    )
    return path


@needs_model
def test_static_stand_gravity_compensation(om, tmp_path):
    qpos0 = _keyframe_qpos(om)
    traj = np.repeat(qpos0[None, :], 40, axis=0)
    npz = _synth_npz(tmp_path, om, traj)
    rep = fo.analyze_clip(om, npz, {}, None, mu=0.8, support_band=0.03)
    assert rep.verdict == "PASS", rep.checks
    # required vertical ground force == weight (gravity compensation)
    fz_mid = rep.fz[5:-5]
    assert np.allclose(fz_mid, rep.weight, rtol=0.01), (fz_mid.mean(), rep.weight)
    # CoP well inside the double-support polygon
    assert np.nanmax(rep.cop_excess[1:-1]) < -0.01
    # nearly no tangential demand
    assert np.nanmax(rep.fric_ratio[1:-1]) < 0.05
    # standing gravity compensation is far from any torque limit
    assert rep.checks["torque"].peak < 0.6
    # stored joint_vel (zeros) matches finite differences
    assert rep.fd_vs_stored_vel < 1e-3


@needs_model
def test_rotated_static_cop_equals_com_local_convention(om):
    """Free-joint rotational qfrc is BODY-LOCAL: CoP must land on the CoM
    projection when the root is yawed+shifted. Guards the frame convention."""
    m = om.model
    qpos = _keyframe_qpos(om)
    a = np.deg2rad(60.0)
    yaw = np.array([np.cos(a / 2), 0, 0, np.sin(a / 2)])
    q0 = qpos[3:7].copy()
    out = np.zeros(4)
    fo.mujoco.mju_mulQuat(out, yaw, q0)
    qpos[3:7] = out
    qpos[0] += 0.3
    qpos[1] -= 0.2
    d = fo.mujoco.MjData(m)
    d.qpos[:] = qpos
    d.qvel[:] = 0
    d.qacc[:] = 0
    fo.mujoco.mj_inverse(m, d)
    f = np.array(d.qfrc_inverse[0:3])
    tau_local = np.array(d.qfrc_inverse[3:6])
    R = np.zeros(9)
    fo.mujoco.mju_quat2Mat(R, qpos[3:7])
    tau_w = R.reshape(3, 3) @ tau_local
    px, py = fo.cop_from_wrench(f, tau_w, qpos[0:3], 0.0)
    fo.mujoco.mj_forward(m, d)
    rid = fo.mujoco.mj_name2id(m, fo.mujoco.mjtObj.mjOBJ_BODY, fo.ROOT_BODY)
    com = np.array(d.subtree_com[rid])
    assert np.hypot(px - com[0], py - com[1]) < 1e-3
    assert f[2] == pytest.approx(om.weight, rel=1e-6)


@needs_model
def test_free_fall_zero_residual(om):
    m = om.model
    d = fo.mujoco.MjData(m)
    d.qpos[:] = _keyframe_qpos(om)
    d.qpos[2] += 5.0
    d.qvel[:] = 0
    d.qacc[:] = 0
    d.qacc[0:3] = m.opt.gravity  # uniform free fall, configuration held rigid
    fo.mujoco.mj_inverse(m, d)
    assert np.max(np.abs(d.qfrc_inverse)) < 1e-6


@needs_model
def test_finite_difference_matches_analytic(om):
    fps, T = 50.0, 60
    tgrid = np.arange(T) / fps
    A, w = 0.3, 2 * np.pi * 1.5
    j = 5  # one Isaac joint column
    qpos = np.repeat(_keyframe_qpos(om)[None, :], T, axis=0)
    col = om.joint_qposadr[j]
    base = qpos[0, col]
    qpos[:, col] = base + A * np.sin(w * tgrid)
    # root: constant linear velocity + constant yaw rate
    vx, wz = 0.4, 0.7
    qpos[:, 0] = qpos[0, 0] + vx * tgrid
    for t in range(T):
        yaw = np.array([np.cos(wz * tgrid[t] / 2), 0, 0, np.sin(wz * tgrid[t] / 2)])
        out = np.zeros(4)
        fo.mujoco.mju_mulQuat(out, yaw, qpos[0, 3:7])
        qpos[t, 3:7] = out
    qvel, qacc = fo.differentiate(om, qpos, 1.0 / fps)
    dof = om.joint_dofadr[j]
    mid = slice(2, T - 2)
    assert np.allclose(qvel[mid, dof], A * w * np.cos(w * tgrid[mid]), atol=2e-2)
    assert np.allclose(qacc[mid, dof], -A * w * w * np.sin(w * tgrid[mid]), atol=0.5)
    assert np.allclose(qvel[mid, 0], vx, atol=1e-6)
    # yaw about world z: local z == world z for a yaw-only rotation of q0~identity-ish
    assert np.allclose(np.abs(qvel[mid, 5]), wz, atol=0.02)
    assert np.max(np.abs(qacc[mid, 3:6])) < 0.5


@needs_model
def test_violent_oscillation_fails_torque(om, tmp_path):
    fps, T = 50.0, 50
    tgrid = np.arange(T) / fps
    qpos0 = _keyframe_qpos(om)
    traj = np.repeat(qpos0[None, :], T, axis=0)
    j = fo.ISAAC_JOINT_NAMES.index("right_shoulder_pitch_joint")
    col = om.joint_qposadr[j]
    traj[:, col] = traj[0, col] + 0.5 * np.sin(2 * np.pi * 8.0 * tgrid)  # ~1263 rad/s^2
    npz = _synth_npz(tmp_path, om, traj, name="violent.npz")
    rep = fo.analyze_clip(om, npz, {}, None, mu=0.8, support_band=0.03)
    assert rep.verdict == "FAIL"
    tq = rep.checks["torque"]
    assert tq.sustained and tq.peak > 1.0
    assert rep.doses["torque"] >= fo.DOSE_TAU_FAIL
    assert any(r["joint"] == "right_shoulder_pitch_joint" and r["peak_util"] > 1.0 for r in rep.top_joints)


@needs_model
def test_body_order_is_required(om, tmp_path):
    qpos0 = _keyframe_qpos(om)
    npz = _synth_npz(tmp_path, om, np.repeat(qpos0[None, :], 10, axis=0), name="noorder.npz")
    # strip embedded body_names -> resolution must fail loudly, not guess
    d = dict(np.load(npz, allow_pickle=True))
    d.pop("body_names")
    bad = tmp_path / "noorder2.npz"
    np.savez(bad, **d)
    with pytest.raises(ValueError, match="body order"):
        fo.analyze_clip(om, bad, {}, None, mu=0.8, support_band=0.03)
