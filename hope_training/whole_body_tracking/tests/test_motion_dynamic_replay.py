"""Tests for scripts/motion_dynamic_replay.py (L2 motion-only dynamic replay).

Two tiers, mirroring tests/test_audit_self_collision.py:
  * pure tiers (numpy only)  — contract table, npz loader fail-closed, timeline,
    support-polygon geometry, JSON sanitizer; run anywhere;
  * mujoco tiers             — a PROGRAMMATIC minimal free-base two-link MJCF
    (named to satisfy MujocoRobot's pelvis/torso/feet/racket contract) plus a
    synthetic sine reference; skipped where mujoco is absent.

The contract encoded here:
  1. the Zhiyuan 2026-07-31 A3 nominal table is complete for all 29 body DoFs,
     with the two head DoFs explicitly pinned to the named HOPE fallback;
  2. the loader is fail-closed: missing keys, wrong joint width, NaN all raise;
  3. the timeline appends the hold_after / hold_between segments exactly and the
     --pair timeline is one continuous no-reset sequence;
  4. a smoke replay on the minimal model is PASS with mj_step_calls == ticks *
     substeps, strictly increasing sim time, all required JSON fields finite;
  5. an extreme reference (violent leg swing) triggers the fall verdict;
  6. NaN injection (poisoned npz) exits fail-closed with code 2.

Real-MJCF validation is a pod-side concern, deliberately not tested here.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

WBT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WBT / "scripts"))

import motion_dynamic_replay as mdr  # noqa: E402

HAVE_MUJOCO = mdr.mujoco is not None
needs_mujoco = pytest.mark.skipif(not HAVE_MUJOCO, reason="mujoco not installed")


# ---------------------------------------------------------------------------
# minimal free-base two-link model + synthetic clips
# ---------------------------------------------------------------------------
# Body/site/actuator names satisfy mujoco_eval_onnx.MujocoRobot's hard contract:
# pelvis_link freejoint at qpos 0, torso_Link anchor, right_racket site, the two
# *_ankle_roll_Link feet, and one "<joint>_motor" actuator per contract joint.
MINI_MJCF = """
<mujoco model="mini_two_link">
  <compiler angle="radian"/>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="pelvis_link" pos="0 0 0.22">
      <freejoint name="root"/>
      <geom name="pelvis_geom" type="sphere" size="0.08" mass="4"/>
      <body name="torso_Link" pos="0 0 0.12">
        <geom name="torso_geom" type="sphere" size="0.04" mass="1"/>
        <site name="right_racket" pos="0 0 0.05"/>
      </body>
      <body name="left_ankle_roll_Link" pos="0 0.1 -0.2">
        <joint name="left_leg_joint" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
        <geom name="left_foot" type="box" size="0.06 0.03 0.02" mass="1"/>
      </body>
      <body name="right_ankle_roll_Link" pos="0 -0.1 -0.2">
        <joint name="right_leg_joint" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
        <geom name="right_foot" type="box" size="0.06 0.03 0.02" mass="1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="left_leg_joint_motor" joint="left_leg_joint" ctrlrange="-60 60"/>
    <motor name="right_leg_joint_motor" joint="right_leg_joint" ctrlrange="-60 60"/>
  </actuator>
</mujoco>
"""

MINI_CONTRACT = {
    "name": "mini_two_link",
    "joint_names": ["left_leg_joint", "right_leg_joint"],
    "body_names": ["pelvis_link", "torso_Link"],
    "kp": [60.0, 60.0],
    "kd": [2.0, 2.0],
    "effort_limits": [60.0, 60.0],
    "velocity_limits": [20.0, 20.0],
}
ROOT_Z0 = 0.22
FPS = 50


def _write_mini_assets(tmp_path: Path):
    mjcf = tmp_path / "mini.xml"
    mjcf.write_text(MINI_MJCF)
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps(MINI_CONTRACT))
    return mjcf, contract


def _write_clip(tmp_path: Path, joint_pos: np.ndarray, name: str = "clip.npz",
                fps: int = FPS, root_z: float = ROOT_Z0) -> Path:
    T, J = joint_pos.shape
    body_pos = np.zeros((T, 1, 3))
    body_pos[:, 0, 2] = root_z
    body_quat = np.zeros((T, 1, 4))
    body_quat[:, 0, 0] = 1.0
    path = tmp_path / name
    np.savez(path, joint_pos=joint_pos, joint_vel=np.zeros_like(joint_pos),
             fps=np.array([fps]), body_pos_w=body_pos, body_quat_w=body_quat,
             body_names=np.array(["pelvis_link"], dtype=object))
    return path


def _sine_clip(frames: int = 50, amp: float = 0.03, hz: float = 1.0) -> np.ndarray:
    t = np.arange(frames) / FPS
    q = amp * np.sin(2.0 * math.pi * hz * t)
    return np.stack([q, q], axis=1)


def _extreme_clip(frames: int = 100) -> np.ndarray:
    """Violent alternating +-1.4 rad square wave: guaranteed to topple the mini."""
    q = np.zeros((frames, 2))
    q[10:] = 1.4
    q[40:] = -1.4
    q[70:] = 1.4
    return q


def _run_cli(tmp_path: Path, clip_paths, out_name: str, extra=()):
    mjcf, contract = _write_mini_assets(tmp_path)
    out = tmp_path / out_name
    argv = []
    if len(clip_paths) == 1:
        argv += ["--motion", str(clip_paths[0])]
    else:
        argv += ["--pair", str(clip_paths[0]), str(clip_paths[1])]
    argv += ["--mjcf", str(mjcf), "--contract-json", str(contract),
             "--out", str(out),
             # mini-model fall thresholds: nominal root_z is 0.22 m
             "--fall-root-z-min", "0.15", "--fall-tilt-rad", "0.5"]
    argv += list(extra)
    code = mdr.main(argv)
    payload = json.loads(out.read_text()) if out.is_file() else None
    return code, payload


# ---------------------------------------------------------------------------
# pure tier: vendor contract table
# ---------------------------------------------------------------------------

EXPECTED_PLANT = {
    # base joint: (vendor Kp, vendor Kd, vendor effort, retained HOPE velocity, vendor armature)
    "hip_yaw_joint": (80.0, 3.0, 220.0, 12.0, 0.066472),
    "hip_roll_joint": (120.0, 4.0, 220.0, 12.0, 0.066472),
    "hip_pitch_joint": (80.0, 3.0, 220.0, 12.0, 0.066472),
    "knee_joint": (250.0, 8.0, 320.0, 14.6, 0.120340),
    "ankle_pitch_joint": (50.0, 2.0, 118.2, 10.8, 0.064449),
    "ankle_roll_joint": (50.0, 2.0, 54.75, 19.3, 0.020129),
    "waist_yaw_joint": (85.0, 3.0, 220.0, 12.0, 0.066472),
    "waist_roll_joint": (50.0, 2.0, 46.0, 22.7, 0.014623),
    "waist_pitch_joint": (50.0, 2.0, 118.0, 9.2, 0.088220),
    "shoulder_pitch_joint": (40.0, 3.0, 60.0, 13.6, 0.012085),
    "shoulder_roll_joint": (40.0, 3.0, 60.0, 13.6, 0.012085),
    "shoulder_yaw_joint": (30.0, 2.0, 24.0, 15.7, 0.004968),
    "elbow_joint": (30.0, 2.0, 24.0, 15.7, 0.004968),
    "wrist_roll_joint": (30.0, 2.0, 24.0, 15.7, 0.004968),
    "wrist_pitch_joint": (20.0, 2.0, 6.0, 12.7, 0.0008100893338),
    "wrist_yaw_joint": (20.0, 2.0, 6.0, 12.7, 0.0008100893338),
}

HEAD_HOPE_FALLBACK = {
    "head_yaw_joint": (40.0, 2.0, 6.0, 12.7, 0.0008100893338),
    "head_pitch_joint": (40.0, 2.0, 6.0, 12.7, 0.0008100893338),
}


def _base_joint_name(joint: str) -> str:
    for prefix in ("left_", "right_"):
        if joint.startswith(prefix):
            return joint[len(prefix):]
    return joint


def test_a3_contract_covers_all_31_joints():
    c = mdr.a3_contract()
    assert c.name == "agibot_a3_zhiyuan_20260731_nominal"
    assert len(c.joint_names) == 31
    assert list(c.joint_names) == list(mdr.ISAAC_JOINT_NAMES)
    assert c.armature is not None
    for arr in (c.kp, c.kd, c.effort_limits, c.velocity_limits, c.armature):
        assert arr.shape == (31,)
        assert np.isfinite(arr).all()
        assert (arr > 0).all()


def test_a3_contract_matches_every_vendor_family_and_named_head_fallback():
    c = mdr.a3_contract()
    idx = {n: i for i, n in enumerate(c.joint_names)}
    assert c.armature is not None

    expected_families = set(EXPECTED_PLANT) | set(HEAD_HOPE_FALLBACK)
    assert {_base_joint_name(name) for name in c.joint_names} == expected_families
    for joint in c.joint_names:
        expected = (EXPECTED_PLANT | HEAD_HOPE_FALLBACK)[_base_joint_name(joint)]
        i = idx[joint]
        actual = (
            c.kp[i], c.kd[i], c.effort_limits[i], c.velocity_limits[i], c.armature[i],
        )
        assert actual == expected, joint

    # Both sides of every wrist axis are covered explicitly.
    for side in ("left", "right"):
        roll_i = idx[f"{side}_wrist_roll_joint"]
        assert (c.kp[roll_i], c.kd[roll_i], c.effort_limits[roll_i], c.armature[roll_i]) == (
            30.0, 2.0, 24.0, 0.004968,
        )
        for axis in ("pitch", "yaw"):
            i = idx[f"{side}_wrist_{axis}_joint"]
            assert (c.kp[i], c.kd[i], c.effort_limits[i], c.armature[i]) == (
                20.0, 2.0, 6.0, 0.0008100893338,
            )


def test_a3_contract_uses_new_rounded_nonconflict_armatures_and_sku_fallbacks():
    c = mdr.a3_contract()
    idx = {n: i for i, n in enumerate(c.joint_names)}
    assert c.kp[idx["waist_yaw_joint"]] == 85.0
    assert c.effort_limits[idx["waist_pitch_joint"]] == 118.0
    assert c.armature[idx["left_hip_pitch_joint"]] != 0.06646569891
    assert c.armature[idx["right_ankle_pitch_joint"]] != 0.06444060531
    for side in ("left", "right"):
        for axis in ("pitch", "yaw"):
            i = idx[f"{side}_wrist_{axis}_joint"]
            assert c.kp[i] == 20.0
            assert c.effort_limits[i] == 6.0
            assert c.armature[i] == 0.0008100893338


def test_a3_contract_implies_vendor_action_scale_for_every_joint():
    c = mdr.a3_contract()
    idx = {n: i for i, n in enumerate(c.joint_names)}

    def action_scale(joint: str) -> float:
        joint_idx = idx[joint]
        return 0.25 * c.effort_limits[joint_idx] / c.kp[joint_idx]

    for joint in c.joint_names:
        expected = (EXPECTED_PLANT | HEAD_HOPE_FALLBACK)[_base_joint_name(joint)]
        assert action_scale(joint) == pytest.approx(0.25 * expected[2] / expected[0])

    assert action_scale("waist_yaw_joint") == pytest.approx(0.25 * 220.0 / 85.0)
    assert action_scale("waist_pitch_joint") == pytest.approx(0.59)
    for side in ("left", "right"):
        assert action_scale(f"{side}_wrist_roll_joint") == pytest.approx(0.2)
        for axis in ("pitch", "yaw"):
            assert action_scale(f"{side}_wrist_{axis}_joint") == pytest.approx(0.075)


def test_build_robot_binds_vendor_contract_armature(monkeypatch):
    contract = mdr.a3_contract()
    captured = {}

    class FakeRobot:
        def __init__(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

    monkeypatch.setattr(mdr, "MujocoRobot", FakeRobot)
    robot = mdr.build_robot("unused.xml", contract, 0.005)

    assert isinstance(robot, FakeRobot)
    assert captured["kwargs"]["joint_armature"] is contract.armature


def test_run_replay_applies_vendor_kp_kd_arrays_to_pd(monkeypatch):
    contract = mdr.a3_contract()
    captured = {}

    class StopAfterFirstPd(RuntimeError):
        pass

    class FakeRobot:
        feet_bid = ()
        model = SimpleNamespace(opt=SimpleNamespace(timestep=0.005))
        data = SimpleNamespace(time=0.0)

        def apply_pd_and_step(self, q_des, kp, kd, substeps):
            captured.update(q_des=q_des, kp=kp, kd=kd, substeps=substeps)
            raise StopAfterFirstPd

    timeline = SimpleNamespace(
        fps=50,
        q_des=np.zeros((1, len(contract.joint_names))),
        segments=[{"start_tick": 0, "end_tick": 1}],
    )
    monkeypatch.setattr(mdr, "mujoco", object())
    with pytest.raises(StopAfterFirstPd):
        mdr.run_replay(
            FakeRobot(), timeline, substeps=4,
            contract=contract, thresholds=mdr.Thresholds(),
        )

    assert captured["kp"] is contract.kp
    assert captured["kd"] is contract.kd
    assert captured["substeps"] == 4
    np.testing.assert_allclose(
        captured["kp"][[contract.joint_names.index("waist_yaw_joint"),
                        contract.joint_names.index("left_wrist_pitch_joint")]],
        [85.0, 20.0],
    )
    np.testing.assert_allclose(
        captured["kd"][[contract.joint_names.index("waist_yaw_joint"),
                        contract.joint_names.index("right_wrist_yaw_joint")]],
        [3.0, 2.0],
    )


def test_contract_json_roundtrip(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps(MINI_CONTRACT))
    c = mdr.contract_from_json(str(path))
    assert c.name == "mini_two_link"
    assert c.joint_names == ("left_leg_joint", "right_leg_joint")
    np.testing.assert_allclose(c.kp, [60.0, 60.0])
    assert c.armature is None


def test_contract_json_optional_armature_roundtrip_and_shape_guard(tmp_path):
    raw = dict(MINI_CONTRACT, armature=[0.01, 0.02])
    path = tmp_path / "with_armature.json"
    path.write_text(json.dumps(raw))
    np.testing.assert_allclose(mdr.contract_from_json(str(path)).armature, [0.01, 0.02])

    path.write_text(json.dumps(dict(raw, armature=[0.01])))
    with pytest.raises(ValueError, match="armature"):
        mdr.contract_from_json(str(path))


def test_contract_json_bad_shapes_fail(tmp_path):
    bad = dict(MINI_CONTRACT)
    bad["kp"] = [60.0]  # one value for two joints
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="kp"):
        mdr.contract_from_json(str(path))


# ---------------------------------------------------------------------------
# pure tier: fail-closed loader
# ---------------------------------------------------------------------------

def test_load_motion_rejects_missing_key(tmp_path):
    path = tmp_path / "nokeys.npz"
    np.savez(path, joint_pos=np.zeros((5, 2)), fps=np.array([50]))
    with pytest.raises(ValueError, match="body_pos_w"):
        mdr.load_motion(str(path), 2)


def test_load_motion_rejects_wrong_joint_width(tmp_path):
    clip = _write_clip(tmp_path, np.zeros((5, 2)))
    with pytest.raises(ValueError, match="contract expects 31"):
        mdr.load_motion(str(clip), 31)


def test_load_motion_rejects_nan(tmp_path):
    q = np.zeros((5, 2))
    q[3, 1] = np.nan
    clip = _write_clip(tmp_path, q, name="nan.npz")
    with pytest.raises(ValueError, match="fail-closed"):
        mdr.load_motion(str(clip), 2)


def test_load_motion_rejects_wrong_root_body(tmp_path):
    path = tmp_path / "wrongroot.npz"
    T = 5
    np.savez(path, joint_pos=np.zeros((T, 2)), joint_vel=np.zeros((T, 2)),
             fps=np.array([50]), body_pos_w=np.zeros((T, 1, 3)),
             body_quat_w=np.tile([1.0, 0, 0, 0], (T, 1, 1)),
             body_names=np.array(["torso_Link"], dtype=object))
    with pytest.raises(ValueError, match="pelvis_link"):
        mdr.load_motion(str(path), 2)


# ---------------------------------------------------------------------------
# pure tier: timeline
# ---------------------------------------------------------------------------

def _mini_clip_obj(tmp_path, frames=50, name="c.npz"):
    return mdr.load_motion(str(_write_clip(tmp_path, _sine_clip(frames), name=name)), 2)


def test_timeline_single_clip_hold_ticks(tmp_path):
    clip = _mini_clip_obj(tmp_path)
    tl = mdr.build_timeline([clip], hold_after_s=1.0, hold_between_s=1.0)
    assert tl.q_des.shape == (50 + 50, 2)          # 50 frames + 1.0 s * 50 Hz hold
    assert tl.is_clip_tick[:50].all()
    assert not tl.is_clip_tick[50:].any()
    assert [s["kind"] for s in tl.segments] == ["clip", "hold"]
    # hold holds the LAST clip frame
    np.testing.assert_allclose(tl.q_des[50:], np.tile(clip.joint_pos[-1], (50, 1)))


def test_timeline_pair_is_one_no_reset_sequence(tmp_path):
    a = _mini_clip_obj(tmp_path, frames=50, name="a.npz")
    b = _mini_clip_obj(tmp_path, frames=40, name="b.npz")
    tl = mdr.build_timeline([a, b], hold_after_s=1.0, hold_between_s=0.5)
    assert tl.q_des.shape[0] == 50 + 25 + 40 + 50
    kinds = [s["kind"] for s in tl.segments]
    assert kinds == ["clip", "hold", "clip", "hold"]
    # contiguous tick ranges — one continuous sequence, no gaps for resets
    for prev, cur in zip(tl.segments, tl.segments[1:]):
        assert prev["end_tick"] == cur["start_tick"]
    assert tl.segments[-1]["end_tick"] == tl.q_des.shape[0]


def test_timeline_rejects_fps_mismatch(tmp_path):
    a = _mini_clip_obj(tmp_path, name="a.npz")
    b = mdr.load_motion(
        str(_write_clip(tmp_path, _sine_clip(30), name="b.npz", fps=30)), 2)
    with pytest.raises(ValueError, match="fps"):
        mdr.build_timeline([a, b], hold_after_s=1.0, hold_between_s=1.0)


# ---------------------------------------------------------------------------
# pure tier: support polygon + sanitizer
# ---------------------------------------------------------------------------

def test_com_support_margin_inside_and_outside():
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    inside = mdr.com_support_margin(np.array([0.5, 0.5]), square)
    assert inside == pytest.approx(0.5)
    outside = mdr.com_support_margin(np.array([2.0, 0.5]), square)
    assert outside == pytest.approx(-1.0)
    # degenerate supports can never contain the CoM -> negative
    two_pts = np.array([[0.0, 0.0], [1.0, 0.0]])
    assert mdr.com_support_margin(np.array([0.5, 0.2]), two_pts) == pytest.approx(-0.2)
    assert mdr.com_support_margin(np.array([0.5, 0.5]), np.empty((0, 2))) is None


def test_sanitize_json_flags_nonfinite():
    nonfinite = []
    out = mdr.sanitize_json(
        {"a": 1.0, "b": float("nan"), "c": [np.inf, np.float64(2.0)],
         "d": np.array([1.0, 2.0]), "e": np.bool_(True)},
        nonfinite)
    assert out["a"] == 1.0 and out["b"] is None
    assert out["c"] == [None, 2.0]
    assert out["d"] == [1.0, 2.0] and out["e"] is True
    assert len(nonfinite) == 2  # b and c[0], recorded with key paths
    assert any("b" in k for k in nonfinite)


# ---------------------------------------------------------------------------
# mujoco tier: smoke replay on the minimal model
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("smoke")
    clip = _write_clip(tmp_path, _sine_clip(50))
    code, payload = _run_cli(tmp_path, [clip], "out.json",
                             extra=["--hold-after-s", "1.0"])
    return code, payload


@needs_mujoco
def test_smoke_verdict_pass(smoke):
    code, payload = smoke
    assert payload is not None
    assert payload["verdict"]["overall"] == "PASS", payload["verdict"]["gates"]
    assert code == 0
    assert payload["reset_count"] == 1


@needs_mujoco
def test_smoke_mj_step_calls_and_monotonic_time(smoke):
    _, payload = smoke
    ticks = 50 + 50  # clip + 1.0 s hold at 50 Hz
    assert payload["planned_ticks"] == ticks
    assert payload["executed_ticks"] == ticks
    assert payload["mj_step_calls"] == ticks * 4          # default substeps
    assert payload["expected_mj_step_calls"] == ticks * 4
    assert payload["sim_time_monotonic"] is True
    assert payload["sim_time_s"] == pytest.approx(
        payload["mj_step_calls"] * payload["sim_dt_s"], rel=1e-9)
    assert payload["sim_dt_s"] == pytest.approx(1.0 / (50 * 4))


REQUIRED_SCALARS = [
    ("mj_step_calls",), ("sim_time_s",), ("root_z_min_m",),
    ("tilt_rad", "p50"), ("tilt_rad", "p95"), ("tilt_rad", "p99"), ("tilt_rad", "max"),
    ("feet", "left", "contact_frac"), ("feet", "left", "normal_force_mean_n"),
    ("feet", "left", "longest_airtime_s"), ("feet", "left", "slip_cum_m"),
    ("feet", "left", "support_speed_peak_mps"),
    ("feet", "right", "contact_frac"), ("feet", "right", "normal_force_mean_n"),
    ("feet", "right", "longest_airtime_s"), ("feet", "right", "slip_cum_m"),
    ("stance_min_foot_dist_m",), ("com_support_margin_min_m",),
    ("legs_crossed", "min_separation_m",),
    ("saturation", "q_soft_limit_samples"), ("saturation", "qdot_limit_hits"),
    ("saturation", "torque_limit_hits"),
    ("tracking_err_rad", "mean"), ("tracking_err_rad", "p95"), ("tracking_err_rad", "max"),
    ("recover", "ready_err_max_rad"), ("recover", "ready_err_mean_rad"),
    ("recover", "root_lin_speed_mps"), ("recover", "root_ang_speed_radps"),
]


@needs_mujoco
def test_smoke_json_required_fields_present_and_finite(smoke):
    _, payload = smoke
    for path in REQUIRED_SCALARS:
        node = payload
        for key in path:
            assert key in node, f"missing {'.'.join(path)}"
            node = node[key]
        assert node is not None and math.isfinite(float(node)), f"non-finite {'.'.join(path)}"
    assert isinstance(payload["fall"]["fell"], bool)
    assert isinstance(payload["legs_crossed"]["crossed"], bool)
    assert payload["mjcf"]["sha256"] and len(payload["mjcf"]["sha256"]) == 64
    assert "does NOT replace policy replay" in payload["disclaimer"]


@needs_mujoco
def test_smoke_physical_plausibility(smoke):
    _, payload = smoke
    assert payload["fall"]["fell"] is False
    assert payload["root_z_min_m"] > 0.15
    for side in ("left", "right"):
        foot = payload["feet"][side]
        assert foot["contact_frac"] > 0.9
        assert foot["normal_force_mean_n"] > 1.0        # actually bearing weight
        assert foot["slip_cum_m"] < 0.05
    assert payload["tracking_err_rad"]["p95"] < 0.2
    # ±0.03 rad sine sits far inside the ±1.35 rad soft limits
    assert payload["saturation"]["q_soft_limit_samples"] == 0
    assert payload["legs_crossed"]["crossed"] is False
    assert payload["legs_crossed"]["min_separation_m"] > 0.05
    assert payload["com_support_margin_min_m"] > 0.0
    assert payload["stance_min_foot_dist_m"] > 0.1


@needs_mujoco
def test_smoke_gate_structure(smoke):
    _, payload = smoke
    gates = payload["verdict"]["gates"]
    names = [g["name"] for g in gates]
    assert names == ["finite", "timeline_complete", "mj_step_accounting", "no_fall",
                     "foot_contact", "foot_slip", "no_crossed_legs",
                     "com_support_margin", "tracking", "recover_ready"]
    for g in gates:
        assert isinstance(g["ok"], bool) and isinstance(g["reason"], str) and g["reason"]


# ---------------------------------------------------------------------------
# mujoco tier: fall, NaN fail-closed, pair mode
# ---------------------------------------------------------------------------

@needs_mujoco
def test_extreme_reference_triggers_fall(tmp_path):
    clip = _write_clip(tmp_path, _extreme_clip(100), name="extreme.npz")
    code, payload = _run_cli(tmp_path, [clip], "out.json")
    assert code == 2
    assert payload["fall"]["fell"] is True
    assert payload["fall"]["first_tick"] is not None
    assert payload["fall"]["reason"]
    assert payload["verdict"]["overall"] == "FAIL"
    no_fall = next(g for g in payload["verdict"]["gates"] if g["name"] == "no_fall")
    assert no_fall["ok"] is False and "fell" in no_fall["reason"]
    # the run stops at the fall -> fewer executed ticks, accounting still exact
    assert payload["executed_ticks"] <= payload["planned_ticks"]
    assert payload["mj_step_calls"] == payload["executed_ticks"] * 4


@needs_mujoco
def test_extreme_reference_counts_torque_saturation(tmp_path):
    clip = _write_clip(tmp_path, _extreme_clip(100), name="extreme.npz")
    _, payload = _run_cli(tmp_path, [clip], "out.json")
    # kp*1.4 = 84 Nm > 60 Nm limit: the total-PD clip must have engaged
    assert payload["saturation"]["torque_limit_hits"] > 0
    assert payload["saturation"]["torque_limit_peak_ratio"] > 1.0


@needs_mujoco
def test_nan_npz_is_fail_closed(tmp_path, capsys):
    q = _sine_clip(50)
    q[20, 0] = np.nan
    clip = _write_clip(tmp_path, q, name="poison.npz")
    code, payload = _run_cli(tmp_path, [clip], "out.json")
    assert code == 2
    assert payload is None          # refused before any sim step, no JSON written
    assert "fail-closed" in capsys.readouterr().err


@needs_mujoco
def test_pair_mode_no_reset_continuous(tmp_path):
    a = _write_clip(tmp_path, _sine_clip(50), name="fh.npz")
    b = _write_clip(tmp_path, _sine_clip(40), name="bh.npz")
    code, payload = _run_cli(
        tmp_path, [a, b], "pair.json",
        extra=["--hold-after-s", "1.0", "--hold-between-s", "0.5"])
    assert code == 0, payload["verdict"]["gates"]
    assert payload["reset_count"] == 1
    ticks = 50 + 25 + 40 + 50
    assert payload["executed_ticks"] == ticks
    assert payload["mj_step_calls"] == ticks * 4
    kinds = [s["kind"] for s in payload["segments"]]
    assert kinds == ["clip", "hold", "clip", "hold"]
    assert all(s["executed"] for s in payload["segments"])
    # per-clip tracking stats exist for both strokes
    clip_segs = [s for s in payload["segments"] if s["kind"] == "clip"]
    assert len(clip_segs) == 2
    for s in clip_segs:
        assert s["track_err_rad"]["p95"] is not None
