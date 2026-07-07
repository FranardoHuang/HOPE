"""Unit tests for scripts/audit_motion_npz.py (the L0 static motion-feasibility audit).

Pure CPU, NO Isaac/torch imports — the module under test is loaded directly by file
path (same pattern as test_stage1_wiring.py). All motion data is synthetic: a toy
3-joint URDF (limits [-1, 1] rad, 5 rad/s) plus hand-built npz arrays, so every
expected frame index / severity below is computable by hand.

Covered cases (task spec 2026-07-08):
  * clean clip                        -> PASS, no repair suggestion
  * head overspeed cluster            -> FAIL + "trim first K frames" suggestion
  * mid-clip overspeed                -> FAIL + "slow-play <= X" (X = limit/peak)
  * first-frame ghost velocity        -> FAIL; annotations first_frame_exempt -> WARN
  * position pinned at the hard limit -> WARN (saturation, retarget-clamp signature)
  * mid-clip position beyond limits   -> FAIL + reject/regenerate (trim/slow can't fix)
  * URDF parsing + trim slice command details (phase-shift reminder)

Run:  python3 -m pytest hope_training/whole_body_tracking/tests/test_audit_motion_npz.py -q
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

# --- load the module under test by path (scripts/ is not a package) -----------
_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_motion_npz.py"
_spec = importlib.util.spec_from_file_location("audit_motion_npz", _SCRIPT)
audit = importlib.util.module_from_spec(_spec)
sys.modules["audit_motion_npz"] = audit
_spec.loader.exec_module(audit)

FPS = 50
JOINTS = ["j0_joint", "j1_joint", "j2_joint"]
JOINT_SPEC = ",".join(JOINTS)  # --joint-names style comma list (3-dof toy robot)

TOY_URDF = """<?xml version="1.0"?>
<robot name="toy">
  <joint name="j0_joint" type="revolute">
    <limit lower="-1.0" upper="1.0" effort="10" velocity="5.0"/>
  </joint>
  <joint name="j1_joint" type="revolute">
    <limit lower="-1.0" upper="1.0" effort="10" velocity="5.0"/>
  </joint>
  <joint name="j2_joint" type="revolute">
    <limit lower="-1.0" upper="1.0" effort="10" velocity="5.0"/>
  </joint>
  <joint name="welded" type="fixed"/>
</robot>
"""


@pytest.fixture()
def toy_urdf(tmp_path):
    p = tmp_path / "toy.urdf"
    p.write_text(TOY_URDF)
    return str(p)


@pytest.fixture()
def limits(toy_urdf):
    return audit.parse_urdf_limits(toy_urdf)


def _write_npz(path, q, dq, fps=FPS, base_lin=None):
    q = np.asarray(q, dtype=np.float32)
    dq = np.asarray(dq, dtype=np.float32)
    t, j = q.shape
    body_lin = np.zeros((t, 1, 3), dtype=np.float32)
    if base_lin is not None:
        body_lin[:, 0, 0] = np.asarray(base_lin, dtype=np.float32)
    np.savez(
        path,
        fps=np.array([fps]),
        joint_pos=q,
        joint_vel=dq,
        body_pos_w=np.zeros((t, 1, 3), dtype=np.float32),
        body_quat_w=np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (t, 1, 1)),
        body_lin_vel_w=body_lin,
        body_ang_vel_w=np.zeros((t, 1, 3), dtype=np.float32),
    )
    return str(path)


def _findings(rep, check, level=None):
    return [
        f for f in rep.findings
        if f.check == check and (level is None or f.level == level)
    ]


# ------------------------------------------------------------------ URDF parse
def test_parse_urdf_limits(limits):
    assert set(limits) == set(JOINTS)  # the fixed joint is skipped
    j0 = limits["j0_joint"]
    assert (j0.lower, j0.upper, j0.velocity) == (-1.0, 1.0, 5.0)


# ------------------------------------------------------------------ 1. clean
def test_clean_clip_passes(tmp_path, limits):
    T = 30
    q = np.full((T, 3), 0.1)
    dq = np.zeros((T, 3))
    f = _write_npz(tmp_path / "clean.npz", q, dq)
    rep = audit.audit_clip(f, limits, joint_names=JOINT_SPEC)
    assert rep.verdict == "PASS"
    assert not [x for x in rep.findings if x.level != "PASS"]
    assert rep.suggestion.kind == "none"


# ---------------------------------------------------- 2. head overspeed -> trim
def test_head_overspeed_suggests_trim(tmp_path, limits):
    # joint0 ramps 0.2 rad/frame over frames 0-3 => FD 10 rad/s > limit 5;
    # stored joint_vel mirrors it on frames 0-2. All FAILs live in frames 0-3;
    # frame 4 is healthy (dq=0, base still) -> expect trim_head K=4.
    T = 30
    q = np.zeros((T, 3))
    q[0:4, 0] = [0.0, 0.2, 0.4, 0.6]
    q[4:, 0] = 0.6
    dq = np.zeros((T, 3))
    dq[0:3, 0] = 10.0
    f = _write_npz(tmp_path / "head.npz", q, dq)
    rep = audit.audit_clip(f, limits, joint_names=JOINT_SPEC)

    assert rep.verdict == "FAIL"
    assert _findings(rep, "vel_limit_stored", "FAIL")
    assert _findings(rep, "vel_limit_fd", "FAIL")
    assert _findings(rep, "first_frame_vel", "FAIL")  # 10 rad/s ghost, no exemption
    assert rep.suggestion.kind == "trim_head"
    assert rep.suggestion.trim_head == 4
    joined = "\n".join(rep.suggestion.lines)
    assert "d[k][4:]" in joined            # equivalent python slice command
    assert "phase" in joined               # registry phase-shift reminder


# ------------------------------------------------- 3. mid overspeed -> slow-play
def test_mid_overspeed_suggests_slowdown(tmp_path, limits):
    # a single mid-clip stored-velocity spike: 6 rad/s vs limit 5 -> X = 5/6
    T = 30
    q = np.full((T, 3), 0.1)
    dq = np.zeros((T, 3))
    dq[15, 0] = 6.0
    f = _write_npz(tmp_path / "mid.npz", q, dq)
    rep = audit.audit_clip(f, limits, joint_names=JOINT_SPEC)

    assert rep.verdict == "FAIL"
    fails = _findings(rep, "vel_limit_stored", "FAIL")
    assert fails and fails[0].frames == [15]
    assert rep.suggestion.kind == "slow_down"
    assert rep.suggestion.slow_factor == pytest.approx(5.0 / 6.0)


# ------------------------------------- 4. first-frame ghost: FAIL vs exemption
def test_first_frame_ghost_fail_and_exemption(tmp_path, limits):
    # frame-0 stored velocity 3 rad/s: legal vs the 5 rad/s URDF limit, but a
    # cold-start ghost per the first-frame health check (> 2 rad/s = FAIL).
    T = 30
    q = np.full((T, 3), 0.1)
    dq = np.zeros((T, 3))
    dq[0, 1] = 3.0
    f = _write_npz(tmp_path / "ghost.npz", q, dq)

    rep = audit.audit_clip(f, limits, joint_names=JOINT_SPEC)
    assert rep.verdict == "FAIL"
    assert _findings(rep, "first_frame_vel", "FAIL")
    assert rep.suggestion.kind == "trim_head"   # ghost is FAIL -> it counts, trim 1
    assert rep.suggestion.trim_head == 1

    ann = {"ghost": {"first_frame_exempt": True, "phase": 0.5}}
    rep2 = audit.audit_clip(f, limits, joint_names=JOINT_SPEC, annotations=ann)
    assert rep2.verdict == "WARN"               # downgraded, nothing else FAILs
    ff = _findings(rep2, "first_frame_vel")
    assert ff and ff[0].level == "WARN" and "DOWNGRADED" in ff[0].message
    assert not _findings(rep2, "first_frame_vel", "FAIL")
    assert rep2.suggestion.kind == "none"       # exempted ghost is not "over the limit"


def test_annotations_yaml_roundtrip(tmp_path, limits):
    # same as above but through a real yaml file + the loader (works with or
    # without PyYAML thanks to the built-in minimal parser)
    T = 30
    q = np.full((T, 3), 0.1)
    dq = np.zeros((T, 3))
    dq[0, 1] = 3.0
    f = _write_npz(tmp_path / "ghost2.npz", q, dq)
    ann_file = tmp_path / "ann.yaml"
    ann_file.write_text(
        "# test registry\n"
        "clips:\n"
        "  ghost2:\n"
        "    phase: 0.5   # contact mid-clip\n"
        "    first_frame_exempt: true\n"
    )
    ann = audit.load_annotations(str(ann_file))
    assert "ghost2" in ann
    rep = audit.audit_clip(f, limits, joint_names=JOINT_SPEC, annotations=ann)
    assert rep.verdict == "WARN"
    assert rep.first_frame_exempt is True
    assert rep.contact_frame == int(round(0.5 * (T - 1)))


# --------------------------------------------- 5. position saturation -> WARN
def test_position_saturation_warns(tmp_path, limits):
    # joint0 creeps to (upper - 0.002) and pins there for 8 frames: inside the
    # hard limit (no FAIL) but within the 0.005 rad saturation tolerance ->
    # WARN with the retarget-clamping signature. Slow ramp keeps velocities legal.
    T = 40
    fps = 10
    target = 1.0 - 0.002
    q = np.zeros((T, 3))
    ramp = np.linspace(0.0, target, 21)         # 0.0499 rad/frame @10fps ~ 0.5 rad/s
    q[:21, 0] = ramp
    q[21:, 0] = target
    dq = np.gradient(q, axis=0) * fps
    f = _write_npz(tmp_path / "sat.npz", q, dq, fps=fps)
    rep = audit.audit_clip(f, limits, joint_names=JOINT_SPEC)

    assert rep.verdict == "WARN"
    sat = _findings(rep, "pos_saturation", "WARN")
    assert sat and sat[0].joint == "j0_joint"
    assert len(sat[0].frames) >= audit.SAT_MIN_RUN
    assert not _findings(rep, "pos_limit")       # never crossed the hard limit
    assert rep.suggestion.kind == "none"


# ------------------------------- 6. mid-clip position FAIL -> reject/regenerate
def test_mid_position_violation_suggests_regenerate(tmp_path, limits):
    T = 30
    q = np.full((T, 3), 0.1)
    q[15, 0] = 1.1                               # beyond the +1.0 hard limit, mid-clip
    dq = np.zeros((T, 3))
    f = _write_npz(tmp_path / "posmid.npz", q, dq)
    rep = audit.audit_clip(f, limits, joint_names=JOINT_SPEC)

    assert rep.verdict == "FAIL"
    pos = _findings(rep, "pos_limit", "FAIL")
    assert pos and 15 in pos[0].frames
    assert rep.suggestion.kind == "regenerate"   # neither trim nor slow-play fixes it


# ----------------------------------------------------------- CLI smoke + exits
def test_cli_exit_codes_and_reports(tmp_path, toy_urdf):
    clean = _write_npz(tmp_path / "ok.npz", np.full((30, 3), 0.1), np.zeros((30, 3)))
    bad_q = np.full((30, 3), 0.1)
    bad_dq = np.zeros((30, 3))
    bad_dq[15, 0] = 6.0
    bad = _write_npz(tmp_path / "bad.npz", bad_q, bad_dq)
    md, js = tmp_path / "r.md", tmp_path / "r.json"

    code = audit.main([
        clean, bad,
        "--urdf", toy_urdf,
        "--joint-names", JOINT_SPEC,
        "--annotations", "none",
        "--md", str(md), "--json", str(js), "--quiet",
    ])
    assert code == 2                              # worst clip FAILs
    text = md.read_text()
    assert "ok | **PASS**" in text and "bad | **FAIL**" in text

    import json as _json
    data = _json.loads(js.read_text())
    verdicts = {c["stem"]: c["verdict"] for c in data["clips"]}
    assert verdicts == {"ok": "PASS", "bad": "FAIL"}
    assert data["exit_code"] == 2

    code_ok = audit.main([
        clean, "--urdf", toy_urdf, "--joint-names", JOINT_SPEC,
        "--annotations", "none", "--quiet",
    ])
    assert code_ok == 0


def test_mini_yaml_fallback_parser():
    # the built-in parser must survive the real registry's comment/multi-line style
    text = (
        "# header comment\n"
        "grip:\n"
        "  session:\n"
        "    alpha_z_deg: 5\n"
        "clips:\n"
        "  hope_backhand_v5_cal:\n"
        "    phase: 0.362\n"
        "    status: verified\n"
        "    method: long text\n"
        "      continuation line that must be ignored\n"
        "    first_frame_exempt: true\n"
        "    rally_yaw_deg: -40  # trailing comment\n"
    )
    clips = audit._mini_yaml_clips(text)
    assert clips["hope_backhand_v5_cal"]["phase"] == "0.362"
    assert clips["hope_backhand_v5_cal"]["first_frame_exempt"] == "true"
    assert clips["hope_backhand_v5_cal"]["rally_yaw_deg"] == "-40"
    assert "session" not in clips                 # grip section not swallowed


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
