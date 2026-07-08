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

Check 7, support-foot slip / foot skate (task spec 2026-07-09):
  * clean planted feet                -> PASS (informational peak reported)
  * mid-clip slip in the WARN band    -> WARN
  * mid-clip slip > 0.15 m/s          -> FAIL + transplant_legs suggestion (never trim/slow)
  * ... even when a mid velocity FAIL would otherwise suggest slow-play (veto)
  * head-window-only slip             -> FAIL routed to the ordinary trim path
  * missing body order                -> fail-loud FAIL + reaudit suggestion;
                                         --body-order none = explicit skip
  * body-order sources: sidecar file next to the npz, explicit file/list
    (explicit spec beats the npz-embedded body_names key), count mismatch loud

REAL-ASSET REGRESSION (no motion npz ships with the laptop checkout — run on the
pod after pulling; calibration numbers are the franco/yikang 定案 2026-07-09):
    source /workspace/franco/env.sh && cd $HOPE_WBT && \
    python scripts/audit_motion_npz.py \
        /workspace/franco/motion_work/motions/regen_0708_candidates/hope_*hand_v4rg_cal.npz \
        /workspace/franco/motion_work/motions/regen_0708_candidates/hope_*hand_v5rg_cal.npz \
        /workspace/franco/motion_work/motions/v5_height_fix/hope_*hand_v5hL_cal.npz \
        --body-order /workspace/franco/body_order_isaac.txt
    EXPECT for check 7: hope_backhand_v5rg_cal FAIL (imagined-leg MID-CLIP slip peak
    0.30-0.35 m/s -> transplant_legs suggestion); hope_forehand_v5rg_cal PASS
    (0.034, light); all v4rg + v5hL PASS (true/pinned legs, ~0.010 clean).

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


# toy body table: col 0 = pelvis (base_lin), cols 1/2 = the two feet (check 7)
BODIES = ["pelvis_link", "left_ankle_roll_Link", "right_ankle_roll_Link"]
BODY_SPEC = ",".join(BODIES)


def _write_npz(path, q, dq, fps=FPS, base_lin=None, feet=None, body_names=BODIES):
    """feet: optional (T, 2, 3) world trajectories of the two ankle links (default
    both planted at the origin = zero slip). body_names embeds the npz `body_names`
    key; pass None to omit it (fail-loud body-order tests)."""
    q = np.asarray(q, dtype=np.float32)
    dq = np.asarray(dq, dtype=np.float32)
    t, j = q.shape
    nb = len(BODIES)
    body_lin = np.zeros((t, nb, 3), dtype=np.float32)
    if base_lin is not None:
        body_lin[:, 0, 0] = np.asarray(base_lin, dtype=np.float32)
    body_pos = np.zeros((t, nb, 3), dtype=np.float32)
    if feet is not None:
        body_pos[:, 1:3, :] = np.asarray(feet, dtype=np.float32)
    extra = {} if body_names is None else {"body_names": np.array(body_names)}
    np.savez(
        path,
        fps=np.array([fps]),
        joint_pos=q,
        joint_vel=dq,
        body_pos_w=body_pos,
        body_quat_w=np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (t, nb, 1)),
        body_lin_vel_w=body_lin,
        body_ang_vel_w=np.zeros((t, nb, 3), dtype=np.float32),
        **extra,
    )
    return str(path)


def _sliding_feet(T, start, stop, step=0.006, foot=0):
    """(T,2,3) planted feet (z=0); the chosen foot translates +x by `step` m/frame
    over frames [start, stop] (inclusive), i.e. slip = step*FPS m/s on samples
    start..stop-1 while remaining inside the z support band."""
    feet = np.zeros((T, 2, 3), dtype=np.float64)
    x = np.zeros(T)
    for i in range(start + 1, T):
        x[i] = x[i - 1] + (step if i <= stop else 0.0)
    feet[:, foot, 0] = x
    return feet


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


# --------------------------------------------- adversarial-review regressions
def test_long_inline_joint_list_does_not_crash(tmp_path, limits):
    # a full 31-name comma list (~700 chars) must not hit ENAMETOOLONG via Path()
    many = ",".join(f"long_name_{i:02d}_joint" * 3 for i in range(31))
    q = np.full((10, 3), 0.1)
    f = _write_npz(tmp_path / "c.npz", q, np.zeros((10, 3)))
    rep = audit.audit_clip(f, limits, joint_names=many)
    assert rep.verdict == "FAIL"                  # count mismatch -> fail-loud
    assert _findings(rep, "load", "FAIL")


def test_nan_arrays_fail_not_pass(tmp_path, limits):
    q = np.full((20, 3), np.nan)
    f = _write_npz(tmp_path / "nan.npz", q, q.copy())
    rep = audit.audit_clip(f, limits, joint_names=JOINT_SPEC)
    assert rep.verdict == "FAIL"
    assert rep.suggestion.kind == "regenerate"


def test_short_clip_overlapping_trims_regenerate(tmp_path, limits):
    # T=15 <= 2*EDGE_WINDOW: a FAIL frame near the middle lands in BOTH edge
    # windows; combined trim would cover the whole clip -> must regenerate,
    # never emit an empty-slice trim command
    T = 15
    q = np.full((T, 3), 0.1)
    dq = np.zeros((T, 3))
    dq[8, 0] = 6.0                                # > 5 rad/s toy limit
    f = _write_npz(tmp_path / "short.npz", q, dq)
    rep = audit.audit_clip(f, limits, joint_names=JOINT_SPEC)
    assert rep.verdict == "FAIL"
    assert rep.suggestion.kind == "regenerate"
    assert (rep.suggestion.trim_head or 0) + (rep.suggestion.trim_tail or 0) == 0


def test_markdown_pipes_escaped(tmp_path, limits):
    # |velocity| in finding messages must not break GFM table cells
    q = np.full((30, 3), 0.1)
    dq = np.zeros((30, 3))
    dq[15, 0] = 6.0
    f = _write_npz(tmp_path / "m.npz", q, dq)
    rep = audit.audit_clip(f, limits, joint_names=JOINT_SPEC)
    md = audit.report_markdown([rep], {
        "generated_utc": "t", "urdf": "u", "soft_factor": 0.9, "annotations": None,
    })
    table = [ln for ln in md.splitlines() if ln.startswith("| vel_limit")]
    assert table and all("\\|velocity\\|" in ln for ln in table)


# ------------------------------------------- 7. support-foot slip (foot skate)
def test_foot_skate_clean_pass(tmp_path, limits):
    # 0.02 m/s slide (0.0004 m/frame @50fps): inside the PASS band (<= 0.05);
    # calibration anchor: healthy v4/v5hL real/pinned legs measure ~0.010
    T = 30
    q, dq = np.full((T, 3), 0.1), np.zeros((T, 3))
    f = _write_npz(tmp_path / "fs_ok.npz", q, dq, feet=_sliding_feet(T, 12, 20, step=0.0004))
    rep = audit.audit_clip(f, limits, joint_names=JOINT_SPEC)
    assert rep.verdict == "PASS"
    info = _findings(rep, "foot_skate", "PASS")
    assert info and max(x.value for x in info if x.value is not None) == pytest.approx(0.02, abs=1e-3)


def test_foot_skate_mid_warn(tmp_path, limits):
    # 0.08 m/s mid-clip slide: WARN band (0.05, 0.15]
    T = 30
    q, dq = np.full((T, 3), 0.1), np.zeros((T, 3))
    f = _write_npz(tmp_path / "fs_warn.npz", q, dq, feet=_sliding_feet(T, 14, 21, step=0.0016))
    rep = audit.audit_clip(f, limits, joint_names=JOINT_SPEC)
    assert rep.verdict == "WARN"
    w = _findings(rep, "foot_skate", "WARN")
    assert w and w[0].value == pytest.approx(0.08, abs=1e-3)
    assert rep.suggestion.kind == "none"


def test_foot_skate_mid_fail_suggests_transplant(tmp_path, limits):
    # 0.30 m/s mid-clip slide on the left foot (the v5-backhand imagined-leg
    # signature, 定案 0.30-0.35 deadly): FAIL + transplant/reshoot, NEVER trim/slow
    T = 30
    q, dq = np.full((T, 3), 0.1), np.zeros((T, 3))
    f = _write_npz(tmp_path / "fs_mid.npz", q, dq, feet=_sliding_feet(T, 14, 21, step=0.006))
    rep = audit.audit_clip(f, limits, joint_names=JOINT_SPEC)

    assert rep.verdict == "FAIL"
    fails = _findings(rep, "foot_skate", "FAIL")
    assert fails and fails[0].joint == "left_ankle_roll_Link"
    assert fails[0].value == pytest.approx(0.30, abs=1e-2)
    assert 15 in fails[0].frames
    assert rep.suggestion.kind == "transplant_legs"
    joined = "\n".join(rep.suggestion.lines)
    assert "腿姿病" in joined and "transplant_legs.py --mode pinned" in joined
    assert rep.suggestion.trim_head == 0 and rep.suggestion.trim_tail == 0
    assert rep.suggestion.slow_factor is None


def test_foot_skate_veto_beats_slowdown(tmp_path, limits):
    # a mid-clip stored-velocity FAIL alone would suggest slow-play; a coexisting
    # mid-clip skate FAIL must veto it (slow-play only slows the same slide)
    T = 30
    q, dq = np.full((T, 3), 0.1), np.zeros((T, 3))
    dq[15, 0] = 6.0                                   # > 5 rad/s toy limit
    f = _write_npz(tmp_path / "fs_veto.npz", q, dq, feet=_sliding_feet(T, 14, 21, step=0.006))
    rep = audit.audit_clip(f, limits, joint_names=JOINT_SPEC)
    assert rep.verdict == "FAIL"
    assert _findings(rep, "vel_limit_stored", "FAIL")
    assert rep.suggestion.kind == "transplant_legs"


def test_foot_skate_head_routes_to_trim(tmp_path, limits):
    # slip confined to the head window (samples 0-7, frames 0-8 < EDGE_WINDOW):
    # the historical head-glitch band -> graded as foot_skate_head and repaired
    # via the ordinary trim path, not transplant
    T = 30
    q, dq = np.full((T, 3), 0.1), np.zeros((T, 3))
    f = _write_npz(tmp_path / "fs_head.npz", q, dq, feet=_sliding_feet(T, 0, 8, step=0.006))
    rep = audit.audit_clip(f, limits, joint_names=JOINT_SPEC)

    assert rep.verdict == "FAIL"
    assert _findings(rep, "foot_skate_head", "FAIL")
    assert not _findings(rep, "foot_skate", "FAIL")   # nothing beyond the head window
    assert rep.suggestion.kind == "trim_head"
    assert rep.suggestion.trim_head == 9              # first frame after the FAIL cluster 0-8


def test_missing_body_order_fails_loud(tmp_path, limits):
    # no embedded body_names, no sidecar, no --body-order: check 7 must FAIL the
    # clip (never a silent skip) and point at re-auditing, not regenerating
    T = 30
    q, dq = np.full((T, 3), 0.1), np.zeros((T, 3))
    f = _write_npz(tmp_path / "fs_noorder.npz", q, dq, body_names=None)
    rep = audit.audit_clip(f, limits, joint_names=JOINT_SPEC)
    assert rep.verdict == "FAIL"
    fails = _findings(rep, "foot_skate", "FAIL")
    assert fails and "body order" in fails[0].message
    assert rep.suggestion.kind == "reaudit"

    # explicit opt-out is the only silent path
    rep2 = audit.audit_clip(f, limits, joint_names=JOINT_SPEC, body_order="none")
    assert rep2.verdict == "PASS"
    assert rep2.suggestion.kind == "none"


def test_body_order_sidecar_and_explicit_sources(tmp_path, limits):
    # sidecar body_order.txt next to the npz (the csv_to_npz_mujoco convention)
    T = 30
    q, dq = np.full((T, 3), 0.1), np.zeros((T, 3))
    f = _write_npz(tmp_path / "fs_sidecar.npz", q, dq, body_names=None)
    (tmp_path / "body_order.txt").write_text("# discover-map output\n" + "\n".join(BODIES) + "\n")
    rep = audit.audit_clip(f, limits, joint_names=JOINT_SPEC)
    assert rep.verdict == "PASS"

    # explicit file wins even where a sidecar/embedded key exists
    order_file = tmp_path / "custom_order.txt"
    order_file.write_text("\n".join(BODIES) + "\n")
    rep2 = audit.audit_clip(f, limits, joint_names=JOINT_SPEC, body_order=str(order_file))
    assert rep2.verdict == "PASS"

    # explicit inline list BEATS the npz-embedded key: a list without the ankle
    # links must fail loud even though the embedded body_names are fine
    f2 = _write_npz(tmp_path / "fs_embedded.npz", q, dq)          # embedded names OK
    rep3 = audit.audit_clip(f2, limits, joint_names=JOINT_SPEC, body_order="a,b,c")
    assert rep3.verdict == "FAIL"
    fails = _findings(rep3, "foot_skate", "FAIL")
    assert fails and "left_ankle_roll_Link" in fails[0].message

    # count mismatch fails loud too
    short_file = tmp_path / "short_order.txt"
    short_file.write_text("pelvis_link\nleft_ankle_roll_Link\n")
    rep4 = audit.audit_clip(f2, limits, joint_names=JOINT_SPEC, body_order=str(short_file))
    assert rep4.verdict == "FAIL"
    fails = _findings(rep4, "foot_skate", "FAIL")
    assert fails and "count" in fails[0].message


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
