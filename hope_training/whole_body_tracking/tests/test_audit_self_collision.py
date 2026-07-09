"""Tests for scripts/audit_self_collision.py (L1 vendor-MJCF self-collision audit).

Three tiers, mirroring tests/test_feasibility_oracle.py:
  * pure tiers (numpy only)   — the backswing-frame rule; run anywhere;
  * mujoco tiers              — need `mujoco` + the vendor MJCF; skipped otherwise;
  * real-asset tier           — needs the production npz (pod only); skipped otherwise.

The contract encoded here:
  1. the racket<->torso pair is COLLIDABLE in the vendor MJCF (the branch's whole
     premise) and the loader refuses to run if a future MJCF ever hides it;
  2. neutral poses are self-collision clean, and the floor is not a self-collision;
  3. a SYNTHESISED self-colliding clip (right arm folded into the torso) is graded
     FAIL, names the racket<->torso pair, and yields the 晚六 redistribute_stroke
     repair suggestion;
  4. a synthesised grazing contact (1 frame, shallow) is graded WARN, not FAIL;
  5. the clearance bisection agrees with analytic distances, and — the reason it
     exists — is correct at frames where raw mj_geomDistance returns a bogus 0.0;
  6. a live production asset (hope_backhand_v4rg_cal) is PASS with zero contacts.

Pod run (mjeval venv, CPU only):
    /workspace/hope_mjeval_venv/bin/python -m pytest \
        hope_training/whole_body_tracking/tests/test_audit_self_collision.py -q
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

WBT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WBT / "scripts"))

import audit_self_collision as sc  # noqa: E402

MJCF = os.environ.get("SELFCOL_MJCF") or str(sc._default_mjcf())
HAVE_MUJOCO = sc.mujoco is not None
HAVE_MJCF = Path(MJCF).is_file()
needs_model = pytest.mark.skipif(
    not (HAVE_MUJOCO and HAVE_MJCF),
    reason=f"mujoco installed={HAVE_MUJOCO}, mjcf found={HAVE_MJCF} ({MJCF})",
)

# A pose (right shoulder pitch/roll/yaw, elbow) that buries the racket ~188 mm
# inside the torso. Found by a joint-limit grid search on the vendor MJCF.
JAM_POSE = {
    "right_shoulder_pitch_joint": -1.92,
    "right_shoulder_roll_joint": -0.364,
    "right_shoulder_yaw_joint": 1.862,
    "right_elbow_joint": -0.96,
}
REAL_CLIP = Path(
    "/workspace/franco/motion_work/motions/regen_0708_candidates/hope_backhand_v4rg_cal.npz"
)
BODY_ORDER = Path("/workspace/franco/body_order_isaac.txt")
needs_real_clip = pytest.mark.skipif(
    not (HAVE_MUJOCO and HAVE_MJCF and REAL_CLIP.is_file() and BODY_ORDER.is_file()),
    reason="production npz / body order not present (pod-only regression)",
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _stand_qpos(sm) -> np.ndarray:
    """The vendor 'stand' keyframe as a full qpos vector."""
    return np.array(sm.model.key_qpos[0], dtype=float)


def _write_clip(tmp_path: Path, qpos: np.ndarray, sm, name="clip.npz",
                fps=50, body_names=("pelvis_link",)) -> Path:
    """Turn a (T, nq) qpos trajectory into the npz layout the auditor consumes.

    Only column 0 of body_pos_w / body_quat_w is ever read (the root pose), but the
    body-order resolver checks the width, so the remaining columns are zero-filled.
    """
    T = qpos.shape[0]
    n_bodies = len(body_names)
    joint_pos = qpos[:, sm.joint_qposadr]
    body_pos = np.zeros((T, n_bodies, 3))
    body_quat = np.zeros((T, n_bodies, 4))
    body_pos[:, 0, :] = qpos[:, 0:3]
    body_quat[:, 0, :] = qpos[:, 3:7]
    p = tmp_path / name
    np.savez(p, joint_pos=joint_pos, joint_vel=np.zeros_like(joint_pos), fps=np.array([fps]),
             body_pos_w=body_pos, body_quat_w=body_quat,
             body_names=np.array(body_names, dtype=object))
    return p


def _jam_qpos(sm, frames: int, jam_frames) -> np.ndarray:
    """Stand pose everywhere, JAM_POSE on `jam_frames`."""
    m = sm.model
    base = _stand_qpos(sm)
    qpos = np.tile(base, (frames, 1))
    for jname, val in JAM_POSE.items():
        jid = sc.mujoco.mj_name2id(m, sc.mujoco.mjtObj.mjOBJ_JOINT, jname)
        qpos[np.asarray(jam_frames), m.jnt_qposadr[jid]] = val
    return qpos


# ---------------------------------------------------------------------------
# 1. pure: the deepest-drawback frame rule
# ---------------------------------------------------------------------------

def test_backswing_frame_is_the_turnaround_not_the_global_max():
    # racket starts far away (ready pose), draws back to a turnaround at t=6,
    # then swings forward to contact at t=10. A plain argmax over [0,10] would
    # return frame 0; the turnaround is what we want.
    x = np.array([9.0, 5.0, 2.0, 1.0, 1.5, 2.5, 3.0, 2.0, 1.0, 0.4, 0.0])
    p = np.stack([x, np.zeros_like(x), np.zeros_like(x)], axis=1)
    assert sc.backswing_frame(p, contact=10) == 6
    assert int(np.argmax(np.linalg.norm(p - p[10], axis=1))) == 0  # the wrong answer


def test_backswing_frame_edges():
    p = np.zeros((5, 3))
    assert sc.backswing_frame(p, contact=0) == 0     # contact at frame 0
    assert sc.backswing_frame(p, contact=4) == 4     # flat: no recession, no walk-back


def test_backswing_frame_ignores_sub_eps_wobble():
    x = np.array([0.30, 0.30 + sc.BACKSWING_EPS / 2, 0.20, 0.0])
    p = np.stack([x, np.zeros_like(x), np.zeros_like(x)], axis=1)
    assert sc.backswing_frame(p, contact=3) == 1  # the 1e-5 wobble does not extend the walk


# ---------------------------------------------------------------------------
# 2. model: the required pair is visible, the floor is not a self-collision
# ---------------------------------------------------------------------------

@needs_model
def test_racket_torso_pair_is_collidable():
    sm = sc.load_selfcol_model(MJCF)
    for g in sc.RACKET_GEOMS:
        for t in sc.TORSO_GEOMS:
            assert sc.geom_pair_enabled(sm.model, sm.geom_ids[g], sm.geom_ids[t])


@needs_model
def test_geom_pair_enabled_honours_same_body_parent_and_exclude():
    sm = sc.load_selfcol_model(MJCF)
    m = sm.model
    gid = lambda n: sc.mujoco.mj_name2id(m, sc.mujoco.mjtObj.mjOBJ_GEOM, n)
    # same body (racket face and handle both hang off right_wrist_yaw_Link)
    assert not sc.geom_pair_enabled(m, gid("right_racket_collision"),
                                    gid("right_racket_handle_collision"))
    # vendor <exclude>: torso_Link <-> right_shoulder_roll_Link
    assert not sc.geom_pair_enabled(m, gid("torso_collision"),
                                    gid("right_shoulder_roll_collision"))
    # parent-child: right_hip_yaw_Link -> right_knee_Link is also an <exclude>;
    # use a true parent-child pair with no exclude entry instead
    assert not sc.geom_pair_enabled(m, gid("head_yaw_collision"), gid("head_pitch_collision"))
    # floor was neutralised at load, so nothing collides with it any more
    assert not sc.geom_pair_enabled(m, gid("floor"), gid("right_ankle_roll_collision"))


@needs_model
def test_loader_refuses_when_required_pair_is_excluded():
    sm = sc.load_selfcol_model(MJCF)
    m = sm.model
    # blind the racket by clearing its collision masks, then re-run the guard
    for g in sc.RACKET_GEOMS:
        m.geom_contype[sm.geom_ids[g]] = 0
        m.geom_conaffinity[sm.geom_ids[g]] = 0
    with pytest.raises(RuntimeError, match="racket-torso"):
        sc._assert_pairs_visible(m, sm.geom_ids)


@needs_model
def test_neutral_poses_are_self_collision_clean(tmp_path):
    """Stand keyframe: the only real contacts are foot-floor, and those are excluded."""
    sm = sc.load_selfcol_model(MJCF)
    d = sc.mujoco.MjData(sm.model)
    qpos = np.tile(_stand_qpos(sm), (3, 1))
    hits, colliding = sc.collect_hits(sm, d, qpos)
    assert hits == []
    assert not colliding.any()


# ---------------------------------------------------------------------------
# 3. synthetic self-collision: sustained -> FAIL, graze -> WARN
# ---------------------------------------------------------------------------

@needs_model
def test_synthetic_racket_into_torso_fails_and_names_the_pair(tmp_path):
    sm = sc.load_selfcol_model(MJCF)
    qpos = _jam_qpos(sm, frames=8, jam_frames=[3, 4, 5])  # sustained: 3 frames
    clip = _write_clip(tmp_path, qpos, sm)
    rep = sc.audit_clip(str(clip), sm, annotations={}, body_order=None)

    assert rep.verdict == sc.FAIL
    pairs = {(h.body1, h.body2) for h in rep.hits}
    assert ("right_wrist_yaw_Link", "torso_Link") in pairs, pairs

    main = next(h for h in rep.hits if h.body1 == "right_wrist_yaw_Link" and h.body2 == "torso_Link")
    # the RACKET itself must be implicated, not merely its body (the hand shares it)
    assert main.involves(sc.RACKET_GEOMS, sc.TORSO_GEOMS)
    assert ("right_racket_collision", "torso_collision") in main.geom_pairs, main.geom_pairs
    assert main.level == sc.FAIL
    assert main.sustained and main.longest_run == 3
    assert main.frames == [3, 4, 5]
    assert main.depth_peak > 0.1  # racket buried ~188 mm

    # no "clean" line may be emitted for the main item when it actually fired
    assert not any(f.check == "racket_torso" for f in rep.findings)
    assert rep.suggestion.kind == "redistribute_stroke"
    assert any("腰偏航" in ln for ln in rep.suggestion.lines)


@needs_model
def test_synthetic_isolated_shallow_contact_is_a_warn_graze(tmp_path):
    """一帧 + 浅 = 擦碰 = WARN. Depth is dialled just past first touch."""
    sm = sc.load_selfcol_model(MJCF)
    m = sm.model
    d = sc.mujoco.MjData(m)
    stand = _stand_qpos(sm)
    adrs = {j: m.jnt_qposadr[sc.mujoco.mj_name2id(m, sc.mujoco.mjtObj.mjOBJ_JOINT, j)]
            for j in JAM_POSE}

    def pose_at(alpha: float) -> np.ndarray:
        """Interpolate stand -> jam. alpha=0 is clearly clear, alpha=1 is 188 mm deep."""
        q = stand.copy()
        for j, val in JAM_POSE.items():
            q[adrs[j]] = stand[adrs[j]] + alpha * (val - stand[adrs[j]])
        return q

    def depth_at(alpha: float) -> float:
        d.qpos[:] = pose_at(alpha)
        sc.mujoco.mj_forward(m, d)
        return max((-d.contact[i].dist for i in range(d.ncon)), default=-1.0)

    lo, hi = 0.0, 1.0  # invariant: depth_at(lo) <= 0 < depth_at(hi)
    assert depth_at(lo) < 0 < depth_at(hi), "bracket must straddle first touch"
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if depth_at(mid) > 0:
            hi = mid
        else:
            lo = mid
        if 0 < depth_at(hi) < sc.DEEP_FAIL_M / 5:
            break
    graze_depth = depth_at(hi)
    assert 0 < graze_depth < sc.DEEP_FAIL_M, graze_depth

    qpos = np.tile(stand, (6, 1))
    qpos[2] = pose_at(hi)  # single frame, barely touching

    clip = _write_clip(tmp_path, qpos, sm, name="graze.npz")
    rep = sc.audit_clip(str(clip), sm, annotations={}, body_order=None)

    assert rep.verdict == sc.WARN, [f.message for f in rep.findings]
    assert all(h.level == sc.WARN for h in rep.hits)
    assert all(h.longest_run == 1 for h in rep.hits)
    assert any("擦碰" in f.message for f in rep.findings if f.level == sc.WARN)


@needs_model
def test_deep_single_frame_contact_is_fail_not_graze(tmp_path):
    """深穿 escalates even at one frame: geometry is passing through, not brushing."""
    sm = sc.load_selfcol_model(MJCF)
    qpos = _jam_qpos(sm, frames=5, jam_frames=[2])  # one frame, ~188 mm deep
    clip = _write_clip(tmp_path, qpos, sm, name="deep.npz")
    rep = sc.audit_clip(str(clip), sm, annotations={}, body_order=None)

    main = next(h for h in rep.hits if h.body1 == "right_wrist_yaw_Link" and h.body2 == "torso_Link")
    assert main.involves(sc.RACKET_GEOMS, sc.TORSO_GEOMS)
    assert main.longest_run == 1 and not main.sustained
    assert main.deep and main.level == sc.FAIL
    assert rep.verdict == sc.FAIL


# ---------------------------------------------------------------------------
# 4. the clearance bisection (the mj_geomDistance workaround)
# ---------------------------------------------------------------------------

@needs_model
def test_clearance_matches_analytic_distance_for_primitive_geoms():
    """Two spheres a known distance apart: the bisection must recover it."""
    xml = """
    <mujoco>
      <worldbody>
        <body name="a" pos="0 0 0"><geom name="ga" type="sphere" size="0.1"/></body>
        <body name="b" pos="0.5 0 0"><geom name="gb" type="sphere" size="0.15"/></body>
      </worldbody>
    </mujoco>
    """
    m = sc.mujoco.MjModel.from_xml_string(xml)
    d = sc.mujoco.MjData(m)
    sc.mujoco.mj_forward(m, d)
    ga = sc.mujoco.mj_name2id(m, sc.mujoco.mjtObj.mjOBJ_GEOM, "ga")
    gb = sc.mujoco.mj_name2id(m, sc.mujoco.mjtObj.mjOBJ_GEOM, "gb")
    dist, sat = sc.geom_clearance(m, d, ga, gb)
    assert not sat
    assert dist == pytest.approx(0.5 - 0.1 - 0.15, abs=2 * sc.CLEARANCE_TOL)


@needs_model
def test_clearance_saturates_beyond_distmax():
    xml = """
    <mujoco>
      <worldbody>
        <body name="a" pos="0 0 0"><geom name="ga" type="sphere" size="0.05"/></body>
        <body name="b" pos="5 0 0"><geom name="gb" type="sphere" size="0.05"/></body>
      </worldbody>
    </mujoco>
    """
    m = sc.mujoco.MjModel.from_xml_string(xml)
    d = sc.mujoco.MjData(m)
    sc.mujoco.mj_forward(m, d)
    ga = sc.mujoco.mj_name2id(m, sc.mujoco.mjtObj.mjOBJ_GEOM, "ga")
    gb = sc.mujoco.mj_name2id(m, sc.mujoco.mjtObj.mjOBJ_GEOM, "gb")
    dist, sat = sc.geom_clearance(m, d, ga, gb)
    assert sat and dist == pytest.approx(sc.CLEARANCE_DISTMAX)


@needs_model
def test_clearance_reports_penetration_depth_when_jammed():
    sm = sc.load_selfcol_model(MJCF)
    m = sm.model
    d = sc.mujoco.MjData(m)
    q = _jam_qpos(sm, 1, [0])
    d.qpos[:] = q[0]
    sc.mujoco.mj_forward(m, d)
    dist, sat = sc.geom_clearance(m, d, sm.geom_ids["right_racket_collision"],
                                  sm.geom_ids["torso_collision"])
    assert not sat and dist < -0.1


@needs_model
@needs_real_clip
def test_bisection_repairs_the_mujoco_mesh_distance_defect():
    """Regression for the MuJoCo 3.10 mesh-mesh defect the module docstring documents.

    v4rg backhand frame 35: raw mj_geomDistance(distmax=0.6) returns exactly 0.0 for
    racket-vs-torso while the frame has NO contact. The bisection must return the
    true distance (independently bracketed at 0.2532-0.2644 by a separating-axis
    argument over the collision hulls).
    """
    sm = sc.load_selfcol_model(MJCF)
    m = sm.model
    d = sc.mujoco.MjData(m)
    z = np.load(REAL_CLIP)
    qpos = sc.build_qpos(sm, np.asarray(z["joint_pos"], float),
                         np.asarray(z["body_pos_w"], float)[:, 0],
                         np.asarray(z["body_quat_w"], float)[:, 0])
    d.qpos[:] = qpos[35]
    sc.mujoco.mj_forward(m, d)
    assert d.ncon == 0, "frame 35 must be contact-free for this regression to mean anything"

    rf, tor = sm.geom_ids["right_racket_collision"], sm.geom_ids["torso_collision"]
    raw = sc.mujoco.mj_geomDistance(m, d, rf, tor, 0.6, None)
    assert raw == pytest.approx(0.0, abs=1e-9), (
        "the MuJoCo defect this workaround exists for has disappeared — if mujoco was "
        "upgraded, re-measure and consider simplifying geom_clearance()"
    )
    dist, sat = sc.geom_clearance(m, d, rf, tor)
    assert not sat
    assert 0.2532 <= dist <= 0.2644, dist


# ---------------------------------------------------------------------------
# 5. live production asset: PASS with zero contacts
# ---------------------------------------------------------------------------

@needs_real_clip
def test_production_backhand_v4rg_is_self_collision_clean():
    sm = sc.load_selfcol_model(MJCF)
    ann = sc.load_annotations(str(sc._default_annotations()))
    rep = sc.audit_clip(str(REAL_CLIP), sm, annotations=ann, body_order=str(BODY_ORDER))

    assert rep.verdict == sc.PASS, [f.message for f in rep.findings if f.level != sc.PASS]
    assert rep.hits == []
    assert rep.suggestion.kind == "none"
    # the main item must be positively reported, not merely absent
    assert any(f.check == "racket_torso" and f.level == sc.PASS for f in rep.findings)
    # backswing frame lands strictly inside the backswing, and the table is populated
    assert rep.contact_frame is not None and 0 < rep.backswing_frame < rep.contact_frame
    rt = rep.clearance("racket-torso")
    assert rt is not None and rt.min_dist > sc.CLEARANCE_WARN_M


@needs_real_clip
def test_cli_exit_code_is_zero_on_a_clean_clip(tmp_path, capsys):
    rc = sc.main([str(REAL_CLIP), "--body-order", str(BODY_ORDER), "--quiet",
                  "--json", str(tmp_path / "out.json"),
                  "--baseline-md", str(tmp_path / "base.md")])
    assert rc == 0
    import json
    payload = json.loads((tmp_path / "out.json").read_text())
    assert payload["exit_code"] == 0
    assert payload["clips"][0]["verdict"] == "PASS"
    assert payload["clips"][0]["collisions"] == []
    assert (tmp_path / "base.md").read_text().startswith("# 反手引拍最深帧")


@needs_model
def test_cli_exit_code_is_two_on_a_self_colliding_clip(tmp_path):
    sm = sc.load_selfcol_model(MJCF)
    clip = _write_clip(tmp_path, _jam_qpos(sm, 6, [2, 3]), sm, name="bad.npz")
    out = tmp_path / "bad.json"
    rc = sc.main([str(clip), "--annotations", "none", "--quiet", "--json", str(out)])
    assert rc == 2
    import json
    clip_json = json.loads(out.read_text())["clips"][0]
    # exit 2 must come from a real self-collision, not from a load/body-order FAIL
    assert clip_json["verdict"] == "FAIL"
    assert {f["check"] for f in clip_json["findings"]} == {"self_collision"}
    assert any(c["level"] == "FAIL" and "right_racket_collision|torso_collision" in c["geom_pairs"]
               for c in clip_json["collisions"])


# ---------------------------------------------------------------------------
# 6. fail-loud posture (never a silent skip)
# ---------------------------------------------------------------------------

@needs_model
def test_unresolved_body_order_is_a_loud_fail(tmp_path):
    sm = sc.load_selfcol_model(MJCF)
    qpos = np.tile(_stand_qpos(sm), (3, 1))
    T = qpos.shape[0]
    p = tmp_path / "nobodies.npz"
    np.savez(p, joint_pos=qpos[:, sm.joint_qposadr], joint_vel=np.zeros((T, 31)),
             fps=np.array([50]), body_pos_w=np.zeros((T, 2, 3)), body_quat_w=np.zeros((T, 2, 4)))
    rep = sc.audit_clip(str(p), sm, annotations={}, body_order=None)
    assert rep.verdict == sc.FAIL
    assert rep.suggestion.kind == "reaudit"
    assert any("body order unresolved" in f.message for f in rep.findings)


@needs_model
def test_wrong_root_body_is_a_loud_fail(tmp_path):
    """Body column 0 must be the pelvis; the root pose is read from it."""
    sm = sc.load_selfcol_model(MJCF)
    qpos = np.tile(_stand_qpos(sm), (3, 1))
    clip = _write_clip(tmp_path, qpos, sm, name="wrongroot.npz", body_names=("torso_Link",))
    rep = sc.audit_clip(str(clip), sm, annotations={}, body_order=None)
    assert rep.verdict == sc.FAIL
    assert any("expected the root" in f.message for f in rep.findings)


@needs_model
def test_corrupt_npz_is_a_fail_not_a_crash(tmp_path):
    sm = sc.load_selfcol_model(MJCF)
    qpos = np.tile(_stand_qpos(sm), (3, 1))
    clip = _write_clip(tmp_path, qpos, sm, name="nan.npz")
    d = dict(np.load(clip, allow_pickle=True))
    d["joint_pos"][1, 0] = np.nan
    np.savez(clip, **d)
    rep = sc.audit_clip(str(clip), sm, annotations={}, body_order=None)
    assert rep.verdict == sc.FAIL
    assert rep.suggestion.kind == "regenerate"


@needs_model
def test_hand_only_contact_does_not_masquerade_as_the_racket_main_item(tmp_path):
    """The hand and the racket share right_wrist_yaw_Link.

    A knuckle brushing the torso must NOT be reported as 拍体-躯干: the main item is
    matched on geoms, so `racket_torso: clean` still has to be emitted.
    """
    sm = sc.load_selfcol_model(MJCF)
    m = sm.model
    hit = sc.PairHit(
        body1="right_wrist_yaw_Link", body2="torso_Link", frames=[1],
        depth_peak=0.001, depth_peak_frame=1, runs=[(1, 1)],
        geom_pairs=[("right_hand_finger_collision", "torso_collision")],
    )
    assert not hit.involves(sc.RACKET_GEOMS, sc.TORSO_GEOMS)
    # ... but it IS part of the arm chain, so the stroke-redistribution advice stands
    assert hit.involves(sc.ARM_CHAIN_GEOMS + ("right_hand_finger_collision",), sc.TRUNK_GEOMS)

    racket_hit = sc.PairHit(
        body1="right_wrist_yaw_Link", body2="torso_Link", frames=[1],
        depth_peak=0.001, depth_peak_frame=1, runs=[(1, 1)],
        geom_pairs=[("right_racket_collision", "torso_collision")],
    )
    assert racket_hit.involves(sc.RACKET_GEOMS, sc.TORSO_GEOMS)
