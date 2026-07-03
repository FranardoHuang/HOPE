"""Tests for the 175-D deploy_parity obs builder + racket FK (pure numpy)."""

import numpy as np

from hope_wbc_runner.obs_builder import (
    N_JOINTS,
    OBS_DIM_175,
    RacketTarget,
    build_obs,
    build_obs_175,
    synthetic_state_from_refs,
)
from hope_wbc_runner.racket_fk import racket_pos_pelvis


def _refs():
    quats = np.tile([1.0, 0.0, 0.0, 0.0], (14, 1))
    pos = np.zeros((14, 3))
    pos[0] = [0.0, 0.0, 0.95]   # pelvis
    pos[7] = [0.0, 0.0, 1.15]   # torso/anchor
    return {
        "joint_pos": np.linspace(-0.3, 0.3, N_JOINTS),
        "joint_vel": np.zeros(N_JOINTS),
        "body_pos_w": pos,
        "body_quat_w": quats,
    }


def _target(y=-0.30):
    return RacketTarget(
        pos_w=np.array([0.65, y, 0.85]), vel_w=np.array([1.5, 1.4, 0.7]),
        swing_sign=1.0 if y < 0 else -1.0, time_to_strike=1.2)


def _both(refs, tgt):
    default_q = np.zeros(N_JOINTS)
    state = synthetic_state_from_refs(refs, default_q)
    la = np.linspace(-1.0, 1.0, N_JOINTS)
    return (build_obs(refs, state, tgt, la, default_q),
            build_obs_175(refs, state, tgt, la, default_q), state)


def test_obs_is_175_dim():
    o180, o175, _ = _both(_refs(), _target())
    assert o175.shape == (OBS_DIM_175,)


def test_175_layout_matches_180_with_blocks_removed():
    """175 = 180 with motion_anchor_pos_b (3, at [62:65]) and base_target_pos_b
    (2, at [167:169]) removed, and the racket-target block re-referenced. All
    other blocks must be bitwise identical."""
    o180, o175, _ = _both(_refs(), _target())
    # command block
    assert np.array_equal(o175[:62], o180[:62])
    # anchor ori 6D: 180 has anchor_pos at [62:65] then ori at [65:71]
    assert np.array_equal(o175[62:68], o180[65:71])
    # base_ang_vel + joint_pos + joint_vel + last_action + proj_grav (3+31+31+31+3 = 99)
    assert np.array_equal(o175[68:167], o180[71:170])
    # racket target vel / tts / swing tails
    assert np.array_equal(o175[-5:], o180[-5:])


def test_175_racket_target_is_fk_relative():
    refs = _refs()
    tgt = _target()
    o180, o175, state = _both(refs, tgt)
    fk_w = state.base_pos_w + racket_pos_pelvis(state.q)   # identity base quat
    expected = tgt.pos_w - fk_w                             # identity yaw
    assert np.allclose(o175[167:170], expected, atol=1e-12)
    # and it must differ from the 180 base-relative block (unless FK == base, impossible)
    assert not np.allclose(o175[167:170], o180[170:173])


def test_racket_fk_zero_pose_plausible():
    """At q=0 the racket must be in front-right of the pelvis at roughly chest
    height offset (the URDF chain reaches x>0, y<0 for the RIGHT arm)."""
    p = racket_pos_pelvis(np.zeros(N_JOINTS))
    assert np.all(np.isfinite(p))
    reach = np.linalg.norm(p)
    assert 0.2 < reach < 1.2, reach
    assert p[1] < 0.0, p        # right side of the pelvis


def test_racket_fk_responds_to_arm_joints_only():
    q = np.zeros(N_JOINTS)
    p0 = racket_pos_pelvis(q)
    q_leg = q.copy(); q_leg[0] = 0.7      # a leg/hip joint (not in the chain)
    assert np.allclose(racket_pos_pelvis(q_leg), p0)
    q_arm = q.copy(); q_arm[24] = 0.7     # right elbow (in the chain)
    assert not np.allclose(racket_pos_pelvis(q_arm), p0)
