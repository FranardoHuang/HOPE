"""Tests for the 180-D obs builder (pure numpy, no ROS / no ONNX)."""

import numpy as np

from hope_wbc_runner.obs_builder import (
    N_JOINTS,
    OBS_DIM,
    RacketTarget,
    build_obs,
    synthetic_state_from_refs,
)


def _refs():
    """Minimal synthetic reference frame (14 tracked bodies, identity quats)."""
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
        pos_w=np.array([0.40, y, 0.88]), vel_w=np.array([2.4, 0.0, 0.8]),
        swing_sign=1.0 if y < 0 else -1.0, time_to_strike=1.2)


def test_obs_is_180_dim():
    refs = _refs()
    default_q = np.zeros(N_JOINTS)
    state = synthetic_state_from_refs(refs, default_q)
    obs = build_obs(refs, state, _target(), np.zeros(N_JOINTS), default_q)
    assert obs.shape == (OBS_DIM,)


def test_trailing_fields_are_tts_and_swing_type():
    refs = _refs()
    default_q = np.zeros(N_JOINTS)
    state = synthetic_state_from_refs(refs, default_q)
    # forehand
    obs = build_obs(refs, state, _target(y=-0.30), np.zeros(N_JOINTS), default_q)
    assert obs[-1] == 1.0          # swing_type forehand
    assert abs(obs[-2] - 1.2) < 1e-9   # time_to_strike
    # backhand
    obs_b = build_obs(refs, state, _target(y=+0.30), np.zeros(N_JOINTS), default_q)
    assert obs_b[-1] == -1.0       # swing_type backhand


def test_command_block_is_ref_joint_pos_vel():
    refs = _refs()
    default_q = np.zeros(N_JOINTS)
    state = synthetic_state_from_refs(refs, default_q)
    obs = build_obs(refs, state, _target(), np.zeros(N_JOINTS), default_q)
    assert np.allclose(obs[:N_JOINTS], refs["joint_pos"])
    assert np.allclose(obs[N_JOINTS:2 * N_JOINTS], refs["joint_vel"])


def test_racket_target_vel_w_passthrough_when_yaw_zero():
    # with identity base orientation, racket_target_vel_w appears unchanged in obs.
    refs = _refs()
    default_q = np.zeros(N_JOINTS)
    state = synthetic_state_from_refs(refs, default_q)
    tgt = _target()
    obs = build_obs(refs, state, tgt, np.zeros(N_JOINTS), default_q)
    # tail layout: base_target_pos_b(2), racket_target_pos_b(3), racket_target_vel_w(3), tts(1), swing(1)
    vel_w = obs[-5:-2]
    assert np.allclose(vel_w, tgt.vel_w)
    racket_pos_b = obs[-8:-5]   # target minus pelvis (yaw identity): (0.4,-0.3, 0.88-0.95)
    assert np.allclose(racket_pos_b, [0.40, -0.30, 0.88 - 0.95])


def test_synthetic_state_tracks_reference():
    refs = _refs()
    default_q = np.zeros(N_JOINTS)
    state = synthetic_state_from_refs(refs, default_q)
    assert np.allclose(state.q, refs["joint_pos"])
    assert np.allclose(state.base_pos_w, refs["body_pos_w"][0])
    assert np.allclose(state.torso_pos_w, refs["body_pos_w"][7])
