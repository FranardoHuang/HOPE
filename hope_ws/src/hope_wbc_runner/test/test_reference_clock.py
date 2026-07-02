"""Tests for the reference clock (pure, no ROS / no ONNX)."""

from hope_wbc_runner.reference_clock import (
    ClipLayout,
    clip_id_from_swing_sign,
    clip_phase,
    swing_sign_from_target_y,
    time_step_for,
)


def _layout():
    return ClipLayout(seg_len=(95, 105), strike_phase=(0.36, 0.50), step_dt=0.02)


def test_seg_start_and_total():
    L = _layout()
    assert L.seg_start == (0, 95)
    assert L.total == 200


def test_strike_frames_match_training():
    L = _layout()
    assert L.strike_frame(0) == 0 + round(0.36 * 94)     # forehand -> 34
    assert L.strike_frame(1) == 95 + round(0.50 * 104)   # backhand -> 147


def test_swing_sign_and_clip_from_target_y():
    assert swing_sign_from_target_y(-0.30) == 1.0   # forehand on -y
    assert swing_sign_from_target_y(+0.30) == -1.0  # backhand on +y
    assert clip_id_from_swing_sign(1.0) == 0
    assert clip_id_from_swing_sign(-1.0) == 1


def test_time_step_aligns_strike_at_tts_zero():
    L = _layout()
    # forehand: at tts=0 the reference is exactly the strike frame
    assert time_step_for(L, 0, 0.0) == L.strike_frame(0)
    # backhand likewise
    assert time_step_for(L, 1, 0.0) == L.strike_frame(1)


def test_time_step_clamped_to_clip_range():
    L = _layout()
    # huge tts -> wind-up clamps to clip start
    assert time_step_for(L, 0, 100.0) == L.seg_start[0]
    assert time_step_for(L, 1, 100.0) == L.seg_start[1]
    # very negative tts -> follow-through clamps to clip end (never crosses into another clip)
    assert time_step_for(L, 0, -100.0) == L.seg_start[0] + L.seg_len[0] - 1   # 94
    assert time_step_for(L, 1, -100.0) == L.seg_start[1] + L.seg_len[1] - 1   # 199


def test_time_step_monotonic_as_tts_decreases():
    L = _layout()
    seq = [time_step_for(L, 1, tts) for tts in (1.0, 0.5, 0.25, 0.0)]
    assert seq == sorted(seq)   # advances toward the strike frame


def test_clip_phase_bounds():
    L = _layout()
    assert clip_phase(L, 0, L.seg_start[0]) == 0.0
    assert clip_phase(L, 0, L.seg_start[0] + L.seg_len[0] - 1) == 1.0
    assert 0.0 <= clip_phase(L, 1, 147) <= 1.0
