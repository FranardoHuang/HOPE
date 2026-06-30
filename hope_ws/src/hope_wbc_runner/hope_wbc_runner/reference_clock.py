"""Reference-motion clock (pure logic, ROS-free, testable).

The 180-D observation needs the reference motion frame at a ``time_step``. In
sim2sim the time_step was a free-running clock. In DEPLOYMENT the planner tells
us WHEN the strike happens (``time_to_strike``), so we drive time_step so the
clip's strike frame lands at tts = 0.

Clip layout (must match the ONNX export's clip order):
    clip 0 = forehand (frames seg_start[0] .. seg_start[0]+seg_len[0]-1)
    clip 1 = backhand (frames seg_start[1] .. seg_start[1]+seg_len[1]-1)
For model_15200: seg_len = [95, 105]  ->  seg_start = [0, 95], total T = 200.

Strike frame within a clip = seg_start[c] + round(strike_phase[c] * (seg_len[c]-1)).
Strike phases (controller-side constants, must match training):
    forehand 0.36, backhand 0.50.

time_step(tts) = strike_frame - tts / step_dt   (clamped to the clip's range)
  * tts large  -> time_step near seg_start (wind-up)
  * tts = 0    -> time_step = strike_frame (contact)
  * tts < 0    -> time_step advances into the follow-through, up to seg_end-1
"""

from dataclasses import dataclass


FOREHAND = "forehand"
BACKHAND = "backhand"


@dataclass
class ClipLayout:
    seg_len: tuple = (95, 105)            # (forehand, backhand) frame counts
    strike_phase: tuple = (0.36, 0.50)    # (forehand, backhand) contact phase
    step_dt: float = 0.02                 # 50 Hz control step

    @property
    def seg_start(self):
        s, out = 0, []
        for n in self.seg_len:
            out.append(s)
            s += n
        return tuple(out)

    @property
    def total(self):
        return sum(self.seg_len)

    def strike_frame(self, clip_id):
        n = self.seg_len[clip_id]
        return self.seg_start[clip_id] + int(round(self.strike_phase[clip_id] * (n - 1)))


def swing_sign_from_target_y(target_y: float) -> float:
    """+1 forehand (target on -y) / -1 backhand (target on +y). Matches training's
    swing_type obs field and planner_imitate's Y-sign convention."""
    return 1.0 if target_y < 0.0 else -1.0


def clip_id_from_swing_sign(swing_sign: float) -> int:
    return 0 if swing_sign > 0.0 else 1


def time_step_for(layout: ClipLayout, clip_id: int, time_to_strike: float) -> int:
    """Reference frame index so the clip's strike frame aligns with tts=0.

    Clamped to the clip's [seg_start, seg_end-1] so we never index another clip's
    frames (the wind-up holds at the first frame; the follow-through holds at the
    last frame until the next command).
    """
    sf = layout.strike_frame(clip_id)
    raw = sf - time_to_strike / max(layout.step_dt, 1e-6)
    lo = layout.seg_start[clip_id]
    hi = layout.seg_start[clip_id] + layout.seg_len[clip_id] - 1
    return int(max(lo, min(hi, round(raw))))


def clip_phase(layout: ClipLayout, clip_id: int, time_step: int) -> float:
    """0..1 progress through the clip at this time_step (for logging)."""
    lo = layout.seg_start[clip_id]
    n = layout.seg_len[clip_id]
    return float(max(0.0, min(1.0, (time_step - lo) / max(n - 1, 1))))
