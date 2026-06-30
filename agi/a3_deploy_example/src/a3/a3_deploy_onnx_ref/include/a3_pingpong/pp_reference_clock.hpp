// Reference-motion clock for model_15200. Ported from
// hope_ws/.../hope_wbc_runner/reference_clock.py. Drives the baked-in clip's
// time_step so the strike frame lands at time_to_strike == 0.
//
// Clip layout (must match the ONNX export's clip order):
//   clip 0 = forehand  frames [0 .. 94]
//   clip 1 = backhand  frames [95 .. 199]   (seg_len = {95,105}, total 200)
// Strike frame within clip c = seg_start[c] + round(strike_phase[c]*(seg_len[c]-1)).
// Strike phases: forehand 0.36, backhand 0.50.
#pragma once

#include <algorithm>
#include <cmath>

namespace a3_pingpong {

struct ClipLayout {
  int seg_len[2] = {95, 105};
  double strike_phase[2] = {0.36, 0.50};
  double step_dt = 0.02;  // 50 Hz

  int seg_start(int clip_id) const { return clip_id == 0 ? 0 : seg_len[0]; }
  int total() const { return seg_len[0] + seg_len[1]; }

  int strike_frame(int clip_id) const {
    const int n = seg_len[clip_id];
    return seg_start(clip_id) +
           static_cast<int>(std::lround(strike_phase[clip_id] * (n - 1)));
  }

  // Reference frame index so the clip's strike frame aligns with tts==0,
  // clamped to the clip's [seg_start, seg_end-1] range.
  int time_step_for(int clip_id, double time_to_strike) const {
    const int sf = strike_frame(clip_id);
    const double raw = sf - time_to_strike / std::max(step_dt, 1e-6);
    const int lo = seg_start(clip_id);
    const int hi = seg_start(clip_id) + seg_len[clip_id] - 1;
    int ts = static_cast<int>(std::lround(raw));
    return std::max(lo, std::min(hi, ts));
  }

  double clip_phase(int clip_id, int time_step) const {
    const int lo = seg_start(clip_id);
    const int n = seg_len[clip_id];
    double p = static_cast<double>(time_step - lo) / std::max(n - 1, 1);
    return std::max(0.0, std::min(1.0, p));
  }
};

// +1 forehand (target on -y) / -1 backhand (target on +y).
inline double swing_sign_from_target_y(double target_y) { return target_y < 0.0 ? 1.0 : -1.0; }
inline int clip_id_from_swing_sign(double swing_sign) { return swing_sign > 0.0 ? 0 : 1; }

}  // namespace a3_pingpong
