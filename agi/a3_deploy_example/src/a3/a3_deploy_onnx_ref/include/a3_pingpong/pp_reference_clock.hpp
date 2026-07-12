// Reference-motion clock. Ported from hope_ws/.../hope_wbc_runner/reference_clock.py.
// Drives the baked-in clip's time_step so the strike frame lands at time_to_strike == 0.
//
// The DEFAULTS below are the LEGACY v1 layout (model_15200: forehand frames [0..94],
// backhand [95..199], phases {0.36,0.50}) and MUST match the clips baked into the loaded
// ONNX. New exports carry clip_seg_lengths/clip_strike_phases metadata and PpPolicy
// OVERRIDES these fields from it at load (e.g. v2-hopex: seg_len {139,132}, phases
// {0.47,0.33}). Do NOT hand-edit per model — bake the metadata instead.
// Strike frame within clip c = seg_start[c] + round(strike_phase[c]*(seg_len[c]-1)).
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

enum class PpPlannerTtsDecision { kTooLate, kWaiting, kEngage };

inline PpPlannerTtsDecision EvaluateExactWindupTts(
    double time_to_strike, double engage_min_tts_s,
    double clip_max_windup_s) {
  if (!std::isfinite(time_to_strike) ||
      !std::isfinite(engage_min_tts_s) || engage_min_tts_s < 0.0 ||
      !std::isfinite(clip_max_windup_s) || clip_max_windup_s <= 0.0) {
    return PpPlannerTtsDecision::kTooLate;
  }
  const double late_cutoff = std::min(engage_min_tts_s, 0.9 * clip_max_windup_s);
  if (time_to_strike < late_cutoff) return PpPlannerTtsDecision::kTooLate;
  if (time_to_strike > clip_max_windup_s) return PpPlannerTtsDecision::kWaiting;
  return PpPlannerTtsDecision::kEngage;
}

// Formal face179/schema-2 commands atomically bind side with position,
// velocity, face normal, and TTS. Legacy actors retain their historical
// base-relative-y inference. Returning false lets the policy stand rather than
// guess if a formal packet somehow reaches it without an explicit +/-1 side.
inline bool resolve_planner_swing_sign(bool require_explicit_side,
                                       bool has_explicit_side,
                                       double explicit_sign,
                                       double target_y,
                                       double split_y,
                                       double hysteresis_y,
                                       double& resolved_sign) {
  if (!require_explicit_side) {
    resolved_sign = swing_sign_from_target_y(target_y);
    return true;
  }
  if (!has_explicit_side || !std::isfinite(explicit_sign) ||
      (explicit_sign != -1.0 && explicit_sign != 1.0) ||
      !std::isfinite(target_y) || !std::isfinite(split_y) ||
      !std::isfinite(hysteresis_y) || hysteresis_y < 0.0) {
    return false;
  }
  // The planner proposes the clip from mocap yaw, while the policy target
  // frame is the runner's boot-yaw-aligned IMU. Outside the shared Schmitt
  // overlap, a proposal inconsistent with the runner's actual observation
  // basis is rejected. Inside [split-h, split+h], either proposal is an
  // explicitly ambiguous overlap and remains subject to target/face gates.
  if (target_y < split_y - hysteresis_y && explicit_sign != 1.0) return false;
  if (target_y > split_y + hysteresis_y && explicit_sign != -1.0) return false;
  resolved_sign = explicit_sign;
  return true;
}

}  // namespace a3_pingpong
