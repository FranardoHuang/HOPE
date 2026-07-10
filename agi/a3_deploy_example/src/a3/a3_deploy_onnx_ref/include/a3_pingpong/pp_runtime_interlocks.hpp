#pragma once

#include "a3_deploy/strict_finite_math.hpp"

#include <Eigen/Dense>

namespace a3_pingpong {

// Scripted-only phase controls (`0/1/,/.`) are unsafe once a publish-capable MOTION swing is
// active: level changes can race the command callback, while changing swing_speed retroactively
// rescales elapsed time from the old origin and jumps the clip phase. Planner mode rejects these
// keys separately because the planner owns phase/time. SHADOW and process-wide no-publish remain
// useful for diagnostics and may exercise the controls without sending a robot command.
inline bool ScriptedPhaseHotkeyBlocked(bool process_no_publish,
                                       bool motion_mode,
                                       int swing_level) noexcept {
  return !process_no_publish && motion_mode && swing_level == 1;
}

// The synchronous A3 state contract always carries one estimated effort per
// joint.  Missing or non-finite feedback in a publish-capable mode is a state
// fault, not permission to skip the measured-effort half of the safety gate.
inline bool MeasuredEffortFeedbackValid(const Eigen::VectorXd& tau_est,
                                        int expected_dof) noexcept {
  return tau_est.size() == expected_dof && tau_est.array().isFinite().all();
}

}  // namespace a3_pingpong
