#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>

#include "a3_pingpong/pp_phase_governor.hpp"

namespace {

a3_pingpong::PpPhaseGovernorProfile Profile() {
  a3_pingpong::PpPhaseGovernorProfile p;
  p.contract_version = "phase_governor_v1";
  p.schema_version = 1;
  p.profile_sha256 = std::string(64, 'a');
  p.policy_dt_s = 0.02;
  p.min_tts_s = 0.1;
  p.max_tts_s = 2.0;
  p.max_phase_rate_per_s = 4.0;
  p.max_phase_acceleration_per_s2 = 20.0;
  p.max_deadline_revision_delta_s = 0.25;
  p.max_position_revision_delta_m = 0.1;
  p.max_velocity_revision_delta_mps = 0.5;
  p.max_normal_revision_delta_rad = 0.2;
  p.normal_unit_tolerance = 1e-4;
  p.early_deadline_tolerance_s = 1e-9;
  return p;
}

a3_pingpong::PpPhaseRevision Revision(std::uint64_t revision = 1,
                                      double tts = 0.8) {
  a3_pingpong::PpPhaseRevision r;
  r.control_epoch = 7;
  r.task_id = 10;
  r.task_revision = revision;
  r.command_sequence = 99 + revision;
  r.source_monotonic_s = 5.0 + 0.1 * revision;
  r.target_position_m = a3_pingpong::Vec3(0.2, -0.1, 0.9);
  r.target_velocity_mps = a3_pingpong::Vec3(-2.0, 0.0, -0.2);
  r.target_normal = a3_pingpong::Vec3(1.0, 0.0, 0.0);
  r.desired_tts_s = tts;
  return r;
}

TEST(PpPhaseGovernor, ModelAndRuntimeTaskRevisionAreDoubleKeyed) {
  EXPECT_NO_THROW(a3_pingpong::RequirePpPlannerTaskRevisionDoubleKey(false, false));
  EXPECT_NO_THROW(a3_pingpong::RequirePpPlannerTaskRevisionDoubleKey(true, true));
  EXPECT_THROW(a3_pingpong::RequirePpPlannerTaskRevisionDoubleKey(true, false),
               std::runtime_error);
  EXPECT_THROW(a3_pingpong::RequirePpPlannerTaskRevisionDoubleKey(false, true),
               std::runtime_error);
}

TEST(PpPhaseGovernor, PhaseRateAndAccelerationAreBoundedAndMonotonic) {
  const auto profile = Profile();
  a3_pingpong::PpPhaseGovernor governor(profile);
  ASSERT_EQ(governor.Begin(Revision(), 20.0),
            a3_pingpong::PpPhaseDecision::kAccepted);
  double prior_phase = 0.0;
  double prior_rate = 0.0;
  for (int i = 0; i < 150; ++i) {
    governor.Advance();
    EXPECT_GE(governor.phase(), prior_phase);
    EXPECT_GE(governor.phase_rate_per_s(), 0.0);
    EXPECT_LE(governor.phase_rate_per_s(), profile.max_phase_rate_per_s);
    EXPECT_LE(std::fabs(governor.phase_rate_per_s() - prior_rate),
              profile.max_phase_acceleration_per_s2 * profile.policy_dt_s + 1e-12);
    prior_phase = governor.phase();
    prior_rate = governor.phase_rate_per_s();
  }
  EXPECT_DOUBLE_EQ(governor.phase(), 1.0);
}

TEST(PpPhaseGovernor, SameTaskRevisionIsAtomicAndEnvelopeBound) {
  a3_pingpong::PpPhaseGovernor governor(Profile());
  ASSERT_EQ(governor.Begin(Revision(), 20.0),
            a3_pingpong::PpPhaseDecision::kAccepted);
  for (int i = 0; i < 10; ++i) governor.Advance();
  const double phase = governor.phase();

  auto accepted = Revision(2, governor.remaining_tts_s() - 0.05);
  accepted.target_position_m[0] += 0.04;
  accepted.target_velocity_mps[0] += 0.2;
  accepted.target_normal = a3_pingpong::Vec3(std::cos(0.1), std::sin(0.1), 0.0);
  EXPECT_EQ(governor.Revise(accepted), a3_pingpong::PpPhaseDecision::kAccepted);
  EXPECT_DOUBLE_EQ(governor.phase(), phase);
  EXPECT_EQ(governor.visible_revision().task_revision, 2u);

  auto rejected = Revision(3, governor.remaining_tts_s());
  rejected.target_position_m[0] += 0.20;
  EXPECT_EQ(governor.Revise(rejected),
            a3_pingpong::PpPhaseDecision::kTargetEnvelope);
  EXPECT_EQ(governor.visible_revision().task_revision, 2u);
  EXPECT_DOUBLE_EQ(governor.phase(), phase);
}

TEST(PpPhaseGovernor, TaskBaselineEnvelopeIsLatestValueSafeAndCannotRatchet) {
  a3_pingpong::PpPhaseGovernor governor(Profile());
  ASSERT_EQ(governor.Begin(Revision(), 20.0),
            a3_pingpong::PpPhaseDecision::kAccepted);

  auto edge_a = Revision(2);
  edge_a.target_position_m[0] = 0.285;
  edge_a.target_velocity_mps[0] = -1.55;
  edge_a.target_normal =
      a3_pingpong::Vec3(std::cos(0.19), std::sin(0.19), 0.0);
  ASSERT_EQ(governor.Revise(edge_a),
            a3_pingpong::PpPhaseDecision::kAccepted);

  // A latest-value mailbox may skip revision 3. Revision 4 is at the
  // opposite edge of the begin-task ball: every visible-to-visible delta is
  // larger than the configured gate, but the task-wide tuple is still legal.
  auto edge_b = Revision(4);
  edge_b.target_position_m[0] = 0.115;
  edge_b.target_velocity_mps[0] = -2.45;
  edge_b.target_normal =
      a3_pingpong::Vec3(std::cos(-0.19), std::sin(-0.19), 0.0);
  ASSERT_EQ(governor.Revise(edge_b),
            a3_pingpong::PpPhaseDecision::kAccepted);
  EXPECT_EQ(governor.visible_revision().task_revision, 4u);

  // A small step from the latest visible tuple cannot ratchet the actor
  // outside the immutable begin-task ball.
  auto outside = Revision(5);
  outside.target_position_m[0] = 0.03;
  EXPECT_EQ(governor.Revise(outside),
            a3_pingpong::PpPhaseDecision::kTargetEnvelope);
  EXPECT_EQ(governor.visible_revision().task_revision, 4u);
}

TEST(PpPhaseGovernor, DeadlineEnvelopeUsesBeginTaskAbsoluteDeadline) {
  a3_pingpong::PpPhaseGovernor governor(Profile());
  ASSERT_EQ(governor.Begin(Revision(), 20.0),
            a3_pingpong::PpPhaseDecision::kAccepted);
  for (int i = 0; i < 10; ++i) governor.Advance();
  const double baseline_deadline = 20.8;

  auto later = Revision(2, baseline_deadline + 0.20 - governor.local_monotonic_s());
  ASSERT_EQ(governor.Revise(later),
            a3_pingpong::PpPhaseDecision::kAccepted);

  // Skip revision 3 and cross to the opposite edge. Although this moves 0.40
  // seconds from the current visible deadline, it remains within +/-0.25 s of
  // the immutable begin-task deadline and must therefore be accepted.
  auto earlier = Revision(4, baseline_deadline - 0.20 - governor.local_monotonic_s());
  ASSERT_EQ(governor.Revise(earlier),
            a3_pingpong::PpPhaseDecision::kAccepted);

  auto outside = Revision(5, baseline_deadline - 0.26 - governor.local_monotonic_s());
  EXPECT_EQ(governor.Revise(outside),
            a3_pingpong::PpPhaseDecision::kDeadlineEnvelope);
  EXPECT_EQ(governor.visible_revision().task_revision, 4u);
}

TEST(PpPhaseGovernor, ConsumerCanBeginFromLatestPositiveRevision) {
  a3_pingpong::PpPhaseGovernor governor(Profile());
  EXPECT_EQ(governor.Begin(Revision(3), 20.0),
            a3_pingpong::PpPhaseDecision::kMalformed);
  EXPECT_EQ(governor.BeginConsumerSnapshot(Revision(3), 20.0),
            a3_pingpong::PpPhaseDecision::kAccepted);
  EXPECT_TRUE(governor.active());
  EXPECT_EQ(governor.visible_revision().task_revision, 3u);
}

TEST(PpPhaseGovernor, LaterDeadlineOnlySlowsNextStepAndImpossibleEarlyRejects) {
  auto profile = Profile();
  // Isolate the dynamic-reachability gate from the independent absolute
  // begin-deadline envelope so this assertion cannot be masked by precedence.
  profile.max_deadline_revision_delta_s = 2.0;
  a3_pingpong::PpPhaseGovernor governor(profile);
  ASSERT_EQ(governor.Begin(Revision(), 20.0),
            a3_pingpong::PpPhaseDecision::kAccepted);
  for (int i = 0; i < 10; ++i) governor.Advance();
  const double prior_rate = governor.phase_rate_per_s();
  EXPECT_EQ(governor.Revise(Revision(2, governor.remaining_tts_s() + 0.10)),
            a3_pingpong::PpPhaseDecision::kAccepted);
  governor.Advance();
  EXPECT_LE(governor.phase_rate_per_s(), prior_rate);

  auto impossible = Revision(3, 0.1);
  EXPECT_EQ(governor.Revise(impossible),
            a3_pingpong::PpPhaseDecision::kUnreachableDeadline);
  EXPECT_EQ(governor.visible_revision().task_revision, 2u);
}

TEST(PpPhaseGovernor, UrgentDeadlineGoldenTracePointFourSeconds) {
  a3_pingpong::PpPhaseGovernor governor(Profile());
  ASSERT_EQ(governor.Begin(Revision(1, 0.4), 20.0),
            a3_pingpong::PpPhaseDecision::kAccepted);
  for (int tick = 1; tick <= 19; ++tick) {
    governor.Advance();
    if (tick == 1) {
      EXPECT_NEAR(governor.phase(), 0.004, 1e-12);
      EXPECT_NEAR(governor.phase_rate_per_s(), 0.4, 1e-12);
    } else if (tick == 10) {
      EXPECT_NEAR(governor.phase(), 0.3895454545454533, 1e-12);
      EXPECT_NEAR(governor.phase_rate_per_s(), 3.3909090909090445, 1e-12);
    } else if (tick == 13) {
      EXPECT_NEAR(governor.phase(), 0.5970261994949448, 1e-12);
      EXPECT_NEAR(governor.phase_rate_per_s(), 3.791445707070643, 1e-12);
    }
  }
  EXPECT_NEAR(governor.phase(), 0.92, 1e-12);
  EXPECT_NEAR(governor.phase_rate_per_s(), 4.0, 1e-12);
  governor.Advance();
  EXPECT_DOUBLE_EQ(governor.phase(), 1.0);
}

TEST(PpPhaseGovernor, UrgentDeadlineGoldenTracePointFiveSeconds) {
  a3_pingpong::PpPhaseGovernor governor(Profile());
  ASSERT_EQ(governor.Begin(Revision(1, 0.5), 20.0),
            a3_pingpong::PpPhaseDecision::kAccepted);
  for (int tick = 1; tick <= 24; ++tick) {
    governor.Advance();
    if (tick == 20) {
      EXPECT_NEAR(governor.phase(), 0.809682744022909, 1e-12);
      EXPECT_NEAR(governor.phase_rate_per_s(), 2.378965699713386, 1e-12);
    } else if (tick == 21) {
      // The urgent-deadline branch asks for max rate here; acceleration still
      // bounds the realized step to +0.4 phase/s.
      EXPECT_NEAR(governor.phase(), 0.8612620580171767, 1e-12);
      EXPECT_NEAR(governor.phase_rate_per_s(), 2.778965699713386, 1e-12);
    }
  }
  EXPECT_NEAR(governor.phase(), 0.9244206860057322, 1e-12);
  EXPECT_NEAR(governor.phase_rate_per_s(), 3.9789656997133855, 1e-12);
  governor.Advance();
  EXPECT_DOUBLE_EQ(governor.phase(), 1.0);
}

TEST(PpPhaseGovernor, FinalPolicyIntervalAcceptsRevisionThenClosesAtContact) {
  auto profile = Profile();
  profile.min_tts_s = 0.02;
  profile.early_deadline_tolerance_s = 1e-6;
  a3_pingpong::PpPhaseGovernor governor(profile);
  ASSERT_EQ(governor.Begin(Revision(1, 0.5), 20.0),
            a3_pingpong::PpPhaseDecision::kAccepted);
  std::uint64_t revision = 1;
  std::size_t accepted = 0;
  double float32_truth_tts = static_cast<float>(0.5);
  for (int tick = 1; tick <= 24; ++tick) {
    governor.Advance();
    float32_truth_tts = static_cast<float>(
        static_cast<float>(float32_truth_tts) - static_cast<float>(0.02));
    if (tick >= 20) {
      ++revision;
      auto update = Revision(revision, float32_truth_tts);
      EXPECT_EQ(governor.Revise(update),
                a3_pingpong::PpPhaseDecision::kAccepted);
      ++accepted;
    }
  }
  EXPECT_EQ(accepted, 5u);
  EXPECT_LT(governor.phase(), 1.0);
  EXPECT_EQ(governor.visible_revision().task_revision, 6u);
  EXPECT_DOUBLE_EQ(governor.visible_revision().desired_tts_s, 0.02);

  governor.Advance();
  EXPECT_DOUBLE_EQ(governor.phase(), 1.0);
  EXPECT_DOUBLE_EQ(governor.remaining_tts_s(), 0.0);
  governor.Advance();
  EXPECT_DOUBLE_EQ(governor.remaining_tts_s(), 0.0);
  EXPECT_EQ(governor.Revise(Revision(7, 0.02)),
            a3_pingpong::PpPhaseDecision::kPostContact);
  EXPECT_EQ(governor.visible_revision().task_revision, 6u);
}

TEST(PpPhaseGovernor, TtsToleranceAcceptsOnlyTheBoundEdgeAndCanonicalizesIt) {
  auto profile = Profile();
  profile.min_tts_s = 0.02;
  profile.early_deadline_tolerance_s = 1e-6;
  profile.max_phase_rate_per_s = 100.0;
  profile.max_phase_acceleration_per_s2 = 10000.0;
  a3_pingpong::PpPhaseGovernor accepted(profile);
  EXPECT_EQ(accepted.Begin(Revision(1, 0.02 - 0.5e-6), 20.0),
            a3_pingpong::PpPhaseDecision::kAccepted);
  EXPECT_DOUBLE_EQ(accepted.visible_revision().desired_tts_s, 0.02);

  a3_pingpong::PpPhaseGovernor rejected(profile);
  EXPECT_EQ(rejected.Begin(Revision(1, 0.02 - 2.0e-6), 20.0),
            a3_pingpong::PpPhaseDecision::kMalformed);
}

}  // namespace
