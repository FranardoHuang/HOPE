#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <limits>
#include <thread>
#include <vector>

#include "a3_pingpong/pp_planner_input.hpp"
#include "a3_pingpong/pp_reference_clock.hpp"
#include "a3_pingpong/pp_task_revision_gate.hpp"

namespace {

std::vector<double> FormalRacket(bool valid, std::uint64_t epoch,
                                 std::uint64_t sequence, double source,
                                 std::uint64_t base_sequence_ref = 1) {
  return {3, valid ? 1.0 : 0.0, valid ? -1.0 : 0.0,
          0.7, -0.4, 0.8, 1.5, 1.4, 0.7, 1.0, 0.0, 0,
          0.6, 0.8, 0.0, 0.0,
          static_cast<double>(epoch), static_cast<double>(sequence),
          static_cast<double>(base_sequence_ref), source};
}

std::vector<double> TaskRacket(bool valid, std::uint64_t epoch,
                               std::uint64_t sequence, double source,
                               std::uint64_t task_id,
                               std::uint64_t task_revision,
                               std::uint64_t base_sequence_ref = 1) {
  auto row = FormalRacket(valid, epoch, sequence, source, base_sequence_ref);
  row[0] = 4.0;
  row.push_back(static_cast<double>(task_id));
  row.push_back(static_cast<double>(task_revision));
  return row;
}

std::vector<double> FormalBase(bool valid, std::uint64_t epoch,
                              std::uint64_t sequence, double source) {
  return {2, valid ? 1.0 : 0.0, 1.0, 2.0, 0.95, 1.0, 0.0, 0.0, 0.0,
          static_cast<double>(epoch), static_cast<double>(sequence), source};
}

a3_pingpong::PpTaskRevisionEnvelope TaskEnvelope(
    std::uint64_t epoch, std::uint64_t task_id, std::uint64_t revision,
    double side = 1.0, int clip = 0) {
  a3_pingpong::PpTaskRevisionEnvelope out;
  out.control_epoch = epoch;
  out.task_id = task_id;
  out.task_revision = revision;
  out.swing_sign = side;
  out.clip_id = clip;
  return out;
}

bool FormalPairEligible(a3_pingpong::PpRacketTargetInput& racket,
                        a3_pingpong::PpBasePoseInput& base) {
  const auto r = racket.Latest();
  a3_pingpong::PpBaseSample b;
  return r.has_valid && !r.invalid_after && r.cmd.has_formal_epoch &&
      base.ExactFormal(
          r.cmd.control_epoch, r.cmd.base_sequence_ref, b, 1.0) &&
      b.has_formal_epoch && r.cmd.control_epoch == b.control_epoch &&
      r.cmd.base_sequence_ref == b.base_sequence;
}

TEST(PpPlannerInput, ExactFloat64CounterBoundaryRejectsFractionAndOverflow) {
  std::uint64_t out = 99;
  EXPECT_TRUE(a3_pingpong::PpParseExactCounter(0.0, out));
  EXPECT_EQ(out, 0u);
  EXPECT_TRUE(a3_pingpong::PpParseExactCounter(
      static_cast<double>(a3_pingpong::kPpMaxExactFloat64Integer), out));
  EXPECT_EQ(out, a3_pingpong::kPpMaxExactFloat64Integer);
  EXPECT_FALSE(a3_pingpong::PpParseExactCounter(-1.0, out));
  EXPECT_FALSE(a3_pingpong::PpParseExactCounter(1.5, out));
  EXPECT_FALSE(a3_pingpong::PpParseExactCounter(
      static_cast<double>(a3_pingpong::kPpMaxExactFloat64Integer) + 1.0, out));
  EXPECT_FALSE(a3_pingpong::PpParseExactCounter(
      std::numeric_limits<double>::infinity(), out));
}

TEST(PpPlannerInput, RacketRejectsWrongSchemaAndNonFiniteValidPayload) {
  a3_pingpong::PpRacketTargetInput input;
  std::vector<double> msg = {2, 1, 1, 0.7, -0.4, 0.8, 1.5, 1.4, 0.7, 1.0, 0.0};
  input.SetFromFlat(msg);
  EXPECT_FALSE(input.Latest().has_valid);

  msg[0] = 1;
  msg[4] = std::numeric_limits<double>::quiet_NaN();
  input.SetFromFlat(msg);
  EXPECT_FALSE(input.Latest().has_valid);

  msg[4] = -0.4;
  input.SetFromFlat(msg);
  const auto snap = input.Latest();
  ASSERT_TRUE(snap.has_valid);
  EXPECT_NEAR(snap.cmd.pos_w[0], 0.7, 1e-12);
  EXPECT_NEAR(snap.cmd.vel_w[1], 1.4, 1e-12);
}

TEST(PpPlannerInput, InvalidRacketMessageRevokesWithoutClockTickSeparation) {
  a3_pingpong::PpRacketTargetInput input;
  const std::vector<double> valid = {1, 1, 1, 0.7, -0.4, 0.8, 1.5, 1.4, 0.7, 1.0, 0.0};
  input.SetFromFlat(valid);
  std::vector<double> invalid = valid;
  invalid[1] = 0;
  invalid[3] = 99.0;
  input.SetFromFlat(invalid);

  const auto snap = input.Latest();
  ASSERT_TRUE(snap.has_valid);
  EXPECT_TRUE(snap.invalid_after);
  EXPECT_NEAR(snap.cmd.pos_w[0], 0.7, 1e-12);
}

TEST(PpPlannerInput, FormalRacketRejectsRecognizedSchemaDowngrade) {
  a3_pingpong::PpRacketTargetInput input;
  const double source = a3_pingpong::PpNowMonotonicSec() - 0.01;
  input.SetFromFlat(FormalRacket(true, 4, 7, source));
  const auto before = input.Latest();
  ASSERT_TRUE(before.has_valid);
  ASSERT_FALSE(before.invalid_after);

  const std::vector<double> legacy_face179 = {
      2, 1, -1, 0.7, -0.4, 0.8, 1.5, 1.4, 0.7, 1.0, 0.0, 0,
      0.6, 0.8, 0.0, 0.0};
  input.SetFromFlat(legacy_face179);
  const auto after = input.Latest();
  ASSERT_TRUE(after.has_valid);
  EXPECT_TRUE(after.invalid_after);
  EXPECT_EQ(after.cmd.control_epoch, 4u);
  EXPECT_GT(after.generation, before.generation);
}

TEST(PpPlannerInput, Face179SchemaBindsUnitNormalAndZeroRhoAtomically) {
  a3_pingpong::PpRacketTargetInput input;
  const std::vector<double> valid = {
      2, 1, -1, 0.7, -0.4, 0.8, 1.5, 1.4, 0.7, 1.0, 0.0, 0,
      0.6, 0.8, 0.0, 0.0};
  input.SetFromFlat(valid);
  const auto snap = input.Latest();
  ASSERT_TRUE(snap.has_valid);
  EXPECT_GE(snap.valid_age_s, 0.0);
  EXPECT_TRUE(snap.cmd.has_face_command);
  EXPECT_TRUE(snap.cmd.has_explicit_side);
  EXPECT_DOUBLE_EQ(snap.cmd.swing_sign, -1.0);
  EXPECT_NEAR(snap.cmd.normal_cmd[0], 0.6, 1e-12);
  EXPECT_NEAR(snap.cmd.normal_cmd[1], 0.8, 1e-12);
  EXPECT_DOUBLE_EQ(snap.cmd.rho, 0.0);
}

TEST(PpPlannerInput, FormalRacketEpochSequenceInvalidAndRecoveryAreCausal) {
  a3_pingpong::PpRacketTargetInput input;
  double source = a3_pingpong::PpNowMonotonicSec() - 0.01;
  input.SetFromFlat(FormalRacket(true, 7, 10, source));
  auto snap = input.Latest();
  ASSERT_TRUE(snap.has_valid);
  EXPECT_TRUE(snap.cmd.has_formal_epoch);
  EXPECT_EQ(snap.cmd.control_epoch, 7u);
  EXPECT_EQ(snap.cmd.command_sequence, 10u);
  EXPECT_EQ(snap.cmd.base_sequence_ref, 1u);
  EXPECT_FALSE(snap.invalid_after);

  source = a3_pingpong::PpNowMonotonicSec() - 0.005;
  input.SetFromFlat(FormalRacket(false, 8, 11, source));
  snap = input.Latest();
  ASSERT_TRUE(snap.has_valid);
  EXPECT_TRUE(snap.invalid_after);

  source = a3_pingpong::PpNowMonotonicSec() - 0.001;
  input.SetFromFlat(FormalRacket(true, 8, 12, source));
  snap = input.Latest();
  ASSERT_TRUE(snap.has_valid);
  EXPECT_FALSE(snap.invalid_after);
  EXPECT_EQ(snap.cmd.control_epoch, 8u);
  EXPECT_EQ(snap.cmd.command_sequence, 12u);

  // Proven-old delayed traffic cannot replace the recovered tuple or refresh age.
  input.SetFromFlat(FormalRacket(true, 7, 11, source - 0.01));
  const auto after_old = input.Latest();
  EXPECT_EQ(after_old.cmd.control_epoch, 8u);
  EXPECT_EQ(after_old.cmd.command_sequence, 12u);
  EXPECT_EQ(after_old.generation, snap.generation);
}

TEST(PpPlannerInput, FormalRacketSameSourceTickInvalidStillRevokes) {
  a3_pingpong::PpRacketTargetInput input;
  const double source = a3_pingpong::PpNowMonotonicSec() - 0.01;
  input.SetFromFlat(FormalRacket(true, 1, 1, source));
  ASSERT_FALSE(input.Latest().invalid_after);

  // A higher sequence with the same source tick is not causally ordered.  It
  // becomes an anonymous revoke; invalid_after is explicit state rather than
  // `last_invalid_wall > last_valid_wall`.
  input.SetFromFlat(FormalRacket(false, 1, 2, source));
  EXPECT_TRUE(input.Latest().invalid_after);
}

TEST(PpPlannerInput, FormalRacketRequiresExactTwentyAndExactBaseSequenceRef) {
  a3_pingpong::PpRacketTargetInput input;
  const double source = a3_pingpong::PpNowMonotonicSec() - 0.01;
  auto old_exact19 = FormalRacket(true, 1, 1, source, 4);
  old_exact19.erase(old_exact19.begin() + 18);
  input.SetFromFlat(old_exact19);
  EXPECT_FALSE(input.Latest().has_valid);

  auto fractional_ref = FormalRacket(true, 1, 2, source + 0.001, 4);
  fractional_ref[18] = 4.5;
  input.SetFromFlat(fractional_ref);
  EXPECT_FALSE(input.Latest().has_valid);

  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  input.SetFromFlat(FormalRacket(
      true, 1, 3, a3_pingpong::PpNowMonotonicSec(),
      a3_pingpong::kPpMaxExactFloat64Integer));
  const auto snap = input.Latest();
  ASSERT_TRUE(snap.has_valid);
  EXPECT_EQ(snap.cmd.base_sequence_ref,
            a3_pingpong::kPpMaxExactFloat64Integer);
}

TEST(PpPlannerInput, TaskSchemaCarriesExactIdentityAfterSchemaThreePrefix) {
  a3_pingpong::PpRacketTargetInput input;
  const double source = a3_pingpong::PpNowMonotonicSec() - 0.01;
  input.SetFromFlat(TaskRacket(true, 7, 11, source, 3, 9, 5));
  const auto snap = input.Latest();
  ASSERT_TRUE(snap.has_valid);
  EXPECT_FALSE(snap.invalid_after);
  EXPECT_TRUE(snap.cmd.has_formal_epoch);
  EXPECT_TRUE(snap.cmd.has_task_contract);
  EXPECT_TRUE(snap.cmd.has_task_identity);
  EXPECT_EQ(snap.cmd.control_epoch, 7u);
  EXPECT_EQ(snap.cmd.command_sequence, 11u);
  EXPECT_EQ(snap.cmd.base_sequence_ref, 5u);
  EXPECT_EQ(snap.cmd.task_id, 3u);
  EXPECT_EQ(snap.cmd.task_revision, 9u);
}

TEST(PpPlannerInput, TaskSchemaRequiresExactTwentyTwoAndValidPositivePair) {
  const double source = a3_pingpong::PpNowMonotonicSec() - 0.02;
  const auto expect_no_valid = [source](std::vector<double> row) {
    a3_pingpong::PpRacketTargetInput input;
    input.SetFromFlat(row);
    EXPECT_FALSE(input.Latest().has_valid);
  };

  auto short_row = TaskRacket(true, 1, 1, source, 1, 1);
  short_row.pop_back();
  expect_no_valid(short_row);

  auto long_row = TaskRacket(true, 1, 1, source, 1, 1);
  long_row.push_back(0.0);
  expect_no_valid(long_row);

  auto fractional_task = TaskRacket(true, 1, 1, source, 1, 1);
  fractional_task[20] = 1.5;
  expect_no_valid(fractional_task);

  auto overflow_revision = TaskRacket(true, 1, 1, source, 1, 1);
  overflow_revision[21] =
      static_cast<double>(a3_pingpong::kPpMaxExactFloat64Integer) + 1.0;
  expect_no_valid(overflow_revision);

  expect_no_valid(TaskRacket(true, 1, 1, source, 0, 0));
  expect_no_valid(TaskRacket(false, 1, 1, source, 0, 1));
  expect_no_valid(TaskRacket(false, 1, 1, source, 1, 0));
}

TEST(PpPlannerInput, TaskSchemaInvalidKeepsPositiveIdentityOrUsesZeroPair) {
  a3_pingpong::PpRacketTargetInput input;
  double source = a3_pingpong::PpNowMonotonicSec() - 0.03;
  input.SetFromFlat(TaskRacket(true, 2, 1, source, 4, 1));
  ASSERT_TRUE(input.Latest().has_valid);
  const auto engaged_revoke_generation =
      input.Latest().revocation_generation;

  source += 0.005;
  input.SetFromFlat(TaskRacket(false, 2, 2, source, 4, 2));
  auto snap = input.Latest();
  ASSERT_TRUE(snap.has_valid);
  EXPECT_TRUE(snap.invalid_after);
  ASSERT_TRUE(snap.has_latest_event);
  EXPECT_FALSE(snap.latest_event.valid);
  EXPECT_TRUE(snap.latest_event.has_task_contract);
  EXPECT_EQ(snap.latest_event.task_id, 4u);
  EXPECT_EQ(snap.latest_event.task_revision, 2u);
  EXPECT_EQ(snap.cmd.task_id, 4u);
  EXPECT_EQ(snap.cmd.task_revision, 1u);
  EXPECT_EQ(snap.revocation_generation, engaged_revoke_generation);

  // A later valid revision of the same task may recover transport validity.
  source += 0.005;
  input.SetFromFlat(TaskRacket(true, 2, 3, source, 4, 3));
  snap = input.Latest();
  ASSERT_TRUE(snap.has_valid);
  EXPECT_FALSE(snap.invalid_after);
  EXPECT_EQ(snap.cmd.task_revision, 3u);
  EXPECT_EQ(snap.revocation_generation, engaged_revoke_generation);

  // Zero/zero is the only anonymous invalid identity and globally revokes.
  source += 0.005;
  input.SetFromFlat(TaskRacket(false, 2, 4, source, 0, 0));
  snap = input.Latest();
  ASSERT_TRUE(snap.has_valid);
  EXPECT_TRUE(snap.invalid_after);
  ASSERT_TRUE(snap.has_latest_event);
  EXPECT_EQ(snap.latest_event.task_id, 0u);
  EXPECT_EQ(snap.latest_event.task_revision, 0u);
  EXPECT_EQ(snap.revocation_generation, engaged_revoke_generation + 1);
}

TEST(PpPlannerInput, GlobalRevokeGenerationSurvivesValidRecoveryBetweenTicks) {
  a3_pingpong::PpRacketTargetInput input;
  double source = a3_pingpong::PpNowMonotonicSec() - 0.01;
  input.SetFromFlat(TaskRacket(true, 7, 1, source, 1, 1));
  const auto engaged = input.Latest();
  ASSERT_TRUE(engaged.has_valid);
  ASSERT_FALSE(engaged.invalid_after);

  source += 0.002;
  input.SetFromFlat(TaskRacket(false, 7, 2, source, 0, 0));
  const auto revoked = input.Latest();
  ASSERT_TRUE(revoked.invalid_after);
  ASSERT_EQ(revoked.revocation_generation,
            engaged.revocation_generation + 1);

  // Simulate latest-value transport coalescing both callbacks before the next
  // policy tick. The latest row is valid again, but the authority-loss edge
  // remains independently observable by the active consumer.
  source += 0.002;
  input.SetFromFlat(TaskRacket(true, 7, 3, source, 2, 1));
  const auto recovered = input.Latest();
  ASSERT_TRUE(recovered.has_valid);
  ASSERT_FALSE(recovered.invalid_after);
  EXPECT_EQ(recovered.cmd.task_id, 2u);
  EXPECT_EQ(recovered.revocation_generation,
            engaged.revocation_generation + 1);
}

TEST(PpPlannerInput, SchemaThreeDowngradeAfterTaskSchemaRevokes) {
  a3_pingpong::PpRacketTargetInput input;
  double source = a3_pingpong::PpNowMonotonicSec() - 0.02;
  input.SetFromFlat(TaskRacket(true, 3, 1, source, 1, 1));
  auto snap = input.Latest();
  ASSERT_TRUE(snap.has_valid);
  ASSERT_FALSE(snap.invalid_after);

  source += 0.005;
  input.SetFromFlat(FormalRacket(true, 3, 2, source));
  snap = input.Latest();
  ASSERT_TRUE(snap.has_valid);
  EXPECT_TRUE(snap.invalid_after);
  EXPECT_TRUE(snap.cmd.has_task_contract);

  // Schema 3 remains a downgrade even with a newer epoch and sequence.
  input.SetFromFlat(FormalRacket(
      true, 4, 100, a3_pingpong::PpNowMonotonicSec()));
  EXPECT_TRUE(input.Latest().invalid_after);

  // Only a causally fresh schema-4 row can recover the established protocol.
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  input.SetFromFlat(TaskRacket(
      true, 4, 101, a3_pingpong::PpNowMonotonicSec(), 1, 1));
  snap = input.Latest();
  ASSERT_TRUE(snap.has_valid);
  EXPECT_FALSE(snap.invalid_after);
  EXPECT_EQ(snap.cmd.control_epoch, 4u);
  EXPECT_EQ(snap.cmd.task_id, 1u);
}

TEST(PpTaskRevisionGate, EngageConsumesOnceAndRevisionsAreStrict) {
  using D = a3_pingpong::PpTaskRevisionDecision;
  a3_pingpong::PpTaskRevisionGate gate;
  const auto first = TaskEnvelope(7, 1, 1, -1.0, 1);
  EXPECT_EQ(gate.TryEngage(first), D::kDisarmed);
  ASSERT_TRUE(gate.Rearm(7));
  EXPECT_EQ(gate.TryEngage(first), D::kEngaged);
  EXPECT_TRUE(gate.active());
  EXPECT_EQ(gate.last_consumed_task_id(), 1u);
  EXPECT_DOUBLE_EQ(gate.frozen_swing_sign(), -1.0);
  EXPECT_EQ(gate.frozen_clip_id(), 1);

  EXPECT_EQ(gate.TryEngage(TaskEnvelope(7, 1, 2, -1.0, 1)),
            D::kAlreadyActive);
  EXPECT_EQ(gate.TryRevision(TaskEnvelope(7, 1, 1, -1.0, 1)),
            D::kOldOrDuplicate);
  EXPECT_EQ(gate.TryRevision(TaskEnvelope(7, 1, 2, -1.0, 1)),
            D::kRevisionAccepted);
  EXPECT_EQ(gate.active_revision(), 2u);
  EXPECT_EQ(gate.TryRevision(TaskEnvelope(7, 1, 2, -1.0, 1)),
            D::kOldOrDuplicate);
}

TEST(PpTaskRevisionGate, SideClipAndTaskStayFrozenWhileActive) {
  using D = a3_pingpong::PpTaskRevisionDecision;
  a3_pingpong::PpTaskRevisionGate gate;
  ASSERT_TRUE(gate.Rearm(7));
  ASSERT_EQ(gate.TryEngage(TaskEnvelope(7, 2, 1, 1.0, 0)), D::kEngaged);
  EXPECT_EQ(gate.TryRevision(TaskEnvelope(7, 2, 2, -1.0, 0)),
            D::kSideOrClipChanged);
  EXPECT_EQ(gate.TryRevision(TaskEnvelope(7, 2, 2, 1.0, 1)),
            D::kSideOrClipChanged);
  EXPECT_EQ(gate.TryRevision(TaskEnvelope(7, 3, 2, 1.0, 0)),
            D::kDifferentTask);
  EXPECT_EQ(gate.active_revision(), 1u);
  EXPECT_DOUBLE_EQ(gate.frozen_swing_sign(), 1.0);
  EXPECT_EQ(gate.frozen_clip_id(), 0);
}

TEST(PpTaskRevisionGate, CompleteNeverMakesOldTaskEngageableAgain) {
  using D = a3_pingpong::PpTaskRevisionDecision;
  a3_pingpong::PpTaskRevisionGate gate;
  ASSERT_TRUE(gate.Rearm(7));
  ASSERT_EQ(gate.TryEngage(TaskEnvelope(7, 4, 1)), D::kEngaged);
  EXPECT_TRUE(gate.Complete(7, 4));
  EXPECT_FALSE(gate.active());
  EXPECT_FALSE(gate.Complete(7, 4));
  EXPECT_EQ(gate.TryEngage(TaskEnvelope(7, 4, 99)), D::kConsumed);
  EXPECT_EQ(gate.TryEngage(TaskEnvelope(7, 3, 1)), D::kConsumed);
  EXPECT_EQ(gate.TryEngage(TaskEnvelope(7, 5, 1)), D::kEngaged);
}

TEST(PpTaskRevisionGate, InvalidRevisionHoldsActiveButGlobalRevokeDisarms) {
  using D = a3_pingpong::PpTaskRevisionDecision;
  a3_pingpong::PpTaskRevisionGate gate;
  ASSERT_TRUE(gate.Rearm(7));
  ASSERT_EQ(gate.TryEngage(TaskEnvelope(7, 1, 1)), D::kEngaged);
  EXPECT_EQ(gate.ObserveInvalid(7, 1, 2), D::kTaskInvalidObserved);
  EXPECT_TRUE(gate.active());
  EXPECT_EQ(gate.active_revision(), 2u);
  EXPECT_EQ(gate.TryRevision(TaskEnvelope(7, 1, 2)), D::kOldOrDuplicate);
  EXPECT_EQ(gate.TryRevision(TaskEnvelope(7, 1, 3)), D::kRevisionAccepted);
  EXPECT_EQ(gate.ObserveInvalid(7, 0, 1), D::kMalformed);
  EXPECT_TRUE(gate.active());
  EXPECT_EQ(gate.ObserveInvalid(7, 0, 0), D::kGlobalRevoke);
  EXPECT_FALSE(gate.active());
  EXPECT_FALSE(gate.armed());
  EXPECT_EQ(gate.TryEngage(TaskEnvelope(7, 2, 1)), D::kDisarmed);
  ASSERT_TRUE(gate.Rearm(7));
  EXPECT_EQ(gate.TryEngage(TaskEnvelope(7, 1, 4)), D::kConsumed);
}

TEST(PpTaskRevisionGate, EpochAdvanceDisarmsAndRegressionCannotMutateState) {
  using D = a3_pingpong::PpTaskRevisionDecision;
  a3_pingpong::PpTaskRevisionGate gate;
  ASSERT_TRUE(gate.Rearm(7));
  ASSERT_EQ(gate.TryEngage(TaskEnvelope(7, 3, 1)), D::kEngaged);
  EXPECT_EQ(gate.TryRevision(TaskEnvelope(8, 3, 2)), D::kDisarmed);
  EXPECT_EQ(gate.control_epoch(), 8u);
  EXPECT_FALSE(gate.armed());
  EXPECT_FALSE(gate.active());
  EXPECT_EQ(gate.TryEngage(TaskEnvelope(8, 1, 1)), D::kDisarmed);
  ASSERT_TRUE(gate.Rearm(8));
  ASSERT_EQ(gate.TryEngage(TaskEnvelope(8, 1, 1)), D::kEngaged);
  EXPECT_EQ(gate.TryRevision(TaskEnvelope(7, 1, 2)), D::kEpochRegressed);
  EXPECT_EQ(gate.control_epoch(), 8u);
  EXPECT_TRUE(gate.active());
  EXPECT_EQ(gate.active_revision(), 1u);
}

TEST(PpTaskRevisionGate, MalformedIdentityAndSideNeverConsume) {
  using D = a3_pingpong::PpTaskRevisionDecision;
  a3_pingpong::PpTaskRevisionGate gate;
  ASSERT_TRUE(gate.Rearm(9));
  EXPECT_EQ(gate.TryEngage(TaskEnvelope(9, 0, 0)), D::kMalformed);
  EXPECT_EQ(gate.TryEngage(TaskEnvelope(9, 1, 1, 0.0, 0)), D::kMalformed);
  EXPECT_EQ(gate.TryEngage(TaskEnvelope(9, 1, 1, 1.0, -1)), D::kMalformed);
  EXPECT_EQ(gate.last_consumed_task_id(), 0u);
  EXPECT_FALSE(gate.active());
  EXPECT_EQ(gate.TryEngage(TaskEnvelope(9, 1, 1)), D::kEngaged);
}

TEST(PpTaskRevisionGate, LatestFutureTaskRevisionCanEngageAfterCurrentCompletes) {
  using D = a3_pingpong::PpTaskRevisionDecision;
  a3_pingpong::PpTaskRevisionGate gate;
  ASSERT_TRUE(gate.Rearm(12));
  ASSERT_EQ(gate.TryEngage(TaskEnvelope(12, 1, 1)), D::kEngaged);

  // A future ball may keep refining while task 1 is still in follow-through;
  // it must not preempt the active physical ball or consume task 2.
  EXPECT_EQ(gate.TryRevision(TaskEnvelope(12, 2, 1)), D::kDifferentTask);
  EXPECT_EQ(gate.TryRevision(TaskEnvelope(12, 2, 2)), D::kDifferentTask);
  ASSERT_TRUE(gate.Complete(12, 1));

  // The latest-value mailbox may now contain only revision 3.  Consumer-side
  // engage accepts that complete fresh snapshot exactly once; producer-side
  // lifecycle tests separately require every new task to originate at rev 1.
  EXPECT_EQ(gate.TryEngage(TaskEnvelope(12, 2, 3)), D::kEngaged);
  ASSERT_TRUE(gate.Complete(12, 2));
  EXPECT_EQ(gate.TryEngage(TaskEnvelope(12, 2, 4)), D::kConsumed);
  EXPECT_EQ(gate.TryEngage(TaskEnvelope(12, 1, 99)), D::kConsumed);
}

TEST(PpPlannerInput, AnonymousMalformedPoisonsPreBarrierDelayedValid) {
  a3_pingpong::PpRacketTargetInput input;
  const double old_source = a3_pingpong::PpNowMonotonicSec() - 0.02;
  input.SetFromFlat(FormalRacket(true, 1, 1, old_source));
  ASSERT_TRUE(input.Latest().has_valid);

  auto malformed = FormalRacket(true, 1, 2, old_source + 0.001);
  malformed[17] = 1.5;  // envelope cannot establish an exact sequence
  input.SetFromFlat(malformed);
  ASSERT_TRUE(input.Latest().invalid_after);

  // This was generated before the anonymous revoke was received, even though
  // its sequence is higher than the last accepted command.
  input.SetFromFlat(FormalRacket(true, 1, 3, old_source + 0.002));
  EXPECT_TRUE(input.Latest().invalid_after);

  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  const double fresh_source = a3_pingpong::PpNowMonotonicSec();
  input.SetFromFlat(FormalRacket(true, 1, 4, fresh_source));
  EXPECT_FALSE(input.Latest().invalid_after);
  EXPECT_EQ(input.Latest().cmd.command_sequence, 4u);
}

TEST(PpPlannerInput, ParsableMalformedBodyAdvancesFormalOrderingAsInvalid) {
  a3_pingpong::PpRacketTargetInput input;
  double source = a3_pingpong::PpNowMonotonicSec() - 0.02;
  input.SetFromFlat(FormalRacket(true, 2, 5, source));
  auto malformed = FormalRacket(true, 2, 6, source + 0.005);
  malformed[3] = std::numeric_limits<double>::quiet_NaN();
  input.SetFromFlat(malformed);
  EXPECT_TRUE(input.Latest().invalid_after);
  // Same envelope is now proven old and cannot become a valid replacement.
  input.SetFromFlat(FormalRacket(true, 2, 6, source + 0.005));
  EXPECT_TRUE(input.Latest().invalid_after);
}

TEST(PpPlannerInput, Face179RejectsMalformedOrNonOpponentFacingCommand) {
  a3_pingpong::PpRacketTargetInput input;
  std::vector<double> msg = {
      2, 1, -1, 0.7, -0.4, 0.8, 1.5, 1.4, 0.7, 1.0, 0.0, 0,
      0.6, 0.8, 0.0, 0.0};
  msg.pop_back();
  input.SetFromFlat(msg);
  EXPECT_FALSE(input.Latest().has_valid);

  msg.push_back(0.0);
  msg[12] = 1.0;
  input.SetFromFlat(msg);
  EXPECT_FALSE(input.Latest().has_valid);

  msg[12] = 0.6;
  msg[15] = 0.1;
  input.SetFromFlat(msg);
  EXPECT_FALSE(input.Latest().has_valid);

  msg[15] = 0.0;
  msg[11] = 1.0;
  input.SetFromFlat(msg);
  EXPECT_FALSE(input.Latest().has_valid);

  msg[11] = 0.0;
  msg[12] = -1.0;
  msg[13] = 0.0;
  input.SetFromFlat(msg);
  EXPECT_FALSE(input.Latest().has_valid);

  msg[12] = 0.0;
  msg[14] = 1.0;
  input.SetFromFlat(msg);
  EXPECT_FALSE(input.Latest().has_valid);

  msg = {2, 1, 0, 0.7, -0.4, 0.8, 1.5, 1.4, 0.7, 1.0, 0.0, 0,
         0.6, 0.8, 0.0, 0.0};
  input.SetFromFlat(msg);
  EXPECT_FALSE(input.Latest().has_valid);
}

TEST(PpPlannerInput, MalformedFace179RevokesEarlierValidCommand) {
  const std::vector<double> valid = {
      2, 1, -1, 0.7, -0.4, 0.8, 1.5, 1.4, 0.7, 1.0, 0.0, 0,
      0.6, 0.8, 0.0, 0.0};
  const auto expect_revoked = [&valid](std::vector<double> malformed) {
    a3_pingpong::PpRacketTargetInput input;
    input.SetFromFlat(valid);
    ASSERT_TRUE(input.Latest().has_valid);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    input.SetFromFlat(malformed);
    const auto snap = input.Latest();
    ASSERT_TRUE(snap.has_valid);
    EXPECT_TRUE(snap.invalid_after);
    EXPECT_NEAR(snap.cmd.normal_cmd[0], 0.6, 1e-12);
  };

  auto wrong_length = valid;
  wrong_length.pop_back();
  expect_revoked(wrong_length);

  auto nonfinite = valid;
  nonfinite[12] = std::numeric_limits<double>::quiet_NaN();
  expect_revoked(nonfinite);

  auto nonunit = valid;
  nonunit[12] = 1.0;
  expect_revoked(nonunit);

  auto nonzero_rho = valid;
  nonzero_rho[15] = 0.1;
  expect_revoked(nonzero_rho);

  auto missing_side = valid;
  missing_side[2] = 0.0;
  expect_revoked(missing_side);

  auto fractional_side = valid;
  fractional_side[2] = 0.5;
  expect_revoked(fractional_side);

  auto unsupported_frame = valid;
  unsupported_frame[11] = 1.0;
  expect_revoked(unsupported_frame);

  auto wrong_face = valid;
  wrong_face[12] = -0.6;
  wrong_face[13] = -0.8;
  expect_revoked(wrong_face);

  auto unknown_schema = valid;
  unknown_schema[0] = 3.0;
  expect_revoked(unknown_schema);

  auto fractional_schema = valid;
  fractional_schema[0] = 1.5;
  expect_revoked(fractional_schema);

  auto nonfinite_schema = valid;
  nonfinite_schema[0] = std::numeric_limits<double>::quiet_NaN();
  expect_revoked(nonfinite_schema);

  expect_revoked({});
}

TEST(PpPlannerInput, HugeFiniteOrFractionalSchemaIsIgnoredWithoutConversion) {
  a3_pingpong::PpRacketTargetInput input;
  std::vector<double> msg = {1e300, 1, 1, 0.7, -0.4, 0.8, 1.5, 1.4, 0.7, 1.0, 0.0};
  input.SetFromFlat(msg);
  EXPECT_FALSE(input.Latest().has_valid);
  msg[0] = 1.5;
  input.SetFromFlat(msg);
  EXPECT_FALSE(input.Latest().has_valid);
}

TEST(PpPlannerInput, FormalSideIsConsumedWhileLegacyKeepsYInference) {
  double sign = 0.0;
  EXPECT_TRUE(a3_pingpong::resolve_planner_swing_sign(
      false, false, 1.0, +0.2, 0.0, 0.04, sign));
  EXPECT_DOUBLE_EQ(sign, -1.0);
  // Inside the shared overlap either planner proposal is accepted.
  EXPECT_TRUE(a3_pingpong::resolve_planner_swing_sign(
      true, true, 1.0, +0.02, 0.0, 0.04, sign));
  EXPECT_DOUBLE_EQ(sign, 1.0);
  // Outside it, the proposal must agree with the runner's actual policy frame.
  EXPECT_FALSE(a3_pingpong::resolve_planner_swing_sign(
      true, true, 1.0, +0.05, 0.0, 0.04, sign));
  EXPECT_TRUE(a3_pingpong::resolve_planner_swing_sign(
      true, true, -1.0, +0.05, 0.0, 0.04, sign));
  EXPECT_DOUBLE_EQ(sign, -1.0);
  EXPECT_TRUE(a3_pingpong::resolve_planner_swing_sign(
      true, true, 1.0, -0.04, 0.0, 0.04, sign));  // exact boundary is overlap
  EXPECT_FALSE(a3_pingpong::resolve_planner_swing_sign(
      true, false, 1.0, -0.2, 0.0, 0.04, sign));
  EXPECT_FALSE(a3_pingpong::resolve_planner_swing_sign(
      true, true, 0.0, -0.2, 0.0, 0.04, sign));
}

TEST(PpPlannerInput, Formal179WaitsForItsSelectedClipWindupWindow) {
  constexpr double fh_max_windup = 1.30;
  constexpr double bh_max_windup = 0.88;
  constexpr double min_tts = 1.00;
  EXPECT_EQ(a3_pingpong::EvaluateExactWindupTts(1.89, min_tts, fh_max_windup),
            a3_pingpong::PpPlannerTtsDecision::kWaiting);
  EXPECT_EQ(a3_pingpong::EvaluateExactWindupTts(1.89, min_tts, bh_max_windup),
            a3_pingpong::PpPlannerTtsDecision::kWaiting);
  EXPECT_EQ(a3_pingpong::EvaluateExactWindupTts(1.30, min_tts, fh_max_windup),
            a3_pingpong::PpPlannerTtsDecision::kEngage);
  EXPECT_EQ(a3_pingpong::EvaluateExactWindupTts(1.30, min_tts, bh_max_windup),
            a3_pingpong::PpPlannerTtsDecision::kWaiting);
  EXPECT_EQ(a3_pingpong::EvaluateExactWindupTts(0.88, min_tts, bh_max_windup),
            a3_pingpong::PpPlannerTtsDecision::kEngage);
  EXPECT_EQ(a3_pingpong::EvaluateExactWindupTts(0.79, min_tts, bh_max_windup),
            a3_pingpong::PpPlannerTtsDecision::kTooLate);
}

TEST(PpPlannerInput, StaleOrRevokedTupleCannotStartAfterWaiting) {
  using D = a3_pingpong::PpPlannerFreshnessDecision;
  EXPECT_EQ(a3_pingpong::EvaluatePpPlannerFreshness(0.51, 0.50, false, 0.10),
            D::kStale);
  EXPECT_EQ(a3_pingpong::EvaluatePpPlannerFreshness(0.20, 0.50, true, 0.10),
            D::kRevoked);
  EXPECT_EQ(a3_pingpong::EvaluatePpPlannerFreshness(0.05, 0.50, true, 0.10),
            D::kFresh);  // existing short invalid-flutter grace
  EXPECT_EQ(a3_pingpong::EvaluatePpPlannerFreshness(
                0.01, 0.50, true, 0.25, true),
            D::kRevoked);  // formal schema 2 never inherits legacy grace
}

TEST(PpPlannerInput, BaseRejectsInvalidPoseAndNormalizesQuaternion) {
  a3_pingpong::PpBasePoseInput input;
  a3_pingpong::PpBaseSample sample;
  std::vector<double> msg = {1, 1, 1.0, 2.0, 0.95, 0.0, 0.0, 0.0, 0.0};
  input.SetFromFlat(msg);
  EXPECT_FALSE(input.Latest(sample, 1.0));

  msg[5] = 2.0;
  input.SetFromFlat(msg);
  ASSERT_TRUE(input.Latest(sample, 1.0));
  EXPECT_NEAR(sample.pos[2], 0.95, 1e-12);
  EXPECT_NEAR(sample.quat.norm(), 1.0, 1e-12);
  EXPECT_NEAR(sample.quat[0], 1.0, 1e-12);

  const double valid_before_revoke = sample.last_invalid_wall_s;
  msg[1] = 0.0;
  input.SetFromFlat(msg);
  EXPECT_FALSE(input.Latest(sample, 1.0));
  msg[1] = 1.0;
  input.SetFromFlat(msg);
  ASSERT_TRUE(input.Latest(sample, 1.0));
  EXPECT_GT(sample.last_invalid_wall_s, valid_before_revoke);
}

TEST(PpPlannerInput, FormalBasePlausibilityRejectsWorkspaceAndSourceTimeJumps) {
  a3_pingpong::PpFormalBasePlausibilityConfig cfg;
  a3_pingpong::PpBaseSample first;
  first.pos = a3_pingpong::Vec3(-0.5, -0.75, 0.9);
  first.quat = a3_pingpong::Vec4(1.0, 0.0, 0.0, 0.0);
  first.has_formal_epoch = true;
  first.source_monotonic_s = 10.0;
  EXPECT_TRUE(a3_pingpong::PpFormalBasePosePlausible(first, cfg));
  auto boundary = first;
  boundary.pos = cfg.min_source;
  EXPECT_TRUE(a3_pingpong::PpFormalBasePosePlausible(boundary, cfg));
  auto absurd = first;
  absurd.pos[0] = 1000.0;
  EXPECT_FALSE(a3_pingpong::PpFormalBasePosePlausible(absurd, cfg));
  auto high = first;
  high.pos[2] = 1.500001;
  EXPECT_FALSE(a3_pingpong::PpFormalBasePosePlausible(high, cfg));

  auto near = first;
  near.source_monotonic_s = 10.01;
  near.pos[0] += 0.12;  // <= 5 cm + 8 m/s * 10 ms
  EXPECT_TRUE(a3_pingpong::PpFormalBaseTransitionPlausible(first, near, cfg));
  auto teleport = near;
  teleport.pos[0] = first.pos[0] + 0.20;
  EXPECT_FALSE(a3_pingpong::PpFormalBaseTransitionPlausible(first, teleport, cfg));
  auto antipodal = near;
  antipodal.pos = first.pos;
  antipodal.quat = -first.quat;
  EXPECT_TRUE(a3_pingpong::PpFormalBaseTransitionPlausible(first, antipodal, cfg));
  auto angular_jump = near;
  angular_jump.pos = first.pos;
  angular_jump.quat = a3_pingpong::Vec4(0.0, 0.0, 0.0, 1.0);
  EXPECT_FALSE(a3_pingpong::PpFormalBaseTransitionPlausible(
      first, angular_jump, cfg));
  auto duplicate_time = near;
  duplicate_time.source_monotonic_s = first.source_monotonic_s;
  EXPECT_FALSE(a3_pingpong::PpFormalBaseTransitionPlausible(
      first, duplicate_time, cfg));
}

TEST(PpPlannerInput, ImplausibleNewBaseRevokesButOldDelayedDoesNotPoisonBaseline) {
  a3_pingpong::PpBasePoseInput base;
  const double t0 = a3_pingpong::PpNowMonotonicSec() - 0.30;
  base.SetFromFlat(FormalBase(true, 1, 1, t0));
  a3_pingpong::PpBaseSample sample;
  ASSERT_TRUE(base.Latest(sample, 1.0));

  // Proven-old delayed garbage is discarded before plausibility and cannot
  // revoke the accepted/latest baseline.
  auto old_absurd = FormalBase(true, 1, 1, t0);
  old_absurd[2] = 1000.0;
  base.SetFromFlat(old_absurd);
  ASSERT_TRUE(base.Latest(sample, 1.0));
  EXPECT_EQ(sample.base_sequence, 1u);

  // A causally new finite teleport is geometry evidence: clear history and
  // revoke. The rejected pose must not become the continuity baseline.
  auto new_absurd = FormalBase(true, 1, 2, t0 + 0.01);
  new_absurd[2] = 1000.0;
  base.SetFromFlat(new_absurd);
  EXPECT_FALSE(base.Latest(sample, 1.0));
  EXPECT_FALSE(base.ExactFormal(1, 1, sample, 1.0));

  // A later physically reachable source recovers against the last accepted
  // good pose, not the rejected 1000 m candidate.
  base.SetFromFlat(FormalBase(true, 1, 3, t0 + 0.20));
  ASSERT_TRUE(base.Latest(sample, 1.0));
  EXPECT_EQ(sample.base_sequence, 3u);
  EXPECT_TRUE(base.PosePlausible(sample));
}

TEST(PpPlannerInput, FormalBaseCarriesEpochAndRejectsOldRecovery) {
  a3_pingpong::PpBasePoseInput input;
  a3_pingpong::PpBaseSample sample;
  double source = a3_pingpong::PpNowMonotonicSec() - 0.01;
  input.SetFromFlat(FormalBase(true, 4, 8, source));
  ASSERT_TRUE(input.Latest(sample, 1.0));
  EXPECT_TRUE(sample.has_formal_epoch);
  EXPECT_EQ(sample.control_epoch, 4u);
  EXPECT_EQ(sample.base_sequence, 8u);

  source = a3_pingpong::PpNowMonotonicSec() - 0.005;
  input.SetFromFlat(FormalBase(false, 5, 9, source));
  EXPECT_FALSE(input.Latest(sample, 1.0));
  input.SetFromFlat(FormalBase(true, 4, 10, source + 0.001));
  EXPECT_FALSE(input.Latest(sample, 1.0));

  // The rejected partial-order event above deliberately installs a
  // receive-time poison barrier. Cross that wall-clock barrier before sending
  // a causally fresh recovery; equality must remain rejected by production.
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  source = a3_pingpong::PpNowMonotonicSec();
  input.SetFromFlat(FormalBase(true, 5, 11, source));
  ASSERT_TRUE(input.Latest(sample, 1.0));
  EXPECT_EQ(sample.control_epoch, 5u);
  EXPECT_EQ(sample.base_sequence, 11u);
}

TEST(PpPlannerInput, FormalBaseRejectsRecognizedSchemaDowngrade) {
  a3_pingpong::PpBasePoseInput input;
  a3_pingpong::PpBaseSample sample;
  const double source = a3_pingpong::PpNowMonotonicSec() - 0.01;
  input.SetFromFlat(FormalBase(true, 2, 3, source));
  ASSERT_TRUE(input.Latest(sample, 1.0));
  const auto revoke_before = sample.revocation_generation;

  input.SetFromFlat({1, 1, 9.0, 9.0, 0.95, 1.0, 0.0, 0.0, 0.0});
  EXPECT_FALSE(input.Latest(sample, 1.0));
  EXPECT_FALSE(input.ExactFormal(2, 3, sample, 1.0));
  EXPECT_EQ(sample.control_epoch, 2u);
  EXPECT_EQ(sample.revocation_generation, revoke_before + 1);
}

TEST(PpPlannerInput, ActiveBaseLeaseRejectsEpochFastRecovery) {
  a3_pingpong::PpBasePoseInput input;
  a3_pingpong::PpBaseSample engaged;
  const double source = a3_pingpong::PpNowMonotonicSec() - 0.03;
  input.SetFromFlat(FormalBase(true, 1, 1, source));
  ASSERT_TRUE(input.Latest(engaged, 1.0));

  input.SetFromFlat(FormalBase(false, 2, 2, source + 0.01));
  input.SetFromFlat(FormalBase(true, 2, 3, a3_pingpong::PpNowMonotonicSec()));
  a3_pingpong::PpBaseSample recovered;
  ASSERT_TRUE(input.Latest(recovered, 1.0));
  EXPECT_EQ(recovered.control_epoch, 2u);
  EXPECT_EQ(recovered.revocation_generation,
            engaged.revocation_generation + 1);
  EXPECT_TRUE(a3_pingpong::PpFormalBaseLeaseChanged(
      recovered, engaged.control_epoch, engaged.revocation_generation));
}

TEST(PpPlannerInput, ActiveBaseLeaseSeesSameEpochHiddenLocalRevoke) {
  a3_pingpong::PpBasePoseInput input;
  a3_pingpong::PpBaseSample engaged;
  const double source = a3_pingpong::PpNowMonotonicSec() - 0.03;
  input.SetFromFlat(FormalBase(true, 5, 1, source));
  ASSERT_TRUE(input.Latest(engaged, 1.0));

  // Anonymous malformed input revokes locally, then a valid same-epoch packet
  // recovers before the next policy sample.  The independent revoke counter
  // preserves the hidden edge.
  input.SetFromFlat({99.0});
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  input.SetFromFlat(FormalBase(true, 5, 2, a3_pingpong::PpNowMonotonicSec()));
  a3_pingpong::PpBaseSample recovered;
  ASSERT_TRUE(input.Latest(recovered, 1.0));
  EXPECT_EQ(recovered.control_epoch, engaged.control_epoch);
  EXPECT_EQ(recovered.revocation_generation,
            engaged.revocation_generation + 1);
  EXPECT_TRUE(a3_pingpong::PpFormalBaseLeaseChanged(
      recovered, engaged.control_epoch, engaged.revocation_generation));
}

TEST(PpPlannerInput, ActiveBaseLeaseAllowsSameEpochNormalRefresh) {
  a3_pingpong::PpBasePoseInput input;
  a3_pingpong::PpBaseSample engaged;
  const double source = a3_pingpong::PpNowMonotonicSec() - 0.02;
  input.SetFromFlat(FormalBase(true, 9, 4, source));
  ASSERT_TRUE(input.Latest(engaged, 1.0));

  input.SetFromFlat(FormalBase(true, 9, 5, a3_pingpong::PpNowMonotonicSec()));
  a3_pingpong::PpBaseSample refreshed;
  ASSERT_TRUE(input.Latest(refreshed, 1.0));
  EXPECT_GT(refreshed.generation, engaged.generation);
  EXPECT_EQ(refreshed.revocation_generation, engaged.revocation_generation);
  EXPECT_FALSE(a3_pingpong::PpFormalBaseLeaseChanged(
      refreshed, engaged.control_epoch, engaged.revocation_generation));
  EXPECT_TRUE(a3_pingpong::PpFormalBaseLeaseUsable(
      refreshed, true, engaged.control_epoch, engaged.revocation_generation));
}

TEST(PpPlannerInput, ActiveBaseLeaseRejectsStaleOrNonFormalCurrentSample) {
  a3_pingpong::PpBasePoseInput input;
  a3_pingpong::PpBaseSample engaged;
  const double source = a3_pingpong::PpNowMonotonicSec() - 0.02;
  input.SetFromFlat(FormalBase(true, 11, 4, source));
  ASSERT_TRUE(input.Latest(engaged, 1.0));

  EXPECT_FALSE(a3_pingpong::PpFormalBaseLeaseUsable(
      engaged, false, engaged.control_epoch, engaged.revocation_generation));
  auto downgraded = engaged;
  downgraded.has_formal_epoch = false;
  EXPECT_FALSE(a3_pingpong::PpFormalBaseLeaseUsable(
      downgraded, true, engaged.control_epoch, engaged.revocation_generation));
}

TEST(PpPlannerInput, SameEpochTupleUsesExactHistoryWithoutLatestBaseStarvation) {
  auto transaction = std::make_shared<std::mutex>();
  a3_pingpong::PpRacketTargetInput racket(transaction);
  a3_pingpong::PpBasePoseInput base(transaction);
  const double t0 = a3_pingpong::PpNowMonotonicSec() - 0.05;

  auto base1 = FormalBase(true, 1, 1, t0);
  base1[2] = 1.1;
  base.SetFromFlat(base1);
  racket.SetFromFlat(FormalRacket(true, 1, 1, t0, 1));
  ASSERT_TRUE(FormalPairEligible(racket, base));

  // A normal newer same-epoch base must not starve a racket that exactly
  // references the still-fresh Bn entry.
  auto base2 = FormalBase(true, 1, 2, t0 + 0.01);
  base2[2] = 1.2;
  base.SetFromFlat(base2);
  EXPECT_TRUE(FormalPairEligible(racket, base));
  a3_pingpong::PpBaseSample exact;
  ASSERT_TRUE(base.ExactFormal(1, 1, exact, 1.0));
  EXPECT_DOUBLE_EQ(exact.pos[0], 1.1);
  a3_pingpong::PpBaseSample latest;
  ASSERT_TRUE(base.Latest(latest, 1.0));
  EXPECT_DOUBLE_EQ(latest.pos[0], 1.2);

  // DDS may deliver a new racket before the base it references. It remains
  // ineligible until that exact base sequence arrives, then matures without
  // depending on cross-topic receive order.
  racket.SetFromFlat(FormalRacket(true, 1, 2, t0 + 0.02, 3));
  EXPECT_FALSE(FormalPairEligible(racket, base));
  base.SetFromFlat(FormalBase(
      true, 1, 3, a3_pingpong::PpNowMonotonicSec()));
  EXPECT_TRUE(FormalPairEligible(racket, base));
}

TEST(PpPlannerInput, FormalHistoryClearsOnRevokeButNotProvenOldEvent) {
  a3_pingpong::PpBasePoseInput base;
  const double t0 = a3_pingpong::PpNowMonotonicSec() - 0.05;
  base.SetFromFlat(FormalBase(true, 1, 1, t0));
  base.SetFromFlat(FormalBase(true, 1, 2, t0 + 0.01));
  a3_pingpong::PpBaseSample exact;
  ASSERT_TRUE(base.ExactFormal(1, 1, exact, 1.0));

  // Proven-old delayed traffic is ignored and cannot clear live history.
  base.SetFromFlat(FormalBase(false, 1, 1, t0));
  EXPECT_TRUE(base.ExactFormal(1, 1, exact, 1.0));

  // A causally new invalid is a revoke and clears every retained valid.
  base.SetFromFlat(FormalBase(false, 1, 3, t0 + 0.02));
  EXPECT_FALSE(base.ExactFormal(1, 1, exact, 1.0));
  EXPECT_FALSE(base.ExactFormal(1, 2, exact, 1.0));

  a3_pingpong::PpBasePoseInput malformed;
  malformed.SetFromFlat(FormalBase(true, 1, 1, t0));
  auto bad_body = FormalBase(true, 1, 2, t0 + 0.01);
  bad_body[2] = std::numeric_limits<double>::quiet_NaN();
  malformed.SetFromFlat(bad_body);
  EXPECT_FALSE(malformed.ExactFormal(1, 1, exact, 1.0));
}

TEST(PpPlannerInput, PoisonedOrderedEnvelopeCannotReuseSequenceWithNewStamp) {
  a3_pingpong::PpBasePoseInput base;
  const double t0 = a3_pingpong::PpNowMonotonicSec() - 0.05;
  base.SetFromFlat(FormalBase(true, 1, 1, t0));
  base.SetFromFlat({99.0});  // anonymous poison barrier and history clear

  // This envelope is structurally newer but its source predates the poison.
  // It must still consume sequence 2 even though its body is revoked.
  base.SetFromFlat(FormalBase(true, 1, 2, t0 + 0.01));
  a3_pingpong::PpBaseSample sample;
  EXPECT_FALSE(base.Latest(sample, 1.0));

  // Reusing sequence 2 with a post-barrier stamp must not revive it.
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  base.SetFromFlat(FormalBase(
      true, 1, 2, a3_pingpong::PpNowMonotonicSec()));
  EXPECT_FALSE(base.Latest(sample, 1.0));
  EXPECT_FALSE(base.ExactFormal(1, 2, sample, 1.0));
}

TEST(PpPlannerInput, ConstantLatencyRacketAndBaseCrossFixedPoisonBarrier) {
  a3_pingpong::PpRacketTargetInput racket;
  double before = a3_pingpong::PpNowMonotonicSec();
  racket.SetFromFlat(FormalRacket(true, 1, 1, before - 0.02));
  before = a3_pingpong::PpNowMonotonicSec();
  racket.SetFromFlat({99.0});
  const double after = a3_pingpong::PpNowMonotonicSec();
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  racket.SetFromFlat(FormalRacket(true, 1, 2, before - 0.005));
  EXPECT_TRUE(racket.Latest().invalid_after);
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  racket.SetFromFlat(FormalRacket(true, 1, 3, after + 0.005));
  EXPECT_FALSE(racket.Latest().invalid_after);
  EXPECT_EQ(racket.Latest().cmd.command_sequence, 3u);

  a3_pingpong::PpBasePoseInput base;
  before = a3_pingpong::PpNowMonotonicSec();
  base.SetFromFlat(FormalBase(true, 1, 1, before - 0.02));
  before = a3_pingpong::PpNowMonotonicSec();
  base.SetFromFlat({99.0});
  const double base_after = a3_pingpong::PpNowMonotonicSec();
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  base.SetFromFlat(FormalBase(true, 1, 2, before - 0.005));
  a3_pingpong::PpBaseSample sample;
  EXPECT_FALSE(base.Latest(sample, 1.0));
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  base.SetFromFlat(FormalBase(true, 1, 3, base_after + 0.005));
  ASSERT_TRUE(base.Latest(sample, 1.0));
  EXPECT_EQ(sample.base_sequence, 3u);
  EXPECT_TRUE(base.ExactFormal(1, 3, sample, 1.0));
}

TEST(PpPlannerInput, FormalHistoryAgingAndBoundedEvictionFailClosed) {
  a3_pingpong::PpBasePoseInput aged;
  const double now = a3_pingpong::PpNowMonotonicSec();
  aged.SetFromFlat(FormalBase(true, 1, 1, now - 0.5));
  a3_pingpong::PpBaseSample exact;
  EXPECT_FALSE(aged.ExactFormal(1, 1, exact, 0.2));
  EXPECT_FALSE(aged.ExactFormal(1, 1, exact,
                               std::numeric_limits<double>::quiet_NaN()));

  a3_pingpong::PpBasePoseInput bounded;
  const double t0 = a3_pingpong::PpNowMonotonicSec() - 0.1;
  for (std::size_t i = 0;
       i <= a3_pingpong::kPpFormalBaseHistoryCapacity; ++i) {
    bounded.SetFromFlat(FormalBase(
        true, 1, static_cast<std::uint64_t>(i + 1),
        t0 + static_cast<double>(i) * 1e-6));
  }
  EXPECT_FALSE(bounded.ExactFormal(1, 1, exact, 1.0));
  EXPECT_TRUE(bounded.ExactFormal(1, 2, exact, 1.0));
}

TEST(PpPlannerInput, AllPerTopicOrderedEpochTwoInterleavingsStayRevokedUntilPair) {
  const std::array<const char*, 6> interleavings = {
      "RRBB", "RBRB", "RBBR", "BRRB", "BRBR", "BBRR"};
  for (const char* order : interleavings) {
    auto transaction = std::make_shared<std::mutex>();
    a3_pingpong::PpRacketTargetInput racket(transaction);
    a3_pingpong::PpBasePoseInput base(transaction);
    const double t0 = a3_pingpong::PpNowMonotonicSec() - 0.05;
    racket.SetFromFlat(FormalRacket(true, 1, 1, t0));
    base.SetFromFlat(FormalBase(true, 1, 1, t0));
    ASSERT_TRUE(FormalPairEligible(racket, base));
    int r_index = 0, b_index = 0;
    for (int i = 0; i < 4; ++i) {
      if (order[i] == 'R') {
        ++r_index;
        racket.SetFromFlat(FormalRacket(
            r_index == 2, 2, 1 + r_index, t0 + 0.01 * r_index,
            r_index == 2 ? 3 : 2));
      } else {
        ++b_index;
        base.SetFromFlat(FormalBase(
            b_index == 2, 2, 1 + b_index, t0 + 0.01 * b_index));
      }
      if (i < 3) EXPECT_FALSE(FormalPairEligible(racket, base)) << order << " step " << i;
    }
    EXPECT_TRUE(FormalPairEligible(racket, base)) << order;
  }
}

TEST(PpPlannerInput, BaseRecoveryCannotResurrectDelayedOldRacketTuple) {
  auto transaction = std::make_shared<std::mutex>();
  a3_pingpong::PpRacketTargetInput racket(transaction);
  a3_pingpong::PpBasePoseInput base(transaction);
  const double t0 = a3_pingpong::PpNowMonotonicSec() - 0.05;
  racket.SetFromFlat(FormalRacket(true, 1, 1, t0));
  base.SetFromFlat(FormalBase(true, 1, 1, t0));
  ASSERT_TRUE(FormalPairEligible(racket, base));
  base.SetFromFlat(FormalBase(false, 2, 2, t0 + 0.01));
  EXPECT_FALSE(FormalPairEligible(racket, base));
  base.SetFromFlat(FormalBase(true, 2, 3, t0 + 0.02));
  EXPECT_FALSE(FormalPairEligible(racket, base));
  racket.SetFromFlat(FormalRacket(true, 1, 2, t0 + 0.01, 1));
  EXPECT_FALSE(FormalPairEligible(racket, base));
  racket.SetFromFlat(FormalRacket(false, 2, 3, t0 + 0.02, 2));
  EXPECT_FALSE(FormalPairEligible(racket, base));
  racket.SetFromFlat(FormalRacket(
      true, 2, 4, a3_pingpong::PpNowMonotonicSec(), 3));
  EXPECT_TRUE(FormalPairEligible(racket, base));
}

TEST(PpPlannerInput, TryCommitIfUnchangedLinearizesBothMailboxes) {
  auto transaction = std::make_shared<std::mutex>();
  a3_pingpong::PpRacketTargetInput racket(transaction);
  a3_pingpong::PpBasePoseInput base(transaction);
  // Use one safely historical source-time origin for the whole test. Every
  // formal event below has a deterministic positive offset from this origin,
  // independent of host monotonic-clock resolution and test execution speed.
  double source = a3_pingpong::PpNowMonotonicSec() - 0.20;
  racket.SetFromFlat(FormalRacket(true, 1, 1, source));
  base.SetFromFlat(FormalBase(true, 1, 1, source));
  auto rs = racket.Latest();
  a3_pingpong::PpBaseSample bs;
  ASSERT_TRUE(base.Latest(bs, 1.0));

  racket.SetFromFlat(FormalRacket(false, 2, 2, source + 0.005));
  bool called = false;
  EXPECT_FALSE(a3_pingpong::PpWithPlannerInputsIfUnchanged(
      racket, rs.generation, base, rs.cmd.control_epoch,
      rs.cmd.base_sequence_ref, 1.0,
      [&](const a3_pingpong::PpBaseSample&,
          const a3_pingpong::PpBaseSample&) { called = true; return true; }));
  EXPECT_FALSE(called);

  // This case tests generation linearization, not clock resolution. Keep the
  // recovery and refresh strictly newer than every event above while still
  // safely in the past.
  source += 0.05;
  racket.SetFromFlat(FormalRacket(true, 2, 3, source, 2));
  base.SetFromFlat(FormalBase(true, 2, 2, source));
  rs = racket.Latest();
  ASSERT_TRUE(base.Latest(bs, 1.0));
  // A normal high-rate refresh after capture advances latest generation but
  // retains the exact referenced base, so commit must not starve.
  base.SetFromFlat(FormalBase(true, 2, 3, source + 0.01));
  ASSERT_TRUE(racket.GenerationCurrent(rs.generation));
  a3_pingpong::PpBaseSample exact_before_commit;
  ASSERT_TRUE(base.ExactFormal(
      rs.cmd.control_epoch, rs.cmd.base_sequence_ref,
      exact_before_commit, 1.0));
  EXPECT_EQ(exact_before_commit.base_sequence, 2u);
  a3_pingpong::PpBaseSample latest_before_commit;
  ASSERT_TRUE(base.Latest(latest_before_commit, 1.0));
  EXPECT_EQ(latest_before_commit.base_sequence, 3u);
  EXPECT_TRUE(a3_pingpong::PpWithPlannerInputsIfUnchanged(
      racket, rs.generation, base, rs.cmd.control_epoch,
      rs.cmd.base_sequence_ref, 1.0,
      [&](const a3_pingpong::PpBaseSample& exact,
          const a3_pingpong::PpBaseSample& latest) {
        called = true;
        return exact.base_sequence == 2 && latest.base_sequence == 3;
      }));
  EXPECT_TRUE(called);

  // A sampled revoke clears history even if the racket generation is stable.
  base.SetFromFlat(FormalBase(
      false, 2, 4, a3_pingpong::PpNowMonotonicSec()));
  called = false;
  EXPECT_FALSE(a3_pingpong::PpWithPlannerInputsIfUnchanged(
      racket, rs.generation, base, rs.cmd.control_epoch,
      rs.cmd.base_sequence_ref, 1.0,
      [&](const a3_pingpong::PpBaseSample&,
          const a3_pingpong::PpBaseSample&) { called = true; return true; }));
  EXPECT_FALSE(called);
}

}  // namespace
