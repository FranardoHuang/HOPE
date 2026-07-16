#include "vrpn_mocap/source_timestamp.hpp"

#include <cassert>
#include <cstdint>
#include <limits>
#include <stdexcept>

using vrpn_mocap::ParseSourceTimestampMode;
using vrpn_mocap::ResolveSourceTimestamp;
using vrpn_mocap::SourceTimestampMode;
using vrpn_mocap::SourceTimestampRejectReason;

int main()
{
  assert(ParseSourceTimestampMode("receipt") == SourceTimestampMode::kReceipt);
  assert(ParseSourceTimestampMode("vrpn_packet") == SourceTimestampMode::kVrpnPacket);
  bool invalid_mode_threw = false;
  try
  {
    (void)ParseSourceTimestampMode("vrpn");
  }
  catch (const std::invalid_argument &)
  {
    invalid_mode_threw = true;
  }
  assert(invalid_mode_threw);

  // Receipt mode preserves the historical path and ignores all packet-only inputs.
  const auto receipt = ResolveSourceTimestamp(
      SourceTimestampMode::kReceipt, -1, -1, 123456789LL,
      std::numeric_limits<double>::quiet_NaN());
  assert(receipt.publish);
  assert(receipt.stamp_nanoseconds == 123456789LL);

  const int64_t now = 1700000000123456000LL;
  const auto exact = ResolveSourceTimestamp(
      SourceTimestampMode::kVrpnPacket, 1700000000LL, 123456LL, now, 0.0);
  assert(exact.publish);
  assert(exact.stamp_nanoseconds == now);
  assert(exact.absolute_skew_nanoseconds == 0);

  const auto old_but_allowed = ResolveSourceTimestamp(
      SourceTimestampMode::kVrpnPacket, 1700000000LL, 100000LL, now, 0.025);
  assert(old_but_allowed.publish);
  assert(old_but_allowed.absolute_skew_nanoseconds == 23456000LL);

  const auto future_but_allowed = ResolveSourceTimestamp(
      SourceTimestampMode::kVrpnPacket, 1700000000LL, 145000LL, now, 0.025);
  assert(future_but_allowed.publish);
  assert(future_but_allowed.absolute_skew_nanoseconds == 21544000LL);

  const auto too_old = ResolveSourceTimestamp(
      SourceTimestampMode::kVrpnPacket, 1700000000LL, 100000LL, now, 0.020);
  assert(!too_old.publish);
  assert(too_old.reject_reason == SourceTimestampRejectReason::kAbsoluteSkewExceeded);

  const auto negative_seconds = ResolveSourceTimestamp(
      SourceTimestampMode::kVrpnPacket, -1, 0, now, 0.1);
  assert(!negative_seconds.publish);
  assert(
      negative_seconds.reject_reason == SourceTimestampRejectReason::kInvalidPacketSeconds);

  for (const int64_t invalid_microseconds : {-1LL, 1000000LL})
  {
    const auto invalid = ResolveSourceTimestamp(
        SourceTimestampMode::kVrpnPacket, 1, invalid_microseconds, now, 0.1);
    assert(!invalid.publish);
    assert(
        invalid.reject_reason ==
        SourceTimestampRejectReason::kInvalidPacketMicroseconds);
  }

  const auto overflow = ResolveSourceTimestamp(
      SourceTimestampMode::kVrpnPacket, std::numeric_limits<int64_t>::max(), 0, now,
      0.1);
  assert(!overflow.publish);
  assert(overflow.reject_reason == SourceTimestampRejectReason::kPacketTimestampOverflow);

  const auto invalid_now = ResolveSourceTimestamp(
      SourceTimestampMode::kVrpnPacket, 1, 0, -1, 0.1);
  assert(!invalid_now.publish);
  assert(invalid_now.reject_reason == SourceTimestampRejectReason::kInvalidReceiptClock);

  for (const double invalid_max : {
           -0.1, std::numeric_limits<double>::infinity(),
           std::numeric_limits<double>::quiet_NaN()})
  {
    const auto invalid = ResolveSourceTimestamp(
        SourceTimestampMode::kVrpnPacket, 1, 0, 1000000000LL, invalid_max);
    assert(!invalid.publish);
    assert(
        invalid.reject_reason == SourceTimestampRejectReason::kInvalidMaxAbsoluteSkew);
  }

  return 0;
}
