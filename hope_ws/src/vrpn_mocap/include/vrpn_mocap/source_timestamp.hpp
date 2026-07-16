// MIT License
//
// Copyright (c) 2022 Alvin Sun
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef VRPN_MOCAP__SOURCE_TIMESTAMP_HPP_
#define VRPN_MOCAP__SOURCE_TIMESTAMP_HPP_

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace vrpn_mocap
{

enum class SourceTimestampMode
{
  kReceipt,
  kVrpnPacket,
};

enum class SourceTimestampRejectReason
{
  kNone,
  kInvalidPacketSeconds,
  kInvalidPacketMicroseconds,
  kPacketTimestampOverflow,
  kInvalidReceiptClock,
  kInvalidMaxAbsoluteSkew,
  kAbsoluteSkewExceeded,
};

struct SourceTimestampDecision
{
  bool publish;
  int64_t stamp_nanoseconds;
  SourceTimestampRejectReason reject_reason;
  int64_t absolute_skew_nanoseconds;
};

inline SourceTimestampMode ParseSourceTimestampMode(const std::string &value)
{
  if (value == "receipt")
  {
    return SourceTimestampMode::kReceipt;
  }
  if (value == "vrpn_packet")
  {
    return SourceTimestampMode::kVrpnPacket;
  }
  throw std::invalid_argument(
      "source_timestamp_mode must be exactly 'receipt' or 'vrpn_packet'");
}

inline const char *SourceTimestampRejectReasonName(SourceTimestampRejectReason reason)
{
  switch (reason)
  {
  case SourceTimestampRejectReason::kNone:
    return "none";
  case SourceTimestampRejectReason::kInvalidPacketSeconds:
    return "invalid_packet_seconds";
  case SourceTimestampRejectReason::kInvalidPacketMicroseconds:
    return "invalid_packet_microseconds";
  case SourceTimestampRejectReason::kPacketTimestampOverflow:
    return "packet_timestamp_overflow";
  case SourceTimestampRejectReason::kInvalidReceiptClock:
    return "invalid_receipt_clock";
  case SourceTimestampRejectReason::kInvalidMaxAbsoluteSkew:
    return "invalid_max_absolute_skew";
  case SourceTimestampRejectReason::kAbsoluteSkewExceeded:
    return "absolute_skew_exceeded";
  }
  return "unknown";
}

inline bool IsValidMaxAbsoluteSkewSeconds(double max_absolute_skew_seconds)
{
  constexpr long double kNanosecondsPerSecond = 1000000000.0L;
  return std::isfinite(max_absolute_skew_seconds) && max_absolute_skew_seconds >= 0.0 &&
         static_cast<long double>(max_absolute_skew_seconds) <=
             static_cast<long double>(std::numeric_limits<int64_t>::max()) /
                 kNanosecondsPerSecond;
}

inline SourceTimestampDecision RejectSourceTimestamp(
    SourceTimestampRejectReason reason, int64_t absolute_skew_nanoseconds = 0)
{
  return SourceTimestampDecision{false, 0, reason, absolute_skew_nanoseconds};
}

/**
 * Resolve the ROS message timestamp without depending on ROS or VRPN types.
 *
 * Receipt mode intentionally ignores the packet fields and the VRPN-only skew
 * setting. This preserves the historical behavior: stamp every accepted sample
 * with the local clock at callback receipt.
 *
 * VRPN-packet mode assumes the VRPN server clock and the ROS clock are already
 * synchronized. It never falls back to receipt time. Malformed, overflowing,
 * or excessively old/future packet timestamps suppress the sample.
 */
inline SourceTimestampDecision ResolveSourceTimestamp(
    SourceTimestampMode mode, int64_t packet_seconds, int64_t packet_microseconds,
    int64_t receipt_now_nanoseconds, double max_absolute_skew_seconds)
{
  if (mode == SourceTimestampMode::kReceipt)
  {
    return SourceTimestampDecision{
        true, receipt_now_nanoseconds, SourceTimestampRejectReason::kNone, 0};
  }

  if (packet_seconds < 0)
  {
    return RejectSourceTimestamp(SourceTimestampRejectReason::kInvalidPacketSeconds);
  }
  if (packet_microseconds < 0 || packet_microseconds >= 1000000)
  {
    return RejectSourceTimestamp(SourceTimestampRejectReason::kInvalidPacketMicroseconds);
  }
  if (receipt_now_nanoseconds < 0)
  {
    return RejectSourceTimestamp(SourceTimestampRejectReason::kInvalidReceiptClock);
  }
  if (!IsValidMaxAbsoluteSkewSeconds(max_absolute_skew_seconds))
  {
    return RejectSourceTimestamp(SourceTimestampRejectReason::kInvalidMaxAbsoluteSkew);
  }

  constexpr int64_t kNanosecondsPerSecond = 1000000000LL;
  constexpr int64_t kNanosecondsPerMicrosecond = 1000LL;
  const int64_t fractional_nanoseconds = packet_microseconds * kNanosecondsPerMicrosecond;
  if (packet_seconds >
      (std::numeric_limits<int64_t>::max() - fractional_nanoseconds) /
          kNanosecondsPerSecond)
  {
    return RejectSourceTimestamp(SourceTimestampRejectReason::kPacketTimestampOverflow);
  }
  const int64_t packet_nanoseconds =
      packet_seconds * kNanosecondsPerSecond + fractional_nanoseconds;
  const int64_t absolute_skew_nanoseconds = packet_nanoseconds >= receipt_now_nanoseconds
                                                ? packet_nanoseconds - receipt_now_nanoseconds
                                                : receipt_now_nanoseconds - packet_nanoseconds;
  const int64_t max_absolute_skew_nanoseconds = static_cast<int64_t>(
      static_cast<long double>(max_absolute_skew_seconds) *
      static_cast<long double>(kNanosecondsPerSecond));
  if (absolute_skew_nanoseconds > max_absolute_skew_nanoseconds)
  {
    return RejectSourceTimestamp(
        SourceTimestampRejectReason::kAbsoluteSkewExceeded, absolute_skew_nanoseconds);
  }
  return SourceTimestampDecision{
      true, packet_nanoseconds, SourceTimestampRejectReason::kNone,
      absolute_skew_nanoseconds};
}

}  // namespace vrpn_mocap

#endif  // VRPN_MOCAP__SOURCE_TIMESTAMP_HPP_
