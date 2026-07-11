// Lightweight validator for the formal 179-D face-command ONNX contract.
// Kept independent of ONNX Runtime so unit tests can exercise every semantic
// metadata guard and the planner normal gate without loading a model file.
#pragma once

#include <array>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <locale.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "a3_pingpong/pp_sha256.hpp"

namespace a3_pingpong {

struct PpFaceNormalEnvelopeMetadata {
  std::string schema_version;
  std::string frame;
  std::string face_convention;
  std::string pairing;
  std::string algorithm;
  std::string bank_row_unit_tolerance;
  std::string runtime_unit_tolerance;
  std::string runtime_dot_tolerance;
  std::string clip_order;
  std::string mount_normal_sign_per_clip;
  std::string centers;
  std::string reference_normals;
  std::string min_dots;
  std::string row_counts;
  std::string train_bank_sha256;
  std::string source_family_sha256;
  std::string payload_sha256;
};

struct PpFace179MetadataContract {
  std::string face_command_enabled;
  std::string face_command_pairing;
  std::string face_obs_convention;
  std::string question_bank_exact;
  std::string bank_schema_version;
  std::string bank_split;
  std::string train_bank_sha256;
  std::string source_family_sha256;
  std::string mount_normal_sign_per_clip;
  PpFaceNormalEnvelopeMetadata normal_envelope;
};

struct PpFaceNormalEnvelope {
  std::array<std::array<double, 3>, 2> centers{};
  std::array<std::array<double, 3>, 2> reference_normals{};
  std::array<double, 2> min_dots{};
  std::array<int, 2> row_counts{};
  std::array<double, 2> mount_normal_signs{1.0, -1.0};
  double runtime_unit_tolerance = 0.0;
  double runtime_dot_tolerance = 0.0;
  std::string train_bank_sha256;
  std::string source_family_sha256;
  std::string payload_sha256;

  double Dot(int clip, double x, double y, double z) const {
    if (clip < 0 || clip >= static_cast<int>(centers.size()))
      return -std::numeric_limits<double>::infinity();
    const auto& center = centers[static_cast<std::size_t>(clip)];
    return center[0] * x + center[1] * y + center[2] * z;
  }

  std::array<double, 3> WireBToRawA(int clip, double x, double y, double z) const {
    if (clip < 0 || clip >= static_cast<int>(mount_normal_signs.size()))
      throw std::out_of_range("formal face179 clip id is outside forehand/backhand mapping");
    const double sign = mount_normal_signs[static_cast<std::size_t>(clip)];
    return {sign * x, sign * y, sign * z};
  }

  bool AllowsRawA(int clip, double x, double y, double z) const {
    if (clip < 0 || clip >= static_cast<int>(centers.size()) ||
        !std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
      return false;
    const double norm = std::sqrt(x * x + y * y + z * z);
    if (!std::isfinite(norm) || std::fabs(norm - 1.0) > runtime_unit_tolerance)
      return false;
    const auto& reference = reference_normals[static_cast<std::size_t>(clip)];
    const double reference_dot = reference[0] * x + reference[1] * y + reference[2] * z;
    return reference_dot > runtime_dot_tolerance &&
           Dot(clip, x, y, z) + runtime_dot_tolerance >=
               min_dots[static_cast<std::size_t>(clip)];
  }
};

inline bool PpIsLowerHexSha256(const std::string& s) {
  if (s.size() != 64) return false;
  for (const char c : s)
    if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
  return true;
}

inline std::vector<std::string> PpEnvelopeSplit(const std::string& value, char delimiter) {
  std::vector<std::string> result;
  std::stringstream stream(value);
  std::string token;
  while (std::getline(stream, token, delimiter)) result.push_back(token);
  if (value.empty() || value.back() == delimiter) result.push_back("");
  return result;
}

inline bool PpEnvelopePortableFiniteDouble(const std::string& value, double* output) {
  // libc++ releases used by the macOS source-gate still omit floating-point from_chars. Keep a
  // strict C-locale grammar fallback there; production libstdc++ uses std::from_chars below.
  if (value.empty() || output == nullptr) return false;
  std::size_t i = 0;
  bool negative = false;
  if (value[i] == '-') { negative = true; ++i; }
  if (i == value.size()) return false;
  bool have_digit = false;
  while (i < value.size() && value[i] >= '0' && value[i] <= '9') {
    have_digit = true;
    ++i;
  }
  if (i < value.size() && value[i] == '.') {
    ++i;
    while (i < value.size() && value[i] >= '0' && value[i] <= '9') {
      have_digit = true;
      ++i;
    }
  }
  if (!have_digit) return false;
  int exponent = 0;
  bool exponent_negative = false;
  if (i < value.size() && (value[i] == 'e' || value[i] == 'E')) {
    ++i;
    if (i < value.size() && (value[i] == '+' || value[i] == '-')) {
      exponent_negative = value[i] == '-';
      ++i;
    }
    const std::size_t exponent_begin = i;
    while (i < value.size() && value[i] >= '0' && value[i] <= '9') {
      if (exponent > 10000) return false;
      exponent = exponent * 10 + static_cast<int>(value[i] - '0');
      ++i;
    }
    if (i == exponent_begin) return false;
  }
  if (i != value.size()) return false;
  (void)negative;
  (void)exponent_negative;
  const locale_t c_locale = newlocale(LC_NUMERIC_MASK, "C", nullptr);
  if (c_locale == nullptr) return false;
  errno = 0;
  char* parsed_end = nullptr;
  const double parsed = strtod_l(value.c_str(), &parsed_end, c_locale);
  freelocale(c_locale);
  if (errno == ERANGE || parsed_end != value.c_str() + value.size() || !std::isfinite(parsed))
    return false;
  *output = parsed;
  return true;
}

inline double PpEnvelopeFiniteDouble(const std::string& value, const char* field) {
  double parsed = 0.0;
#if defined(_LIBCPP_VERSION)
  const bool valid = PpEnvelopePortableFiniteDouble(value, &parsed);
#else
  const char* begin = value.data();
  const char* end = begin + value.size();
  const auto result = std::from_chars(begin, end, parsed, std::chars_format::general);
  const bool valid = result.ec == std::errc{} && result.ptr == end && std::isfinite(parsed);
#endif
  if (!valid)
    throw std::invalid_argument(std::string("formal face179 ") + field +
                                " contains a malformed/non-finite value");
  return parsed;
}

inline int PpEnvelopePositiveInt(const std::string& value, const char* field) {
  long parsed = 0;
  const char* begin = value.data();
  const char* end = begin + value.size();
  const auto result = std::from_chars(begin, end, parsed, 10);
  if (result.ec != std::errc{} || result.ptr != end || parsed <= 0 ||
      parsed > std::numeric_limits<int>::max() || value != std::to_string(parsed))
    throw std::invalid_argument(std::string("formal face179 ") + field +
                                " is not a canonical positive integer");
  return static_cast<int>(parsed);
}

inline std::string BuildPpFaceNormalEnvelopePayload(
    const PpFaceNormalEnvelopeMetadata& c) {
  std::ostringstream out;
  out << "stage1_normal_envelope_schema_version=" << c.schema_version << '\n'
      << "stage1_normal_envelope_frame=" << c.frame << '\n'
      << "stage1_normal_envelope_face_convention=" << c.face_convention << '\n'
      << "stage1_normal_envelope_pairing=" << c.pairing << '\n'
      << "stage1_normal_envelope_algorithm=" << c.algorithm << '\n'
      << "stage1_normal_envelope_bank_row_unit_tolerance="
      << c.bank_row_unit_tolerance << '\n'
      << "stage1_normal_envelope_runtime_unit_tolerance="
      << c.runtime_unit_tolerance << '\n'
      << "stage1_normal_envelope_runtime_dot_tolerance="
      << c.runtime_dot_tolerance << '\n'
      << "stage1_normal_envelope_clip_order=" << c.clip_order << '\n'
      << "stage1_normal_envelope_mount_normal_sign_per_clip="
      << c.mount_normal_sign_per_clip << '\n'
      << "stage1_normal_envelope_centers=" << c.centers << '\n'
      << "stage1_normal_envelope_reference_normals=" << c.reference_normals << '\n'
      << "stage1_normal_envelope_min_dots=" << c.min_dots << '\n'
      << "stage1_normal_envelope_row_counts=" << c.row_counts << '\n'
      << "stage1_normal_envelope_train_bank_sha256=" << c.train_bank_sha256 << '\n'
      << "stage1_normal_envelope_source_family_sha256=" << c.source_family_sha256 << '\n';
  return out.str();
}

inline std::array<std::array<double, 3>, 2> PpEnvelopeTwoVectors(
    const std::string& value, const char* field) {
  const auto rows = PpEnvelopeSplit(value, ';');
  if (rows.size() != 2)
    throw std::invalid_argument(std::string("formal face179 ") + field +
                                " must contain exactly two clip vectors");
  std::array<std::array<double, 3>, 2> result{};
  for (std::size_t clip = 0; clip < rows.size(); ++clip) {
    const auto values = PpEnvelopeSplit(rows[clip], ',');
    if (values.size() != 3)
      throw std::invalid_argument(std::string("formal face179 ") + field +
                                  " clip vector must contain exactly three values");
    for (std::size_t axis = 0; axis < values.size(); ++axis)
      result[clip][axis] = PpEnvelopeFiniteDouble(values[axis], field);
  }
  return result;
}

inline PpFaceNormalEnvelope ParsePpFaceNormalEnvelope(
    const PpFaceNormalEnvelopeMetadata& c,
    const std::string& expected_train_bank_sha256,
    const std::string& expected_source_family_sha256) {
  if (c.schema_version != "1" || c.frame != "world_table_frame0" ||
      c.face_convention != "mount_plusY_A" || c.pairing != "shared_plus_y" ||
      c.algorithm != "per_clip_sign_preserving_spherical_mean_cap_v1" ||
      c.bank_row_unit_tolerance != "0.0002" ||
      c.runtime_unit_tolerance != "0.000001" ||
      c.runtime_dot_tolerance != "0.000001" ||
      c.clip_order != "forehand,backhand" ||
      c.mount_normal_sign_per_clip != "1,-1")
    throw std::invalid_argument(
        "formal face179 normal envelope has the wrong schema/frame/convention/algorithm/"
        "tolerance/clip order/sign table");
  if (c.train_bank_sha256 != expected_train_bank_sha256 ||
      c.source_family_sha256 != expected_source_family_sha256)
    throw std::invalid_argument(
        "formal face179 normal envelope is not bound to the ONNX train bank/source family");
  if (!PpIsLowerHexSha256(c.payload_sha256) ||
      PpSha256Hex(BuildPpFaceNormalEnvelopePayload(c)) != c.payload_sha256)
    throw std::invalid_argument(
        "formal face179 normal-envelope payload SHA-256 is missing or inconsistent");

  PpFaceNormalEnvelope result;
  result.centers = PpEnvelopeTwoVectors(c.centers, "normal-envelope centers");
  result.reference_normals =
      PpEnvelopeTwoVectors(c.reference_normals, "normal-envelope reference normals");
  const auto thresholds = PpEnvelopeSplit(c.min_dots, ',');
  const auto counts = PpEnvelopeSplit(c.row_counts, ',');
  if (thresholds.size() != 2 || counts.size() != 2)
    throw std::invalid_argument(
        "formal face179 normal-envelope thresholds/counts must contain exactly two clips");
  result.runtime_unit_tolerance =
      PpEnvelopeFiniteDouble(c.runtime_unit_tolerance, "runtime unit tolerance");
  result.runtime_dot_tolerance =
      PpEnvelopeFiniteDouble(c.runtime_dot_tolerance, "runtime dot tolerance");
  result.mount_normal_signs = {1.0, -1.0};
  result.train_bank_sha256 = c.train_bank_sha256;
  result.source_family_sha256 = c.source_family_sha256;
  result.payload_sha256 = c.payload_sha256;
  const double bank_unit_tolerance =
      PpEnvelopeFiniteDouble(c.bank_row_unit_tolerance, "bank row unit tolerance");
  for (std::size_t clip = 0; clip < 2; ++clip) {
    result.min_dots[clip] =
        PpEnvelopeFiniteDouble(thresholds[clip], "normal-envelope min dots");
    result.row_counts[clip] =
        PpEnvelopePositiveInt(counts[clip], "normal-envelope row counts");
    const auto& center = result.centers[clip];
    const auto& reference = result.reference_normals[clip];
    const double center_norm = std::sqrt(
        center[0] * center[0] + center[1] * center[1] + center[2] * center[2]);
    const double reference_norm = std::sqrt(
        reference[0] * reference[0] + reference[1] * reference[1] +
        reference[2] * reference[2]);
    const double center_reference = center[0] * reference[0] +
                                    center[1] * reference[1] +
                                    center[2] * reference[2];
    if (std::fabs(center_norm - 1.0) > 1e-12 ||
        std::fabs(reference_norm - 1.0) > bank_unit_tolerance ||
        center_reference <= 0.0 || result.min_dots[clip] <= 0.0 ||
        result.min_dots[clip] > 1.0)
      throw std::invalid_argument(
          "formal face179 normal envelope contains a non-unit/flipped/invalid spherical cap");
  }
  return result;
}

inline PpFaceNormalEnvelope ValidatePpFace179MetadataContract(
    const PpFace179MetadataContract& c) {
  if (c.face_command_enabled != "1")
    throw std::invalid_argument(
        "formal face179 ONNX requires face_command_enabled=1");
  if (c.face_command_pairing != "shared_plus_y")
    throw std::invalid_argument(
        "formal face179 ONNX requires face_command_pairing=shared_plus_y");
  if (c.face_obs_convention != "mount_plusY_A")
    throw std::invalid_argument(
        "formal face179 ONNX requires face_obs_convention=mount_plusY_A");
  if (c.question_bank_exact != "1")
    throw std::invalid_argument(
        "formal face179 ONNX requires stage1_question_bank_exact=1");
  if (c.bank_schema_version != "3")
    throw std::invalid_argument(
        "formal face179 ONNX requires stage1_bank_schema_version=3");
  if (c.bank_split != "train")
    throw std::invalid_argument(
        "formal face179 ONNX requires stage1_bank_split=train");
  if (!PpIsLowerHexSha256(c.train_bank_sha256))
    throw std::invalid_argument(
        "formal face179 ONNX requires a lowercase stage1_train_bank_sha256");
  if (!PpIsLowerHexSha256(c.source_family_sha256))
    throw std::invalid_argument(
        "formal face179 ONNX requires a lowercase stage1_source_family_sha256");
  if (c.mount_normal_sign_per_clip != "1,-1")
    throw std::invalid_argument(
        "formal face179 ONNX requires mount_normal_sign_per_clip=1,-1");
  if (c.normal_envelope.mount_normal_sign_per_clip != c.mount_normal_sign_per_clip)
    throw std::invalid_argument(
        "formal face179 normal envelope sign table disagrees with ONNX metadata");
  return ParsePpFaceNormalEnvelope(
      c.normal_envelope, c.train_bank_sha256, c.source_family_sha256);
}

}  // namespace a3_pingpong
