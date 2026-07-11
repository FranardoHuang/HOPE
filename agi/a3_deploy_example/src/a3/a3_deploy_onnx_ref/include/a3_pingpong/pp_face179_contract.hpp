// Lightweight validator for the formal 179-D face-command ONNX contract.
// Kept independent of ONNX Runtime so unit tests can exercise every semantic
// metadata guard and the planner normal gate without loading a model file.
#pragma once

#include <array>
#include <cmath>
#include <limits>
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
  PpFaceNormalEnvelopeMetadata normal_envelope;
};

struct PpFaceNormalEnvelope {
  std::array<std::array<double, 3>, 2> centers{};
  std::array<std::array<double, 3>, 2> reference_normals{};
  std::array<double, 2> min_dots{};
  std::array<int, 2> row_counts{};
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

  bool Allows(int clip, double x, double y, double z) const {
    if (clip < 0 || clip >= static_cast<int>(centers.size()) ||
        !std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
      return false;
    const double norm = std::sqrt(x * x + y * y + z * z);
    if (!std::isfinite(norm) || std::fabs(norm - 1.0) > runtime_unit_tolerance)
      return false;
    return Dot(clip, x, y, z) + runtime_dot_tolerance >=
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

inline double PpEnvelopeFiniteDouble(const std::string& value, const char* field) {
  std::size_t consumed = 0;
  double parsed = 0.0;
  try {
    parsed = std::stod(value, &consumed);
  } catch (const std::exception&) {
    throw std::invalid_argument(std::string("formal face179 ") + field +
                                " contains a non-numeric value");
  }
  if (consumed != value.size() || !std::isfinite(parsed))
    throw std::invalid_argument(std::string("formal face179 ") + field +
                                " contains a malformed/non-finite value");
  return parsed;
}

inline int PpEnvelopePositiveInt(const std::string& value, const char* field) {
  std::size_t consumed = 0;
  long parsed = 0;
  try {
    parsed = std::stol(value, &consumed, 10);
  } catch (const std::exception&) {
    throw std::invalid_argument(std::string("formal face179 ") + field +
                                " is not a canonical positive integer");
  }
  if (consumed != value.size() || parsed <= 0 ||
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
      c.clip_order != "forehand,backhand")
    throw std::invalid_argument(
        "formal face179 normal envelope has the wrong schema/frame/convention/algorithm/tolerance/clip order");
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
  return ParsePpFaceNormalEnvelope(
      c.normal_envelope, c.train_bank_sha256, c.source_family_sha256);
}

}  // namespace a3_pingpong
