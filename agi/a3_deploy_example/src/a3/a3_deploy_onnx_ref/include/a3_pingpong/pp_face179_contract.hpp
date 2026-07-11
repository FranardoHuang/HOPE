// Lightweight validator for the formal 179-D face-command ONNX contract.
// Kept independent of ONNX Runtime so unit tests can exercise every semantic
// metadata guard without constructing or rewriting a model file.
#pragma once

#include <stdexcept>
#include <string>

namespace a3_pingpong {

struct PpFace179MetadataContract {
  std::string face_command_enabled;
  std::string face_command_pairing;
  std::string face_obs_convention;
  std::string question_bank_exact;
  std::string bank_schema_version;
  std::string bank_split;
  std::string train_bank_sha256;
  std::string source_family_sha256;
};

inline bool PpIsLowerHexSha256(const std::string& s) {
  if (s.size() != 64) return false;
  for (const char c : s)
    if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
  return true;
}

inline void ValidatePpFace179MetadataContract(
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
}

}  // namespace a3_pingpong
