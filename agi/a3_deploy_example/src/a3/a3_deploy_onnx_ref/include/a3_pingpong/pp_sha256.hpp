// Small dependency-free SHA-256 used to verify content-bound ONNX metadata.
// This is not an authenticity signature; it detects missing/reordered/tampered
// envelope fields and binds them to the exact train-bank/source-family hashes.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace a3_pingpong {

inline std::uint32_t PpSha256Rotr(std::uint32_t value, unsigned shift) {
  return (value >> shift) | (value << (32U - shift));
}

inline std::string PpSha256Hex(const std::string& input) {
  static constexpr std::array<std::uint32_t, 64> k = {
      0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
      0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
      0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
      0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
      0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
      0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
      0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
      0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
      0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
      0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
      0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
      0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
      0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
      0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
      0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
      0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};
  std::array<std::uint32_t, 8> h = {
      0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
      0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};

  std::vector<std::uint8_t> bytes(input.begin(), input.end());
  const std::uint64_t bit_length = static_cast<std::uint64_t>(bytes.size()) * 8U;
  bytes.push_back(0x80U);
  while ((bytes.size() % 64U) != 56U) bytes.push_back(0U);
  for (int shift = 56; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>((bit_length >> shift) & 0xffU));

  for (std::size_t offset = 0; offset < bytes.size(); offset += 64U) {
    std::array<std::uint32_t, 64> w{};
    for (std::size_t i = 0; i < 16U; ++i) {
      const std::size_t j = offset + 4U * i;
      w[i] = (static_cast<std::uint32_t>(bytes[j]) << 24U) |
             (static_cast<std::uint32_t>(bytes[j + 1U]) << 16U) |
             (static_cast<std::uint32_t>(bytes[j + 2U]) << 8U) |
             static_cast<std::uint32_t>(bytes[j + 3U]);
    }
    for (std::size_t i = 16U; i < 64U; ++i) {
      const std::uint32_t s0 = PpSha256Rotr(w[i - 15U], 7U) ^
                               PpSha256Rotr(w[i - 15U], 18U) ^ (w[i - 15U] >> 3U);
      const std::uint32_t s1 = PpSha256Rotr(w[i - 2U], 17U) ^
                               PpSha256Rotr(w[i - 2U], 19U) ^ (w[i - 2U] >> 10U);
      w[i] = w[i - 16U] + s0 + w[i - 7U] + s1;
    }

    std::uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
    std::uint32_t e = h[4], f = h[5], g = h[6], hh = h[7];
    for (std::size_t i = 0; i < 64U; ++i) {
      const std::uint32_t s1 = PpSha256Rotr(e, 6U) ^ PpSha256Rotr(e, 11U) ^
                               PpSha256Rotr(e, 25U);
      const std::uint32_t ch = (e & f) ^ ((~e) & g);
      const std::uint32_t temp1 = hh + s1 + ch + k[i] + w[i];
      const std::uint32_t s0 = PpSha256Rotr(a, 2U) ^ PpSha256Rotr(a, 13U) ^
                               PpSha256Rotr(a, 22U);
      const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t temp2 = s0 + maj;
      hh = g;
      g = f;
      f = e;
      e = d + temp1;
      d = c;
      c = b;
      b = a;
      a = temp1 + temp2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d;
    h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
  }

  // Handwritten lowercase hex is locale-independent.  std::ostringstream can inherit a global
  // num_put facet with grouping, which would make a metadata digest process-locale dependent.
  static constexpr char hex[] = "0123456789abcdef";
  std::string out;
  out.reserve(64);
  for (const std::uint32_t word : h) {
    for (int shift = 28; shift >= 0; shift -= 4)
      out.push_back(hex[(word >> shift) & 0x0fU]);
  }
  return out;
}

}  // namespace a3_pingpong
