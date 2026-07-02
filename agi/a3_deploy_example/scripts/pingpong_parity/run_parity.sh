#!/usr/bin/env bash
# Standalone (NO ROS / NO AimRT) verification of the model_15200 ping-pong
# front-end. Proves:
#   1. C++ ONNX inference == Python ONNX inference on identical obs (parity)
#   2. metadata-driven decode q = default_q + action*scale matches Python
#   3. end-to-end C++ PpPolicy::ComputeCommand is finite/bounded + emits the
#      first-tick debug dump (joint pos/vel, IMU/gravity, per-block obs stats,
#      action stats, q_des/kp/kd, q_des-clamp + secondary-IMU status).
#
# Requires: g++ (C++20), Eigen3 (/usr/include/eigen3), the bundled onnxruntime in
# thirdparty/onnxruntime/, and a python with onnxruntime+numpy (PYBIN env or
# python3). Run from the a3_deploy_example repo root, or pass MODEL/PYBIN.
set -euo pipefail
cd "$(dirname "$0")/../.."                    # -> a3_deploy_example root
MODEL="${MODEL:-assets/a3_runtime/models/model_15200.onnx}"
PYBIN="${PYBIN:-python3}"
ORT="thirdparty/onnxruntime/onnxruntime-linux-x64-1.19.2"
INC=(-Isrc/a3/a3_deploy_onnx_ref/include -I/usr/include/eigen3 -I"$ORT/include")
LNK=(-L"$ORT/lib" -lonnxruntime -Wl,-rpath,"$PWD/$ORT/lib")
OUT="$(mktemp -d)"; D=scripts/pingpong_parity
echo "model=$MODEL  out=$OUT"

g++ -std=c++20 -O2 "$D/pp_parity_harness.cpp" "${INC[@]}" "${LNK[@]}" -o "$OUT/parity"
g++ -std=c++20 -O2 "$D/pp_e2e_harness.cpp"    "${INC[@]}" "${LNK[@]}" -o "$OUT/e2e"
"$PYBIN" "$D/gen_python_ref.py" "$MODEL" "$OUT"

for name in zeros seed; do
  ts=$([ "$name" = zeros ] && echo 0 || echo 3)
  "$OUT/parity" "$MODEL" "$OUT/obs_$name.txt" "$ts" > "$OUT/cpp_$name.txt" 2>/dev/null
done
"$PYBIN" - "$OUT" <<'PY'
import sys, numpy as np
O = sys.argv[1]; ok = True
for name in ("zeros", "seed"):
    c = open(f"{O}/cpp_{name}.txt").read().splitlines()
    cact = np.array([float(x) for x in c[0][4:].split()])
    pact = np.array([float(x) for x in open(f"{O}/py_act_{name}.txt").read().split()])
    d = np.abs(cact - pact).max()
    print(f"[{name}] C++ vs Python ONNX  max|delta|={d:.2e}")
    ok &= d < 1e-4
print("PARITY:", "PASS" if ok else "FAIL")
PY

echo "=== end-to-end ComputeCommand + first-tick dump ==="
"$OUT/e2e" "$MODEL" 1
