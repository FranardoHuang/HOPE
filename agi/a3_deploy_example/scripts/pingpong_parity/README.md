# Ping-pong front-end parity / smoke harness (no ROS, no AimRT)

Standalone verification of the `model_15200` ping-pong front-end used by
`a3_deploy_onnx_ref_pingpong`. Builds the **real** `PpOnnxPolicy` / `PpPolicy`
header code against the bundled onnxruntime and Eigen — it does **not** need ROS 2
or AimRT, so it runs on any dev box and in CI.

## Run

```bash
# from agi/a3_deploy_example/ (or anywhere; the script cd's to repo root)
PYBIN=/path/to/python-with-onnxruntime bash scripts/pingpong_parity/run_parity.sh
# optional: MODEL=dist/a3_deploy_x86_64/models/model_15200.onnx
```

## What it proves

1. **C++ ONNX inference == Python ONNX inference** on identical obs vectors
   (`max|Δ|` ~1e-6, float32 noise) — `pp_parity_harness.cpp` vs `gen_python_ref.py`.
2. **Decode parity**: `q = default_q + action ⊙ action_scale` (metadata-driven)
   matches Python.
3. **End-to-end** `PpPolicy::ComputeCommand` on a nominal state is finite/bounded
   (dims=31, `q_des` in range, `kp∈[20,250]`, `kd∈[2,8]`, neck passive), and emits
   the **first-tick debug dump** (`pp_e2e_harness.cpp`).

## Files

| file | role |
|---|---|
| `run_parity.sh` | build + run + diff (self-contained) |
| `gen_python_ref.py` | Python ONNX reference vectors (zeros@ts0, seed-12345@ts3) |
| `pp_parity_harness.cpp` | C++ ONNX inference + decode on a given obs file |
| `pp_e2e_harness.cpp` | C++ `ComputeCommand` end-to-end + first-tick dump |
| `first_tick_sample.txt` | a captured first-tick dump (for AGI staff review) |

Last run: `PARITY: PASS` (zeros 2.4e-7, seed 9.5e-7); e2e `PASS (finite, bounded)`.
