#!/usr/bin/env python3
"""Generate Python ONNX reference vectors for the C++<->Python parity harness.

Writes, into OUTDIR:
  obs_zeros.txt / obs_seed.txt   - identical obs[180] fed to both C++ and Python
  py_act_zeros.txt / py_act_seed.txt - Python ONNX action[31]
  py_tq_zeros.txt  / py_tq_seed.txt  - Python decoded q = default + act*scale

Cases: zeros obs @ time_step=0, and seed-12345 std-normal obs @ time_step=3.
Floats are written as the exact double of their float32 value so a C++ read
(double) + cast to float reproduces the identical float32 input.

usage: gen_python_ref.py <model.onnx> <outdir>
"""
import sys
import numpy as np
import onnxruntime as ort


def main():
    model, outdir = sys.argv[1], sys.argv[2]
    sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    md = sess.get_modelmeta().custom_metadata_map
    default_q = np.array([float(x) for x in md["default_joint_pos"].split(",")])
    scale = np.array([float(x) for x in md["action_scale"].split(",")])
    # obs dim from the model itself (175 = deploy_parity, 180 = full) — a hardcoded 180
    # made this script reject every deploy_parity model.
    obs_dim = int(sess.get_inputs()[0].shape[-1])
    print(f"[gen_python_ref] obs_dim={obs_dim} ({md.get('actor_obs_contract', 'unknown contract')})")

    def writevec(path, v):
        open(path, "w").write(" ".join(repr(float(x)) for x in np.asarray(v).reshape(-1)))

    def run(obs, ts):
        feed = {"obs": obs.astype(np.float32),
                "time_step": np.array([[ts]], dtype=np.float32)}
        return sess.run(["actions"], feed)[0].reshape(-1).astype(np.float64)

    cases = {"zeros": (np.zeros((1, obs_dim)), 0),
             "seed": (np.random.default_rng(12345).standard_normal((1, obs_dim)), 3)}
    for name, (obs, ts) in cases.items():
        writevec(f"{outdir}/obs_{name}.txt", obs.astype(np.float32).astype(np.float64))
        act = run(obs.astype(np.float32), ts)
        writevec(f"{outdir}/py_act_{name}.txt", act)
        writevec(f"{outdir}/py_tq_{name}.txt", default_q + act * scale)
        print(f"[{name}] ts={ts} act min/max/mean="
              f"{act.min():.5f}/{act.max():.5f}/{act.mean():.5f}")


if __name__ == "__main__":
    main()
