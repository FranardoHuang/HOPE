"""Read-only audit: compare the HOPE-trained policy's actuator contract (ONNX metadata) against the
official Agibot A3 deploy constants (a3_policy_parameters.hpp).

Produces the joint-by-joint table:
  joint_name | train_default | deploy_default | train_kp | deploy_kp | train_kd | deploy_kd | action_scale | status

Nothing is modified. The training side is read from the exported ONNX metadata (Isaac articulation
order, 31 joints incl. head). The deploy side is parsed from the C++ constexpr arrays in
a3_policy_parameters.hpp (29-DOF "MuJoCo policy view" order, no head). Joints are matched by NAME.
The 2 head joints exist only on the training side (deploy fills them via ExpandToBackend).

    python scripts/inspect_a3_deploy_contract.py [--onnx <policy.onnx>] [--hpp <a3_policy_parameters.hpp>]
"""
import argparse
import os
import re

import numpy as np

# 29-DOF MuJoCo policy-view joint order, verbatim from a3_policy_parameters.hpp header (lines 11-15).
DEPLOY_JOINT_ORDER = [
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
]


def parse_hpp_array(text, name):
    """Extract the numeric values of `constexpr std::array<double, N> name = { ... };`."""
    m = re.search(r"std::array<double,\s*\d+>\s*" + re.escape(name) + r"\s*=\s*\{(.*?)\};",
                  text, re.S)
    if not m:
        raise SystemExit(f"[FATAL] could not find array {name} in the hpp")
    body = re.sub(r"//[^\n]*", "", m.group(1))           # strip // comments
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", body)
    return np.array([float(x) for x in nums], float)


def read_onnx_meta(onnx_path):
    import onnxruntime as ort

    s = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    md = s.get_modelmeta().custom_metadata_map
    obs_dim = int(s.get_inputs()[0].shape[1])
    act_dim = int(s.get_outputs()[0].shape[1])
    g = lambda k: [float(v) for v in md[k].split(",")]
    return dict(
        obs_dim=obs_dim, act_dim=act_dim,
        joint_names=md["joint_names"].split(","),
        default=np.array(g("default_joint_pos")), action_scale=np.array(g("action_scale")),
        kp=np.array(g("joint_stiffness")), kd=np.array(g("joint_damping")),
        input_names=[i.name for i in s.get_inputs()],
        output_names=[o.name for o in s.get_outputs()],
    )


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    wbt = os.path.dirname(here)
    repo = os.path.dirname(os.path.dirname(wbt))
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", default=os.path.join(
        wbt, "logs/rsl_rl/agibot_a3_hope/2026-06-27_18-14-06_basecouple03_resume/exported/policy.onnx"))
    p.add_argument("--hpp", default=os.path.join(
        repo, "agi/a3_deploy_example/src/a3/a3_deploy_onnx_ref/include/a3_policy_parameters.hpp"))
    args = p.parse_args()

    tr = read_onnx_meta(args.onnx)
    hpp = open(args.hpp).read()
    dep = {
        "default": parse_hpp_array(hpp, "a3_default_angles"),
        "kp": parse_hpp_array(hpp, "a3_kps"),
        "kd": parse_hpp_array(hpp, "a3_kds"),
        "action_scale": parse_hpp_array(hpp, "a3_action_scale"),
    }
    for k, v in dep.items():
        assert v.shape == (29,), f"deploy {k} has {v.shape}, expected (29,)"
    dep_idx = {n: i for i, n in enumerate(DEPLOY_JOINT_ORDER)}

    print("=" * 118)
    print("HOPE policy (ONNX) vs Agibot A3 deploy (a3_policy_parameters.hpp) — actuator contract")
    print("=" * 118)
    print(f"TRAIN ONNX : obs_dim={tr['obs_dim']}  act_dim={tr['act_dim']}  "
          f"inputs={tr['input_names']}  outputs={len(tr['output_names'])}")
    print(f"DEPLOY     : action_dim=29 (no head)  obs=obs_dict[1570] (HITTER tokenizer)  "
          f"single output 'action'")
    print("-" * 118)
    hdr = ("joint", "tr_def", "dep_def", "tr_kp", "dep_kp", "tr_kd", "dep_kd", "act_scale", "status")
    print(f"{hdr[0]:28s}{hdr[1]:>9s}{hdr[2]:>9s}{hdr[3]:>8s}{hdr[4]:>8s}{hdr[5]:>7s}{hdr[6]:>7s}{hdr[7]:>11s}  {hdr[8]}")
    print("-" * 118)

    mism = 0
    head_joints = []
    for i, jn in enumerate(tr["joint_names"]):
        td, tkp, tkd, tas = tr["default"][i], tr["kp"][i], tr["kd"][i], tr["action_scale"][i]
        if jn in dep_idx:
            di = dep_idx[jn]
            dd, dkp, dkd, das = dep["default"][di], dep["kp"][di], dep["kd"][di], dep["action_scale"][di]
            # The ONNX metadata stores defaults/scales at 3-decimal precision (exporter's
            # list_to_csv_str(decimals=3)), so sub-1e-3 differences are storage rounding, not real
            # gain mismatches. kp/kd are integers/clean -> compare tight.
            ok = (abs(td - dd) < 1.1e-3 and abs(tkp - dkp) < 1e-4 and abs(tkd - dkd) < 1e-4
                  and abs(tas - das) < 1.1e-3)
            status = "OK" if ok else "*** MISMATCH ***"
            mism += not ok
            print(f"{jn:28s}{td:9.4f}{dd:9.4f}{tkp:8.1f}{dkp:8.1f}{tkd:7.1f}{dkd:7.1f}{tas:11.4f}  {status}")
        else:
            head_joints.append(jn)
            print(f"{jn:28s}{td:9.4f}{'--':>9s}{tkp:8.1f}{'--':>8s}{tkd:7.1f}{'--':>7s}{tas:11.4f}  "
                  f"HEAD: deploy fills via ExpandToBackend")

    print("-" * 118)
    shared = len(tr["joint_names"]) - len(head_joints)
    print(f"shared joints: {shared}/29   gain/default/scale MISMATCHES among shared: {mism}")
    print(f"train-only (head) joints: {head_joints}")
    print("VERDICT (gains): " + ("ALL shared joints match the deploy a3_kps/a3_kds/defaults/action_scale"
                                  if mism == 0 else f"{mism} shared-joint mismatches — INVESTIGATE"))
    print("NOTE: obs (180 vs 1570) and action dim (31 vs 29) + joint ORDER still differ — see the audit doc.")
    print("=" * 118)


if __name__ == "__main__":
    main()
