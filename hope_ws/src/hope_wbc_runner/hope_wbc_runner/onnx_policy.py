"""ONNX policy wrapper for model_15200 (ported from mujoco_eval_onnx.py OnnxPolicy).

Loads the exported ``policy.onnx``:
  * inputs : obs[1,180] (float32), time_step[1,1] (float32)
  * outputs: actions[1,31] (the deterministic MEAN action), followed by the
             reference-motion side-outputs (joint_pos, joint_vel, body_pos_w,
             body_quat_w, body_lin_vel_w, body_ang_vel_w) indexed by time_step.
  * metadata: joint_names, default_joint_pos, action_scale, joint_stiffness (kp),
              joint_damping (kd), body_names, anchor_body_name, observation_names.

learned_std.npy: NOT needed for deterministic inference. The ONNX emits the MEAN
action directly; std is only used to add exploration dither (mean + scale*std*N).
We default to deterministic (no dither, no std). std is loaded ONLY if a non-zero
dither scale is requested (kept available for parity testing, off by default).
"""

import numpy as np

EXPECTED_OBS_DIM = 180
EXPECTED_JOINTS = 31


class OnnxPolicy:
    def __init__(self, onnx_path, std_path=None):
        import onnxruntime as ort  # lazy: only needed when actually running the policy

        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        ins = {i.name: i.shape for i in self.sess.get_inputs()}
        if "obs" not in ins or "time_step" not in ins:
            raise ValueError(f"unexpected ONNX inputs {list(ins)}; expected obs + time_step")
        self.obs_dim = int(ins["obs"][1])
        if self.obs_dim != EXPECTED_OBS_DIM:
            raise ValueError(f"obs dim {self.obs_dim} != {EXPECTED_OBS_DIM}")
        self.out_names = [o.name for o in self.sess.get_outputs()]

        md = self.sess.get_modelmeta().custom_metadata_map
        required = ["joint_names", "default_joint_pos", "action_scale",
                    "joint_stiffness", "joint_damping", "body_names"]
        missing = [k for k in required if k not in md or not md[k].strip()]
        if missing:
            raise ValueError(
                "ONNX missing metadata keys: " + ", ".join(missing) +
                ". Re-export with scripts/play.py (attach_onnx_metadata).")
        self.joint_names = md["joint_names"].split(",")
        self.default_q = np.array([float(v) for v in md["default_joint_pos"].split(",")], np.float64)
        self.action_scale = np.array([float(v) for v in md["action_scale"].split(",")], np.float64)
        self.kp = np.array([float(v) for v in md["joint_stiffness"].split(",")], np.float64)
        self.kd = np.array([float(v) for v in md["joint_damping"].split(",")], np.float64)
        self.body_names = md["body_names"].split(",")
        if len(self.joint_names) != EXPECTED_JOINTS:
            raise ValueError(f"expected {EXPECTED_JOINTS} joints, got {len(self.joint_names)}")

        # Per-clip reference-clock layout, baked at export by attach_onnx_metadata
        # (clip_seg_lengths / clip_strike_phases). None on older exports -> the runner
        # falls back to its yaml config (and warns: a stale yaml mis-times every strike).
        self.clip_seg_lengths = None
        self.clip_strike_phases = None
        if md.get("clip_seg_lengths", "").strip():
            self.clip_seg_lengths = tuple(int(float(v)) for v in md["clip_seg_lengths"].split(","))
        if md.get("clip_strike_phases", "").strip():
            self.clip_strike_phases = tuple(float(v) for v in md["clip_strike_phases"].split(","))

        # std only if explicitly requested (deterministic by default -> not loaded)
        self.std = None
        if std_path:
            self.std = np.load(std_path).astype(np.float64).reshape(-1)

    def refs(self, time_step: int) -> dict:
        """Reference motion at time_step (obs-independent side outputs)."""
        obs0 = np.zeros((1, self.obs_dim), np.float32)
        ts = np.array([[float(time_step)]], np.float32)
        out = self.sess.run(None, {"obs": obs0, "time_step": ts})
        return {self.out_names[i]: out[i][0] for i in range(len(self.out_names))}

    def mean_action(self, obs: np.ndarray, time_step: int) -> np.ndarray:
        """Deterministic mean action (31,), Isaac articulation order."""
        ts = np.array([[float(time_step)]], np.float32)
        out = self.sess.run(None, {"obs": obs[None].astype(np.float32), "time_step": ts})
        return out[0][0].astype(np.float64)

    def action(self, obs, time_step, dither_scale=0.0, rng=None):
        """Mean action, optionally dithered (dither_scale>0 needs std; default 0 = deterministic)."""
        mean = self.mean_action(obs, time_step)
        if dither_scale <= 0.0:
            return mean
        if self.std is None:
            raise ValueError("dither_scale>0 requested but learned_std.npy not provided")
        rng = rng or np.random.default_rng(0)
        return mean + dither_scale * self.std * rng.standard_normal(len(mean))

    def target_q(self, action: np.ndarray) -> np.ndarray:
        """Decode raw action -> joint position target. target_q = default_q + action*action_scale."""
        return self.default_q + action * self.action_scale
