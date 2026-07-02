# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import json
import torch

import onnx

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl.exporter import _OnnxPolicyExporter

from whole_body_tracking.tasks.tracking.mdp import MotionCommand
from whole_body_tracking.tasks.tracking.actor_observation_contract import (
    infer_actor_observation_contract,
    removed_terms_vs_full,
)


def export_motion_policy_as_onnx(
    env: ManagerBasedRLEnv,
    actor_critic: object,
    path: str,
    normalizer: object | None = None,
    filename="policy.onnx",
    verbose=False,
):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxMotionPolicyExporter(env, actor_critic, normalizer, verbose)
    policy_exporter.export(path, filename)


class _OnnxMotionPolicyExporter(_OnnxPolicyExporter):
    def __init__(self, env: ManagerBasedRLEnv, actor_critic, normalizer=None, verbose=False):
        super().__init__(actor_critic, normalizer, verbose)
        cmd: MotionCommand = env.command_manager.get_term("motion")

        self.joint_pos = cmd.motion.joint_pos.to("cpu")
        self.joint_vel = cmd.motion.joint_vel.to("cpu")
        self.body_pos_w = cmd.motion.body_pos_w.to("cpu")
        self.body_quat_w = cmd.motion.body_quat_w.to("cpu")
        self.body_lin_vel_w = cmd.motion.body_lin_vel_w.to("cpu")
        self.body_ang_vel_w = cmd.motion.body_ang_vel_w.to("cpu")
        self.time_step_total = self.joint_pos.shape[0]

    def forward(self, x, time_step):
        time_step_clamped = torch.clamp(time_step.long().squeeze(-1), max=self.time_step_total - 1)
        return (
            self.actor(self.normalizer(x)),
            self.joint_pos[time_step_clamped],
            self.joint_vel[time_step_clamped],
            self.body_pos_w[time_step_clamped],
            self.body_quat_w[time_step_clamped],
            self.body_lin_vel_w[time_step_clamped],
            self.body_ang_vel_w[time_step_clamped],
        )

    def export(self, path, filename):
        self.to("cpu")
        obs = torch.zeros(1, self.actor[0].in_features)
        time_step = torch.zeros(1, 1)
        torch.onnx.export(
            self,
            (obs, time_step),
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=["obs", "time_step"],
            output_names=[
                "actions",
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ],
            dynamic_axes={},
        )


def list_to_csv_str(arr, *, decimals: int = 3, delimiter: str = ",") -> str:
    fmt = f"{{:.{decimals}f}}"
    return delimiter.join(
        fmt.format(x) if isinstance(x, (int, float)) else str(x) for x in arr  # numbers → format, strings → as-is
    )


def attach_onnx_metadata(env: ManagerBasedRLEnv, run_path: str, path: str, filename="policy.onnx") -> None:
    onnx_path = os.path.join(path, filename)

    observation_names = env.observation_manager.active_terms["policy"]
    observation_history_lengths: list[int] = []

    if env.observation_manager.cfg.policy.history_length is not None:
        observation_history_lengths = [env.observation_manager.cfg.policy.history_length] * len(observation_names)
    else:
        for name in observation_names:
            term_cfg = env.observation_manager.cfg.policy.to_dict()[name]
            history_length = term_cfg["history_length"]
            observation_history_lengths.append(1 if history_length == 0 else history_length)

    metadata = {
        "run_path": run_path,
        "joint_names": env.scene["robot"].data.joint_names,
        # NOMINAL PD gains — must come from default_joint_stiffness/damping, NOT joint_stiffness.
        # data.joint_stiffness holds the LIVE PhysX drive gains: explicit actuators
        # (IdealPDActuatorCfg) null them (their PD is applied by the actuator model, not the PhysX
        # drive) and randomize_actuator_gains DR rewrites them per env at reset. Exporting
        # data.joint_stiffness[0] therefore baked kp=kd=0 for every explicit joint (limp
        # arms/waist/ankles on deploy) and a random env-0 DR draw for the implicit ones.
        # data.default_joint_stiffness is written once from the actuator configs at init
        # ("for implicit and explicit actuators" — articulation.py) and DR only reads it.
        "joint_stiffness": env.scene["robot"].data.default_joint_stiffness[0].cpu().tolist(),
        "joint_damping": env.scene["robot"].data.default_joint_damping[0].cpu().tolist(),
        "default_joint_pos": env.scene["robot"].data.default_joint_pos_nominal.cpu().tolist(),
        "command_names": env.command_manager.active_terms,
        "observation_names": observation_names,
        "observation_history_lengths": observation_history_lengths,
        "action_scale": env.action_manager.get_term("joint_pos")._scale[0].cpu().tolist(),
        "anchor_body_name": env.command_manager.get_term("motion").cfg.anchor_body_name,
        "body_names": env.command_manager.get_term("motion").cfg.body_names,
    }

    # FAIL-FAST: a non-positive nominal gain means the export would deploy a limp joint (the
    # kp=kd=0 bug that felled the 2026-07-02 explicitpd_ft bring-up). Refuse to bake it.
    _kp = metadata["joint_stiffness"]
    _kd = metadata["joint_damping"]
    if min(_kp) <= 0.0 or min(_kd) <= 0.0:
        bad = [
            f"{n}(kp={p:.3g},kd={d:.3g})"
            for n, p, d in zip(metadata["joint_names"], _kp, _kd)
            if p <= 0.0 or d <= 0.0
        ]
        raise ValueError(
            "attach_onnx_metadata: non-positive nominal PD gain(s) — deploy would command zero "
            "torque on: " + ", ".join(bad)
        )

    # Per-clip layout for the deploy reference clock (pp_reference_clock.hpp). The C++ runner
    # falls back to a hardcoded legacy layout when these keys are absent, which drove the baked
    # v2 clips at v1 frame indices (strike served ~0.6 s early; "backhand" spliced across the
    # clip boundary). seg_len comes from the baked MotionLoader segments; strike phases from the
    # racket_target command (per-clip when configured).
    motion_cmd = env.command_manager.get_term("motion")
    metadata["clip_seg_lengths"] = ",".join(str(int(n)) for n in motion_cmd.motion.seg_len.cpu().tolist())
    try:
        rt_cfg = env.command_manager.get_term("racket_target").cfg
        phases = getattr(rt_cfg, "strike_phase_per_clip", None)
        if phases is None:
            phases = [rt_cfg.strike_phase] * motion_cmd.motion.num_segments
        metadata["clip_strike_phases"] = ",".join(f"{float(p):.4f}" for p in phases)
    except (KeyError, ValueError):
        pass  # task without a racket_target command (plain tracking) — clock keys not needed
    actor_contract = infer_actor_observation_contract(env)
    if actor_contract is not None:
        metadata.update(
            {
                "actor_obs_contract": actor_contract.name,
                "actor_obs_mode": actor_contract.obs_mode,
                "actor_obs_total_dim": actor_contract.total_dim,
                "actor_obs_term_dims": [term.dim for term in actor_contract.terms],
                "actor_obs_term_sources_json": json.dumps(
                    {term.name: term.deploy_source for term in actor_contract.terms},
                    separators=(",", ":"),
                ),
                "actor_obs_removed_vs_full_json": json.dumps(
                    [term.name for term in removed_terms_vs_full(actor_contract)],
                    separators=(",", ":"),
                ),
            }
        )

    model = onnx.load(onnx_path)

    for k, v in metadata.items():
        entry = onnx.StringStringEntryProto()
        entry.key = k
        entry.value = list_to_csv_str(v) if isinstance(v, list) else str(v)
        model.metadata_props.append(entry)

    onnx.save(model, onnx_path)
