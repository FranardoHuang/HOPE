# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import json
import hashlib
import math
import torch

import onnx

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl.exporter import _OnnxPolicyExporter

from whole_body_tracking.tasks.tracking.mdp import MotionCommand
from whole_body_tracking.tasks.tracking.actor_observation_contract import (
    infer_actor_observation_contract,
    removed_terms_vs_full,
)
from whole_body_tracking.utils.training_contract import (
    ACTOR_LEG_REF_MASK_PROVENANCE_KEY,
    TRAINING_CONTRACT_SCHEMA_VERSION,
    RUNTIME_EXECUTION_KEYS,
    bind_actor_leg_ref_mask_metadata,
    bind_planner_task_revision_metadata,
    checkpoint_claims_contract,
    checkpoint_contract_lineage_exact,
    planner_task_revision_metadata,
    require_checkpoint_contract_binding,
    resolve_motion_body_lin_vel_points,
    runtime_execution_facts,
    validate_schema3_contract,
    validate_schema3_contract_structure,
)
from whole_body_tracking.utils.stage1_normal_envelope import (
    FORMAL_CLIP_ORDER,
    derive_stage1_normal_envelope,
)


def is_empirical_normalizer(normalizer: object | None) -> bool:
    """Return whether ``normalizer`` is a real observation transform rather than a dead Identity.

    Several historical rsl_rl lineages exposed ``actor_obs_normalizer`` even when the effective
    module was ``nn.Identity``. Object presence is therefore not export provenance.
    """
    return normalizer is not None and not isinstance(normalizer, torch.nn.Identity)


def export_motion_policy_as_onnx(
    env: ManagerBasedRLEnv,
    actor_critic: object,
    path: str,
    normalizer: object | None = None,
    filename="policy.onnx",
    verbose=False,
) -> bool:
    """Export the actor/reference graph and return whether obs normalization was baked in."""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    obs_norm_baked = is_empirical_normalizer(normalizer)
    policy_exporter = _OnnxMotionPolicyExporter(env, actor_critic, normalizer, verbose)
    policy_exporter.export(path, filename)
    return obs_norm_baked


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


def list_to_csv_str(arr, *, decimals=None, delimiter: str = ",") -> str:
    """Serialize metadata arrays without silently changing the deploy contract.

    The historical three-decimal default rounded action scales/default poses/gains.  Full Python
    round-trip precision is now the default; callers may still request fixed decimals for display.
    """
    def encode(value) -> str:
        if isinstance(value, bool):
            return "1" if value else "0"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            return format(value, ".17g") if decimals is None else f"{value:.{decimals}f}"
        return str(value)

    return delimiter.join(encode(value) for value in arr)


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contract_value(value):
    """Match train.py's JSON primitive conversion without importing its Hydra entry module."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        return [_contract_value(item) for item in value]
    except TypeError:
        return str(value)


def attach_onnx_metadata(
    env: ManagerBasedRLEnv,
    run_path: str,
    path: str,
    filename="policy.onnx",
    *,
    obs_norm_baked: bool,
    trained_with_obs_norm: bool,
    source_checkpoint_path: str,
) -> None:
    if source_checkpoint_path is None or not str(source_checkpoint_path).strip():
        raise ValueError(
            "attach_onnx_metadata: source_checkpoint_path is mandatory; runtime environment "
            "facts cannot establish provenance for actor bytes"
        )
    onnx_path = os.path.join(path, filename)
    actor_contract = infer_actor_observation_contract(env)
    runtime_facts = runtime_execution_facts(env, actor_contract)
    observation_names = runtime_facts["actor_obs_term_names"]
    observation_history_lengths = runtime_facts["observation_history_lengths"]
    flat_qdes_limits = [
        value for pair in runtime_facts["qdes_joint_pos_limits"] for value in pair
    ]

    metadata = {
        "hope_metadata_schema_version": "2",
        "run_path": run_path,
        "joint_names": runtime_facts["joint_names"],
        "articulation_joint_names": runtime_facts["articulation_joint_names"],
        "action_joint_ids": runtime_facts["action_joint_ids"],
        # NOMINAL PD gains — must come from default_joint_stiffness/damping, NOT joint_stiffness.
        # data.joint_stiffness holds the LIVE PhysX drive gains: explicit actuators
        # (IdealPDActuatorCfg) null them (their PD is applied by the actuator model, not the PhysX
        # drive) and randomize_actuator_gains DR rewrites them per env at reset. Exporting
        # data.joint_stiffness[0] therefore baked kp=kd=0 for every explicit joint (limp
        # arms/waist/ankles on deploy) and a random env-0 DR draw for the implicit ones.
        # data.default_joint_stiffness is written once from the actuator configs at init
        # ("for implicit and explicit actuators" — articulation.py) and DR only reads it.
        "joint_stiffness": runtime_facts["joint_stiffness"],
        "joint_damping": runtime_facts["joint_damping"],
        "joint_effort_limits": runtime_facts["joint_effort_limits"],
        "joint_actuator_types": runtime_facts["joint_actuator_types"],
        "joint_armature": runtime_facts["joint_armature"],
        "joint_friction_coefficients": runtime_facts["joint_friction_coefficients"],
        "joint_velocity_limits": runtime_facts["joint_velocity_limits"],
        "joint_friction_backend": runtime_facts["joint_friction_backend"],
        "joint_friction_semantics": runtime_facts["joint_friction_semantics"],
        "joint_friction_units": runtime_facts["joint_friction_units"],
        "default_joint_pos": runtime_facts["default_joint_pos"],
        "qdes_joint_pos_limits": flat_qdes_limits,
        "command_names": env.command_manager.active_terms,
        "observation_names": observation_names,
        "observation_history_lengths": observation_history_lengths,
        "actor_obs_term_names": observation_names,
        "action_scale": runtime_facts["action_scale"],
        "action_use_default_offset": "1" if runtime_facts["action_use_default_offset"] else "0",
        "anchor_body_name": runtime_facts["anchor_body_name"],
        "anchor_body_index": runtime_facts["anchor_body_index"],
        "body_names": runtime_facts["body_names"],
        "body_indices": runtime_facts["body_indices"],
        "motion_kinematics_exact": (
            "1" if runtime_facts["motion_kinematics_exact"] else "0"
        ),
        "motion_body_lin_vel_points": list(
            resolve_motion_body_lin_vel_points(
                runtime_facts["motion_kinematics_contracts"]
            )
        ),
        "physics_step_dt_s": format(runtime_facts["physics_step_dt_s"], ".17g"),
        "policy_step_dt_s": format(runtime_facts["policy_step_dt_s"], ".17g"),
        "control_decimation": runtime_facts["control_decimation"],
        # Mandatory export-path truth: evaluator/runner must not guess from sidecar presence.
        # The graph calls actor(normalizer(x)); this is 1 exactly when a real normalizer object
        # was supplied to the exporter, otherwise its internal Identity leaves raw obs unchanged.
        "obs_norm_baked": "1" if obs_norm_baked else "0",
        # Training-contract truth, independent of export packaging. A raw graph from a normalized
        # run requires the matching sidecar; a baked graph must skip it. Consumers can therefore
        # fail loudly instead of guessing from whether obs_norm.npz happens to be present.
        "trained_with_obs_norm": "1" if trained_with_obs_norm else "0",
        # Backward-compatible spelling consumed by the current MuJoCo evaluator.
        "empirical_normalization": "1" if trained_with_obs_norm else "0",
        # Becomes 1 only after a schema-v3 source checkpoint contract is fully replay-verified.
        "training_contract_exact": "0",
    }

    # Timeout semantics are part of the rollout distribution. Rally tasks use 16 s while older
    # tasks use 10 s; guessing changes both swing count and the all-attempt denominator after falls.
    episode_length_s = float(env.cfg.episode_length_s)
    if not math.isfinite(episode_length_s) or episode_length_s <= 0.0:
        raise ValueError(
            f"attach_onnx_metadata: invalid env episode_length_s={episode_length_s!r}"
        )
    metadata["episode_length_s"] = format(episode_length_s, ".17g")
    qdes_clamp = bool(runtime_facts["qdes_clamp"])
    metadata["qdes_clamp"] = "1" if qdes_clamp else "0"
    # Mask/unmasked provenance may only come from the checkpoint-bound contract below. Runtime
    # facts are used for equality checking, never to mint provenance for historical actor bytes.
    training_contract = None
    training_contract_schema = 0
    training_contract_sha256 = None
    training_contract_lineage_exact = False
    planner_revision_metadata_json = None
    if source_checkpoint_path is not None:
        checkpoint_path = os.path.abspath(str(source_checkpoint_path))
        if not os.path.isfile(checkpoint_path):
            raise ValueError(
                f"attach_onnx_metadata: source checkpoint does not exist: {checkpoint_path}"
            )
        metadata["source_checkpoint_sha256"] = _sha256_file(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        training_contract_path = os.path.join(
            os.path.dirname(checkpoint_path), "params", "training_contract.json"
        )
        if os.path.isfile(training_contract_path):
            try:
                with open(training_contract_path, encoding="utf-8") as stream:
                    training_contract = json.load(stream)
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"attach_onnx_metadata: invalid training contract {training_contract_path}: "
                    f"{exc}"
                ) from exc
            if not isinstance(training_contract, dict):
                raise ValueError("attach_onnx_metadata: training contract root must be an object")
            training_contract_sha256 = _sha256_file(training_contract_path)
            try:
                training_contract_schema = int(training_contract.get("schema_version", 0))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "attach_onnx_metadata: invalid training-contract schema version"
                ) from exc
            if training_contract_schema == TRAINING_CONTRACT_SCHEMA_VERSION:
                validate_schema3_contract_structure(training_contract)
                try:
                    require_checkpoint_contract_binding(
                        checkpoint,
                        schema=training_contract_schema,
                        sha256=training_contract_sha256,
                        require_lineage_exact=False,
                    )
                except ValueError as exc:
                    raise ValueError(f"attach_onnx_metadata: {exc}") from exc
                training_contract_lineage_exact = checkpoint_contract_lineage_exact(checkpoint)
                if training_contract_lineage_exact:
                    # An exact-lineage claim keeps the stronger formal motion/body-order gate.
                    # Diagnostic schema-3 checkpoints remain fully bound but export with
                    # training_contract_exact=0 instead of being promoted or rejected outright.
                    validate_schema3_contract(training_contract)
            elif training_contract_schema in (1, 2):
                if checkpoint_claims_contract(checkpoint):
                    try:
                        require_checkpoint_contract_binding(
                            checkpoint,
                            schema=training_contract_schema,
                            sha256=training_contract_sha256,
                            require_lineage_exact=False,
                        )
                    except ValueError as exc:
                        raise ValueError(f"attach_onnx_metadata: {exc}") from exc
            else:
                raise ValueError(
                    "attach_onnx_metadata: unsupported training-contract schema "
                    f"{training_contract_schema}; supported diagnostic schemas are 1/2 and "
                    f"formal schema is {TRAINING_CONTRACT_SCHEMA_VERSION}"
                )
            # Only consume the profile for export after the adjacent checkpoint contract binding
            # above has succeeded. The structural validator may inspect it, but cannot mint output.
            planner_revision_metadata_json = planner_task_revision_metadata(training_contract)
            metadata["training_contract_sha256"] = training_contract_sha256
            metadata["training_contract_schema_version"] = str(training_contract_schema)
        elif checkpoint_claims_contract(checkpoint):
            raise ValueError(
                "attach_onnx_metadata: checkpoint claims a training-contract binding but the "
                f"adjacent sidecar is missing: {training_contract_path}"
            )
        if training_contract is None:
            # Never infer historical checkpoint semantics from the current export environment.
            bind_actor_leg_ref_mask_metadata(metadata, None)

    # FAIL-FAST: a non-positive nominal gain means the export would deploy a limp joint (the
    # kp=kd=0 bug that felled the 2026-07-02 explicitpd_ft bring-up). Refuse to bake it.
    _kp = metadata["joint_stiffness"]
    _kd = metadata["joint_damping"]
    _effort = metadata["joint_effort_limits"]
    _armature = metadata["joint_armature"]
    _friction = metadata["joint_friction_coefficients"]
    _velocity = metadata["joint_velocity_limits"]
    if (len(_effort) != len(_kp)
            or any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in _effort)):
        raise ValueError(
            "attach_onnx_metadata: missing/non-positive per-joint effort limits; command "
            "envelope cannot be proven"
        )
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
    joint_count = len(metadata["joint_names"])
    for key, values, strictly_positive in (
        ("joint_armature", _armature, False),
        ("joint_friction_coefficients", _friction, False),
        ("joint_velocity_limits", _velocity, True),
    ):
        invalid = (
            len(values) != joint_count
            or any(
                not math.isfinite(float(value))
                or (float(value) <= 0.0 if strictly_positive else float(value) < 0.0)
                for value in values
            )
        )
        if invalid:
            qualifier = "positive" if strictly_positive else "non-negative"
            raise ValueError(
                f"attach_onnx_metadata: {key} must have {joint_count} finite {qualifier} values"
            )

    # Per-clip layout for the deploy reference clock (pp_reference_clock.hpp). The C++ runner
    # falls back to a hardcoded legacy layout when these keys are absent, which drove the baked
    # v2 clips at v1 frame indices (strike served ~0.6 s early; "backhand" spliced across the
    # clip boundary). seg_len comes from the baked MotionLoader segments; strike phases from the
    # racket_target command (per-clip when configured).
    motion_cmd = env.command_manager.get_term("motion")
    runtime_planner_revision = (
        motion_cmd.planner_revision_hard_contract()
        if callable(getattr(motion_cmd, "planner_revision_hard_contract", None))
        else None
    )
    checkpoint_planner_revision = (
        training_contract.get("planner_task_revision")
        if training_contract_schema == TRAINING_CONTRACT_SCHEMA_VERSION
        and training_contract is not None
        else None
    )
    if runtime_planner_revision != checkpoint_planner_revision:
        raise ValueError(
            "attach_onnx_metadata: current export environment planner-task-revision contract "
            "disagrees with the checkpoint-bound schema-3 contract; legacy/OFF checkpoints "
            "cannot inherit revision support from the current environment"
        )
    if planner_revision_metadata_json is not None:
        if actor_contract is None or actor_contract.name != "deploy_parity_face179":
            raise ValueError(
                "attach_onnx_metadata: planner-task-revision metadata is valid only for the "
                "formal deploy_parity_face179 actor"
            )
    bound_planner_revision = bind_planner_task_revision_metadata(metadata, training_contract)
    if bound_planner_revision != planner_revision_metadata_json:
        raise RuntimeError("attach_onnx_metadata: planner revision canonicalization drift")
    metadata["clip_seg_lengths"] = ",".join(
        str(int(n)) for n in runtime_facts["motion_segment_lengths"]
    )
    # Episode/reset semantics are part of evaluation, not incidental trainer settings. Without
    # these keys a 110-D base-Pure model ([0,0] holds) and Rally/V3 model ([25,125], stand ref)
    # are indistinguishable to the MuJoCo evaluator and can be graded on the wrong distribution.
    motion_cfg = motion_cmd.cfg
    # Keep the immutable TRAINING distribution distinct from the native-speed
    # graph/deploy clock. _OnnxMotionPolicyExporter copies the unscaled source
    # arrays and indexes them by an explicit integer time_step, so its baked
    # graph is native-speed even when called directly from a live R14 training
    # env. A speed-curriculum checkpoint therefore remains exportable without
    # pretending it was trained at 1.0x.
    metadata["export_reference_clock_speed"] = "1"
    metadata["training_motion_speed_scale_range_json"] = json.dumps(
        None if training_contract is None else training_contract.get("motion_speed_scale_range"),
        separators=(",", ":"),
    )
    metadata["training_motion_speed_scale_per_clip_json"] = json.dumps(
        None if training_contract is None else training_contract.get("motion_speed_scale_per_clip"),
        separators=(",", ":"),
    )
    motion_files = motion_cfg.motion_file
    if not isinstance(motion_files, (list, tuple)):
        motion_files = [motion_files]
    if len(motion_files) != int(motion_cmd.motion.num_segments):
        raise ValueError(
            "attach_onnx_metadata: motion_file count does not match loaded motion segments"
        )
    metadata["motion_clip_sha256"] = ",".join(
        _sha256_file(os.path.abspath(str(path))) for path in motion_files
    )
    if training_contract is not None:
        trained_motion = [
            {
                "index": int(item.get("index", -1)),
                "basename": str(item.get("basename", "")),
                "sha256": str(item.get("sha256", "")),
                "segment_length": int(item.get("segment_length", -1)),
            }
            for item in (training_contract.get("motion_clips") or [])
        ]
        current_motion = [
            {
                "index": index,
                "basename": os.path.basename(str(motion_path)),
                "sha256": digest,
                "segment_length": int(runtime_facts["motion_segment_lengths"][index]),
            }
            for index, (motion_path, digest) in enumerate(
                zip(motion_files, metadata["motion_clip_sha256"].split(","))
            )
        ]
        if training_contract_schema < TRAINING_CONTRACT_SCHEMA_VERSION:
            # Schema 1/2 had no segment_length. Preserve their diagnostic replay check without
            # granting formal provenance.
            trained_motion_cmp = [item["sha256"] for item in trained_motion]
            current_motion_cmp = [item["sha256"] for item in current_motion]
        else:
            trained_motion_cmp = [
                {key: item[key] for key in ("index", "sha256", "segment_length")}
                for item in trained_motion
            ]
            current_motion_cmp = [
                {key: item[key] for key in ("index", "sha256", "segment_length")}
                for item in current_motion
            ]
        if trained_motion_cmp != current_motion_cmp:
            raise ValueError(
                "attach_onnx_metadata: export motion clips do not match the source checkpoint's "
                f"training contract: trained={trained_motion_cmp}, current={current_motion_cmp}"
            )
    metadata["wrap_teleport"] = "1" if bool(motion_cfg.wrap_teleport) else "0"
    metadata["motion_hold_steps_range"] = ",".join(
        str(int(v)) for v in motion_cfg.hold_steps_range
    )
    metadata["motion_hold_reference"] = "stand"
    metadata["motion_stand_start_prob"] = str(float(motion_cfg.stand_start_prob))
    metadata["motion_stand_start_min_hold"] = str(int(motion_cfg.stand_start_min_hold))
    metadata["motion_stand_start_yaw_range"] = ",".join(
        f"{float(v):.6f}" for v in getattr(motion_cfg, "stand_start_yaw_range", (0.0, 0.0))
    )
    metadata["motion_rsi_hold_root_stand_z"] = (
        "1" if bool(getattr(motion_cfg, "rsi_hold_root_stand_z", False)) else "0"
    )
    metadata["motion_allow_legacy_link_origin_velocity"] = (
        "1"
        if bool(getattr(motion_cfg, "allow_legacy_link_origin_velocity", False))
        else "0"
    )
    try:
        rt_cmd = env.command_manager.get_term("racket_target")
    except KeyError:
        rt_cmd = None  # plain tracking task: no strike clock/geometry metadata
    current_question_bank_contract = None
    validated_question_bank = None
    if rt_cmd is not None:
        rt_cfg = rt_cmd.cfg
        mount_offset = [float(value) for value in rt_cfg.mount_offset]
        if len(mount_offset) != 3 or any(not math.isfinite(value) for value in mount_offset):
            raise ValueError("attach_onnx_metadata: invalid racket control-point mount offset")
        metadata["racket_control_point"] = "pingpang_red_Link_origin_v1"
        metadata["racket_control_point_offset_wrist_m"] = mount_offset
        phases = getattr(rt_cfg, "strike_phase_per_clip", None)
        if phases is None:
            phases = [rt_cfg.strike_phase] * motion_cmd.motion.num_segments
        if len(phases) != int(motion_cmd.motion.num_segments):
            raise ValueError(
                "attach_onnx_metadata: strike_phase_per_clip count does not match motion clips"
            )
        metadata["clip_strike_phases"] = ",".join(f"{float(p):.4f}" for p in phases)
        # Per-clip reference base->racket reach offset at the strike frame (dx0,dy0,dx1,dy1,...).
        # The 177-D hitter_footwork deploy path derives its base station from the frozen racket
        # target as station_xy = racket_target_xy - reach_offset_xy[clip] (base_couple_mode
        # reference_reach) — the C++ runner refuses a 177 model without this key rather than
        # silently feeding a zero station.
        if hasattr(rt_cmd, "_ensure_reference_strike_state"):
            rt_cmd._ensure_reference_strike_state()
            reach = getattr(rt_cmd, "_ref_reach_offset_xy_per_clip", None)
            if reach is not None:
                metadata["ref_reach_offset_xy"] = ",".join(
                    f"{float(v):.6f}" for v in reach.reshape(-1).cpu().tolist()
                )
        # HITTER-PURE sampling geometry (2026-07-07): the 110-D deploy path derives the base
        # station from the racket target the paper's way (§V-B-3 fh/bh heuristic ONLY computes
        # p̂_base) as station_xy = racket_target_xy − (fixed_plane_x, band_center_y)[clip], and
        # its engage/gate logic needs the TRAINED distribution, so bake the station-relative
        # per-clip boxes + the independent station box. Format: per-clip flat
        # "x_lo,x_hi,y_lo,y_hi,z_lo,z_hi" joined by ';' (clip 0 = forehand, 1 = backhand).
        if getattr(rt_cfg, "target_mode", "") == "hitter_pure":
            _ppc = getattr(rt_cfg, "racket_pos_range_per_clip", None)
            _vpc = getattr(rt_cfg, "racket_vel_range_per_clip", None)
            nclips = int(motion_cmd.motion.num_segments)
            if _ppc is None or _vpc is None or len(_ppc) != nclips or len(_vpc) != nclips:
                raise ValueError(
                    "attach_onnx_metadata: HitterPure requires one position and velocity box per clip"
                )
            metadata["hitter_pure_pos_range_per_clip"] = ";".join(
                ",".join(f"{float(lo):.4f},{float(hi):.4f}" for (lo, hi) in clip_rng)
                for clip_rng in _ppc
            )
            metadata["hitter_pure_vel_range_per_clip"] = ";".join(
                ",".join(f"{float(lo):.4f},{float(hi):.4f}" for (lo, hi) in clip_rng)
                for clip_rng in _vpc
            )
            metadata["hitter_pure_base_target_range"] = ",".join(
                f"{float(v):.4f}"
                for v in (*rt_cfg.base_target_x_range, *rt_cfg.base_target_y_range)
            )

        # Bind a bank-trained policy to the exact train-bank bytes and its schema-v3 source family.
        # The held-out exam can then prove it is the same family instead of trusting a filename.
        question_bank_path = str(getattr(rt_cfg, "question_bank", "") or "").strip()
        trained_bank_contract = (
            training_contract.get("question_bank") if training_contract is not None else None
        )
        if trained_bank_contract and not question_bank_path:
            raise ValueError(
                "attach_onnx_metadata: source checkpoint was bank-trained but export env has no "
                "question_bank configured"
            )
        if question_bank_path:
            qb = getattr(rt_cmd, "_question_bank", None)
            if qb is None:
                raise ValueError(
                    "attach_onnx_metadata: question_bank configured but no validated QuestionBank"
                )
            validated_question_bank = qb
            bank_path = os.path.abspath(str(qb.source_path))
            if not os.path.isfile(bank_path):
                raise ValueError(
                    f"attach_onnx_metadata: validated train bank disappeared: {bank_path}"
                )
            metadata["stage1_train_bank_sha256"] = _sha256_file(bank_path)
            bank_meta = dict(qb.metadata or {})
            if bank_meta:
                if int(bank_meta.get("schema_version", -1)) != 3:
                    raise ValueError(
                        "attach_onnx_metadata: non-legacy question bank metadata must be schema v3"
                    )
                if bank_meta.get("split") != "train":
                    raise ValueError(
                        "attach_onnx_metadata: training export bank split must be 'train'"
                    )
                family_sha = str(bank_meta.get("source_family_sha256", "")).strip().lower()
                if len(family_sha) != 64 or any(
                    ch not in "0123456789abcdef" for ch in family_sha
                ):
                    raise ValueError(
                        "attach_onnx_metadata: schema-v3 train bank has invalid source-family SHA"
                    )
                metadata["stage1_bank_schema_version"] = "3"
                metadata["stage1_bank_split"] = "train"
                current_bank_sha = metadata["stage1_train_bank_sha256"]
                expected = {
                    "sha256": current_bank_sha,
                    "schema_version": 3,
                    "split": "train",
                    "source_family_sha256": family_sha,
                    "exact": True,
                }
                current_question_bank_contract = dict(expected)
                if trained_bank_contract is None:
                    # Old checkpoints have no immutable bank binding. Record the inspected bytes,
                    # but never let the current play configuration retroactively relabel history.
                    metadata["stage1_question_bank_exact"] = "0"
                else:
                    mismatch = {
                        key: (trained_bank_contract.get(key), value)
                        for key, value in expected.items()
                        if trained_bank_contract.get(key) != value
                    }
                    if mismatch:
                        raise ValueError(
                            "attach_onnx_metadata: export bank does not match the source "
                            f"checkpoint training contract: {mismatch}"
                        )
                    metadata["stage1_source_family_sha256"] = family_sha
                    metadata["stage1_question_bank_exact"] = "1"
            else:
                # Explicit legacy training can still be reproduced, but cannot claim a formal
                # same-family score because old banks carried neither split nor family provenance.
                current_question_bank_contract = {
                    "sha256": metadata["stage1_train_bank_sha256"],
                    "schema_version": "legacy",
                    "split": "unknown",
                    "source_family_sha256": None,
                    "exact": False,
                }
                metadata["stage1_bank_schema_version"] = "legacy"
                metadata["stage1_bank_split"] = "unknown"
                metadata["stage1_question_bank_exact"] = "0"
    # Face-frame provenance (2026-07-09 单翻病定案): ship the striking-face sign table and the
    # face-obs convention so eval/deploy can introspect the checkpoint instead of guessing. The
    # face_command obs lane (179/181-D tail) is ALWAYS +Y/A-frame (bank rows verbatim); the sign
    # table serves the metric/reference channels and the formal schema-2 physical-B -> raw-A
    # deploy bridge. Empty string = table not configured (scalar sign path). Own narrow guard
    # (NOT the shared try above): a
    # malformed table must fail the export loudly, not silently ship a face model without its
    # provenance keys — lying metadata is the exact failure this hunk exists to prevent.
    # NOTE judge.sh must carry mount_normal_sign_per_clip into the EXPORT env for this key to be
    # truthful on CLI-configured arms — it is in the export carry list, keep it there.
    try:
        rt_cfg2 = env.command_manager.get_term("racket_target").cfg
    except KeyError:
        rt_cfg2 = None  # plain-tracking task — no face channel, no keys
    if rt_cfg2 is not None:
        mns = getattr(rt_cfg2, "mount_normal_sign_per_clip", None) or ()
        metadata["mount_normal_sign_per_clip"] = ",".join(f"{float(s):g}" for s in mns)
        metadata["face_command_enabled"] = (
            "1" if bool(getattr(rt_cfg2, "face_command", False)) else "0"
        )
        metadata["face_command_pairing"] = str(
            getattr(rt_cfg2, "face_command_pairing", "shared_plus_y")
        )
        if bool(getattr(rt_cfg2, "face_command", False)):
            metadata["face_obs_convention"] = "mount_plusY_A"
    if actor_contract is not None:
        metadata.update(
            {
                "actor_obs_contract": runtime_facts["actor_obs_contract"],
                "actor_obs_mode": runtime_facts["actor_obs_mode"],
                "actor_obs_total_dim": runtime_facts["actor_obs_total_dim"],
                "actor_obs_term_dims": runtime_facts["actor_obs_term_dims"],
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
    if actor_contract is not None and actor_contract.name == "deploy_parity_face179":
        if (
            training_contract_schema != TRAINING_CONTRACT_SCHEMA_VERSION
            or training_contract is None
        ):
            raise ValueError(
                "attach_onnx_metadata: formal face179 export requires a checkpoint-bound "
                "schema-3 training contract"
            )
        checkpoint_mount_signs = tuple(
            float(value) for value in training_contract["mount_normal_sign_per_clip"]
        )
        required_bank = {
            "stage1_question_bank_exact": "1",
            "stage1_bank_schema_version": "3",
            "stage1_bank_split": "train",
            "face_command_enabled": "1",
            "face_command_pairing": "shared_plus_y",
            "face_obs_convention": "mount_plusY_A",
            "mount_normal_sign_per_clip": "1,-1",
        }
        mismatch = {
            key: (metadata.get(key), expected)
            for key, expected in required_bank.items()
            if metadata.get(key) != expected
        }
        if mismatch or validated_question_bank is None:
            raise ValueError(
                "attach_onnx_metadata: formal face179 export requires the live exact schema-3 "
                f"train bank and shared +Y/A-frame face channel; mismatch={mismatch}"
            )
        # Read the exact NPZ rows again under a before/after content hash.  The command loader has
        # already validated the full schema-3 physics/motion contract; this second pass derives the
        # deploy gate from the same immutable bytes and prevents a filename/donor ONNX from
        # inventing a wider normal distribution.
        metadata.update(
            derive_stage1_normal_envelope(
                validated_question_bank.source_path,
                expected_train_bank_sha256=metadata["stage1_train_bank_sha256"],
                expected_source_family_sha256=metadata["stage1_source_family_sha256"],
                mount_normal_sign_per_clip=checkpoint_mount_signs,
                clip_order=FORMAL_CLIP_ORDER,
            )
        )
    if actor_contract is not None and actor_contract.name == "hitter_pure":
        required_hp = (
            "clip_seg_lengths", "clip_strike_phases", "motion_clip_sha256",
            "hitter_pure_pos_range_per_clip", "hitter_pure_vel_range_per_clip",
            "hitter_pure_base_target_range", "mount_normal_sign_per_clip",
            "actor_obs_contract", "actor_obs_mode", "actor_obs_total_dim",
            "actor_obs_term_dims", "observation_names", "obs_norm_baked",
            "empirical_normalization", "joint_effort_limits",
        )
        missing_hp = [key for key in required_hp if not str(metadata.get(key, "")).strip()]
        if missing_hp:
            raise ValueError(
                "attach_onnx_metadata: HitterPure export missing required contract keys: "
                + ", ".join(missing_hp)
            )

    if training_contract is not None:
        # Do not let a replay/export environment rewrite immutable evaluation semantics on top of
        # an old actor.  The source checkpoint's training_contract.json is the authority.
        current_contract_facts = {
            **{key: runtime_facts[key] for key in RUNTIME_EXECUTION_KEYS},
            **{
                key: runtime_facts[key]
                for key in (ACTOR_LEG_REF_MASK_PROVENANCE_KEY, "actor_leg_ref_mask")
                if key in runtime_facts
            },
            "episode_length_s": episode_length_s,
            "motion_wrap_teleport": bool(motion_cfg.wrap_teleport),
            "motion_hold_steps_range": [int(v) for v in motion_cfg.hold_steps_range],
            "motion_hold_reference": "stand",
            "motion_stand_start_prob": float(motion_cfg.stand_start_prob),
            "motion_stand_start_min_hold": int(motion_cfg.stand_start_min_hold),
            "motion_stand_start_yaw_range": [
                float(v) for v in getattr(motion_cfg, "stand_start_yaw_range", (0.0, 0.0))
            ],
            "motion_rsi_hold_root_stand_z": bool(
                getattr(motion_cfg, "rsi_hold_root_stand_z", False)
            ),
            "question_bank": current_question_bank_contract,
        }
        if "motion_allow_legacy_link_origin_velocity" in training_contract:
            current_contract_facts["motion_allow_legacy_link_origin_velocity"] = bool(
                getattr(motion_cfg, "allow_legacy_link_origin_velocity", False)
            )
        if rt_cmd is not None:
            rt_cfg_contract = rt_cmd.cfg
            phases = getattr(rt_cmd.cfg, "strike_phase_per_clip", None)
            if phases is None:
                phases = [rt_cmd.cfg.strike_phase] * motion_cmd.motion.num_segments
            current_contract_facts["strike_phase_per_clip"] = [float(v) for v in phases]
            current_contract_facts["racket_control_point"] = "pingpang_red_Link_origin_v1"
            current_contract_facts["racket_control_point_offset_wrist_m"] = [
                float(v) for v in rt_cfg_contract.mount_offset
            ]
            for key in (
                "target_mode", "normal_mode", "racket_pos_range_per_clip",
                "racket_vel_range_per_clip", "base_target_x_range", "base_target_y_range",
                "mount_normal_axis", "mount_normal_sign", "mount_normal_sign_per_clip",
            ):
                current_contract_facts[key] = _contract_value(
                    getattr(rt_cfg_contract, key, None)
                )
            if "face_command_pairing" in training_contract:
                current_contract_facts["face_command_pairing"] = str(
                    getattr(rt_cfg_contract, "face_command_pairing", "shared_plus_y")
                )
            if "face_command_enabled" in training_contract:
                current_contract_facts["face_command_enabled"] = bool(
                    getattr(rt_cfg_contract, "face_command", False)
                )
        else:
            for key in (
                "target_mode", "normal_mode", "racket_pos_range_per_clip",
                "racket_vel_range_per_clip", "base_target_x_range", "base_target_y_range",
                "mount_normal_axis", "mount_normal_sign", "mount_normal_sign_per_clip",
                "racket_control_point", "racket_control_point_offset_wrist_m",
                "strike_phase_per_clip",
            ):
                current_contract_facts[key] = None
            if "face_command_pairing" in training_contract:
                current_contract_facts["face_command_pairing"] = "shared_plus_y"
            if "face_command_enabled" in training_contract:
                current_contract_facts["face_command_enabled"] = False
        keys_to_verify = (
            current_contract_facts.keys()
            if training_contract_schema == TRAINING_CONTRACT_SCHEMA_VERSION
            else (key for key in current_contract_facts if key in training_contract)
        )
        mismatches = {
            key: (training_contract.get(key, "<missing>"), current_contract_facts[key])
            for key in keys_to_verify
            if training_contract.get(key, "<missing>") != current_contract_facts[key]
        }
        if mismatches:
            raise ValueError(
                "attach_onnx_metadata: export environment disagrees with the source checkpoint's "
                f"immutable training/evaluation contract: {mismatches}"
            )
        # keys_to_verify iterates the CURRENT facts, so a key that exists only in the stored
        # contract is never compared: a masked checkpoint re-exported from an unmasked env would
        # silently drop the actor_leg_ref_mask metadata (and with it the evaluator's refusal).
        if training_contract.get("actor_leg_ref_mask") and not current_contract_facts.get(
            "actor_leg_ref_mask"
        ):
            raise ValueError(
                "attach_onnx_metadata: source checkpoint was trained with actor_leg_ref_mask "
                "but the export environment is unmasked; re-export with "
                "task.actor_leg_ref_mask=true so the mask provenance survives"
            )
        bind_actor_leg_ref_mask_metadata(
            metadata,
            training_contract
            if training_contract_schema == TRAINING_CONTRACT_SCHEMA_VERSION
            else None,
        )
        if (
            training_contract_schema == TRAINING_CONTRACT_SCHEMA_VERSION
            and training_contract_lineage_exact
            and training_contract.get("actor_leg_ref_mask") is not True
        ):
            metadata["training_contract_exact"] = "1"

    if obs_norm_baked and not trained_with_obs_norm:
        raise ValueError("obs_norm_baked=1 is impossible when trained_with_obs_norm=0")

    model = onnx.load(onnx_path)
    # Re-attaching metadata must be idempotent. Duplicate keys are consumer-dependent in ONNX.
    del model.metadata_props[:]

    for k, v in metadata.items():
        entry = onnx.StringStringEntryProto()
        entry.key = k
        entry.value = list_to_csv_str(v) if isinstance(v, list) else str(v)
        model.metadata_props.append(entry)

    onnx.save(model, onnx_path)
