#!/usr/bin/env python3
"""Fail-closed Pod2 demo-only strict-resume queue.

This is deliberately separate from ``run_lean_training_queue.py``: the generic
queue remains fresh-only.  Here every row resumes a declared model checkpoint,
preserves the full optimizer state, opts into a hard-contract mismatch, and is
therefore permanently formal-ineligible.  ``plan`` is dry-run; parent
attestation and launch each require distinct simulation-only confirmation
tokens.
"""

from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import shlex
import sys
from typing import Any

import yaml


HERE = Path(__file__).resolve()
GENERIC_PATH = HERE.with_name("run_lean_training_queue.py")
SPEC = importlib.util.spec_from_file_location("lean_queue_for_demo_hotstart", GENERIC_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load the generic lean queue module")
Q = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = Q
SPEC.loader.exec_module(Q)


class DemoQueueError(RuntimeError):
    pass


PARENT_ATTEST_CONFIRM = "SIM_ONLY_ATTEST_DEMO_WARMSTART_PARENTS"
PARENT_INSPECT_CONFIRM = "SIM_ONLY_INSPECT_DEMO_WARMSTART_PARENTS"
LAUNCH_CONFIRM = "SIM_ONLY_LAUNCH_ONE_DEMO_WARMSTART_JOB"
ATTEST_CONFIRM = "SIM_ONLY_ATTEST_ONE_DEMO_WARMSTART_MILESTONE"
EXPECTED_SOURCE = "2c2d70d6d0ccf7b0757aac4dd8e575c2e077607e"
EXPECTED_SOURCE_CHECKOUT = "/workspace/codexschema/nohope_p1_activation_successor_2c2d70d"
EXPECTED_SSH_KEY = "~/.ssh/id_ed25519_runpod"
EXPECTED_PODS = {
    "pod1": {
        "host": "162.43.172.171", "port": 18333,
        "gpus": [0, 1, 2], "max_trainers_per_gpu": 4,
    },
    "pod2": {
        "host": "162.43.172.181", "port": 13146,
        "gpus": [0, 1, 2], "max_trainers_per_gpu": 4,
    },
}
EXPECTED_SOURCE_CONTRACT_FILES = {
    "hope_training/whole_body_tracking/scripts/train.py":
        "e263c70ec037b9e3d9ff5a90b38a5c1e90ee2ac142b6e5e03e9180c496579775",
    "hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/"
    "utils/training_contract.py":
        "25cf79b9341201b96e618c9b449b1aea649b164815cc3051321d451ce1cbd4c4",
}
EXPECTED_IGNORED_RUNTIME_ASSET = {
    "target_relative_path": (
        "hope_training/whole_body_tracking/source/whole_body_tracking/"
        "whole_body_tracking/assets/agibot_a3"
    ),
    "donor": {
        "checkout": "/workspace/codexschema/nohope",
        "commit": "6d93bcb16c422a2f42748c2dc99432559653480b",
        "relative_path": (
            "hope_training/whole_body_tracking/source/whole_body_tracking/"
            "whole_body_tracking/assets/agibot_a3"
        ),
    },
    "file_count": 46, "total_file_bytes": 15378264,
    "tree_content_sha256": (
        "0137f59b1fe45e7d5f8fa731bedca905f5466bc98e8d1354081fe071d60426c6"
    ),
    "symlinks_forbidden": True, "target_must_be_gitignored": True,
}
EXPECTED_MOTION_BINDINGS = {
    "motion_file": "/workspace/codexschema/phase1_fresh_20260711/assets/v4rg_runtime_order_v3/hope_forehand_v4rg_cal.npz",
    "motion_file_2": "/workspace/codexschema/phase1_fresh_20260711/assets/v4rg_runtime_order_v3/hope_backhand_v4rg_cal.npz",
}
EXPECTED_BANK = "/workspace/codexschema/phase1_signed_face_rescue_20260713/assets/schema3_bank_rebind_v2/s1_v4rg_runtime_order_schema3_train_882fea4_rebound.npz"
EXPECTED_EXAM = "/workspace/codexschema/phase1_signed_face_rescue_20260713/papers/signed_face_exam_k100_v1/signed_face_exam_k100.schedule.json"
EXPECTED_PARENT_ITERATION = 3500
EXPECTED_MILESTONES = [3700, 4000, 4500, 5500, 7500]
EXPECTED_SLOTS = [
    "pod2/gpu0", "pod2/gpu1", "pod2/gpu0",
    "pod2/gpu1", "pod2/gpu0", "pod2/gpu1", "pod2/gpu2",
    "pod2/gpu1", "pod2/gpu0",
]
EXPECTED_RECEIPT_PATH = (
    "/workspace/codexschema/phase1_demo_hotstart_20260716/activation/"
    "parent_snapshot_receipt_v2.json"
)
EXPECTED_PARENT_SPECS = {
    "qdot": {
        "original_job_id": "p1_long_no_replay_qdot_w5_seed3",
        "original_run_name": "phase1_long_no_replay_qdot_w5_seed3_20260715",
        "original_run_dir": "/workspace/codexschema/phase1_long_funnel_20260715/runs/phase1_long_no_replay_qdot_w5_seed3",
        "original_rsl_log_dir": f"{EXPECTED_SOURCE_CHECKOUT}/hope_training/whole_body_tracking/logs/rsl_rl/agibot_a3_hope_virtualball/2026-07-15_04-12-03_phase1_long_no_replay_qdot_w5_seed3_20260715",
    },
    "v1v2": {
        "original_job_id": "p1_long_no_replay_v1v2_seed3",
        "original_run_name": "phase1_long_no_replay_v1v2_seed3_20260715",
        "original_run_dir": "/workspace/codexschema/phase1_long_funnel_20260715/runs/phase1_long_no_replay_v1v2_seed3",
        "original_rsl_log_dir": f"{EXPECTED_SOURCE_CHECKOUT}/hope_training/whole_body_tracking/logs/rsl_rl/agibot_a3_hope_virtualball/2026-07-15_04-13-17_phase1_long_no_replay_v1v2_seed3_20260715",
    },
    "control": {
        "original_job_id": "p1_long_no_replay_control_seed3_retry_v2",
        "original_run_name": "phase1_long_no_replay_control_seed3_retry_v2_20260715",
        "original_run_dir": "/workspace/codexschema/phase1_long_funnel_20260715/runs/phase1_long_no_replay_control_seed3_retry_v2",
        "original_rsl_log_dir": f"{EXPECTED_SOURCE_CHECKOUT}/hope_training/whole_body_tracking/logs/rsl_rl/agibot_a3_hope_virtualball/2026-07-15_04-14-14_phase1_long_no_replay_control_seed3_retry_v2_20260715",
    },
}
for _parent_name, _parent_spec in EXPECTED_PARENT_SPECS.items():
    _run = _parent_spec["original_run_dir"]
    _rsl = _parent_spec["original_rsl_log_dir"]
    _snapshot = (
        "/workspace/codexschema/phase1_demo_hotstart_20260716/activation/"
        f"parent_snapshots_v2/{_parent_name}"
    )
    _parent_spec.update({
        "original_pod": "pod2", "original_gpu": 2,
        "live_checkpoint_path": f"{_rsl}/model_3500.pt",
        "live_hard_contract_path": f"{_rsl}/params/training_contract.json",
        "live_queue_claim_path": f"{_run}/queue_claim.json",
        "live_run_binding_path": f"{_run}/run_binding.json",
        "snapshot_checkpoint_path": f"{_snapshot}/model_3500.pt",
        "snapshot_hard_contract_path": f"{_snapshot}/params/training_contract.json",
        "snapshot_queue_claim_path": f"{_snapshot}/queue_claim.json",
        "snapshot_run_binding_path": f"{_snapshot}/run_binding.json",
    })

EXPECTED_BASE_RECIPE = (
    "task=HOPEPingPongVirtualBall", "algo=ppo", "headless=true",
    "logger=tensorboard", "video=false",
    "task.actor_obs_contract=deploy_parity_face179",
    "task.env.episode_length_s=10.0", "task.sim.dt=0.005",
    "task.sim.decimation=4", "task.actions.qdes_clamp=true",
    "task.plant.zero_joint_friction=true", "task.motion.wrap_teleport=false",
    "task.motion.stand_start_prob=0.25", "task.motion.hold_steps_range=[0,100]",
    "task.motion.stand_start_min_hold=25",
    "task.motion.post_swing_buffer_size=4096",
    "task.motion.post_swing_min_fill=256", "task.motion.post_swing_min_hold=25",
    "task.motion.clip_switch_prob=0.0", "task.motion.speed_scale_range=[1.0,1.0]",
    "++task.motion.rsi_skip_settle_frames=0",
    "++task.motion.rsi_hold_root_stand_z=false",
    "++task.motion.stagger_initial_clock=false",
    "++task.motion.stagger_hold_max_steps=150",
    "++task.racket.question_bank_allow_legacy=false",
    "++task.racket.face_command=true", "++task.racket.face_command_obs=true",
    "++task.racket.station_obs=false",
    "++task.racket.mount_normal_sign_per_clip=[1.0,-1.0]",
    "task.racket.target_mode=uniform", "task.racket.normal_mode=velocity",
    "task.racket.adaptive_sigma=true", "task.racket.target_delay_steps=2",
    "task.racket.target_jitter_pos_per_s=0.0",
    "task.racket.target_jitter_vel_per_s=0.0",
    "task.racket.target_noise_white=0.0019",
    "task.racket.target_noise_ar1_sigma=0.0052",
    "task.racket.target_noise_ar1_rho=0.717",
    "task.racket.target_dropout_prob=0.0",
    "task.racket.target_post_strike_dropout_s=0.0",
    "task.racket.target_bias_per_swing=0.0",
    "task.racket.midswing_resample_prob=0.0",
    "task.racket.midswing_resample_tts_floor=0.3",
    "task.racket.vb_spin_mode=minimize", "task.racket.vb_metrics_only=true",
    "++task.racket.rally_legacy_metrics=true",
    "task.racket.achieved_target_mix_prob=0.30",
    "task.racket.achieved_buffer_size=4096", "task.racket.achieved_min_fill=256",
    "task.racket.achieved_jitter_pos=0.03", "task.racket.achieved_jitter_vel=0.15",
    "task.racket.achieved_clamp_inflate=0.20",
    "task.racket.clean_reference_strike_velocity=true",
    "task.racket.clean_strike_vel_window=2", "task.racket.strike_window_s=0.12",
    "task.racket.strike_success_pos_thresh=0.075", "++task.physical_ball=true",
    "task.rewards.racket_position_weight=14.0",
    "task.rewards.racket_position_std=0.2",
    "task.rewards.racket_velocity_weight=10.0",
    "task.rewards.racket_velocity_std=1.0",
    "task.rewards.racket_normal_weight=5.0", "task.rewards.racket_normal_std=0.3",
    "task.rewards.free_wrist_ori_mimic=true",
    "++task.motion.allow_legacy_link_origin_velocity=false",
    "++task.motion.event_timing_mode=disabled",
    "task.racket.strike_phase_per_clip=[0.471,0.338]",
    "++task.racket.face_command_pairing=shared_plus_y",
    "++task.rewards.racket_guidance_weight=0.0",
    "++task.rewards.racket_face_guidance_weight=0.0",
    "++task.rewards.racket_face_guidance_theta_max=3.141592653589793",
    "task.rewards.base_decel_weight=0.0", "task.motion.post_swing_start_prob=0.0",
    "task.rewards.joint_velocity_limit_hinge_margin=0.85",
    "checkpoint_tolerant=false", "checkpoint_allow_missing_contract=false",
    "checkpoint_allow_contract_mismatch=true", "++kit_carb_tasking_thread_count=16",
    "++kit_tbb_thread_count=16",
)
EXPECTED_LONG_BASE_RECIPE = tuple(
    "task.env.episode_length_s=16.0"
    if item == "task.env.episode_length_s=10.0" else item
    for item in EXPECTED_BASE_RECIPE
)

EXPECTED_JOB_SPECS = {
    "demo_qdot_v1v2_face_w0p4": (
        "qdot", "pod2/gpu0", "phase1_demo_qdot_v1v2_face_w0p4_seed3_20260716",
        "/workspace/codexschema/phase1_demo_hotstart_20260716/runs/demo_qdot_v1v2_face_w0p4",
        ("true", "0.25", "-5.0", "-0.4", "-0.3", "false"),
    ),
    "demo_qdot_v1v2_face_w0p2": (
        "qdot", "pod2/gpu1", "phase1_demo_qdot_v1v2_face_w0p2_seed3_20260716",
        "/workspace/codexschema/phase1_demo_hotstart_20260716/runs/demo_qdot_v1v2_face_w0p2",
        ("true", "0.25", "-5.0", "-0.2", "-0.3", "false"),
    ),
    "demo_v1v2_qdot_w5_face_w0p4": (
        "v1v2", "pod2/gpu0", "phase1_demo_v1v2_qdot_w5_face_w0p4_seed3_20260716",
        "/workspace/codexschema/phase1_demo_hotstart_20260716/runs/demo_v1v2_qdot_w5_face_w0p4",
        ("true", "0.25", "-5.0", "-0.4", "-0.3", "false"),
    ),
    "demo_v1v2_qdot_w2p5_face_w0p4_free_arm": (
        "v1v2", "pod2/gpu1",
        "phase1_demo_v1v2_qdot_w2p5_face_w0p4_free_arm_seed3_20260716",
        "/workspace/codexschema/phase1_demo_hotstart_20260716/runs/demo_v1v2_qdot_w2p5_face_w0p4_free_arm",
        ("true", "0.25", "-2.5", "-0.4", "-0.3", "true"),
    ),
    "demo_control_qdot_w5_face_w0p4": (
        "control", "pod2/gpu0", "phase1_demo_control_qdot_w5_face_w0p4_seed3_20260716",
        "/workspace/codexschema/phase1_demo_hotstart_20260716/runs/demo_control_qdot_w5_face_w0p4",
        ("false", "1.0", "-5.0", "-0.4", "-0.3", "false"),
    ),
    "demo_control_full_stack_free_arm_foot_w0p6": (
        "control", "pod2/gpu1",
        "phase1_demo_control_full_stack_free_arm_foot_w0p6_seed3_20260716",
        "/workspace/codexschema/phase1_demo_hotstart_20260716/runs/demo_control_full_stack_free_arm_foot_w0p6",
        ("true", "0.25", "-5.0", "-0.4", "-0.6", "true"),
    ),
    "demo_qdot_long_carry_free_arm_16s": (
        "qdot", "pod2/gpu2",
        "phase1_demo_qdot_long_carry_free_arm_16s_seed3_20260716",
        "/workspace/codexschema/phase1_demo_hotstart_20260716/runs/demo_qdot_long_carry_free_arm_16s",
        ("true", "0.25", "-5.0", "-0.4", "-0.3", "true"),
    ),
    "demo_v1v2_qdot_w2p5_face_w0p4_free_arm_retry_v2": (
        "v1v2", "pod2/gpu1",
        "phase1_demo_v1v2_qdot_w2p5_face_w0p4_free_arm_seed3_20260716_retry_v2",
        "/workspace/codexschema/phase1_demo_hotstart_20260716/runs/demo_v1v2_qdot_w2p5_face_w0p4_free_arm_retry_v2",
        ("true", "0.25", "-2.5", "-0.4", "-0.3", "true"),
    ),
    "demo_control_qdot_w5_face_w0p4_retry_v2": (
        "control", "pod2/gpu0",
        "phase1_demo_control_qdot_w5_face_w0p4_seed3_20260716_retry_v2",
        "/workspace/codexschema/phase1_demo_hotstart_20260716/runs/demo_control_qdot_w5_face_w0p4_retry_v2",
        ("false", "1.0", "-5.0", "-0.4", "-0.3", "false"),
    ),
}
LONG_CARRY_JOB_ID = "demo_qdot_long_carry_free_arm_16s"
RETRY_PREDECESSORS = {
    "demo_v1v2_qdot_w2p5_face_w0p4_free_arm_retry_v2": (
        "demo_v1v2_qdot_w2p5_face_w0p4_free_arm"
    ),
    "demo_control_qdot_w5_face_w0p4_retry_v2": (
        "demo_control_qdot_w5_face_w0p4"
    ),
}
RETRY_IDS = frozenset(RETRY_PREDECESSORS)
REJECTED_PREDECESSOR_IDS = frozenset(RETRY_PREDECESSORS.values())
EXPECTED_TERMINAL_CONTRACTS = {
    "demo_v1v2_qdot_w2p5_face_w0p4_free_arm": {
        "classification": "infrastructure_only_pre_first_iteration",
        "terminal_kind": "pre_marker_exit",
        "terminal_exit_code": 134,
        "process_identity": {
            "pid": 429116, "pgid": 429116,
            "starttime_ticks": 557505718, "absent_verified": True,
        },
        "evidence": {
            "queue_claim_path": "/workspace/codexschema/phase1_demo_hotstart_20260716/runs/demo_v1v2_qdot_w2p5_face_w0p4_free_arm/queue_claim.json",
            "queue_claim_file_sha256": "09d1b842a7c2fbf2468daeb0430451926c6c2821fa7f121c42439da1c704dbff",
            "queue_claim_content_sha256": "d942f1b20df209a68f8f45e567faed6f1df22c607abecdf1f0b71145a1be003e",
            "run_binding_path": "/workspace/codexschema/phase1_demo_hotstart_20260716/runs/demo_v1v2_qdot_w2p5_face_w0p4_free_arm/run_binding.json",
            "run_binding_file_sha256": "3ec302f7bc186d174cd242e9f35e08ec4b12a2d00737325e8851903dc875a0ed",
            "run_binding_content_sha256": "8b9bfcd5b9f0c5af152ae53a20c5f23e9c513d274b5b234d2f46f2c209cac294",
            "run_log_path": "/workspace/codexschema/phase1_demo_hotstart_20260716/runs/demo_v1v2_qdot_w2p5_face_w0p4_free_arm/run.log",
            "run_log_sha256": "ac2e1bdf6da8953f3961b2aa5dfb368da39624204c0be4a4e4141f3f5a69b10e",
            "launch_state_path": "/workspace/codexschema/phase1_demo_hotstart_20260716/runs/demo_v1v2_qdot_w2p5_face_w0p4_free_arm/run.log.launch",
            "launch_state_sha256": "1819713e1475f8b1714e62b7eacb70493cde34bf0113f1460eb3ea4b4987b0b5",
            "leader_identity_path": "/workspace/codexschema/phase1_demo_hotstart_20260716/runs/demo_v1v2_qdot_w2p5_face_w0p4_free_arm/run.log.launch.leader.json",
            "leader_identity_sha256": "e80ab343810d6087d67cbd9b8fc3c848810bfe3cecb72c1155b55ba83635da48",
        },
        "behavior_evidence_eligible": False,
        "old_namespace_relaunch_forbidden": True,
    },
    "demo_control_qdot_w5_face_w0p4": {
        "classification": "infrastructure_only_pre_first_iteration",
        "terminal_kind": "stale_timeout",
        "terminal_exit_code": 125,
        "process_identity": {
            "pid": 429974, "pgid": 429974,
            "starttime_ticks": 557535387, "absent_verified": True,
        },
        "evidence": {
            "queue_claim_path": "/workspace/codexschema/phase1_demo_hotstart_20260716/runs/demo_control_qdot_w5_face_w0p4/queue_claim.json",
            "queue_claim_file_sha256": "eaea2766ccdff35bd26cbb271118b01df9c7a4a20d855054e6c9913c246eb70d",
            "queue_claim_content_sha256": "2e6eb7e0f479d48e34130252cf4c83a22bdfdadfcbfe1813b5b1aa398d29b811",
            "run_binding_path": "/workspace/codexschema/phase1_demo_hotstart_20260716/runs/demo_control_qdot_w5_face_w0p4/run_binding.json",
            "run_binding_file_sha256": "6015ccbbb0d8e9f6eb95cd735e475d002ee1340f70ff4256f39ff54d002b0c32",
            "run_binding_content_sha256": "0942c89b5acac88c9a6432e50cff9836c85f5c0564b3a3dfb8a9f8ece5c25008",
            "run_log_path": "/workspace/codexschema/phase1_demo_hotstart_20260716/runs/demo_control_qdot_w5_face_w0p4/run.log",
            "run_log_sha256": "d62d1055a00de134b1b3476735ef298a838b9e6330e0800fc37b65f4fbf58382",
            "launch_state_path": "/workspace/codexschema/phase1_demo_hotstart_20260716/runs/demo_control_qdot_w5_face_w0p4/run.log.launch",
            "launch_state_sha256": "dfb5a21d9e8fa53f0e8d87b08f3434727575ddf3bcd3994b00c0f2b2f4a4e694",
            "leader_identity_path": "/workspace/codexschema/phase1_demo_hotstart_20260716/runs/demo_control_qdot_w5_face_w0p4/run.log.launch.leader.json",
            "leader_identity_sha256": "eaf24e7646a7a256a21afb5ca0348baf34759c54a8fac49be67433b3d8f8e79f",
            "pre_term_identity_path": "/workspace/codexschema/phase1_demo_hotstart_20260716/runs/demo_control_qdot_w5_face_w0p4/run.log.launch.pre_term.json",
            "pre_term_identity_sha256": "cc3f0c6d7a2f056d5a7cbf81fa2d7af98c8d1339e9eecab7c07eca3879d037ff",
            "pre_kill_identity_path": "/workspace/codexschema/phase1_demo_hotstart_20260716/runs/demo_control_qdot_w5_face_w0p4/run.log.launch.pre_kill.json",
            "pre_kill_identity_sha256": "8905b4505488c53ebec60e644b42670a7222d2b2a91c71023e69e9cfcb4818ba",
        },
        "behavior_evidence_eligible": False,
        "old_namespace_relaunch_forbidden": True,
    },
}
EXPECTED_RETRY_CONTRACTS = {
    retry_id: {
        "retry_of": predecessor_id,
        "retry_ordinal": 1,
        "manual_retry_limit": 1,
        "manual_dispatch_only": True,
        "automatic_retry": False,
        "further_retry_authorized": False,
        "recipe_equal": True,
        "predecessor_status": "rejected",
        "predecessor_terminal": EXPECTED_TERMINAL_CONTRACTS[predecessor_id],
        "launch_sequence": sequence,
    }
    for sequence, (retry_id, predecessor_id) in enumerate(
        RETRY_PREDECESSORS.items(), start=1
    )
}
EXPECTED_LONG_CARRY_SCREENING_CONTRACT = {
    "scientific_question": "three_to_four_swing_balance_debt_in_one_episode",
    "expected_swings_per_episode": [3, 4],
    "episode_length_s": 16.0,
    "milestone_rules": {
        3700: {
            "offset_from_parent": 200,
            "decision_scope": "structure_and_activation_only",
            "sparse_hit_zero_may_stop": False,
        },
        4000: {
            "offset_from_parent": 500,
            "decision_scope": "safety_and_balance_debt",
            "sparse_hit_zero_may_stop": False,
        },
        4500: {
            "offset_from_parent": 1000,
            "decision_scope": "overnight_demo_portfolio_ranking",
            "sparse_hit_zero_may_stop": False,
        },
    },
    "launch_gate": {
        "required_slot": "pod2/gpu2",
        "required_prelaunch_occupancy": 3,
        "fourth_slot_only": True,
    },
}
EXPECTED_DELTA_KEYS = (
    "checkpoint_path", "++task.rewards.free_wrist_vel_mimic",
    "++task.rewards.motion_scale_in_window",
    "task.rewards.joint_velocity_limit_hinge_weight",
    "++task.rewards.racket_face_conditional_guidance_weight",
    "task.rewards.foot_orientation_weight",
    "++task.rewards.free_non_striking_arm_mimic",
)
EXPECTED_PROBE_EVIDENCE = {
    "attempt_id": "inferencefix_2c2d70d_pod2_gpu1_a1",
    "result_file_sha256": (
        "4b12854c5deca075ddf886fea3c5806aa0838b1d2bc9d3739e2fa13cd1840b27"
    ),
    "terminal_status": "passed",
    "reuse_scope": "exact_source_scene_boot_and_runtime_binding_only",
}
EXPECTED_DECISION_CONTRACT = {
    "purpose": "overnight_demo_portfolio_not_causal_or_exact_evidence",
    "descendants_formal_exact_eligible": False,
    "checkpoints_absolute": EXPECTED_MILESTONES,
    "checkpoint_offsets_from_parent": [200, 500, 1000, 2000, 4000],
    "sparse_outcome_zero_before_eligibility_may_stop": False,
    "second_seed_authorized": False, "formal_promotion_authorized": False,
    "real_robot_authorized": False,
}


PARENT_PROGRAM = r'''import base64
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import sys
import torch


class ParentError(RuntimeError):
    pass


def canonical_sha256(value):
    encoded = json.dumps(
        value, allow_nan=False, ensure_ascii=False,
        separators=(",", ":"), sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def require_workspace_path(value, label):
    path = Path(value)
    if not path.is_absolute() or not str(path).startswith("/workspace/"):
        raise ParentError(f"{label} must be an absolute /workspace path")
    return path


def require_no_symlink_components(path, label, *, leaf_may_be_missing=False):
    path = require_workspace_path(path, label)
    current = Path(path.anchor)
    for index, part in enumerate(path.parts[1:]):
        current = current / part
        try:
            item = current.lstat()
        except FileNotFoundError:
            if leaf_may_be_missing and index == len(path.parts[1:]) - 1:
                return
            raise ParentError(f"{label} component missing: {current}")
        if stat.S_ISLNK(item.st_mode):
            raise ParentError(f"{label} contains a symlink component: {current}")
        if index < len(path.parts[1:]) - 1 and not stat.S_ISDIR(item.st_mode):
            raise ParentError(f"{label} parent is not a directory: {current}")


def safe_mkdirs(path, label):
    path = require_workspace_path(path, label)
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        try:
            item = current.lstat()
        except FileNotFoundError:
            try:
                current.mkdir()
            except FileExistsError:
                pass
            item = current.lstat()
        if stat.S_ISLNK(item.st_mode) or not stat.S_ISDIR(item.st_mode):
            raise ParentError(f"{label} is not a real directory: {current}")


def require_publish_target_absent(path, label):
    path = require_workspace_path(path, label)
    current = Path(path.anchor)
    for index, part in enumerate(path.parts[1:]):
        current = current / part
        try:
            item = current.lstat()
        except FileNotFoundError:
            return
        if stat.S_ISLNK(item.st_mode):
            raise ParentError(f"{label} contains a symlink component: {current}")
        if index < len(path.parts[1:]) - 1 and not stat.S_ISDIR(item.st_mode):
            raise ParentError(f"{label} parent is not a directory: {current}")
    raise ParentError(f"{label} already exists; one-shot namespace is consumed")


def file_bytes(path, label):
    path = require_workspace_path(path, label)
    require_no_symlink_components(path, label)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ParentError(f"cannot open {label}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
            raise ParentError(f"{label} must be a non-empty regular file")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        signature = lambda item: (
            item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns
        )
        if signature(before) != signature(after):
            raise ParentError(f"{label} changed while reading")
        payload = b"".join(chunks)
        if len(payload) != before.st_size:
            raise ParentError(f"{label} short read")
        return payload
    finally:
        os.close(descriptor)


def write_exclusive(path, payload, label):
    path = require_workspace_path(path, label)
    safe_mkdirs(path.parent, f"{label} parent")
    require_no_symlink_components(path, label, leaf_may_be_missing=True)
    flags = (
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags, 0o444)
    except FileExistsError as exc:
        raise ParentError(f"{label} already exists; overwrite is forbidden") from exc
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise ParentError(f"{label} write failed")
            view = view[written:]
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def require_read_only_file(path, label):
    path = require_workspace_path(path, label)
    require_no_symlink_components(path, label)
    item = path.lstat()
    if not stat.S_ISREG(item.st_mode) or stat.S_IMODE(item.st_mode) & 0o222:
        raise ParentError(f"{label} must be a read-only regular file")


def json_mapping(raw, label):
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ParentError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ParentError(f"{label} is not a mapping")
    return value


def finite_audit(value):
    seen = set()
    tensors = floating = elements = nonfinite = 0
    def visit(item):
        nonlocal tensors, floating, elements, nonfinite
        if isinstance(item, torch.Tensor):
            tensors += 1
            if torch.is_floating_point(item) or torch.is_complex(item):
                floating += 1
                count = int(item.numel())
                elements += count
                nonfinite += count - int(torch.isfinite(item).sum().item())
            return
        if isinstance(item, dict):
            if id(item) in seen:
                return
            seen.add(id(item))
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            if id(item) in seen:
                return
            seen.add(id(item))
            for child in item:
                visit(child)
    visit(value)
    return {
        "tensor_count": tensors,
        "floating_tensor_count": floating,
        "floating_elements": elements,
        "nonfinite_floating_elements": nonfinite,
    }


def validate_claim(raw, name, item):
    claim = json_mapping(raw, f"{name} queue claim")
    if claim.get("schema_version") != 2:
        raise ParentError(f"{name} queue claim schema is not 2")
    content = claim.get("content")
    if not isinstance(content, dict) or content.get("schema_version") != 1:
        raise ParentError(f"{name} queue claim content schema is not 1")
    digest = claim.get("content_sha256")
    if canonical_sha256(content) != digest:
        raise ParentError(f"{name} queue claim canonical SHA mismatch")
    caller_argv = content.get("training_argv_without_claim")
    full_argv = claim.get("training_argv")
    if (
        not isinstance(caller_argv, list)
        or not all(type(value) is str for value in caller_argv)
        or full_argv != [*caller_argv, f"++training_launch_claim_sha256={digest}"]
    ):
        raise ParentError(f"{name} queue claim training argv is not self-bound")
    expected = {
        "job_id": item["original_job_id"], "pod": item["original_pod"],
        "gpu": item["original_gpu"], "run_name": item["original_run_name"],
        "run_dir": item["original_run_dir"], "runtime_binding": True,
    }
    for key, value in expected.items():
        if content.get(key) != value:
            raise ParentError(f"{name} queue claim {key} mismatch")
    source = content.get("source")
    if not isinstance(source, dict) or (
        source.get("checkout") != item["source_checkout"]
        or source.get("commit") != item["source_commit"]
    ):
        raise ParentError(f"{name} queue claim source mismatch")
    required = (
        f"++training_queue_claim_path={item['live_queue_claim_path']}",
        f"++training_run_binding_path={item['live_run_binding_path']}",
        f"run_name={item['original_run_name']}",
    )
    for override in required:
        if full_argv.count(override) != 1:
            raise ParentError(f"{name} queue claim must contain one {override}")
    expected_train = f"{item['source_checkout']}/hope_training/whole_body_tracking/scripts/train.py"
    if len(full_argv) < 2 or full_argv[1] != expected_train:
        raise ParentError(f"{name} queue claim train.py source mismatch")
    return claim, content, digest


def validate_binding(raw, name, item, claim, claim_content, claim_digest):
    binding = json_mapping(raw, f"{name} run binding")
    if binding.get("schema_version") != 1:
        raise ParentError(f"{name} run binding schema is not 1")
    content = binding.get("content")
    digest = binding.get("content_sha256")
    if not isinstance(content, dict) or canonical_sha256(content) != digest:
        raise ParentError(f"{name} run binding canonical SHA mismatch")
    expected = {
        "schema_version": 1, "job_id": item["original_job_id"],
        "claim_path": item["live_queue_claim_path"],
        "claim_content_sha256": claim_digest,
        "binding_path": item["live_run_binding_path"],
        "rsl_log_dir": item["original_rsl_log_dir"],
        "pod": item["original_pod"], "gpu": item["original_gpu"],
        "source": claim_content["source"],
        "source_state_at_binding": {
            "head": item["source_commit"], "clean": True,
        },
        "run_name": item["original_run_name"],
        "run_dir": item["original_run_dir"],
        "training_argv": claim["training_argv"],
        "purpose": None, "not_science": False,
        "attestable": True, "promotable": True,
    }
    for key, value in expected.items():
        if content.get(key) != value:
            raise ParentError(f"{name} run binding {key} mismatch")
    if content.get("milestones") != claim_content.get("budget", {}).get("milestones"):
        raise ParentError(f"{name} run binding milestones mismatch")
    process = content.get("process")
    if not isinstance(process, dict):
        raise ParentError(f"{name} run binding process missing")
    pid, pgid = process.get("pid"), process.get("pgid")
    if type(pid) is not int or pid <= 0 or pgid != pid:
        raise ParentError(f"{name} run binding process identity invalid")
    if process.get("argv") != claim["training_argv"]:
        raise ParentError(f"{name} run binding process argv mismatch")
    return binding, content, digest


def audit_bundle(name, item, paths):
    raw = file_bytes(Path(paths["checkpoint"]), f"{name} checkpoint")
    hard_raw = file_bytes(Path(paths["hard"]), f"{name} hard contract")
    claim_raw = file_bytes(Path(paths["claim"]), f"{name} queue claim")
    binding_raw = file_bytes(Path(paths["binding"]), f"{name} run binding")
    claim, claim_content, claim_digest = validate_claim(claim_raw, name, item)
    _binding, _binding_content, binding_digest = validate_binding(
        binding_raw, name, item, claim, claim_content, claim_digest
    )
    hard = json_mapping(hard_raw, f"{name} hard contract")
    if hard.get("schema_version") != 3:
        raise ParentError(f"{name} hard contract schema is not 3")
    parent_qdot = hard.get("joint_velocity_limit_hinge_reward")
    parent_face = hard.get("racket_guidance_reward", {}).get("conditional_signed_face")
    if not isinstance(parent_qdot, dict) or not isinstance(parent_face, dict):
        raise ParentError(f"{name} parent lacks qdot/conditional-face hard bindings")
    parent_qdot_weight = parent_qdot.get("weight")
    parent_face_weight = parent_face.get("weight")
    if (
        isinstance(parent_qdot_weight, bool)
        or not isinstance(parent_qdot_weight, (int, float))
        or isinstance(parent_face_weight, bool)
        or not isinstance(parent_face_weight, (int, float))
    ):
        raise ParentError(f"{name} parent hard-bound reward weights are malformed")
    for descendant in item["descendant_contract_values"]:
        if (
            descendant["qdot_weight"] == parent_qdot_weight
            and descendant["face_weight"] == parent_face_weight
        ):
            raise ParentError(
                f"{name} descendant {descendant['job_id']} changes no hard-bound reward"
            )
    try:
        checkpoint = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ParentError(f"cannot load {name} checkpoint bytes: {exc}") from exc
    if not isinstance(checkpoint, dict):
        raise ParentError(f"{name} checkpoint is not a mapping")
    if type(checkpoint.get("iter")) is not int or checkpoint["iter"] != item["iteration"]:
        raise ParentError(f"{name} embedded iteration mismatch")
    model = checkpoint.get("model_state_dict")
    if not isinstance(model, dict) or not model:
        raise ParentError(f"{name} checkpoint lacks non-empty model_state_dict")
    keys = tuple(model)
    if not any(key.startswith("actor.") for key in keys) or not any(
        key.startswith("critic.") for key in keys
    ):
        raise ParentError(f"{name} checkpoint lacks actor/critic model keys")
    optimizer = checkpoint.get("optimizer_state_dict")
    if (
        not isinstance(optimizer, dict) or not optimizer
        or not isinstance(optimizer.get("state"), dict) or not optimizer["state"]
        or not isinstance(optimizer.get("param_groups"), list)
        or not optimizer["param_groups"]
    ):
        raise ParentError(f"{name} checkpoint optimizer state is empty or malformed")
    audit = finite_audit(checkpoint)
    if audit["floating_tensor_count"] <= 0 or audit["nonfinite_floating_elements"] != 0:
        raise ParentError(f"{name} checkpoint floating tensors are not finite")
    infos = checkpoint.get("infos")
    if not isinstance(infos, dict):
        raise ParentError(f"{name} checkpoint infos missing")
    hard_sha = hashlib.sha256(hard_raw).hexdigest()
    if (
        infos.get("training_contract_sha256") != hard_sha
        or infos.get("training_contract_schema_version") != 3
    ):
        raise ParentError(f"{name} checkpoint/hard-contract binding mismatch")
    lineage = infos.get("training_contract_lineage_exact")
    if type(lineage) is not int or lineage != 1:
        raise ParentError(f"{name} parent lineage is not exact")
    if infos.get("training_launch_claim_sha256") != claim_digest:
        raise ParentError(f"{name} checkpoint/launch-claim binding mismatch")
    return {
        "checkpoint_sha256": hashlib.sha256(raw).hexdigest(),
        "hard_contract_sha256": hard_sha,
        "queue_claim_file_sha256": hashlib.sha256(claim_raw).hexdigest(),
        "queue_claim_content_sha256": claim_digest,
        "run_binding_file_sha256": hashlib.sha256(binding_raw).hexdigest(),
        "run_binding_content_sha256": binding_digest,
        "embedded_iteration": checkpoint["iter"],
        "optimizer_state_dict_present": True,
        "optimizer_state_dict_nonempty": True,
        "parent_training_contract_lineage_exact": True,
        "parent_qdot_weight": parent_qdot_weight,
        "parent_conditional_face_weight": parent_face_weight,
        "training_launch_claim_sha256": claim_digest,
        "finite_audit": audit,
        "raw": {
            "checkpoint": raw, "hard": hard_raw,
            "claim": claim_raw, "binding": binding_raw,
        },
    }


def public_parent(item, audited):
    return {
        "original_job_id": item["original_job_id"],
        "original_run_name": item["original_run_name"],
        "original_run_dir": item["original_run_dir"],
        "original_rsl_log_dir": item["original_rsl_log_dir"],
        "original_pod": item["original_pod"], "original_gpu": item["original_gpu"],
        "live_paths": {
            key.removeprefix("live_"): item[key]
            for key in (
                "live_checkpoint_path", "live_hard_contract_path",
                "live_queue_claim_path", "live_run_binding_path",
            )
        },
        "snapshot_paths": {
            key.removeprefix("snapshot_"): item[key]
            for key in (
                "snapshot_checkpoint_path", "snapshot_hard_contract_path",
                "snapshot_queue_claim_path", "snapshot_run_binding_path",
            )
        },
        **{key: value for key, value in audited.items() if key != "raw"},
    }


def main():
    if len(sys.argv) != 2:
        raise ParentError("one base64 JSON specification is required")
    spec = json.loads(base64.b64decode(sys.argv[1], validate=True))
    mode = spec.get("mode")
    if mode not in {"inspect", "attest", "verify"}:
        raise ParentError("mode must be inspect, attest, or verify")
    if mode in {"inspect", "attest"}:
        require_publish_target_absent(
            Path(spec["receipt_path"]), "activation receipt"
        )
        for name, item in sorted(spec["parents"].items()):
            for key, label in (
                ("snapshot_checkpoint_path", "checkpoint"),
                ("snapshot_hard_contract_path", "hard contract"),
                ("snapshot_queue_claim_path", "queue claim"),
                ("snapshot_run_binding_path", "run binding"),
            ):
                require_publish_target_absent(
                    Path(item[key]), f"{name} snapshot {label}"
                )
        live_audits = {}
        for name, item in sorted(spec["parents"].items()):
            live_audits[name] = audit_bundle(name, item, {
                "checkpoint": item["live_checkpoint_path"],
                "hard": item["live_hard_contract_path"],
                "claim": item["live_queue_claim_path"],
                "binding": item["live_run_binding_path"],
            })
        if mode == "attest":
            for name, item in sorted(spec["parents"].items()):
                raw = live_audits[name]["raw"]
                write_exclusive(Path(item["snapshot_checkpoint_path"]), raw["checkpoint"], f"{name} snapshot checkpoint")
                write_exclusive(Path(item["snapshot_hard_contract_path"]), raw["hard"], f"{name} snapshot hard contract")
                write_exclusive(Path(item["snapshot_queue_claim_path"]), raw["claim"], f"{name} snapshot queue claim")
                write_exclusive(Path(item["snapshot_run_binding_path"]), raw["binding"], f"{name} snapshot run binding")
    if mode == "inspect":
        audits = live_audits
    else:
        audits = {}
        for name, item in sorted(spec["parents"].items()):
            for key, label in (
                ("snapshot_checkpoint_path", "checkpoint"),
                ("snapshot_hard_contract_path", "hard contract"),
                ("snapshot_queue_claim_path", "queue claim"),
                ("snapshot_run_binding_path", "run binding"),
            ):
                require_read_only_file(Path(item[key]), f"{name} snapshot {label}")
            audits[name] = audit_bundle(name, item, {
                "checkpoint": item["snapshot_checkpoint_path"],
                "hard": item["snapshot_hard_contract_path"],
                "claim": item["snapshot_queue_claim_path"],
                "binding": item["snapshot_run_binding_path"],
            })
    if mode == "attest":
        for name in sorted(spec["parents"]):
            for key in (
                "checkpoint_sha256", "hard_contract_sha256",
                "queue_claim_file_sha256", "queue_claim_content_sha256",
                "run_binding_file_sha256", "run_binding_content_sha256",
                "training_launch_claim_sha256",
            ):
                if audits[name][key] != live_audits[name][key]:
                    raise ParentError(f"{name} snapshot {key} differs from live source")
    content = {
        "schema_version": 2,
        "purpose": "demo_only_strict_full_state_snapshot_parent_receipt_v2",
        "source_commit": spec["source_commit"],
        "transfer_mode": "strict_full_state_preserve_optimizer",
        "descendant_exact_eligible": False,
        "parents": {
            name: public_parent(spec["parents"][name], audit)
            for name, audit in sorted(audits.items())
        },
    }
    receipt = {
        "schema_version": 2, "content": content,
        "content_sha256": canonical_sha256(content),
    }
    encoded = (
        json.dumps(receipt, allow_nan=False, ensure_ascii=False,
                   separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")
    output = require_workspace_path(spec["receipt_path"], "activation receipt")
    if mode == "attest":
        write_exclusive(output, encoded, "activation receipt")
    elif mode == "verify":
        if file_bytes(output, "activation receipt") != encoded:
            raise ParentError("activation receipt differs from immutable snapshots")
    receipt_sha = hashlib.sha256(encoded).hexdigest()
    expected = spec.get("expected_receipt_sha256")
    if expected is not None and receipt_sha != expected:
        raise ParentError("activation receipt file SHA mismatch")
    print(json.dumps({
        "status": "DEMO_WARMSTART_PARENT_SNAPSHOTS_V2_OK", "mode": mode,
        "receipt_path": str(output), "receipt_file_sha256": receipt_sha,
        "receipt": receipt,
    }, sort_keys=True))


try:
    main()
except ParentError as exc:
    print(f"DEMO_WARMSTART_PARENT_ERROR: {exc}", file=sys.stderr)
    raise SystemExit(2)
'''


SNAPSHOT_RECHECK_PROGRAM = r'''import base64
import hashlib
import json
import os
from pathlib import Path
import stat
import sys


class RecheckError(RuntimeError):
    pass


def recheck(item):
    path = Path(item["path"])
    if not path.is_absolute() or not str(path).startswith("/workspace/"):
        raise RecheckError(f"{item['label']} path is outside /workspace")
    current = Path(path.anchor)
    for index, part in enumerate(path.parts[1:]):
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError as exc:
            raise RecheckError(f"{item['label']} is missing") from exc
        if stat.S_ISLNK(info.st_mode):
            raise RecheckError(f"{item['label']} contains a symlink component")
        if index < len(path.parts[1:]) - 1 and not stat.S_ISDIR(info.st_mode):
            raise RecheckError(f"{item['label']} parent is not a directory")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(fd)
        if (
            not stat.S_ISREG(before.st_mode) or before.st_size <= 0
            or stat.S_IMODE(before.st_mode) & 0o222
        ):
            raise RecheckError(f"{item['label']} is not a read-only regular file")
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(fd)
        signature = lambda value: (
            value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns
        )
        if signature(before) != signature(after) or size != before.st_size:
            raise RecheckError(f"{item['label']} changed while hashing")
    finally:
        os.close(fd)
    observed = digest.hexdigest()
    if observed != item["sha256"]:
        raise RecheckError(f"{item['label']} SHA differs from activated queue")
    return {"label": item["label"], "path": str(path), "sha256": observed}


try:
    if len(sys.argv) != 2:
        raise RecheckError("one base64 JSON specification is required")
    spec = json.loads(base64.b64decode(sys.argv[1], validate=True))
    files = spec.get("files")
    if not isinstance(files, list) or len(files) != 4:
        raise RecheckError("exactly four parent snapshot files are required")
    result = [recheck(item) for item in files]
    print(json.dumps({
        "status": "PARENT_SNAPSHOT_RECHECK_OK", "job_id": spec["job_id"],
        "files": result,
    }, sort_keys=True))
except (RecheckError, OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
    print(f"PARENT_SNAPSHOT_RECHECK_ERROR: {exc}", file=sys.stderr)
    raise SystemExit(2)
'''


FIRST_ITER_PROGRAM = r'''import base64
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import time


class ProofError(RuntimeError):
    pass


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(
        value, allow_nan=False, ensure_ascii=False,
        separators=(",", ":"), sort_keys=True,
    ).encode("utf-8")).hexdigest()


def workspace_path(value, label):
    path = Path(value)
    if not path.is_absolute() or not str(path).startswith("/workspace/"):
        raise ProofError(f"{label} must be an absolute /workspace path")
    current = Path(path.anchor)
    for index, part in enumerate(path.parts[1:]):
        current = current / part
        try:
            item = current.lstat()
        except FileNotFoundError as exc:
            raise ProofError(f"{label} is missing: {current}") from exc
        if stat.S_ISLNK(item.st_mode):
            raise ProofError(f"{label} contains a symlink component")
        if index < len(path.parts[1:]) - 1 and not stat.S_ISDIR(item.st_mode):
            raise ProofError(f"{label} parent is not a directory")
    return path


def read_bytes(value, label):
    path = workspace_path(value, label)
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
            raise ProofError(f"{label} must be a non-empty regular file")
        chunks = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


def read_json(value, label):
    try:
        document = json.loads(read_bytes(value, label).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProofError(f"{label} is not valid JSON") from exc
    if not isinstance(document, dict):
        raise ProofError(f"{label} is not a mapping")
    return document


def proc_identity(expected, proc_root="/proc", getpgid=os.getpgid):
    pid = expected.get("pid")
    pgid = expected.get("pgid")
    starttime = expected.get("starttime_ticks")
    argv = expected.get("argv")
    if (
        type(pid) is not int or pid <= 0 or type(pgid) is not int or pgid != pid
        or type(starttime) is not int or starttime <= 0
        or not isinstance(argv, list) or not argv
        or not all(type(value) is str for value in argv)
    ):
        raise ProofError("run binding process identity is incomplete")
    root = Path(proc_root) / str(pid)
    try:
        stat_text = (root / "stat").read_text(encoding="utf-8")
        cmdline_raw = (root / "cmdline").read_bytes()
        observed_pgid = getpgid(pid)
    except (FileNotFoundError, ProcessLookupError) as exc:
        raise ProofError("bound trainer exited before a post-resume learning iteration") from exc
    close = stat_text.rfind(")")
    fields = [] if close < 0 else stat_text[close + 2:].split()
    if len(fields) <= 19 or not fields[19].isdigit():
        raise ProofError("bound trainer /proc stat is malformed")
    if fields[0] in {"Z", "X", "x"}:
        raise ProofError("bound trainer exited before a post-resume learning iteration")
    observed_starttime = int(fields[19])
    observed_argv = [
        value.decode("utf-8", "surrogateescape")
        for value in cmdline_raw.split(b"\0") if value
    ]
    if observed_pgid != pgid or observed_starttime != starttime:
        raise ProofError("bound trainer PID/PGID/starttime identity drifted or was reused")
    if observed_argv != argv:
        raise ProofError("bound trainer /proc argv differs from run binding")
    return {"pid": pid, "pgid": pgid, "starttime_ticks": starttime}


def first_post_resume_iteration(log):
    iterations = [
        int(match.group(1))
        for match in re.finditer(r"Learning iteration\s+(\d+)(?:/\d+)?", log)
    ]
    later = [value for value in iterations if value > 3500]
    return None if not later else min(later)


def envelope(value, label, schema):
    document = read_json(value, label)
    if document.get("schema_version") != schema:
        raise ProofError(f"{label} schema mismatch")
    content = document.get("content")
    if not isinstance(content, dict) or canonical_sha256(content) != document.get("content_sha256"):
        raise ProofError(f"{label} canonical SHA mismatch")
    return document, content


def write_failure(spec, error):
    process = None
    try:
        _binding, content = envelope(spec["binding_path"], "run binding", 1)
        raw = content.get("process")
        if isinstance(raw, dict):
            pid, pgid = raw.get("pid"), raw.get("pgid")
            if type(pid) is int and pid > 0 and type(pgid) is int and pgid > 0:
                process = {"pid": pid, "pgid": pgid,
                           "starttime_ticks": raw.get("starttime_ticks")}
    except Exception:
        pass
    content = {
        "schema_version": 1, "status": "first_iteration_proof_failed",
        "error": str(error), "job_id": spec["job_id"],
        "claim_content_sha256": spec["claim_content_sha256"],
        "run_binding_path": spec["binding_path"], "process": process,
        "manual_exact_pgid_disposition_required": process is not None,
        "automatic_retry": False, "signal_sent": False,
    }
    document = {
        "schema_version": 1, "content": content,
        "content_sha256": canonical_sha256(content),
    }
    payload = (json.dumps(document, allow_nan=False, ensure_ascii=False,
                          separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")
    path = Path(spec["failure_path"])
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o444)
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)


def main():
    if len(sys.argv) != 2:
        raise ProofError("one base64 JSON specification is required")
    spec = json.loads(base64.b64decode(sys.argv[1], validate=True))
    claim, claim_content = envelope(spec["claim_path"], "queue claim", 2)
    digest = claim.get("content_sha256")
    if digest != spec["claim_content_sha256"]:
        raise ProofError("queue claim differs from launch contract")
    argv = claim.get("training_argv")
    caller = claim_content.get("training_argv_without_claim")
    if argv != [*caller, f"++training_launch_claim_sha256={digest}"]:
        raise ProofError("queue claim argv is not self-bound")
    binding, bound = envelope(spec["binding_path"], "run binding", 1)
    expected_bound = {
        "job_id": spec["job_id"], "claim_path": spec["claim_path"],
        "claim_content_sha256": digest, "binding_path": spec["binding_path"],
        "run_name": spec["run_name"], "run_dir": spec["run_dir"],
        "pod": spec["pod"], "gpu": spec["gpu"], "training_argv": argv,
    }
    for key, value in expected_bound.items():
        if bound.get(key) != value:
            raise ProofError(f"run binding {key} mismatch")
    process = bound.get("process")
    if not isinstance(process, dict) or process.get("argv") != argv:
        raise ProofError("run binding lacks exact isolated PID/PGID")
    exact_process = proc_identity(process)
    hard_path = str(Path(bound["rsl_log_dir"]) / "params/training_contract.json")
    hard = read_json(hard_path, "child hard contract")
    if hard.get("schema_version") != 3:
        raise ProofError("child hard contract schema is not 3")
    qdot = hard.get("joint_velocity_limit_hinge_reward")
    face = hard.get("racket_guidance_reward", {}).get("conditional_signed_face")
    if not isinstance(qdot, dict) or qdot.get("weight") != spec["qdot_weight"]:
        raise ProofError("child qdot hard-contract weight mismatch")
    if not isinstance(face, dict) or face.get("weight") != spec["face_weight"]:
        raise ProofError("child conditional-face hard-contract weight mismatch")
    deadline = time.monotonic() + spec["timeout_s"]
    expected_resume = (
        f"[train.py] RESUMED from checkpoint: {spec['checkpoint_path']} "
        "(continuing at iteration 3500, optimizer=resumed)"
    )
    mismatch = "[train.py] WARNING: explicit hard-contract mismatch override:"
    while True:
        exact_process = proc_identity(process)
        try:
            log = read_bytes(spec["run_log_path"], "run log").decode("utf-8", "replace")
        except ProofError:
            log = ""
        first_later_iteration = first_post_resume_iteration(log)
        if (
            expected_resume in log and mismatch in log
            and first_later_iteration is not None
        ):
            exact_process = proc_identity(process)
            break
        if time.monotonic() >= deadline:
            raise ProofError(
                "run log lacks strict resume, mismatch, or Learning iteration > 3500"
            )
        time.sleep(1)
    content = {
        "schema_version": 1, "status": "strict_full_state_resume_proven",
        "job_id": spec["job_id"], "claim_content_sha256": digest,
        "run_binding_content_sha256": binding["content_sha256"],
        "process": exact_process,
        "checkpoint_path": spec["checkpoint_path"], "parent_iteration": 3500,
        "optimizer": "resumed", "explicit_hard_contract_mismatch": True,
        "first_observed_learning_iteration": first_later_iteration,
        "expected_child_lineage_exact": 0,
        "child_hard_contract_path": hard_path,
        "qdot_weight": spec["qdot_weight"], "face_weight": spec["face_weight"],
    }
    document = {
        "schema_version": 1, "content": content,
        "content_sha256": canonical_sha256(content),
    }
    payload = (json.dumps(document, allow_nan=False, ensure_ascii=False,
                          separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")
    fd = os.open(spec["success_path"], os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o444)
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    print(json.dumps(document, sort_keys=True))


try:
    main()
except (ProofError, OSError, KeyError, TypeError, ValueError) as exc:
    try:
        if len(sys.argv) == 2:
            write_failure(json.loads(base64.b64decode(sys.argv[1], validate=True)), exc)
    except Exception as record_error:
        print(f"FIRST_ITER_FAILURE_RECORD_ERROR: {record_error}", file=sys.stderr)
    print(f"FIRST_ITER_PROOF_ERROR: {exc}", file=sys.stderr)
    raise SystemExit(2)
'''


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, allow_nan=False, ensure_ascii=False,
            separators=(",", ":"), sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _values(job: dict[str, Any]) -> dict[str, str]:
    Q._compile_recipe_override_keys(job, job["id"])
    result: dict[str, str] = {}
    for raw in [*job["recipe"]["base"], *job["recipe"]["delta"]]:
        result[Q._override_key(raw, job["id"])] = raw.partition("=")[2]
    return result


def _require_sha(value: Any, label: str, *, pending: bool) -> str | None:
    if pending and value is None:
        return None
    if not isinstance(value, str) or Q.SHA256.fullmatch(value) is None:
        raise DemoQueueError(f"{label} must be a SHA-256")
    return value


def load_queue(path: Path) -> dict[str, Any]:
    try:
        queue = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise DemoQueueError(f"cannot read queue: {exc}") from exc
    if not isinstance(queue, dict) or queue.get("schema_version") != 1:
        raise DemoQueueError("queue schema_version must be 1")
    if queue.get("simulation_only") is not True:
        raise DemoQueueError("simulation_only must be true")
    if queue.get("dispatch_pods") != ["pod2"]:
        raise DemoQueueError("demo queue must dispatch only to Pod2")
    if queue.get("ssh") != {"key": EXPECTED_SSH_KEY}:
        raise DemoQueueError("SSH key changed")
    if queue.get("pods") != EXPECTED_PODS:
        raise DemoQueueError("exact Pod host/port/GPU/capacity contract changed")
    if queue.get("source_contract_files") != EXPECTED_SOURCE_CONTRACT_FILES:
        raise DemoQueueError("critical source-file hashes changed")

    activation = queue.get("activation_contract")
    if not isinstance(activation, dict):
        raise DemoQueueError("activation_contract is required")
    state = activation.get("state")
    if state not in {"pending_parent_receipt_and_gpu_release", "activated"}:
        raise DemoQueueError("invalid activation state")
    pending = state != "activated"
    expected_preregistration = (
        "blocked_until_parent_snapshot_receipt_and_fourth_slot_evidence"
        if pending else "activated_demo_only_inexact"
    )
    if queue.get("preregistration_status") != expected_preregistration:
        raise DemoQueueError("preregistration_status does not match activation state")
    if queue.get("launch_authorized") is not (not pending):
        raise DemoQueueError("launch_authorized must exactly follow activation state")
    receipt_path = Q._ready_workspace_path(
        activation.get("receipt_path"), "activation receipt path"
    )
    if receipt_path != EXPECTED_RECEIPT_PATH:
        raise DemoQueueError("activation must use the immutable v2 snapshot receipt path")
    _require_sha(
        activation.get("receipt_file_sha256"),
        "activation receipt_file_sha256", pending=pending,
    )
    if activation.get("gpu_release_rule") != (
        "pod2_gpu0_gpu1_occupancy_lte3_and_six_model500_preserved_"
        "allow_fourth_slot_jobs1_2"
    ):
        raise DemoQueueError("activation GPU release rule changed")
    if set(activation) != {
        "state", "receipt_path", "receipt_file_sha256", "gpu_release_rule",
        "automatic_activation", "automatic_retry",
    } or activation.get("automatic_activation") is not False or (
        activation.get("automatic_retry") is not False
    ):
        raise DemoQueueError("activation one-shot/no-retry fields changed")
    if queue.get("strict_full_scene_probe_evidence") != EXPECTED_PROBE_EVIDENCE:
        raise DemoQueueError("strict full-scene probe evidence changed")
    if queue.get("decision_contract") != EXPECTED_DECISION_CONTRACT:
        raise DemoQueueError("demo decision/exactness contract changed")

    parents = queue.get("parents")
    if not isinstance(parents, dict) or list(parents) != ["qdot", "v1v2", "control"]:
        raise DemoQueueError("parents must be ordered qdot, v1v2, control")
    for name, parent in parents.items():
        if not isinstance(parent, dict):
            raise DemoQueueError(f"parent {name} must be a mapping")
        expected_parent = EXPECTED_PARENT_SPECS[name]
        identity_keys = set(expected_parent)
        if any(parent.get(key) != value for key, value in expected_parent.items()):
            raise DemoQueueError(f"parent {name} identity/path contract changed")
        expected_keys = identity_keys | {
            "embedded_iteration", "optimizer_state_dict_required",
            "checkpoint_sha256", "hard_contract_sha256",
            "queue_claim_file_sha256", "queue_claim_content_sha256",
            "run_binding_file_sha256", "run_binding_content_sha256",
            "training_launch_claim_sha256",
        }
        if set(parent) != expected_keys:
            raise DemoQueueError(f"parent {name} has extra or missing fields")
        for prefix in ("live", "snapshot"):
            checkpoint = Q._ready_workspace_path(
                parent[f"{prefix}_checkpoint_path"],
                f"parent {name} {prefix} checkpoint",
            )
            hard = Q._ready_workspace_path(
                parent[f"{prefix}_hard_contract_path"],
                f"parent {name} {prefix} hard contract",
            )
            Q._ready_workspace_path(
                parent[f"{prefix}_queue_claim_path"],
                f"parent {name} {prefix} queue claim",
            )
            Q._ready_workspace_path(
                parent[f"{prefix}_run_binding_path"],
                f"parent {name} {prefix} run binding",
            )
            if PurePosixPath(checkpoint).name != "model_3500.pt":
                raise DemoQueueError(f"parent {name} {prefix} checkpoint is not model_3500.pt")
            if PurePosixPath(hard) != (
                PurePosixPath(checkpoint).parent / "params/training_contract.json"
            ):
                raise DemoQueueError(f"parent {name} {prefix} hard contract is not adjacent")
        if parent.get("embedded_iteration") != EXPECTED_PARENT_ITERATION:
            raise DemoQueueError(f"parent {name} iteration must be 3500")
        if parent.get("optimizer_state_dict_required") is not True:
            raise DemoQueueError(f"parent {name} must preserve optimizer state")
        for key in (
            "checkpoint_sha256", "hard_contract_sha256",
            "queue_claim_file_sha256", "queue_claim_content_sha256",
            "run_binding_file_sha256", "run_binding_content_sha256",
            "training_launch_claim_sha256",
        ):
            _require_sha(parent.get(key), f"parent {name} {key}", pending=pending)
        if (
            not pending
            and parent["training_launch_claim_sha256"]
            != parent["queue_claim_content_sha256"]
        ):
            raise DemoQueueError(
                f"parent {name} checkpoint launch claim differs from claim content SHA"
            )

    jobs = queue.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 9:
        raise DemoQueueError("exactly seven original rows and two retry-v2 rows are required")
    if [job.get("id") for job in jobs] != list(EXPECTED_JOB_SPECS):
        raise DemoQueueError("original/retry job ids or order changed")
    if [job.get("resource", {}).get("required_slot") for job in jobs] != EXPECTED_SLOTS:
        raise DemoQueueError("original/retry GPU bindings changed")
    ids: set[str] = set()
    runs: set[str] = set()
    for job in jobs:
        if not isinstance(job, dict):
            raise DemoQueueError("each job must be a mapping")
        job_id = job.get("id")
        if not isinstance(job_id, str) or not Q.SAFE_ID.fullmatch(job_id) or job_id in ids:
            raise DemoQueueError("job ids must be unique safe identifiers")
        ids.add(job_id)
        parent_name, expected_slot, expected_run_name, expected_run_dir, knobs = (
            EXPECTED_JOB_SPECS[job_id]
        )
        expected_job_fields = {
            "id", "human_name", "action", "status", "blocker", "motion",
            "bank", "exam", "source", "runtime_binding", "warm_start",
            "recipe", "seed", "budget", "milestones", "resource",
            "run_name", "run_dir",
        }
        if job_id == LONG_CARRY_JOB_ID:
            expected_job_fields.add("screening_contract")
        if job_id in REJECTED_PREDECESSOR_IDS:
            expected_job_fields.add("terminal_contract")
        if job_id in RETRY_IDS:
            expected_job_fields.add("retry_contract")
        if set(job) != expected_job_fields:
            raise DemoQueueError(f"{job_id} has extra or missing job fields")
        if job_id == LONG_CARRY_JOB_ID:
            if job.get("screening_contract") != EXPECTED_LONG_CARRY_SCREENING_CONTRACT:
                raise DemoQueueError(f"{job_id} screening contract changed")
        if job_id in REJECTED_PREDECESSOR_IDS:
            expected_status = "rejected"
        else:
            expected_status = "blocked" if pending else "ready"
        if job.get("status") != expected_status:
            raise DemoQueueError(f"{job_id} status does not match activation state")
        if job_id in REJECTED_PREDECESSOR_IDS:
            if not isinstance(job.get("blocker"), str) or not job["blocker"]:
                raise DemoQueueError(f"{job_id} rejected row must explain terminal evidence")
            if job.get("terminal_contract") != EXPECTED_TERMINAL_CONTRACTS[job_id]:
                raise DemoQueueError(f"{job_id} terminal evidence contract changed")
        elif pending and not isinstance(job.get("blocker"), str):
            raise DemoQueueError(f"{job_id} must explain its blocker")
        elif not pending and job.get("blocker") not in (None, ""):
            raise DemoQueueError(f"{job_id} ready row still has a blocker")
        if job_id in RETRY_IDS:
            if job.get("retry_contract") != EXPECTED_RETRY_CONTRACTS[job_id]:
                raise DemoQueueError(f"{job_id} retry contract changed")
        if job.get("runtime_binding") is not True:
            raise DemoQueueError(f"{job_id} requires runtime binding")
        expected_source = {
            "checkout": EXPECTED_SOURCE_CHECKOUT, "commit": EXPECTED_SOURCE,
            "ignored_runtime_asset": EXPECTED_IGNORED_RUNTIME_ASSET,
        }
        if job.get("source") != expected_source:
            raise DemoQueueError(f"{job_id} exact source/asset closure changed")
        if job.get("action") != "signed_face_v4rg_shared_face":
            raise DemoQueueError(f"{job_id} action changed")
        if job.get("motion") != {
            "action": "signed_face_v4rg_shared_face",
            "bindings": EXPECTED_MOTION_BINDINGS,
        }:
            raise DemoQueueError(f"{job_id} v4rg motion binding changed")
        if job.get("bank") != {
            "action": "signed_face_v4rg_shared_face", "train_path": EXPECTED_BANK,
            "train_arg": "++task.racket.question_bank",
        }:
            raise DemoQueueError(f"{job_id} schema-3 bank changed")
        if job.get("exam") != {
            "action": "signed_face_v4rg_shared_face", "path": EXPECTED_EXAM,
            "family": "signed_face_rebound_k100_v1",
        }:
            raise DemoQueueError(f"{job_id} immutable exam changed")
        warm = job.get("warm_start")
        if isinstance(warm, dict) and warm.get("descendant_exact_eligible") is not False:
            raise DemoQueueError(f"{job_id} descendants must remain exact-ineligible")
        expected_warm = {
            "parent": parent_name,
            "transfer_mode": "strict_full_state_preserve_optimizer",
            "checkpoint_tolerant": False, "allow_missing_contract": False,
            "allow_contract_mismatch": True, "descendant_exact_eligible": False,
        }
        if warm != expected_warm:
            raise DemoQueueError(f"{job_id} exact warm-start contract changed")
        expected_delta = (
            f"checkpoint_path={parents[parent_name]['snapshot_checkpoint_path']}",
            f"++task.rewards.free_wrist_vel_mimic={knobs[0]}",
            f"++task.rewards.motion_scale_in_window={knobs[1]}",
            f"task.rewards.joint_velocity_limit_hinge_weight={knobs[2]}",
            f"++task.rewards.racket_face_conditional_guidance_weight={knobs[3]}",
            f"task.rewards.foot_orientation_weight={knobs[4]}",
            f"++task.rewards.free_non_striking_arm_mimic={knobs[5]}",
        )
        recipe = job.get("recipe")
        if not isinstance(recipe, dict) or set(recipe) != {"base", "delta"}:
            raise DemoQueueError(f"{job_id} recipe has extra or missing sections")
        expected_base = (
            EXPECTED_LONG_BASE_RECIPE
            if job_id == LONG_CARRY_JOB_ID else EXPECTED_BASE_RECIPE
        )
        if tuple(recipe.get("base", ())) != expected_base:
            raise DemoQueueError(f"{job_id} full base recipe changed")
        if tuple(recipe.get("delta", ())) != expected_delta:
            raise DemoQueueError(f"{job_id} full delta recipe changed")
        values = _values(job)
        expected_values = {
            Q._override_key(raw, job_id): raw.partition("=")[2]
            for raw in (*expected_base, *expected_delta)
        }
        if values != expected_values:
            raise DemoQueueError(f"{job_id} final Hydra key/value mapping changed")
        parent_path = parents[parent_name]["snapshot_checkpoint_path"]
        required = {
            "checkpoint_path": parent_path,
            "checkpoint_tolerant": "false",
            "checkpoint_allow_missing_contract": "false",
            "checkpoint_allow_contract_mismatch": "true",
            "task.env.episode_length_s": (
                "16.0" if job_id == LONG_CARRY_JOB_ID else "10.0"
            ),
            "task.rewards.racket_position_weight": "14.0",
            "task.rewards.racket_velocity_weight": "10.0",
            "task.rewards.racket_normal_weight": "5.0",
        }
        for key, expected in required.items():
            if values.get(key) != expected:
                raise DemoQueueError(f"{job_id} requires {key}={expected}")
        budget = job.get("budget")
        if budget != {"num_envs": 4096, "max_iterations": 5001, "save_interval": 100}:
            raise DemoQueueError(f"{job_id} budget changed")
        if job.get("milestones") != EXPECTED_MILESTONES:
            raise DemoQueueError(f"{job_id} absolute milestones changed")
        if job.get("seed") != 3:
            raise DemoQueueError(f"{job_id} seed changed")
        if job.get("resource") != {
            "policy": "dispatch_gpu_round_robin", "required_slot": expected_slot,
        }:
            raise DemoQueueError(f"{job_id} required slot changed")
        run_name = job.get("run_name")
        if not isinstance(run_name, str) or not Q.SAFE_ID.fullmatch(run_name) or run_name in runs:
            raise DemoQueueError("run names must be unique safe identifiers")
        runs.add(run_name)
        if run_name != expected_run_name or job.get("run_dir") != expected_run_dir:
            raise DemoQueueError(f"{job_id} run_name/run_dir changed")
        Q._ready_workspace_path(expected_run_dir, f"{job_id} run_dir")
    by_id = {job["id"]: job for job in jobs}
    for retry_id, predecessor_id in RETRY_PREDECESSORS.items():
        retry = by_id[retry_id]
        predecessor = by_id[predecessor_id]
        for field in (
            "action", "motion", "bank", "exam", "source", "runtime_binding",
            "warm_start", "recipe", "seed", "budget", "milestones",
        ):
            if retry[field] != predecessor[field]:
                raise DemoQueueError(
                    f"{retry_id} must remain recipe-identical to {predecessor_id}: {field}"
                )
        if retry["run_dir"] == predecessor["run_dir"] or (
            retry["run_name"] == predecessor["run_name"]
        ):
            raise DemoQueueError(f"{retry_id} must use a fresh run namespace")
    return queue


def _parent_spec(queue: dict[str, Any], *, mode: str) -> dict[str, Any]:
    activation = queue["activation_contract"]
    return {
        "mode": mode,
        "source_commit": EXPECTED_SOURCE,
        "receipt_path": activation["receipt_path"],
        "expected_receipt_sha256": (
            activation["receipt_file_sha256"] if mode == "verify" else None
        ),
        "parents": {
            name: {
                **{
                    key: parent[key]
                    for key in EXPECTED_PARENT_SPECS[name]
                },
                "source_checkout": EXPECTED_SOURCE_CHECKOUT,
                "source_commit": EXPECTED_SOURCE,
                "iteration": parent["embedded_iteration"],
                "descendant_contract_values": [
                    {
                        "job_id": job["id"],
                        "qdot_weight": float(
                            _values(job)["task.rewards.joint_velocity_limit_hinge_weight"]
                        ),
                        "face_weight": float(
                            _values(job)["task.rewards.racket_face_conditional_guidance_weight"]
                        ),
                    }
                    for job in queue["jobs"]
                    if (
                        job["warm_start"]["parent"] == name
                        and job["id"] not in RETRY_IDS
                    )
                ],
            }
            for name, parent in queue["parents"].items()
        },
    }


def _parent_remote(queue: dict[str, Any], *, mode: str) -> str:
    encoded = base64.b64encode(
        json.dumps(_parent_spec(queue, mode=mode), sort_keys=True).encode("utf-8")
    ).decode("ascii")
    return shlex.join([Q.ISAAC_PYTHON, "-c", PARENT_PROGRAM, encoded])


def _demo_claim(
    queue: dict[str, Any], job: dict[str, Any], slot: Any
) -> tuple[dict[str, Any], list[str]]:
    claim, _old_argv = Q._launch_contract(queue, job, slot)
    content = claim["content"]
    parent_name = job["warm_start"]["parent"]
    content["demo_warm_start"] = {
        **job["warm_start"],
        **queue["parents"][parent_name],
    }
    content["activation_receipt"] = {
        "path": queue["activation_contract"]["receipt_path"],
        "file_sha256": queue["activation_contract"]["receipt_file_sha256"],
    }
    content["formal_exact_eligible"] = False
    content["source_contract_files"] = dict(EXPECTED_SOURCE_CONTRACT_FILES)
    if job["id"] == LONG_CARRY_JOB_ID:
        content["screening_contract"] = dict(job["screening_contract"])
    if job["id"] in RETRY_IDS:
        content["retry_contract"] = dict(job["retry_contract"])
    digest = _canonical_sha256(content)
    argv = [
        *content["training_argv_without_claim"],
        f"++training_launch_claim_sha256={digest}",
    ]
    return {
        "schema_version": 2,
        "content": content,
        "content_sha256": digest,
        "training_argv": argv,
    }, argv


def _launch_script(queue: dict[str, Any], job: dict[str, Any], slot: Any) -> str:
    source = job["source"]["checkout"].rstrip("/")
    workdir = f"{source}/{Q.WBT_RELATIVE}"
    run_dir = job["run_dir"].rstrip("/")
    run_parent = str(PurePosixPath(run_dir).parent)
    claim_document, argv = _demo_claim(queue, job, slot)
    claim = json.dumps(
        claim_document, allow_nan=False, ensure_ascii=False,
        separators=(",", ":"), sort_keys=True,
    ) + "\n"
    values = _values(job)
    proof_spec = {
        "job_id": job["id"], "pod": slot.pod, "gpu": slot.gpu,
        "run_name": job["run_name"], "run_dir": run_dir,
        "claim_path": f"{run_dir}/queue_claim.json",
        "binding_path": f"{run_dir}/run_binding.json",
        "run_log_path": f"{run_dir}/run.log",
        "success_path": f"{run_dir}/first_iter_resume_proof.json",
        "failure_path": f"{run_dir}/first_iter_resume_failure.json",
        "claim_content_sha256": claim_document["content_sha256"],
        "checkpoint_path": values["checkpoint_path"],
        "qdot_weight": float(values["task.rewards.joint_velocity_limit_hinge_weight"]),
        "face_weight": float(values["task.rewards.racket_face_conditional_guidance_weight"]),
        "timeout_s": 120,
    }
    proof_encoded = base64.b64encode(
        json.dumps(proof_spec, sort_keys=True).encode("utf-8")
    ).decode("ascii")
    proof_command = shlex.join(["python3", "-c", FIRST_ITER_PROGRAM, proof_encoded])
    parent = queue["parents"][job["warm_start"]["parent"]]
    snapshot_recheck_spec = {
        "job_id": job["id"],
        "files": [
            {
                "label": "parent checkpoint",
                "path": parent["snapshot_checkpoint_path"],
                "sha256": parent["checkpoint_sha256"],
            },
            {
                "label": "parent hard contract",
                "path": parent["snapshot_hard_contract_path"],
                "sha256": parent["hard_contract_sha256"],
            },
            {
                "label": "parent queue claim",
                "path": parent["snapshot_queue_claim_path"],
                "sha256": parent["queue_claim_file_sha256"],
            },
            {
                "label": "parent run binding",
                "path": parent["snapshot_run_binding_path"],
                "sha256": parent["run_binding_file_sha256"],
            },
        ],
    }
    snapshot_recheck_encoded = base64.b64encode(
        json.dumps(snapshot_recheck_spec, sort_keys=True).encode("utf-8")
    ).decode("ascii")
    snapshot_recheck_command = shlex.join(
        ["python3", "-c", SNAPSHOT_RECHECK_PROGRAM, snapshot_recheck_encoded]
    )
    source_hash_checks = "\n".join(
        "test \"$(sha256sum "
        + shlex.quote(f"{source}/{relative}")
        + " | awk '{print $1}')\" = "
        + shlex.quote(digest)
        for relative, digest in EXPECTED_SOURCE_CONTRACT_FILES.items()
    )
    launcher = f"{workdir}/{Q.KIT_LAUNCHER_RELATIVE}"
    launch = shlex.join([launcher, f"{run_dir}/run.log"]) + " " + (
        Q._child_env_command(argv, slot.gpu)
    ) + f" {Q.GPU_LAUNCH_LOCK_FD}>&-"
    body = snapshot_recheck_command + "\n" + source_hash_checks + "\n" + Q._doctor_body(
        queue, job, slot, training_argv=argv
    ) + f"""
count=$(nvidia-smi -i {slot.gpu} --query-compute-apps=pid --format=csv,noheader,nounits | awk {shlex.quote(Q.UNIQUE_NUMERIC_PID_AWK)})
test "$count" -lt {slot.capacity}
mkdir -p {shlex.quote(run_parent)}
mkdir {shlex.quote(run_dir)}
mkdir {shlex.quote(run_dir + '/milestones')}
( set -o noclobber; printf %s {shlex.quote(claim)} > {shlex.quote(run_dir + '/queue_claim.json')} )
export KIT_BOOT_MARKER={shlex.quote(Q.KIT_BOOT_MARKER)}
export KIT_BOOT_TIMEOUT_S={Q.KIT_BOOT_TIMEOUT_SECONDS}
{launch}
{proof_command}
printf '%s\n' phase=first_iter demo_only=true exact_eligible=false strict_full_state_resume_proven=true expected_lineage_exact=0 >> {shlex.quote(run_dir + '/run.log.launch')}
"""
    return Q._gpu_launch_lock_script(slot, body)


def _assign_demo(
    queue: dict[str, Any], occupancy: dict[str, int],
    existing_ids: set[str] | None = None,
) -> list[tuple[dict[str, Any], Any]]:
    """Apply the long-carry slot gate and ordered manual retry gate.

    The generic scheduler correctly enforces capacity, but it intentionally has
    no experiment-specific notion of "only add the fourth colocated trainer".
    Keep that policy local to this demo queue and re-check it from the same live
    occupancy snapshot used for assignment.  Retry sequence 2 cannot be
    selected until sequence 1 has consumed its fresh one-shot namespace.
    """

    claimed = set(existing_ids or set())
    assignments = Q._assign(queue, occupancy, claimed)
    first_retry = next(iter(RETRY_PREDECESSORS))
    second_retry = list(RETRY_PREDECESSORS)[1]
    return [
        (job, slot)
        for job, slot in assignments
        if (
            (job["id"] != LONG_CARRY_JOB_ID or occupancy.get(slot.name) == 3)
            and (job["id"] != second_retry or first_retry in claimed)
        )
    ]


def cmd_plan(queue: dict[str, Any]) -> dict[str, Any]:
    occupancy = {slot.name: 0 for slot in Q.slots(queue)}
    assignments = Q._assign(queue, occupancy)
    return {
        "mode": "plan",
        "dry_run": True,
        "launch_authorized": queue["launch_authorized"],
        "activation_state": queue["activation_contract"]["state"],
        "assignments": [
            {"job_id": job["id"], "resource": slot.name}
            for job, slot in assignments
        ],
        "blocked": [
            {"job_id": job["id"], "reason": job["blocker"]}
            for job in queue["jobs"] if job["status"] == "blocked"
        ],
        "parent_inspect_command": f"--execute --confirm {PARENT_INSPECT_CONFIRM}",
        "parent_attest_command": f"--execute --confirm {PARENT_ATTEST_CONFIRM}",
    }


def cmd_parent_attest(
    queue: dict[str, Any], *, execute: bool, confirm: str | None
) -> dict[str, Any]:
    if queue["activation_contract"]["state"] != "pending_parent_receipt_and_gpu_release":
        raise DemoQueueError("parent-attest is only valid before activation")
    if execute and confirm != PARENT_ATTEST_CONFIRM:
        raise DemoQueueError(f"--execute requires --confirm {PARENT_ATTEST_CONFIRM}")
    remote = _parent_remote(queue, mode="attest")
    inspect_remote = _parent_remote(queue, mode="inspect")
    result: dict[str, Any] = {
        "mode": "parent-attest", "dry_run": not execute,
        "receipt_path": queue["activation_contract"]["receipt_path"],
        "automatic_activation": False, "automatic_retry": False,
    }
    if not execute:
        result["preflight_ssh_argv"] = [
            *Q._ssh_prefix(queue, "pod2"), f"bash -lc {shlex.quote(inspect_remote)}"
        ]
        result["attest_ssh_argv"] = [
            *Q._ssh_prefix(queue, "pod2"), f"bash -lc {shlex.quote(remote)}"
        ]
        result["ssh_argv"] = result["attest_ssh_argv"]
        return result
    result["read_only_preflight"] = json.loads(
        Q._run_ssh(
            queue, "pod2", inspect_remote, timeout=600,
            phase="demo-parent-inspect-before-attest",
        )
    )
    result["remote_result"] = json.loads(
        Q._run_ssh(queue, "pod2", remote, timeout=600, phase="demo-parent-attest")
    )
    return result


def cmd_parent_inspect(
    queue: dict[str, Any], *, execute: bool, confirm: str | None
) -> dict[str, Any]:
    if queue["activation_contract"]["state"] != "pending_parent_receipt_and_gpu_release":
        raise DemoQueueError("parent-inspect is only valid before activation")
    if execute and confirm != PARENT_INSPECT_CONFIRM:
        raise DemoQueueError(f"--execute requires --confirm {PARENT_INSPECT_CONFIRM}")
    remote = _parent_remote(queue, mode="inspect")
    result: dict[str, Any] = {
        "mode": "parent-inspect", "dry_run": not execute,
        "read_only": True, "creates_snapshots": False, "creates_receipt": False,
        "receipt_preview_path": queue["activation_contract"]["receipt_path"],
    }
    if not execute:
        result["ssh_argv"] = [
            *Q._ssh_prefix(queue, "pod2"), f"bash -lc {shlex.quote(remote)}"
        ]
        return result
    result["remote_result"] = json.loads(
        Q._run_ssh(queue, "pod2", remote, timeout=600, phase="demo-parent-inspect")
    )
    return result


def _require_remote_activation(queue: dict[str, Any]) -> dict[str, Any]:
    if queue["activation_contract"]["state"] != "activated":
        raise DemoQueueError("activation state is not activated")
    output = Q._run_ssh(
        queue, "pod2", _parent_remote(queue, mode="verify"),
        timeout=600, phase="demo-parent-verify",
    )
    result = json.loads(output)
    if result.get("receipt_file_sha256") != queue["activation_contract"]["receipt_file_sha256"]:
        raise DemoQueueError("verified receipt SHA differs from activation contract")
    try:
        observed = result["receipt"]["content"]["parents"]
    except (KeyError, TypeError) as exc:
        raise DemoQueueError("verified receipt lacks parent content") from exc
    for name, expected in queue["parents"].items():
        current = observed.get(name)
        if not isinstance(current, dict):
            raise DemoQueueError(f"verified receipt lacks parent {name}")
        bound = {
            "checkpoint_sha256": expected["checkpoint_sha256"],
            "hard_contract_sha256": expected["hard_contract_sha256"],
            "queue_claim_file_sha256": expected["queue_claim_file_sha256"],
            "queue_claim_content_sha256": expected["queue_claim_content_sha256"],
            "run_binding_file_sha256": expected["run_binding_file_sha256"],
            "run_binding_content_sha256": expected["run_binding_content_sha256"],
            "embedded_iteration": expected["embedded_iteration"],
            "optimizer_state_dict_present": True,
            "optimizer_state_dict_nonempty": True,
            "parent_training_contract_lineage_exact": True,
            "training_launch_claim_sha256": expected["training_launch_claim_sha256"],
        }
        for key, value in bound.items():
            if current.get(key) != value:
                raise DemoQueueError(
                    f"verified parent {name}.{key} differs from activated queue"
                )
        expected_live = {
            key.removeprefix("live_"): expected[key]
            for key in (
                "live_checkpoint_path", "live_hard_contract_path",
                "live_queue_claim_path", "live_run_binding_path",
            )
        }
        expected_snapshot = {
            key.removeprefix("snapshot_"): expected[key]
            for key in (
                "snapshot_checkpoint_path", "snapshot_hard_contract_path",
                "snapshot_queue_claim_path", "snapshot_run_binding_path",
            )
        }
        if current.get("live_paths") != expected_live:
            raise DemoQueueError(f"verified parent {name} live paths changed")
        if current.get("snapshot_paths") != expected_snapshot:
            raise DemoQueueError(f"verified parent {name} snapshot paths changed")
    return result


def cmd_fill(
    queue: dict[str, Any], *, execute: bool, confirm: str | None, count: int
) -> dict[str, Any]:
    if queue["launch_authorized"] is not True:
        raise DemoQueueError("launch_authorized is false; fill is blocked")
    if count <= 0:
        raise DemoQueueError("count must be positive")
    if execute and confirm != LAUNCH_CONFIRM:
        raise DemoQueueError(f"--execute requires --confirm {LAUNCH_CONFIRM}")
    if not execute:
        occupancy = {slot.name: 0 for slot in Q.slots(queue)}
        assignments = Q._assign(queue, occupancy)[:count]
        return {
            "mode": "fill", "dry_run": True,
            "jobs": [
                {"job_id": job["id"], "resource": slot.name,
                 "ssh_argv": [*Q._ssh_prefix(queue, slot.pod),
                              f"bash -lc {shlex.quote(_launch_script(queue, job, slot))}"]}
                for job, slot in assignments
            ],
        }
    activation = _require_remote_activation(queue)
    launched: list[dict[str, Any]] = []
    Q.GLOBAL_SCHEDULER_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with Q.GLOBAL_SCHEDULER_LOCK.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        for _ in range(count):
            occupancy, claims = Q.live_snapshot(queue)
            effective = Q._effective_occupancy(queue, occupancy, claims)
            assignments = _assign_demo(queue, effective, set(claims))
            if not assignments:
                break
            job, slot = assignments[0]
            output = Q._run_ssh(
                queue, slot.pod, _launch_script(queue, job, slot),
                timeout=Q.KIT_BOOT_TIMEOUT_SECONDS + 60,
                phase=f"demo-hotstart:{job['id']}",
            )
            launched.append({"job_id": job["id"], "resource": slot.name,
                             "remote_output": output})
    if not launched:
        raise DemoQueueError("no ready job fits an available GPU slot")
    return {"mode": "fill", "dry_run": False,
            "activation": activation, "launched": launched}


def cmd_attest_milestone(
    queue: dict[str, Any], *, job_id: str, milestone: int,
    execute: bool, confirm: str | None,
) -> dict[str, Any]:
    jobs = {job["id"]: job for job in queue["jobs"]}
    if job_id not in jobs or milestone not in EXPECTED_MILESTONES:
        raise DemoQueueError("unknown job or non-preregistered milestone")
    if execute and confirm != ATTEST_CONFIRM:
        raise DemoQueueError(f"--execute requires --confirm {ATTEST_CONFIRM}")
    job = jobs[job_id]
    slot_name = job["resource"]["required_slot"]
    slot = next(slot for slot in Q.slots(queue) if slot.name == slot_name)
    remote = Q._milestone_attestor_script(job, milestone)
    result: dict[str, Any] = {
        "mode": "attest-milestone", "dry_run": not execute,
        "job_id": job_id, "milestone": milestone,
        "expected_lineage_exact": 0,
    }
    if not execute:
        result["ssh_argv"] = [
            *Q._ssh_prefix(queue, slot.pod), f"bash -lc {shlex.quote(remote)}"
        ]
        return result
    _occupancy, claims = Q.live_snapshot(queue)
    claim = claims.get(job_id)
    if claim is None or claim.get("claim_schema_version") != 2:
        raise DemoQueueError("live schema-2 job claim is missing")
    expected, _argv = _demo_claim(queue, job, slot)
    if claim.get("claim_content_sha256") != expected["content_sha256"]:
        raise DemoQueueError("live claim differs from current demo queue")
    remote_output = Q._run_ssh(
        queue, slot.pod, remote, timeout=180,
        phase=f"demo-attest:{job_id}:{milestone}",
    )
    receipt = json.loads(remote_output)
    try:
        lineage = receipt["receipt"]["content"]["hard_contract"]["lineage_exact"]
    except (KeyError, TypeError) as exc:
        raise DemoQueueError("milestone attestor omitted lineage exactness") from exc
    if type(lineage) is not int or lineage != 0:
        raise DemoQueueError("demo warm-start descendant did not remain lineage_exact=0")
    result["remote_result"] = receipt
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan")
    inspect = sub.add_parser("parent-inspect")
    inspect.add_argument("--execute", action="store_true")
    inspect.add_argument("--confirm")
    parent = sub.add_parser("parent-attest")
    parent.add_argument("--execute", action="store_true")
    parent.add_argument("--confirm")
    fill = sub.add_parser("fill")
    fill.add_argument("--count", type=int, default=1)
    fill.add_argument("--execute", action="store_true")
    fill.add_argument("--confirm")
    attest = sub.add_parser("attest-milestone")
    attest.add_argument("--job-id", required=True)
    attest.add_argument("--milestone", type=int, required=True)
    attest.add_argument("--execute", action="store_true")
    attest.add_argument("--confirm")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        queue = load_queue(args.queue.resolve())
        if args.command == "plan":
            result = cmd_plan(queue)
        elif args.command == "parent-inspect":
            result = cmd_parent_inspect(
                queue, execute=args.execute, confirm=args.confirm
            )
        elif args.command == "parent-attest":
            result = cmd_parent_attest(
                queue, execute=args.execute, confirm=args.confirm
            )
        elif args.command == "fill":
            result = cmd_fill(
                queue, execute=args.execute, confirm=args.confirm, count=args.count
            )
        elif args.command == "attest-milestone":
            result = cmd_attest_milestone(
                queue, job_id=args.job_id, milestone=args.milestone,
                execute=args.execute, confirm=args.confirm,
            )
        else:
            raise DemoQueueError(f"unsupported command: {args.command}")
    except (DemoQueueError, Q.QueueError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
