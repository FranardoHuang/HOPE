#!/usr/bin/env python3
"""Validate the runtime-blocked Gate3 serve-sync preregistration.

This is a source/design check only.  Its only child commands are two fixed,
read-only Git identity queries.  It never launches, waits on, or signals a
planner, runner, simulator, publisher, Pod, or robot process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


EXPECTED_PREREG_RELATIVE_PATH = Path("configs/gate3_serve_sync_prereg_20260712.json")
EXPECTED_SCOPE = (
    "source-only fail-closed Gate3 serve synchronization design; no simulator, "
    "publisher, signal, Pod, or robot action"
)
EXPECTED_PLAN_GATE_DEPENDENCY = {
    "merge_commit": "b2067ba72a9ea65d82cccec676d8af223d518bcd",
    "source_path": "scripts/run_gate3_first_tick_harness.py",
    "source_sha256": "612f68bfdd7375838f38d4f89a25fcee5db1e2ce19eac7e55b60bdee47b4d680",
    "legacy_audit_path": "configs/gate3_legacy_process_audit_20260712.json",
    "legacy_audit_sha256": "3dce92d777959b18d7fb0c0d38f3193e366f2aa030830d7fa48f4df3422010dc",
    "role": "plan-only prerequisite with no runtime, publisher, or signal authority",
}
EXPECTED_SOURCE_DEPENDENCIES = [
    {
        "path": "hope_ws/src/hope_planner/hope_planner/node.py",
        "sha256": "52a70da3fbcefbd4365d87b3e31ffa6ef11e7a7dfe630b1d1a6bf9b07dcea75d",
        "role": "planner formal source, frame, epoch, sequence, and revoke semantics",
    },
    {
        "path": "hope_ws/src/hope_planner/hope_planner/node_runtime_contract.py",
        "sha256": "a1f367a19e1ace7584cde030d9b315a3a549aac2ecbf6aad8033eea6930f4d1e",
        "role": "source-clock, base-lease, and runtime configuration validation",
    },
    {
        "path": "hope_ws/src/hope_planner/hope_planner/flat_command_wire.py",
        "sha256": "052384884382fbaffeaf2263121956cf030cca6bb12791fa15ced3142e6a21b9",
        "role": "planner formal racket and base wire encoding",
    },
    {
        "path": "hope_ws/src/hope_planner/config/hope_planner.yaml",
        "sha256": "12e8c8dcd937c208fad28dcfe5d820aef9ad53b1edd32490c9d2c89981792817",
        "role": "arena planner runtime configuration",
    },
    {
        "path": "hope_ws/src/hope_planner/config/hope_planner.sim.yaml",
        "sha256": "efdb726e246fc4264a90ef6f187b19b891763fd5f1827706501708f4d3a19cea",
        "role": "vendor simulator planner runtime configuration and explicit frame blocker",
    },
    {
        "path": (
            "agi/a3_deploy_example/src/a3/a3_deploy_onnx_ref/include/"
            "a3_pingpong/pp_planner_input.hpp"
        ),
        "sha256": "1bb41836c532167ecf561c85d3c913d5ed6be590840214dfa747d5ace3c2eb2b",
        "role": "runner formal mailbox, epoch, sequence, and revoke parsing",
    },
    {
        "path": (
            "agi/a3_deploy_example/src/a3/a3_deploy_onnx_ref/include/"
            "a3_pingpong/pp_policy.hpp"
        ),
        "sha256": "f91b1f5db4e35163abce6923567365bb177c094573de2740f443e3ada1b01de1",
        "role": "runner same-tick planner-policy engage, wait, abort, and active lease gates",
    },
    {
        "path": (
            "agi/a3_deploy_example/src/a3/a3_deploy_onnx_ref/include/"
            "a3_pingpong/pp_frame_math.hpp"
        ),
        "sha256": "4581a3b0b800b4ae63afd6cf387d7fce3d8e7d9d8b7b09745cef7273eba9e24d",
        "role": "runner target frame and face conversion helpers",
    },
    {
        "path": (
            "agi/a3_deploy_example/src/a3/a3_deploy_onnx_ref/include/"
            "a3_pingpong/pp_reference_clock.hpp"
        ),
        "sha256": "d943b2f0f945a7a29ab734bd10d186b896ca2152ba658aecfc563951e3fcadfb",
        "role": "runner per-clip reference timing",
    },
    {
        "path": (
            "agi/a3_deploy_example/src/a3/a3_deploy_onnx_ref/src/a3_deploy/"
            "a3_pingpong_main.cpp"
        ),
        "sha256": "fc4ac19c57ccd3bbc7853785ec01a1d4c3b66666131de89f0f932c8c4f08b389",
        "role": "runner CLI, model, and planner input binding",
    },
    {
        "path": (
            "agi/a3_deploy_example/src/a3/a3_deploy_onnx_ref/include/robot_io/"
            "a3_aimrt_backend.hpp"
        ),
        "sha256": "262981e24c3bd504f5a7e9b3b79725ae25a2b7aa72637c7b7ab23e6b4a3bb951",
        "role": "runner AimRT backend topic ownership and callback interface",
    },
    {
        "path": (
            "agi/a3_deploy_example/src/a3/a3_deploy_onnx_ref/src/robot_io/"
            "a3_aimrt_backend.cpp"
        ),
        "sha256": "4b93621f54f08f47e8c63a4d898abd11b52f1457723c26e5d618742dda9e93bc",
        "role": "runner AimRT backend subscriptions, publications, and callback ordering",
    },
    {
        "path": (
            "agi/a3_deploy_example/src/a3/a3_deploy_onnx_ref/config/"
            "a3_aimrt_config.pingpong_ros2body.yaml"
        ),
        "sha256": "48440e112c1deb7c281eb21b40a0d521dae6ee524826ac3aab62eab06fce734a",
        "role": "runner ping-pong AimRT channel and plugin configuration",
    },
    {
        "path": (
            "agi/a3_deploy_example/src/a3/a3_deploy_onnx_ref/config/"
            "a3_runtime_config.pingpong.yaml"
        ),
        "sha256": "844f6eec122ec9f9eb5c18caba5e5dab4ea31e53b14fe0bf46dd73c0e3f35c42",
        "role": "runner ping-pong runtime backend and timing selection",
    },
    {
        "path": (
            "agi/a3_deploy_example/src/a3/a3_deploy_onnx_ref/include/robot_io/"
            "robot_io_backend.hpp"
        ),
        "sha256": "9c8ade30b76c8c10c8d4274ddf2c1c3dadb882a97516fb1e919298b5862ec1b9",
        "role": "runner backend ownership and lifecycle contract",
    },
    {
        "path": (
            "agi/a3_deploy_example/src/a3/a3_deploy_onnx_ref/include/a3_deploy/"
            "a3_policy_driver.hpp"
        ),
        "sha256": "c118e517845723785b50bafa0525d01b70cc65260abf34d4487f21e14249c572",
        "role": "runner policy driver ownership and compute interface",
    },
    {
        "path": (
            "agi/a3_deploy_example/src/a3/a3_deploy_onnx_ref/src/a3_deploy/"
            "a3_policy_driver.cpp"
        ),
        "sha256": "0116cd16ab319943cbebbeba0d73fce6b805bc3212454810347566608c764cee",
        "role": "runner policy driver lifecycle and backend dispatch",
    },
    {
        "path": "agi/a3_deploy_example/src/a3/a3_deploy_onnx_ref/CMakeLists.txt",
        "sha256": "3f500cfbedcd13a93446785e088c30a6b6b60cc4023405aba1a11e925c8a10c8",
        "role": "runner compiled source, link, plugin, and install closure",
    },
]
EXPECTED_DIAGNOSTIC_LOGS = {
    "authorization_value": False,
    "runner_fragment": "-> MOTION (PUBLISHING)",
    "planner_fragment": "HOPE planner READY: corrected base pose fresh",
    "permitted_use": "human-readable diagnosis after a machine-state failure",
    "forbidden_use": (
        "no stdout or stderr bytes, marker order, marker age, or log inode may "
        "authorize or arm the publisher"
    ),
}
EXPECTED_LOGGING_ENVIRONMENT = {
    "required_exact": {
        "PYTHONUNBUFFERED": "1",
        "RCUTILS_LOGGING_USE_STDOUT": "1",
        "RCUTILS_LOGGING_BUFFERED_STREAM": "0",
    },
    "diagnostic_only": True,
    "may_contribute_to_authorization_guard": False,
}
PLANNER_ACK_FIELDS = [
    "run_nonce",
    "session_nonce",
    "state",
    "clock_id",
    "clock_sample_sequence",
    "status_sequence",
    "emitted_monotonic_ns",
    "source_epoch",
    "base_sequence_anchor",
    "base_sequence_current",
    "base_source_age_ms",
    "base_lease_valid",
    "actor_base_ready",
    "pid",
    "pgid",
    "proc_start_ticks",
    "executable_sha256",
    "argv_sha256",
    "config_closure_sha256",
    "environment_sha256",
    "policy_model_sha256",
    "runtime_closure_sha256",
]
RUNNER_ACK_FIELDS = [
    "run_nonce",
    "session_nonce",
    "state",
    "clock_id",
    "acknowledged_clock_sample_sequence",
    "status_sequence",
    "emitted_monotonic_ns",
    "source_epoch",
    "base_sequence_anchor",
    "base_sequence_current",
    "base_revocation_generation_anchor",
    "base_revocation_generation_current",
    "base_source_age_ms",
    "base_lease_valid",
    "actor_base_ready",
    "runner_actor_runtime_ready",
    "pid",
    "pgid",
    "proc_start_ticks",
    "executable_sha256",
    "argv_sha256",
    "config_closure_sha256",
    "environment_sha256",
    "policy_model_sha256",
    "runtime_closure_sha256",
]
VENDOR_ACK_FIELDS = [
    "run_nonce",
    "session_nonce",
    "backend_session_nonce",
    "state",
    "clock_id",
    "acknowledged_clock_sample_sequence",
    "status_sequence",
    "emitted_monotonic_ns",
    "pid",
    "pgid",
    "proc_start_ticks",
    "executable_sha256",
    "argv_sha256",
    "config_closure_sha256",
    "environment_sha256",
    "vendor_mjcf_sha256",
    "vendor_plant_sha256",
    "runtime_closure_sha256",
    "aimrt_plugin_closure_sha256",
    "transitive_shared_library_closure_sha256",
]
EXPECTED_MACHINE_ACK = {
    "transport": (
        "future machine-readable supervisor-owned channel; stdout and stderr are excluded"
    ),
    "clock_domain": "same_host_CLOCK_MONOTONIC",
    "planner_state": "READY_NO_BALL",
    "runner_state": "WAITING_BALL_READY",
    "vendor_state": "BACKEND_READY_NO_BALL",
    "planner_required_fields": PLANNER_ACK_FIELDS,
    "runner_required_fields": RUNNER_ACK_FIELDS,
    "vendor_required_fields": VENDOR_ACK_FIELDS,
    "joint_acceptance": (
        "exact run_nonce and session_nonce across planner, runner, and vendor backend; every "
        "owned PID, exact PGID, proc start ticks and content SHA still matches; all clocks are "
        "CLOCK_MONOTONIC on the same host and runner/vendor acknowledge the exact planner clock "
        "sample sequence; source_epoch, base_sequence_anchor, and policy_model_sha256 match "
        "between planner and runner; at ACK each current base sequence equals its anchor, runner "
        "current revocation generation equals its anchor, both base leases are valid/fresh and "
        "actor_base_ready=true, and runner_actor_runtime_ready=true; "
        "vendor BACKEND_READY_NO_BALL shares the accepted backend_session_nonce; READY_NO_BALL "
        "precedes WAITING_BALL_READY and all three samples remain inside the reviewed monotonic "
        "freshness window"
    ),
    "base_refresh_rule": (
        "base_sequence_anchor and runner base_revocation_generation_anchor are immutable "
        "readiness evidence, while base_sequence_current, base_source_age_ms and "
        "actor_base_ready are live status; "
        "after ACK current sequence may remain equal between 300 Hz publisher samples or advance "
        "strictly within the same source_epoch while fresh, plausible, lease-valid and "
        "actor-ready, but may never regress below the anchor, jump epoch, or change the runner "
        "revocation generation"
    ),
    "restart_rule": (
        "before arm, any planner, runner, or vendor backend PID, proc_start_ticks, executable, "
        "argv, config, model, plant, MJCF, environment, runtime closure, backend session nonce, "
        "run nonce, or session nonce change invalidates the tuple and requires a new jointly "
        "acknowledged session; after ACK acceptance the same change enters "
        "TERMINAL_DISARMED_FAILED before the next publish"
    ),
}
EXPECTED_ACTIVE_STATUS = {
    "transport": (
        "same supervisor-owned machine-readable channel as the accepted ACK; stdout and stderr "
        "remain excluded"
    ),
    "status_freshness_max_age_ms": 40,
    "post_arm_transition_deadline_ms": 60,
    "post_arm_deadline_origin": (
        "immutable first_publish_monotonic_ns from the pwrite-once first-publish record; deadline "
        "is exactly origin plus 60 ms and can never slide or reset"
    ),
    "publisher_active_state": "ONE_SHOT_ACTIVE",
    "publisher_identity_rule": (
        "the accepted ledger and single-use arm token bind one exact publisher PID, PGID, proc "
        "start ticks, pidfd/cgroup ownership, executable, argv, config, environment and runtime/"
        "transitive closure; exit, restart, exec, environment drift, ownership loss or "
        "substitution enters TERMINAL_DISARMED_FAILED before the next publish"
    ),
    "publisher_cursor_rule": (
        "the content-bound trajectory has exact sample_count=N and the exact owned publisher "
        "exposes supervisor-readable status_sequence, trajectory_sha256, next_sample_index and "
        "last_published_index; before sample i, next=i and last=i-1, after a successful publish "
        "the same process advances exactly once, and terminal success requires next=N and "
        "last=N-1; any duplicate, gap, regression, wrong trajectory, count drift or process "
        "substitution fails before the next publish"
    ),
    "planner": {
        "prearm_state": "READY_NO_BALL",
        "postarm_states": ["BALL_OBSERVED", "COMMANDING"],
        "legal_transitions": [
            "READY_NO_BALL->BALL_OBSERVED",
            "READY_NO_BALL->COMMANDING",
            "BALL_OBSERVED->COMMANDING",
        ],
    },
    "runner": {
        "prearm_state": "WAITING_BALL_READY",
        "postarm_states": ["TRACKING", "ACTOR_ACTIVE"],
        "legal_transitions": [
            "WAITING_BALL_READY->TRACKING",
            "WAITING_BALL_READY->ACTOR_ACTIVE",
            "TRACKING->ACTOR_ACTIVE",
        ],
    },
    "vendor": {
        "prearm_state": "BACKEND_READY_NO_BALL",
        "postarm_states": ["BACKEND_ONE_SHOT_ACTIVE"],
        "legal_transitions": [
            "BACKEND_READY_NO_BALL->BACKEND_ONE_SHOT_ACTIVE",
        ],
    },
    "transition_rule": (
        "prearm states may remain fresh only until the exact post-arm deadline; before every "
        "later publish each owned status_sequence is nondecreasing, any newly observed sequence "
        "strictly increases, and the state is on its declared forward path; after the deadline "
        "all three components must be in a declared postarm state"
    ),
    "base_sequence_rule": (
        "planner and runner report separate immutable base_sequence_anchor and live "
        "base_sequence_current fields; each current sequence is never below its anchor, repeated "
        "observation of the same current sequence is allowed while base_lease_valid=true and "
        "base_source_age_ms is fresh, each newly observed current sequence strictly increases in "
        "the same source_epoch, and any regression, epoch change, implausible pose, expiry, or "
        "invalid lease fails before the next publish"
    ),
    "runner_revocation_rule": (
        "runner base_revocation_generation_anchor is immutable in the accepted ledger and "
        "base_revocation_generation_current must equal it before every publish; any hidden "
        "malformed/implausible revoke, including invalid-to-valid recovery within the same "
        "source_epoch, changes current generation and fails before the next publish"
    ),
    "actor_base_ready_rule": (
        "planner and runner actor_base_ready must be true at ACK and before every publish; true "
        "means a fresh formal same-epoch lease, hard workspace and source-time continuity "
        "plausibility, finite latest pose, runner latest z at or above base_low, and for any "
        "recovery hold the latched engaged epoch/revocation lease remains usable; false or "
        "unknown fails before the next publish"
    ),
    "runner_actor_runtime_ready_rule": (
        "runner_actor_runtime_ready must be true at ACK and before every publish; prearm WAITING "
        "true means publish-capable MOTION mode with no-publish=false, healthy owned supervisor/"
        "backend, level0 with no active clip or deadline fault, yaw capture complete, finite "
        "q/dq/IMU, reviewed upright/still bounds, and exact model/179 observation/action contract; "
        "postarm TRACKING true keeps level0/no active clip while waiting for an accepted command, "
        "whereas ACTOR_ACTIVE true requires exactly one accepted active clip and usable epoch/"
        "revocation lease under the same health/model/observation/safety checks; false, unknown, "
        "multiple clips or any safety abort fails before the next publish"
    ),
    "failure_rule": (
        "unknown state, backward transition, stale status, missed transition deadline, session "
        "or epoch change, base regression or revoke, process exit or restart, ownership or "
        "immutable binding drift enters TERMINAL_DISARMED_FAILED before the next publish"
    ),
}
EXPECTED_PUBLISHER_ARM = {
    "initial_state": "OWNED_DISARMED",
    "arm_transition": "OWNED_DISARMED_to_ARM_COMMITTED",
    "authorization_guard": (
        "supervisor atomic no-replace ledger contains the exact accepted machine-readable "
        "planner READY_NO_BALL, runner WAITING_BALL_READY, and vendor BACKEND_READY_NO_BALL "
        "joint ACK and every owned identity and content binding still matches"
    ),
    "atomic_ledger_required_fields": [
        "run_nonce",
        "session_nonce",
        "accepted_ack_sha256",
        "accepted_monotonic_ns",
        "arm_commit_record_identity_sha256",
        "first_publish_record_identity_sha256",
        "planner_pid_pgid_start_ticks",
        "runner_pid_pgid_start_ticks",
        "vendor_pid_pgid_start_ticks",
        "publisher_pid_pgid_start_ticks",
        "publisher_pidfd_cgroup_identity_sha256",
        "publisher_environment_sha256",
        "publisher_runtime_closure_sha256",
        "publisher_machine_status_schema_sha256",
        "planner_status_sha256",
        "runner_status_sha256",
        "vendor_status_sha256",
        "vendor_backend_session_nonce",
        "vendor_executable_argv_config_mjcf_plant_closure_sha256",
        "publisher_executable_argv_config_sha256",
        "publisher_one_shot_trajectory_sha256",
        "single_use_arm_token_sha256",
        "source_epoch",
        "base_sequence_anchor",
        "base_revocation_generation_anchor",
        "policy_model_sha256",
    ],
    "ledger_creation": (
        "unique run directory plus O_CREAT_O_EXCL O_NOFOLLOW, exclusive flock, pwrite/pread "
        "verification, fsync, and atomic no-replace finalization"
    ),
    "arm_effect": (
        "future supervisor creates exactly one content-bound single-use arm token while moving "
        "the owned publisher from OWNED_DISARMED to ARM_COMMITTED; no diagnostic log marker can "
        "create this token"
    ),
    "arm_commit_record": (
        "on the unique OWNED_DISARMED to ARM_COMMITTED edge, the supervisor creates one "
        "O_CREAT_O_EXCL/O_NOFOLLOW/flock/pwrite/pread/fsync record containing "
        "arm_committed_monotonic_ns, accepted ledger identity, token/session/publisher identity "
        "and trajectory; the ACK ledger pre-binds this record identity, and the record is "
        "immutable and cannot be replaced or refreshed"
    ),
    "first_publish_record": (
        "after consuming the token, the exact owned publisher creates one O_CREAT_O_EXCL/"
        "O_NOFOLLOW/flock/pwrite/pread/fsync record containing the conservative "
        "first_publish_monotonic_ns deadline origin, immutable arm-commit record identity/time, "
        "token/session/trajectory identity and first_sample_index=0; after fsync it re-reads/"
        "verifies the record and revalidates every live first-sample guard immediately before "
        "publishing sample zero, so any drift terminal-fails without a publish; the record is "
        "immutable and cannot be replaced or refreshed"
    ),
    "publish_guard": (
        "immediately before its first trajectory sample, the exact owned publisher atomically "
        "consumes the arm token and revalidates its own PID/PGID/start-ticks/pidfd/cgroup/"
        "executable/argv/config/environment/runtime closure plus the accepted ACK, planner, runner and vendor identities/"
        "closures, backend session, frame, one-shot config, trajectory, source epoch, "
        "base-sequence and runner-revocation anchors, live lease health, actor_base_ready=true, "
        "runner_actor_runtime_ready=true, model and timeouts; a consumed token can never be reused"
    ),
    "per_sample_guard": (
        "before every later 300 Hz trajectory sample, the publisher revalidates its own and "
        "planner, runner and vendor owned pidfd/cgroup liveness, exact process identity and "
        "immutable environment/runtime closure, "
        "legal fresh machine states, session and backend-session identity, source epoch, each "
        "nonregressing base_sequence_current at or above its readiness anchor, fresh "
        "base_source_age_ms, base_lease_valid=true, actor_base_ready=true, "
        "runner_actor_runtime_ready=true, unchanged runner base_revocation_generation_current, "
        "immutable trajectory identity and exact supervisor-observed next/last sample indices; "
        "failure is detected and "
        "enters TERMINAL_DISARMED_FAILED before the next publish"
    ),
    "failure_effect": (
        "any guard failure before or during the one-shot trajectory disarms the publisher and "
        "enters TERMINAL_DISARMED_FAILED without retry or reset"
    ),
}
NONTERMINAL_STATES = [
    "PLAN_ONLY",
    "RUNTIME_BINDINGS_VERIFIED",
    "WAIT_MACHINE_ACK",
    "ACK_ACCEPTED",
    "OWNED_DISARMED",
    "ARM_COMMITTED",
    "ONE_SHOT_ACTIVE",
]
TERMINAL_STATES = ["TERMINAL_DISARMED_SUCCESS", "TERMINAL_DISARMED_FAILED"]
EXPECTED_STATE_MACHINE = {
    "initial_state": "PLAN_ONLY",
    "terminal_states": TERMINAL_STATES,
    "success_path": [
        {
            "state": "PLAN_ONLY",
            "next": "RUNTIME_BINDINGS_VERIFIED",
            "guard": (
                "all runtime binding values are non-null content addresses and independently "
                "verified"
            ),
        },
        {
            "state": "RUNTIME_BINDINGS_VERIFIED",
            "next": "WAIT_MACHINE_ACK",
            "guard": (
                "supervisor owns planner, runner, vendor backend, and disarmed publisher "
                "identities and exact closure bindings"
            ),
        },
        {
            "state": "WAIT_MACHINE_ACK",
            "next": "ACK_ACCEPTED",
            "guard": (
                "machine-readable planner READY_NO_BALL, runner WAITING_BALL_READY, and vendor "
                "BACKEND_READY_NO_BALL satisfy the exact same-session same-host monotonic clock, "
                "readiness-anchor, backend-session, identity, and content rules"
            ),
        },
        {
            "state": "ACK_ACCEPTED",
            "next": "OWNED_DISARMED",
            "guard": (
                "atomic no-replace ledger durably binds the accepted joint ACK, owned publisher "
                "identity, exact closure, frame, one-shot config, and trajectory"
            ),
        },
        {
            "state": "OWNED_DISARMED",
            "next": "ARM_COMMITTED",
            "guard": (
                "supervisor atomically creates the unique content-bound single-use arm token "
                "after revalidating every accepted binding"
            ),
        },
        {
            "state": "ARM_COMMITTED",
            "next": "ONE_SHOT_ACTIVE",
            "guard": (
                "publisher atomically consumes the unique arm token and revalidates every "
                "accepted binding immediately before the first publish"
            ),
        },
        {
            "state": "ONE_SHOT_ACTIVE",
            "next": "TERMINAL_DISARMED_SUCCESS",
            "guard": (
                "before every later sample the per-sample guard passes and the exact owned "
                "publisher cursor status advances once; terminal cursor proves all N bound "
                "300 Hz samples published exactly once with no reset, reuse, second serve, or "
                "further publish"
            ),
        },
    ],
    "failure_edges": [
        {"state": state, "next": "TERMINAL_DISARMED_FAILED"}
        for state in NONTERMINAL_STATES
    ],
    "failure_guard": (
        "any identity, status, nonce, sequence, epoch, clock, frame, closure, model, config, "
        "trajectory, timeout, ownership, ledger, arm-token, process, or publish-count mismatch"
    ),
    "terminal_semantics": (
        "both terminal states are absorbing; they have no outgoing transition, ACK or arm-token "
        "reuse, retry, reset, re-arm, or second serve"
    ),
}
EXPECTED_ONE_SHOT_SERVE = {
    "required_parameters": {
        "one_shot": True,
        "max_serves": 1,
        "rate_hz": 300.0,
        "trajectory_count": 1,
        "auto_reset": False,
    },
    "trajectory_binding": (
        "one finite trajectory is content-addressed together with frame, initial position and "
        "velocity, drag, restitution, table bounds, end condition, maximum flight time, sample "
        "rate, and exact sample-count or terminal rule"
    ),
    "current_source_supports_contract": False,
    "current_source_limitation": (
        "tracked fake_ball_publisher cycles forever and automatically resets after each pause; "
        "it is evidence only and cannot be armed by this preregistration"
    ),
    "publish_count_rule": (
        "publisher emits only the exact N indexed samples from the single bound trajectory after "
        "consuming one arm token; before every sample it revalidates publisher, planner, runner "
        "and vendor owned liveness, legal status freshness/transitions, session/backend session, "
        "source epoch, nonregressing live base sequences at or above immutable anchors, fresh "
        "valid actor-ready bases, runner actor runtime readiness, unchanged runner revocation "
        "generation, trajectory identity and exact supervisor-observed next/last indices, and "
        "fails before the next publish on any mismatch; terminal success requires next=N and "
        "last=N-1, then permanent disarm with no reset, loop, ACK reuse, retry, second serve, or "
        "post-terminal publish"
    ),
    "launch_effect": (
        "BLOCKED_NO_PUBLISHER until reviewed publisher source and content bindings implement "
        "this exact one-shot contract"
    ),
}
EXPECTED_FRAME_CONTRACT = {
    "formal_common_frame_required": True,
    "current_ball_frame": "world",
    "current_base_frame": "odom",
    "current_frames_match": False,
    "fake_ball_source": {
        "path": "hope_ws/src/hope_bringup/scripts/fake_ball_publisher",
        "sha256": "f52d766eaa1f3cc2f0b2badcb2da78cd204c1f952f17f5717e105c9ef1bc055d",
        "evidence": "declared frame_id default is world",
    },
    "vendor_sim_config": {
        "path": (
            "agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/"
            "a3_pingpong_iceoryx_cfg.yaml"
        ),
        "sha256": "48dfcb69542f41dc01b90f6f3e4f4a942e51b881a337110c0fcdf688be75a8bc",
        "evidence": "pelvis pose, twist, and racket pose frame_id are odom",
    },
    "frame_transform_bound": False,
    "fake_ball_frame_parameter_bound": False,
    "launch_effect": (
        "BLOCKED_NO_PUBLISHER until an exact source-target transform or verified identical-frame "
        "publisher argument is content-addressed and bound"
    ),
}
EXPECTED_LEGACY_AUDIT = {
    "path": "agi/a3_deploy_example/scripts/pp_gate3_rally.sh",
    "may_be_used_as_launcher": False,
    "broad_kill_detected": True,
    "publisher_starts_before_machine_ack": True,
    "serve_wait_fails_closed": False,
    "selective_copy_allowed": False,
}
REQUIRED_BINDINGS = (
    "exact_supervisor_source_sha256",
    "exact_supervisor_executable_sha256",
    "planner_machine_status_schema_sha256",
    "runner_machine_status_schema_sha256",
    "vendor_machine_status_schema_sha256",
    "vendor_pidfd_cgroup_identity_sha256",
    "vendor_backend_readiness_session_sha256",
    "supervisor_joint_ack_schema_sha256",
    "supervisor_atomic_ledger_schema_sha256",
    "supervisor_pidfd_cgroup_handshake_sha256",
    "same_host_clock_contract_sha256",
    "unique_run_session_nonce_binding_sha256",
    "runner_executable_sha256",
    "runner_argv_sha256",
    "runner_config_closure_sha256",
    "runner_environment_sha256",
    "runner_model_sha256",
    "runner_aimrt_config_sha256",
    "runner_aimrt_plugin_closure_sha256",
    "runner_transitive_shared_library_closure_sha256",
    "planner_executable_sha256",
    "planner_argv_sha256",
    "planner_config_closure_sha256",
    "planner_environment_sha256",
    "planner_runtime_dependency_closure_sha256",
    "fake_ball_publisher_executable_sha256",
    "fake_ball_publisher_argv_sha256",
    "fake_ball_publisher_config_sha256",
    "fake_ball_publisher_environment_sha256",
    "fake_ball_publisher_machine_status_schema_sha256",
    "fake_ball_publisher_frame_id_binding_sha256",
    "fake_ball_publisher_transitive_shared_library_closure_sha256",
    "fake_ball_publisher_runtime_closure_sha256",
    "fake_ball_publisher_pidfd_cgroup_identity_sha256",
    "fake_ball_publisher_one_shot_config_sha256",
    "fake_ball_publisher_trajectory_sha256",
    "fake_ball_publisher_arm_token_schema_sha256",
    "fake_ball_publisher_arm_commit_record_schema_sha256",
    "fake_ball_publisher_first_publish_record_schema_sha256",
    "fake_ball_publisher_terminal_evidence_sha256",
    "vendor_mjcf_sha256",
    "vendor_plant_sha256",
    "vendor_runtime_executable_sha256",
    "vendor_runtime_config_closure_sha256",
    "vendor_environment_sha256",
    "vendor_aimrt_plugin_closure_sha256",
    "vendor_transitive_shared_library_closure_sha256",
    "frame_transform_source_target_sha256",
    "supervisor_accepted_ack_ledger_runtime_evidence_sha256",
)
EXPECTED_PROHIBITED = {
    "authorize_from_stdout_or_stderr",
    "authorize_from_runner_motion_log_fragment",
    "authorize_from_planner_ready_log_fragment",
    "arm_publisher_without_accepted_machine_joint_ack",
    "publish_without_consumed_single_use_arm_token",
    "reuse_or_reset_one_shot_publisher",
    "rearm_or_retry_after_terminal_state",
    "reuse_ack_after_planner_or_runner_restart",
    "mix_run_or_session_nonce",
    "mix_clock_domain_or_clock_sample_sequence",
    "regress_base_sequence_below_ack_anchor_or_mix_source_epoch",
    "freeze_base_sequence_at_ack",
    "treat_prearm_status_as_postarm_after_deadline",
    "publish_after_vendor_exit_or_restart",
    "publish_after_publisher_exit_or_restart",
    "slide_or_reset_post_arm_transition_deadline",
    "mix_policy_model_or_runtime_closure",
    "launch_with_world_ball_and_odom_base_without_bound_transform_or_exact_frame_override",
    "launch_without_AimRT_publisher_config_transitive_shared_library_and_plugin_closure",
    "pkill",
    "killall",
    "kill_by_name_or_pattern",
    "signal_unverified_pid_or_pgid",
    "real_robot_command",
}


class ContractError(ValueError):
    """The static serve-synchronization contract is unsafe or ambiguous."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise ContractError(f"non-finite JSON constant: {value}")


def _walk_finite(value: Any, label: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ContractError(f"non-finite JSON number at {label}")
    if isinstance(value, dict):
        for key, item in value.items():
            _walk_finite(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _walk_finite(item, f"{label}[{index}]")


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate,
            parse_constant=_reject_constant,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read preregistration: {exc}") from None
    if not isinstance(value, dict):
        raise ContractError("preregistration must be a JSON mapping")
    _walk_finite(value)
    return value


def _run_readonly_git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    allowed = {
        ("rev-parse", "--show-toplevel"),
        (
            "merge-base",
            "--is-ancestor",
            EXPECTED_PLAN_GATE_DEPENDENCY["merge_commit"],
            "HEAD",
        ),
    }
    if args not in allowed:
        raise ContractError("unapproved Git identity query")
    env = os.environ.copy()
    env["GIT_OPTIONAL_LOCKS"] = "0"
    try:
        return subprocess.run(
            ["git", "--no-optional-locks", "-C", str(repo_root), *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=10.0,
            env=env,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ContractError(f"read-only Git identity check failed: {exc}") from None


def validate_repo_context(repo_root: Path, prereg: Path) -> Path:
    """Bind the CLI to the exact worktree top and canonical prereg path."""

    try:
        root = repo_root.expanduser().resolve(strict=True)
    except OSError as exc:
        raise ContractError(f"cannot resolve repo root: {exc}") from None
    if not root.is_dir():
        raise ContractError("repo root must be a directory")
    top_result = _run_readonly_git(root, "rev-parse", "--show-toplevel")
    if top_result.returncode != 0 or not top_result.stdout.strip():
        raise ContractError("repo root is not a readable Git worktree top level")
    try:
        git_top = Path(top_result.stdout.strip()).resolve(strict=True)
    except OSError as exc:
        raise ContractError(f"cannot resolve Git top level: {exc}") from None
    if git_top != root:
        raise ContractError(f"--repo-root is not the exact Git top level: {git_top}")
    try:
        actual = prereg.expanduser().resolve(strict=True)
        expected = (root / EXPECTED_PREREG_RELATIVE_PATH).resolve(strict=True)
    except OSError as exc:
        raise ContractError(f"cannot resolve canonical preregistration: {exc}") from None
    if actual != expected:
        raise ContractError("--prereg must be the canonical file under --repo-root")
    return root


def _resolve_bound_repo_file(repo_root: Path, relative_path: str, label: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ContractError(f"{label} must be a contained relative path")
    try:
        resolved = (repo_root / relative).resolve(strict=True)
        resolved.relative_to(repo_root)
    except (OSError, ValueError) as exc:
        raise ContractError(f"cannot resolve contained {label}: {exc}") from None
    if not resolved.is_file():
        raise ContractError(f"{label} must be a regular file")
    return resolved


def _validate_bound_file(
    repo_root: Path, record: dict[str, Any], label: str
) -> Path:
    path = _resolve_bound_repo_file(repo_root, record["path"], label)
    if sha256_file(path) != record["sha256"]:
        raise ContractError(f"actual source SHA changed: {record['path']}")
    return path


def validate_plan_gate_dependency(repo_root: Path, dependency: dict[str, Any]) -> None:
    if dependency != EXPECTED_PLAN_GATE_DEPENDENCY:
        raise ContractError("plan-only first-tick dependency changed")
    source = _resolve_bound_repo_file(repo_root, dependency["source_path"], "plan gate source")
    legacy = _resolve_bound_repo_file(
        repo_root, dependency["legacy_audit_path"], "legacy process audit"
    )
    if sha256_file(source) != dependency["source_sha256"]:
        raise ContractError("actual plan-gate source SHA changed")
    if sha256_file(legacy) != dependency["legacy_audit_sha256"]:
        raise ContractError("actual legacy-audit SHA changed")
    ancestor = _run_readonly_git(
        repo_root,
        "merge-base",
        "--is-ancestor",
        dependency["merge_commit"],
        "HEAD",
    )
    if ancestor.returncode != 0:
        raise ContractError("bound plan-gate merge commit is not reachable from HEAD")


def validate_source_dependencies(
    repo_root: Path, dependencies: list[dict[str, Any]]
) -> None:
    if dependencies != EXPECTED_SOURCE_DEPENDENCIES:
        raise ContractError("reviewed formal tuple and runner transport source subset changed")
    if len({item["path"] for item in dependencies}) != len(dependencies):
        raise ContractError("duplicate source dependency path")
    for dependency in dependencies:
        _validate_bound_file(repo_root, dependency, dependency["role"])


def validate_frame_evidence(repo_root: Path, frame: dict[str, Any]) -> None:
    if frame != EXPECTED_FRAME_CONTRACT:
        raise ContractError("formal frame blocker contract changed")
    fake_ball = _validate_bound_file(repo_root, frame["fake_ball_source"], "fake-ball source")
    vendor = _validate_bound_file(repo_root, frame["vendor_sim_config"], "vendor sim config")
    fake_text = fake_ball.read_text(encoding="utf-8")
    vendor_text = vendor.read_text(encoding="utf-8")
    if 'self.declare_parameter("frame_id", "world")' not in fake_text:
        raise ContractError("fake-ball world-frame evidence changed")
    if vendor_text.count("frame_id: odom") < 3:
        raise ContractError("vendor odom-frame evidence changed")
    sim_config = _resolve_bound_repo_file(
        repo_root,
        "hope_ws/src/hope_planner/config/hope_planner.sim.yaml",
        "planner simulator config",
    ).read_text(encoding="utf-8")
    for required in (
        'formal_ball_source_frame_id: "world"',
        'formal_base_source_frame_id: "odom"',
        "formal_common_frame_required: true",
    ):
        if required not in sim_config:
            raise ContractError(f"explicit planner frame blocker missing: {required}")


def validate_prereg(path: Path, expected_sha256: str) -> dict[str, Any]:
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha256:
        raise ContractError(
            f"preregistration SHA mismatch: {actual_sha} != {expected_sha256}"
        )
    doc = read_json(path)
    expected_keys = {
        "schema_version",
        "contract_id",
        "status",
        "launch_authorized",
        "real_robot_authorized",
        "scope",
        "plan_gate_dependency",
        "planner_policy_source_dependencies",
        "diagnostic_only_logs",
        "planner_logging_environment",
        "machine_ack_contract",
        "active_status_contract",
        "publisher_arm_contract",
        "state_machine",
        "one_shot_serve_contract",
        "frame_contract",
        "legacy_audit",
        "runtime_bindings",
        "prohibited",
    }
    if set(doc) != expected_keys:
        raise ContractError("top-level key set changed")
    if type(doc["schema_version"]) is not int or doc["schema_version"] != 4:
        raise ContractError("schema version changed")
    if doc["contract_id"] != "gate3-machine-ack-before-publisher-arm-v4":
        raise ContractError("contract id changed")
    if doc["status"] != "preregistered_runtime_blocked":
        raise ContractError("contract must remain preregistered_runtime_blocked")
    if doc["launch_authorized"] is not False or doc["real_robot_authorized"] is not False:
        raise ContractError("design preregistration cannot authorize launch or robot use")
    if doc["scope"] != EXPECTED_SCOPE:
        raise ContractError("source-only scope changed")
    if doc["plan_gate_dependency"] != EXPECTED_PLAN_GATE_DEPENDENCY:
        raise ContractError("plan-only dependency claim changed")
    if doc["planner_policy_source_dependencies"] != EXPECTED_SOURCE_DEPENDENCIES:
        raise ContractError("formal tuple and runner transport source subset claim changed")
    if doc["diagnostic_only_logs"] != EXPECTED_DIAGNOSTIC_LOGS:
        raise ContractError("diagnostic-log non-authority contract changed")
    if doc["planner_logging_environment"] != EXPECTED_LOGGING_ENVIRONMENT:
        raise ContractError("planner diagnostic logging environment changed")
    if doc["machine_ack_contract"] != EXPECTED_MACHINE_ACK:
        raise ContractError("machine-readable joint ACK contract changed")
    if doc["active_status_contract"] != EXPECTED_ACTIVE_STATUS:
        raise ContractError("post-arm machine status contract changed")
    if doc["publisher_arm_contract"] != EXPECTED_PUBLISHER_ARM:
        raise ContractError("publisher arm contract changed")
    if doc["state_machine"] != EXPECTED_STATE_MACHINE:
        raise ContractError("publisher state machine changed")
    if doc["one_shot_serve_contract"] != EXPECTED_ONE_SHOT_SERVE:
        raise ContractError("one-shot serve contract changed")
    if doc["frame_contract"] != EXPECTED_FRAME_CONTRACT:
        raise ContractError("frame mismatch blocker changed")
    if doc["legacy_audit"] != EXPECTED_LEGACY_AUDIT:
        raise ContractError("legacy unsafe-launcher audit changed")

    bindings = doc["runtime_bindings"]
    if not isinstance(bindings, dict) or tuple(bindings) != REQUIRED_BINDINGS:
        raise ContractError("runtime binding key set or order changed")
    if any(value is not None for value in bindings.values()):
        raise ContractError("runtime-blocked preregistration bindings must remain null")
    prohibited = doc["prohibited"]
    if (
        not isinstance(prohibited, list)
        or len(prohibited) != len(EXPECTED_PROHIBITED)
        or set(prohibited) != EXPECTED_PROHIBITED
    ):
        raise ContractError("prohibited action set changed")

    authorization_text = doc["publisher_arm_contract"]["authorization_guard"]
    transition_text = "\n".join(
        f"{item['state']} {item['next']} {item['guard']}"
        for item in doc["state_machine"]["success_path"]
    )
    transition_text += "\n" + doc["state_machine"]["failure_guard"]
    for fragment in (
        doc["diagnostic_only_logs"]["runner_fragment"],
        doc["diagnostic_only_logs"]["planner_fragment"],
    ):
        if fragment in authorization_text or fragment in transition_text:
            raise ContractError("diagnostic log fragment entered authorization semantics")
    for forbidden_word in ("stdout", "stderr", "log fragment", "log marker"):
        if forbidden_word in authorization_text.lower() or forbidden_word in transition_text.lower():
            raise ContractError("diagnostic log channel entered authorization semantics")
    return doc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--expected-prereg-sha256", required=True)
    parser.add_argument("--mode", choices=("design-check", "launch-check"), required=True)
    args = parser.parse_args(argv)
    try:
        repo_root = validate_repo_context(args.repo_root, args.prereg)
        doc = validate_prereg(args.prereg, args.expected_prereg_sha256)
        validate_plan_gate_dependency(repo_root, doc["plan_gate_dependency"])
        validate_source_dependencies(repo_root, doc["planner_policy_source_dependencies"])
        validate_frame_evidence(repo_root, doc["frame_contract"])
    except ContractError as exc:
        print(f"CONTRACT ERROR: {exc}", file=sys.stderr)
        return 2

    missing = [key for key, value in doc["runtime_bindings"].items() if value is None]
    if args.mode == "launch-check":
        print(
            f"LAUNCH BLOCKED: {len(missing)} unresolved exact runtime bindings",
            file=sys.stderr,
        )
        for key in missing:
            print(f"MISSING {key}", file=sys.stderr)
        print(
            "BLOCKER current formal frames differ: ball=world base=odom; no transform or "
            "fake-ball frame argument is bound",
            file=sys.stderr,
        )
        print(
            "BLOCKER machine-readable READY_NO_BALL plus WAITING_BALL_READY plus "
            "BACKEND_READY_NO_BALL ACK, exact four-process ownership, and atomic accepted-ACK "
            "ledger have no reviewed runtime implementation/evidence",
            file=sys.stderr,
        )
        print(
            "BLOCKER bounded post-arm planner/runner/vendor states, base anchor/current/revoke "
            "health, actor runtime readiness, fixed deadline origin, and publisher cursor have "
            "no reviewed runtime implementation/evidence",
            file=sys.stderr,
        )
        print(
            "BLOCKER tracked fake-ball publisher does not implement content-bound "
            "one_shot=true max_serves=1 with a single 300 Hz trajectory and absorbing "
            "terminal disarm",
            file=sys.stderr,
        )
        return 1

    print(
        json.dumps(
            {
                "status": "pass_design_only",
                "launch_authorized": False,
                "machine_ack_runtime_present": False,
                "active_status_runtime_present": False,
                "vendor_backend_status_runtime_present": False,
                "publisher_arm_runtime_present": False,
                "one_shot_serve_ready": False,
                "frame_contract_ready": False,
                "runtime_blocker_count": len(missing),
                "source_dependencies_verified": len(EXPECTED_SOURCE_DEPENDENCIES),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
