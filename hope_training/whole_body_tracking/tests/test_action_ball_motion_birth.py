"""Focused Motion-owned action-ball birth transaction tests.

These tests deliberately use the real dependency-light ``action_ball_runtime``
broker/provider/domain protocol and the real ``MotionCommand`` methods.  Only
IsaacLab is stubbed (through the existing CPU test loader); the fake robot
records the two simulator setters that Motion owns.

The synthetic Motion harness stubs ``action_ball_motion_admission_hard_contract``
only so transaction and exact-resume behavior can be isolated without a
production canonical bank/certificate.  That stub is *not* used as admission
evidence.  Admission fail-closed behavior is checked separately against the
real source AST and the real trusted-root path validator.

Run on a CPU Torch environment:

    python -m pytest \
      hope_training/whole_body_tracking/tests/test_action_ball_motion_birth.py -q
"""

from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import inspect
import math
from pathlib import Path
import textwrap
import types

import pytest
import torch

from test_reward_flags_mdp import commands_mod as C  # noqa: E402


_ACTION_COUNT = 5
_READY_Z_M = 1.02
_POLICY_DT_S = 0.02
_SEGMENT_FRAMES = 61
_REFERENCE_T_HIT_S = 0.42
_REFERENCE_T_CYCLE_S = 1.20


def _digest(label: object) -> str:
    return hashlib.sha256(str(label).encode("utf-8")).hexdigest()


def _pins(runtime):
    return runtime.RuntimePins(
        manifest_sha256=_digest("motion-birth-manifest"),
        sampler_sha256=_digest("motion-birth-sampler"),
        domain_authority_sha256=_digest("motion-birth-domain"),
        physics_sha256=_digest("motion-birth-physics"),
        solver_sha256=_digest("motion-birth-solver"),
    )


def _bindings(runtime, count: int = _ACTION_COUNT):
    return tuple(
        runtime.ActionBinding(
            action_uid=20_000 + slot * 101,
            action_slot=slot,
            motion_path=f"vendor_assets/motions/motion_birth_{slot}.npz",
            motion_sha256=_digest(f"motion-birth-motion-{slot}"),
            profile_sha256=_digest(f"motion-birth-profile-{slot}"),
        )
        for slot in range(count)
    )


def _levels(runtime):
    return runtime.ActionDomainLevels(
        **{
            name: (index + 1) / 100.0
            for index, name in enumerate(runtime.ARM_KEYS)
        }
    )


class _DomainAuthority:
    """Small exact-state adapter for the real broker protocol."""

    def __init__(self, runtime, bindings, mode: str):
        self._runtime = runtime
        self.domain_authority_contract_sha256 = _digest(
            "motion-birth-domain"
        )
        self.state_owner_sha256 = _digest("motion-birth-domain-owner")
        self._bindings = {
            binding.action_uid: binding for binding in bindings
        }
        self._mode = mode
        self._cursors: dict[int, int] = {}
        # Diagnostic only: intentionally absent from serialized state.
        self.claim_invocations = 0

    def state_dict(self):
        return {
            "cursors": [
                [uid, cursor]
                for uid, cursor in sorted(self._cursors.items())
            ]
        }

    def load_state_dict(self, state):
        if type(state) is not dict or set(state) != {"cursors"}:
            raise ValueError("invalid synthetic domain state")
        rows = state["cursors"]
        if not isinstance(rows, list):
            raise ValueError("invalid synthetic domain cursor rows")
        self._cursors = {int(uid): int(cursor) for uid, cursor in rows}

    def claim_for_action(self, action_uid: int):
        self.claim_invocations += 1
        binding = self._bindings[action_uid]
        epoch = self._cursors.get(action_uid, 0)
        levels = _levels(self._runtime)
        claim = self._runtime.ActionDomainClaim(
            authority_contract_sha256=(
                self.domain_authority_contract_sha256
            ),
            arm_catalog_sha256=self._runtime.ARM_CATALOG_SHA256,
            action_uid=action_uid,
            domain_epoch=epoch,
            domain_levels=levels,
            levels_sha256=levels.canonical_sha256,
            profile_sha256=binding.profile_sha256,
            mobility_mode=self._mode,
        )
        self._cursors[action_uid] = epoch + 1
        return claim

    def domain_cursor_for(self, action_uid: int) -> int:
        return self._cursors.get(action_uid, 0)


class _BirthProvider:
    """Real-receipt provider with an optional failure after one row mutates."""

    def __init__(self, runtime, *, bad_at: int | None = None):
        self._runtime = runtime
        self.sampler_contract_sha256 = _digest("motion-birth-sampler")
        self.state_owner_sha256 = _digest("motion-birth-provider-owner")
        self._birth_counts: dict[int, int] = {}
        self._draw_counts: dict[int, int] = {}
        self.bad_at = bad_at
        # Diagnostics only; neither counter is part of provider random tape.
        self.issue_invocations = 0
        self.assert_invocations = 0

    def state_dict(self):
        return {
            "birth_counts": [
                [uid, count]
                for uid, count in sorted(self._birth_counts.items())
            ],
            "draw_counts": [
                [uid, count]
                for uid, count in sorted(self._draw_counts.items())
            ],
        }

    def load_state_dict(self, state):
        if (
            type(state) is not dict
            or set(state) != {"birth_counts", "draw_counts"}
        ):
            raise ValueError("invalid synthetic provider state")
        birth_counts = {
            int(uid): int(count)
            for uid, count in state["birth_counts"]
        }
        draw_counts = {
            int(uid): int(count)
            for uid, count in state["draw_counts"]
        }
        if set(birth_counts) != set(draw_counts):
            raise ValueError("synthetic provider counter domains differ")
        self._birth_counts = birth_counts
        self._draw_counts = draw_counts

    def birth_highwater_for(self, action_uid: int):
        count = self._birth_counts.get(action_uid, 0)
        return (
            count - 1,
            self._draw_counts.get(action_uid, 0),
        )

    def assert_issued_birth(self, receipt):
        self.assert_invocations += 1
        count = self._birth_counts.get(receipt.action_uid, 0)
        draw_count = self._draw_counts.get(receipt.action_uid, 0)
        if (
            receipt.sampler_birth_index >= count
            or receipt.sampler_draw_end > draw_count
        ):
            raise ValueError("receipt is absent from provider tape")

    def __call__(self, request):
        invocation = self.issue_invocations
        self.issue_invocations += 1
        uid = request.action_uid
        birth_index = self._birth_counts.get(uid, 0)
        draw_start = self._draw_counts.get(uid, 0)
        spawn = (
            -0.30 + request.env_id * 1.0e-5,
            -0.12 + request.action_slot * 0.06,
            _READY_Z_M,
        )
        birth_identity = {
            "schema_version": self._runtime.SAMPLER_SCHEMA_VERSION,
            "kind": "base_birth",
            "sampler_contract_sha256": request.pins.sampler_sha256,
            "arm_catalog_sha256": self._runtime.ARM_CATALOG_SHA256,
            "action_uid": request.action_uid,
            "domain_epoch": request.domain_claim.domain_epoch,
            "levels_sha256": request.domain_claim.levels_sha256,
            "profile_sha256": request.binding.profile_sha256,
            "birth_index": birth_index,
            "draw_start": draw_start,
            "draw_end": (
                draw_start + self._runtime.SAMPLER_BIRTH_DRAW_COUNT
            ),
            "mobility_mode": request.mobility_mode,
            "base_yaw_rad": 0.0,
            "base_start_w_m": spawn,
        }
        receipt = self._runtime.ActionBirthReceipt(
            registry_sha256=request.registry_sha256,
            env_id=request.env_id,
            reset_generation=request.reset_generation,
            action_uid=request.action_uid,
            action_slot=request.action_slot,
            domain_epoch=request.domain_claim.domain_epoch,
            domain_claim_sha256=request.domain_claim.canonical_sha256,
            domain_authority_sha256=(
                request.domain_claim.authority_contract_sha256
            ),
            domain_levels=request.domain_claim.domain_levels,
            arm_catalog_sha256=request.domain_claim.arm_catalog_sha256,
            levels_sha256=request.domain_claim.levels_sha256,
            sampler_birth_sha256=self._runtime._sha256_json(
                birth_identity
            ),
            sampler_birth_index=birth_index,
            sampler_draw_start=draw_start,
            sampler_draw_end=(
                draw_start + self._runtime.SAMPLER_BIRTH_DRAW_COUNT
            ),
            mobility_mode=request.mobility_mode,
            base_yaw_rad=0.0,
            base_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            base_spawn_w_m=spawn,
            manifest_sha256=request.pins.manifest_sha256,
            sampler_sha256=request.pins.sampler_sha256,
            profile_sha256=request.binding.profile_sha256,
            motion_sha256=request.binding.motion_sha256,
            physics_sha256=request.pins.physics_sha256,
            solver_sha256=request.pins.solver_sha256,
        )
        # Mutate the exact provider tape before the injected failure.  The
        # broker and Motion rollback paths therefore have real work to undo.
        self._birth_counts[uid] = birth_index + 1
        self._draw_counts[uid] = (
            draw_start + self._runtime.SAMPLER_BIRTH_DRAW_COUNT
        )
        if self.bad_at is not None and invocation == self.bad_at:
            raise RuntimeError("injected bad provider row")
        return receipt


class _FakeRobot:
    def __init__(self, num_envs: int, joint_count: int):
        self.root_write_calls = 0
        self.joint_write_calls = 0
        self.last_root_env_ids = None
        self.last_joint_env_ids = None
        self.data = types.SimpleNamespace(
            root_state_w=torch.zeros(num_envs, 13),
            joint_pos=torch.zeros(num_envs, joint_count),
            joint_vel=torch.zeros(num_envs, joint_count),
            joint_names=tuple(f"joint_{index}" for index in range(joint_count)),
        )

    def write_root_state_to_sim(self, root_state, *, env_ids):
        self.root_write_calls += 1
        self.last_root_env_ids = env_ids.detach().cpu().clone()
        self.data.root_state_w[env_ids] = root_state

    def write_joint_state_to_sim(
        self, joint_pos, joint_vel, *, env_ids
    ):
        self.joint_write_calls += 1
        self.last_joint_env_ids = env_ids.detach().cpu().clone()
        self.data.joint_pos[env_ids] = joint_pos
        self.data.joint_vel[env_ids] = joint_vel


def _counted_balanced_sampler(count: int, clip_order):
    sampler = C._BalancedRoundRobinClipSampler(
        count,
        97,
        clip_order,
        "cpu",
    )
    original_sample = sampler.sample
    sampler.sample_invocations = 0

    def counted_sample(sample_count):
        sampler.sample_invocations += 1
        return original_sample(sample_count)

    sampler.sample = counted_sample
    return sampler


def _motion_harness(
    num_envs: int,
    *,
    provider_bad_at: int | None = None,
):
    runtime = C.MotionCommand._action_ball_runtime_module()
    bindings = _bindings(runtime)
    pins = _pins(runtime)
    broker = runtime.ActionBirthBroker(bindings, pins, "no_move")
    authority = _DomainAuthority(runtime, bindings, "no_move")
    provider = _BirthProvider(runtime, bad_at=provider_bad_at)
    broker.bind_domain_claim_authority(authority)
    broker.bind_provider(provider)

    command = C.MotionCommand.__new__(C.MotionCommand)
    command.num_envs = num_envs
    command.device = "cpu"
    command._env = types.SimpleNamespace(
        step_dt=_POLICY_DT_S,
        max_episode_length=100,
        scene=types.SimpleNamespace(
            env_origins=torch.stack(
                (
                    torch.arange(num_envs, dtype=torch.float32) * 2.0,
                    torch.arange(num_envs, dtype=torch.float32) * -0.5,
                    torch.zeros(num_envs),
                ),
                dim=-1,
            )
        )
    )
    joint_count = 31
    command.robot = _FakeRobot(num_envs, joint_count)
    time_step_total = _ACTION_COUNT * _SEGMENT_FRAMES
    segment_starts = (
        torch.arange(_ACTION_COUNT, dtype=torch.long)
        * _SEGMENT_FRAMES
    )
    ready_pos = torch.zeros(time_step_total, 1, 3)
    ready_pos[:, 0, 2] = _READY_Z_M
    # A real canonical-ready root carries roll/pitch.  Use a pitched ready
    # state with yaw zero so the birth's B_yaw receipt is identity while the
    # physical root quaternion must remain non-identity.
    ready_root_quat = torch.tensor(
        [math.cos(0.1), 0.0, -math.sin(0.1), 0.0],
        dtype=torch.float32,
    )
    ready_quat = ready_root_quat.reshape(1, 1, 4).repeat(
        time_step_total, 1, 1
    )
    command.motion = types.SimpleNamespace(
        num_segments=_ACTION_COUNT,
        seg_start=segment_starts,
        seg_len=torch.full(
            (_ACTION_COUNT,), _SEGMENT_FRAMES, dtype=torch.long
        ),
        time_step_total=time_step_total,
        body_pos_w=ready_pos,
        body_quat_w=ready_quat,
        joint_pos=torch.arange(
            time_step_total * joint_count, dtype=torch.float32
        ).reshape(time_step_total, joint_count)
        / 1_000.0,
    )
    command.clip_id = (
        torch.arange(num_envs, dtype=torch.long) % _ACTION_COUNT
    )
    command.time_steps = torch.zeros(num_envs, dtype=torch.long)
    command.time_steps_f = torch.zeros(num_envs)
    command.speed_scale = torch.zeros(num_envs)
    command.hold_counter = torch.zeros(num_envs, dtype=torch.long)
    command.metrics = {
        "in_hold": torch.zeros(num_envs),
        "sampling_entropy": torch.zeros(num_envs),
        "sampling_top1_prob": torch.zeros(num_envs),
        "sampling_top1_bin": torch.zeros(num_envs),
    }
    command._stagger_hold_pending = None
    command._resampling_from_wrap = False
    command.retiming_active = False
    command._speed_per_clip = None
    command.planner_revision_enabled = False

    clip_order = tuple(binding.motion_path for binding in bindings)
    command._balanced_clip_sampler = _counted_balanced_sampler(
        _ACTION_COUNT, clip_order
    )
    command._motion_files = clip_order
    command._motion_file_sha256 = tuple(
        binding.motion_sha256 for binding in bindings
    )
    command._action_ball_birth_broker = broker
    command._action_ball_runtime_module_bound = runtime
    command._action_ball_trusted_repo_root = Path("/synthetic/trusted/repo")
    command._action_ball_action_uids = broker.ordered_action_uids
    command._action_ball_motion_sha256 = tuple(
        binding.motion_sha256 for binding in bindings
    )
    command._action_ball_ready_root_z = (_READY_Z_M,) * _ACTION_COUNT
    command._action_ball_ready_root_quat = (
        tuple(float(value) for value in ready_root_quat.tolist()),
    ) * _ACTION_COUNT
    command._action_ball_reset_generation = torch.zeros(
        num_envs, dtype=torch.long
    )
    command._action_ball_swing_generation = torch.zeros(
        num_envs, dtype=torch.long
    )
    command._action_ball_birth_receipt_sha256 = [None] * num_envs
    command._action_ball_seen_birth_receipts = set()
    command._action_ball_task_ref_for_env = None
    command._action_ball_task_receipt_resolver = None
    command._action_ball_shared_state_sha256_accessor = None
    command._action_ball_expected_shared_racket_state_sha256 = None
    command._action_ball_active_task_refs = [None] * num_envs
    command._action_ball_task_timing_active = torch.zeros(
        num_envs, dtype=torch.bool
    )
    command._action_ball_task_pending_elapsed_s = torch.zeros(
        num_envs, dtype=torch.float64
    )
    command._action_ball_task_age_s = torch.zeros(
        num_envs, dtype=torch.float64
    )
    command._action_ball_time_to_contact_s = torch.zeros(
        num_envs, dtype=torch.float64
    )
    command._action_ball_teacher_rate = torch.zeros(
        num_envs, dtype=torch.float64
    )
    command._action_ball_scaled_t_hit_s = torch.zeros(
        num_envs, dtype=torch.float64
    )
    command._action_ball_scaled_t_cycle_s = torch.zeros(
        num_envs, dtype=torch.float64
    )
    command._action_ball_pre_swing_wait_s = torch.zeros(
        num_envs, dtype=torch.float64
    )
    command._action_ball_motion_admission_receipt_sha256 = _digest(
        "synthetic-admission"
    )

    # Transaction/resume harness only.  The real opaque-capability recheck is
    # covered by the dedicated AST test below.
    admission_sha = _digest("controlled-motion-admission-stub")
    command.action_ball_motion_admission_hard_contract = types.MethodType(
        lambda self: {"canonical_sha256": admission_sha},
        command,
    )

    command.bin_count = 4
    command.bin_failed_count = torch.zeros(4)
    command._current_bin_failed = torch.zeros(4)
    command._post_swing_root = None
    command._post_swing_joint_pos = None
    command._post_swing_joint_vel = None
    command._post_swing_ptr = 0
    command._post_swing_count = 0
    command._post_swing_first_reset_checked = False
    command._post_swing_teacher_hard_contract = None
    command._post_swing_fail_fast_first_reset = False
    command._post_swing_first_reset_min_adopted_count = 0
    command._post_swing_first_reset_min_adopted_fraction = 0.0
    command._post_swing_first_reset_selection_tolerance = 0.0
    command._post_swing_first_reset_require_readback = False
    command.cfg = types.SimpleNamespace(
        body_names=("root",),
        adaptive_kernel_size=3,
        adaptive_lambda=0.8,
        adaptive_uniform_ratio=0.1,
        adaptive_alpha=0.02,
        post_swing_start_prob=0.0,
        post_swing_buffer_size=16,
        post_swing_min_fill=1,
        post_swing_min_hold=0,
        speed_scale_range=(1.0, 1.0),
        hold_steps_range=(0, 0),
        stand_start_min_hold=0,
        stagger_initial_clock=False,
        wrap_teleport=False,
    )
    return command, runtime, broker, provider, authority


def _reserve_write_commit(command, env_ids):
    transaction = command._reserve_action_ball_true_reset(env_ids)
    rollback_state = command._write_canonical_ready_state(
        env_ids,
        action_ball_base_spawn_w_m=transaction["spawn"],
        action_ball_base_quat_wxyz=transaction["quat"],
    )
    command._commit_action_ball_true_reset(env_ids, transaction)
    return transaction, rollback_state


def _consume_committed(runtime, broker, receipts):
    requests = tuple(
        runtime.BirthConsumeRequest(
            env_id=receipt.env_id,
            reset_generation=receipt.reset_generation,
            action_uid=receipt.action_uid,
            action_slot=receipt.action_slot,
            receipt_sha256=receipt.canonical_sha256,
        )
        for receipt in receipts
    )
    return broker.consume_many_true_reset(
        requests, reset_kind="true_reset"
    )


def _task_receipt(runtime, birth, *, swing_generation: int):
    """Mint one exact synthetic task whose timing matches the Motion harness."""

    base_goal = birth.base_spawn_w_m
    contact = (
        birth.base_spawn_w_m[0] + 0.55,
        birth.base_spawn_w_m[1],
        birth.base_spawn_w_m[2] - 0.12,
    )
    contact_offset = tuple(
        contact[index] - base_goal[index] for index in range(3)
    )
    incoming_velocity = (-4.0, 0.0, 0.0)
    incoming_speed = 4.0
    incoming_direction = (-1.0, 0.0, 0.0)
    spin_magnitude = 0.0
    spin_direction = (0.0, 1.0, 0.0)
    incoming_spin = (0.0, 0.0, 0.0)
    racket_velocity = (3.0, 0.0, 0.0)
    time_to_contact_s = 0.80
    geometry = runtime._contact_geometry.solve_exact_face_contact(
        ball_contact_w_m=contact,
        racket_face_center_velocity_w_mps=racket_velocity,
        solved_raw_a_normal_w=(1.0, 0.0, 0.0),
        mount_normal_sign=1,
        reference_racket_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        reference_racket_angular_velocity_w_radps=(0.0, 0.0, 0.0),
        reference_racket_site_speed_mps=3.0,
        teacher_rate_min=0.8,
        teacher_rate_max=1.2,
    )
    timing = runtime.derive_action_teacher_site_timing(
        racket_site_velocity_w_mps=(
            geometry.racket_site_velocity_w_mps
        ),
        time_to_contact_s=time_to_contact_s,
        reference_t_hit_s=_REFERENCE_T_HIT_S,
        reference_t_cycle_s=_REFERENCE_T_CYCLE_S,
        reference_racket_site_speed_mps=3.0,
        reaction_margin_s=0.05,
        teacher_rate_min=0.8,
        teacher_rate_max=1.2,
    )
    sample_index = swing_generation
    draw_start = (
        1_000 + sample_index * runtime.SAMPLER_SAMPLE_DRAW_COUNT
    )
    draw_end = draw_start + runtime.SAMPLER_SAMPLE_DRAW_COUNT
    sample_identity = {
        "schema_version": runtime.SAMPLER_SCHEMA_VERSION,
        "kind": "swing_sample",
        "sampler_contract_sha256": birth.sampler_sha256,
        "arm_catalog_sha256": birth.arm_catalog_sha256,
        "sample_index": sample_index,
        "action_uid": birth.action_uid,
        "domain_epoch": birth.domain_epoch,
        "domain_levels": birth.domain_levels.to_dict(),
        "birth_id": birth.sampler_birth_sha256,
        "profile_sha256": birth.profile_sha256,
        "levels_sha256": birth.levels_sha256,
        "draw_start": draw_start,
        "draw_end": draw_end,
        "mobility_mode": birth.mobility_mode,
        "base_yaw_rad": birth.base_yaw_rad,
        "base_start_w_m": birth.base_spawn_w_m,
        "base_spawn_latent_w_m": birth.base_spawn_w_m,
        "base_travel_latent_b_yaw_m": (0.0, 0.0, 0.0),
        "base_goal_w_m": base_goal,
        "contact_offset_from_base_goal_b_yaw_m": contact_offset,
        "contact_w_m": contact,
        "time_to_contact_s": time_to_contact_s,
        "incoming_speed_mps": incoming_speed,
        "incoming_direction_b_yaw": incoming_direction,
        "incoming_direction_w": incoming_direction,
        "incoming_velocity_w_mps": incoming_velocity,
        "spin_magnitude_radps": spin_magnitude,
        "spin_direction_b_yaw": spin_direction,
        "spin_direction_w": spin_direction,
        "spin_w_radps": incoming_spin,
        "landing_aim_w_xy_m": (2.5, 0.0),
    }
    return runtime.ActionBallTaskReceipt.from_birth(
        birth,
        sample_sha256=runtime._sha256_json(sample_identity),
        sample_index=sample_index,
        sample_draw_start=draw_start,
        sample_draw_end=draw_end,
        swing_generation=swing_generation,
        base_goal_w_m=base_goal,
        base_spawn_latent_w_m=birth.base_spawn_w_m,
        base_travel_latent_b_yaw_m=(0.0, 0.0, 0.0),
        contact_offset_from_base_goal_b_yaw_m=contact_offset,
        ball_contact_w_m=contact,
        time_to_contact_s=time_to_contact_s,
        incoming_speed_mps=incoming_speed,
        incoming_direction_b_yaw=incoming_direction,
        incoming_velocity_w_mps=incoming_velocity,
        spin_magnitude_radps=spin_magnitude,
        spin_direction_b_yaw=spin_direction,
        incoming_spin_w_radps=incoming_spin,
        landing_aim_w_xy_m=(2.5, 0.0),
        racket_site_target_w_m=geometry.racket_site_target_w_m,
        mount_normal_sign=1,
        racket_normal_w=(1.0, 0.0, 0.0),
        reference_racket_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        reference_racket_angular_velocity_w_radps=(0.0, 0.0, 0.0),
        racket_command_quat_wxyz=(
            geometry.racket_command_quat_wxyz
        ),
        racket_face_center_velocity_w_mps=(
            geometry.racket_face_center_velocity_w_mps
        ),
        racket_site_velocity_w_mps=(
            geometry.racket_site_velocity_w_mps
        ),
        racket_command_angular_velocity_w_radps=(
            geometry.racket_command_angular_velocity_w_radps
        ),
        geometry_source_sha256=geometry.geometry_source_sha256,
        reference_t_hit_s=_REFERENCE_T_HIT_S,
        reference_t_cycle_s=_REFERENCE_T_CYCLE_S,
        reference_racket_site_speed_mps=3.0,
        required_racket_site_speed_mps=(
            timing.required_racket_site_speed_mps
        ),
        reaction_margin_s=0.05,
        teacher_rate_min=0.8,
        teacher_rate_max=1.2,
        teacher_rate=timing.teacher_rate,
        scaled_t_hit_s=timing.scaled_t_hit_s,
        scaled_t_cycle_s=timing.scaled_t_cycle_s,
        pre_swing_wait_s=timing.pre_swing_wait_s,
        solver_residual_m=0.004,
    )


class _TaskAuthority:
    """Synthetic Racket owner for the exact opaque ref/resolve seam."""

    def __init__(self, runtime, broker):
        self._runtime = runtime
        self._broker = broker
        self._tasks = {}
        self._nonce = 0
        self.ref_calls = 0
        self.resolve_calls = 0
        self.digest_calls = 0

    def install(self, receipts):
        self._tasks = {
            receipt.env_id: receipt for receipt in receipts
        }
        self._nonce += 1

    def state_dict(self):
        return {
            "broker": self._broker.state_dict(),
            "tasks": [
                self._tasks[env_id].to_dict()
                for env_id in sorted(self._tasks)
            ],
            "nonce": self._nonce,
        }

    def load_state_dict(self, state):
        if type(state) is not dict or set(state) != {
            "broker",
            "tasks",
            "nonce",
        }:
            raise ValueError("invalid synthetic Racket state")
        tasks = [
            self._runtime.ActionBallTaskReceipt.from_dict(row)
            for row in state["tasks"]
        ]
        self._broker.load_state_dict(state["broker"])
        self._tasks = {receipt.env_id: receipt for receipt in tasks}
        self._nonce = int(state["nonce"])

    def action_ball_task_ref_for_env(self, env_id):
        self.ref_calls += 1
        receipt = self._tasks.get(int(env_id))
        return None if receipt is None else receipt.task_ref()

    def action_ball_resolve_task_ref(self, ref):
        self.resolve_calls += 1
        receipt = self._tasks.get(ref.env_id)
        if receipt is None or receipt.task_ref() != ref:
            raise KeyError("unknown synthetic task ref")
        return receipt

    def action_ball_shared_state_sha256(self):
        self.digest_calls += 1
        return self._runtime._sha256_json(self.state_dict())


def _bind_task_authority(command, runtime, broker):
    authority = _TaskAuthority(runtime, broker)
    command.bind_action_ball_task_authority(
        task_ref_for_env=authority.action_ball_task_ref_for_env,
        resolve_task_ref=authority.action_ball_resolve_task_ref,
        shared_state_sha256=authority.action_ball_shared_state_sha256,
    )
    assert authority.digest_calls == 0
    command.validate_action_ball_task_authority_binding()
    assert authority.digest_calls == 1
    return authority


def _install_current_tasks(
    command,
    runtime,
    broker,
    authority,
    birth_receipts,
    *,
    elapsed_s: float = 0.0,
):
    consumed = _consume_committed(runtime, broker, birth_receipts)
    tasks = tuple(
        _task_receipt(
            runtime,
            receipt,
            swing_generation=int(
                command._action_ball_swing_generation[
                    receipt.env_id
                ].item()
            ),
        )
        for receipt in consumed
    )
    authority.install(tasks)
    env_ids = torch.tensor(
        [receipt.env_id for receipt in tasks], dtype=torch.long
    )
    command._begin_action_ball_task_pending(
        env_ids, elapsed_s=elapsed_s
    )
    return tasks


def _assert_nested_equal(left, right):
    if torch.is_tensor(left) or torch.is_tensor(right):
        assert torch.is_tensor(left) and torch.is_tensor(right)
        assert left.dtype == right.dtype
        assert left.device == right.device
        assert torch.equal(left, right)
        return
    if isinstance(left, dict) or isinstance(right, dict):
        assert isinstance(left, dict) and isinstance(right, dict)
        assert set(left) == set(right)
        for key in left:
            _assert_nested_equal(left[key], right[key])
        return
    if isinstance(left, (tuple, list)) or isinstance(
        right, (tuple, list)
    ):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_nested_equal(left_item, right_item)
        return
    assert left == right


def test_motion_reserves_writes_and_commits_one_4096_env_batch():
    command, runtime, broker, provider, authority = _motion_harness(4096)
    env_ids = torch.arange(4096, dtype=torch.long)

    transaction, rollback_state = _reserve_write_commit(
        command, env_ids
    )

    assert type(transaction["receipts"]) is tuple
    assert len(transaction["receipts"]) == 4096
    assert len(set(transaction["receipt_sha256"])) == 4096
    assert provider.issue_invocations == 4096
    assert authority.claim_invocations == 4096
    assert command.robot.root_write_calls == 1
    assert command.robot.joint_write_calls == 1
    assert torch.equal(command.robot.last_root_env_ids, env_ids)
    assert torch.equal(command.robot.last_joint_env_ids, env_ids)
    assert rollback_state is not None

    expected_position = (
        command._env.scene.env_origins[env_ids]
        + transaction["spawn"]
    )
    assert torch.allclose(
        command.robot.data.root_state_w[:, :3], expected_position
    )
    expected_ready_steps = command.motion.seg_start[command.clip_id]
    expected_root_quat = command.motion.body_quat_w[
        expected_ready_steps, 0
    ]
    assert torch.equal(
        command.robot.data.root_state_w[:, 3:7],
        expected_root_quat,
    )
    assert not torch.equal(expected_root_quat, transaction["quat"])
    assert torch.count_nonzero(
        command.robot.data.root_state_w[:, 7:]
    ).item() == 0
    assert torch.count_nonzero(command.robot.data.joint_vel).item() == 0
    assert torch.equal(
        command._action_ball_reset_generation,
        torch.ones(4096, dtype=torch.long),
    )
    assert torch.count_nonzero(
        command._action_ball_swing_generation
    ).item() == 0
    assert len(command._action_ball_seen_birth_receipts) == 4096
    assert all(
        receipt is not None
        for receipt in command._action_ball_birth_receipt_sha256
    )

    broker_state = broker.state_dict()
    assert broker_state["schema_version"] == (
        runtime.BROKER_STATE_SCHEMA_VERSION
    )
    assert len(broker_state["pending"]) == 4096
    assert {
        row["status"] for row in broker_state["pending"]
    } == {"committed"}


def test_bad_provider_row_rolls_back_broker_tapes_motion_and_sampler():
    bad_at = 2_048
    command, _runtime, broker, provider, authority = _motion_harness(
        4096, provider_bad_at=bad_at
    )
    env_ids = torch.arange(4096, dtype=torch.long)
    # Make the sampler cursor non-zero so a false reset-to-default cannot pass.
    command._balanced_clip_sampler.sample(3)
    command.clip_id[:] = 4
    command.time_steps[:] = 17
    command.time_steps_f[:] = 17.0
    command.speed_scale[:] = 0.75
    command.hold_counter[:] = 9

    def failing_body(self, reset_env_ids):
        self._action_ball_select_or_rewind_action(reset_env_ids)
        return self._reserve_action_ball_true_reset(reset_env_ids)

    command._resample_command_body = types.MethodType(
        failing_body, command
    )

    broker_before = deepcopy(broker.state_dict())
    provider_before = deepcopy(provider.state_dict())
    authority_before = deepcopy(authority.state_dict())
    sampler_before = deepcopy(
        command.balanced_clip_sampler_state_dict()
    )
    clip_before = command.clip_id.clone()
    time_before = command.time_steps.clone()
    time_f_before = command.time_steps_f.clone()
    speed_before = command.speed_scale.clone()
    hold_before = command.hold_counter.clone()
    generation_before = command._action_ball_reset_generation.clone()
    swing_before = command._action_ball_swing_generation.clone()
    receipts_before = list(
        command._action_ball_birth_receipt_sha256
    )
    seen_before = set(command._action_ball_seen_birth_receipts)
    root_writes_before = command.robot.root_write_calls
    joint_writes_before = command.robot.joint_write_calls

    with pytest.raises(RuntimeError, match="injected bad provider row"):
        command._resample_command(env_ids)

    assert provider.issue_invocations == bad_at + 1
    assert broker.state_dict() == broker_before
    assert provider.state_dict() == provider_before
    assert authority.state_dict() == authority_before
    assert command.balanced_clip_sampler_state_dict() == sampler_before
    assert torch.equal(command.clip_id, clip_before)
    assert torch.equal(command.time_steps, time_before)
    assert torch.equal(command.time_steps_f, time_f_before)
    assert torch.equal(command.speed_scale, speed_before)
    assert torch.equal(command.hold_counter, hold_before)
    assert torch.equal(
        command._action_ball_reset_generation, generation_before
    )
    assert torch.equal(
        command._action_ball_swing_generation, swing_before
    )
    assert (
        command._action_ball_birth_receipt_sha256 == receipts_before
    )
    assert command._action_ball_seen_birth_receipts == seen_before
    assert command.robot.root_write_calls == root_writes_before
    assert command.robot.joint_write_calls == joint_writes_before


def test_forged_runtime_birth_and_untrusted_paths_are_rejected(tmp_path):
    command, _runtime, _broker, _provider, _authority = (
        _motion_harness(1)
    )
    transaction = command._reserve_action_ball_true_reset(
        torch.tensor([0], dtype=torch.long)
    )
    real_receipt = transaction["receipts"][0]
    forged = types.SimpleNamespace(**real_receipt.__dict__)

    with pytest.raises(ValueError, match="forged runtime type"):
        command._validate_action_ball_birth_receipt(
            forged,
            env_id=0,
            reset_generation=1,
            action_slot=0,
            action_uid=command._action_ball_action_uids[0],
        )
    command._rollback_action_ball_broker(
        transaction["broker_state_before"],
        original_error=ValueError("test cleanup"),
    )

    trusted_root = tmp_path / "repo"
    trusted_root.mkdir()
    outside = tmp_path / "outside.npz"
    outside.write_bytes(b"outside")
    with pytest.raises(ValueError, match="relative|escape|trusted"):
        C.MotionCommand._action_ball_file_receipt(
            trusted_root,
            "../outside.npz",
            name="escaped motion",
        )
    link = trusted_root / "linked.npz"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable on this platform")
    with pytest.raises(ValueError, match="symlink|regular|trusted"):
        C.MotionCommand._action_ball_file_receipt(
            trusted_root,
            "linked.npz",
            name="symlinked motion",
        )


def test_broker_bind_revalidates_real_opaque_admission_capability():
    bind_tree = ast.parse(
        textwrap.dedent(
            inspect.getsource(
                C.MotionCommand.bind_action_ball_birth_broker
            )
        )
    )
    bind_calls = {
        node.func.attr
        for node in ast.walk(bind_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
    }
    assert {
        "_action_ball_resolve_root",
        "_require_action_ball_motion_admission",
        "_action_ball_file_receipt",
    } <= bind_calls

    require_tree = ast.parse(
        textwrap.dedent(
            inspect.getsource(
                C.MotionCommand._require_action_ball_motion_admission
            )
        )
    )
    require_calls = {
        node.func.attr
        for node in ast.walk(require_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
    }
    assert "require_matching_admission" in require_calls


def test_receipt_timing_hits_exact_contact_and_presents_final_frame_once():
    command, runtime, broker, _provider, _domain = _motion_harness(1)
    task_authority = _bind_task_authority(command, runtime, broker)
    env_ids = torch.tensor([0], dtype=torch.long)
    transaction, _rollback = _reserve_write_commit(command, env_ids)
    tasks = _install_current_tasks(
        command,
        runtime,
        broker,
        task_authority,
        transaction["receipts"],
    )
    task = tasks[0]

    assert not bool(command.action_ball_task_timing_active[0])
    assert (
        float(command.action_ball_time_to_contact_remaining_s[0])
        == 1.0e6
    )

    # True-reset task publication and the first command compute happen before
    # this task has driven physics. Resolution must therefore leave age at 0.
    held, cycle_due = command._advance_action_ball_task_timing()
    assert bool(held[0])
    assert not bool(cycle_due[0])
    assert float(command._action_ball_task_age_s[0]) == 0.0

    for _ in range(round(task.time_to_contact_s / _POLICY_DT_S)):
        held, cycle_due = command._advance_action_ball_task_timing()
    assert not bool(held[0])
    assert not bool(cycle_due[0])
    assert float(command._action_ball_task_age_s[0]) == pytest.approx(
        task.time_to_contact_s,
        abs=1.0e-12,
    )
    assert float(
        command.action_ball_time_to_contact_remaining_s[0]
    ) == pytest.approx(0.0, abs=1.0e-12)
    assert float(command.speed_scale[0]) == pytest.approx(
        task.teacher_rate,
        abs=1.0e-12,
    )
    assert float(command.time_steps_f[0]) == pytest.approx(
        _REFERENCE_T_HIT_S / _POLICY_DT_S,
        abs=1.0e-6,
    )

    cycle_steps = round(
        (
            task.pre_swing_wait_s
            + task.scaled_t_cycle_s
        )
        / _POLICY_DT_S
    )
    for _ in range(
        round(task.time_to_contact_s / _POLICY_DT_S),
        cycle_steps,
    ):
        held, cycle_due = command._advance_action_ball_task_timing()
    assert not bool(cycle_due[0])
    assert int(command.time_steps[0]) == _SEGMENT_FRAMES - 1
    assert float(command.time_steps_f[0]) == pytest.approx(
        _SEGMENT_FRAMES - 1,
        abs=1.0e-6,
    )

    # The next command update detects completion before advancing, so the
    # final reference frame remains visible for exactly the intervening tick.
    _held, cycle_due = command._advance_action_ball_task_timing()
    assert bool(cycle_due[0])
    assert int(command.time_steps[0]) == _SEGMENT_FRAMES - 1


def test_wrap_installs_age_zero_then_counts_exactly_one_physical_tick():
    command, runtime, broker, _provider, _domain = _motion_harness(1)
    task_authority = _bind_task_authority(command, runtime, broker)
    env_ids = torch.tensor([0], dtype=torch.long)
    transaction, _rollback = _reserve_write_commit(command, env_ids)
    first_tasks = _install_current_tasks(
        command,
        runtime,
        broker,
        task_authority,
        transaction["receipts"],
    )
    command._resolve_pending_action_ball_tasks()

    command._event_scheduler = None
    command._adaptive_sampling = types.MethodType(
        lambda self, ids: None, command
    )
    command.canonical_ready_mode = True
    command._multiseg = True
    command._resampling_from_wrap = True
    try:
        command._resample_command_body(env_ids)
    finally:
        command._resampling_from_wrap = False

    assert int(command._action_ball_swing_generation[0]) == 1
    assert not bool(command._action_ball_task_timing_active[0])
    assert float(
        command._action_ball_task_pending_elapsed_s[0]
    ) == _POLICY_DT_S

    second_task = _task_receipt(
        runtime,
        transaction["receipts"][0],
        swing_generation=1,
    )
    task_authority.install((second_task,))
    assert (
        float(command.action_ball_time_to_contact_remaining_s[0])
        == 1.0e6
    )

    # One physics tick occurs between Racket's same-compute install and the
    # next Motion compute. WRAP carries that dt into resolution; newly
    # resolved rows do not add a second dt in the same compute.
    held, due = command._advance_action_ball_task_timing()
    assert bool(held[0])
    assert not bool(due[0])
    assert float(command._action_ball_task_age_s[0]) == pytest.approx(
        _POLICY_DT_S,
        abs=1.0e-12,
    )
    assert float(
        command.action_ball_time_to_contact_remaining_s[0]
    ) == pytest.approx(
        second_task.time_to_contact_s - _POLICY_DT_S,
        abs=1.0e-12,
    )
    assert first_tasks[0].task_ref() != second_task.task_ref()


def test_public_timing_handoff_exposes_teacher_wait_before_first_observation():
    command, runtime, broker, _provider, _domain = _motion_harness(1)
    task_authority = _bind_task_authority(command, runtime, broker)
    env_ids = torch.tensor([0], dtype=torch.long)
    transaction, _rollback = _reserve_write_commit(command, env_ids)
    tasks = _install_current_tasks(
        command,
        runtime,
        broker,
        task_authority,
        transaction["receipts"],
    )
    task = tasks[0]

    assert not bool(command.action_ball_task_timing_active[0])
    assert float(
        command.action_ball_pre_swing_wait_remaining_s[0]
    ) == 0.0

    command.resolve_action_ball_task_timing_now(env_ids)
    assert bool(command.action_ball_task_timing_active[0])
    assert float(
        command.action_ball_pre_swing_wait_remaining_s[0]
    ) == pytest.approx(task.pre_swing_wait_s, abs=1.0e-12)

    held, due = command._advance_action_ball_task_timing()
    assert bool(held[0])
    assert not bool(due[0])
    assert float(
        command.action_ball_pre_swing_wait_remaining_s[0]
    ) == pytest.approx(
        max(task.pre_swing_wait_s - _POLICY_DT_S, 0.0),
        abs=1.0e-12,
    )


def test_receipt_cycle_must_fit_episode_horizon_with_one_close_tick():
    command, runtime, broker, _provider, _domain = _motion_harness(1)
    task_authority = _bind_task_authority(command, runtime, broker)
    command._env.max_episode_length = 79
    env_ids = torch.tensor([0], dtype=torch.long)
    transaction, _rollback = _reserve_write_commit(command, env_ids)
    _install_current_tasks(
        command,
        runtime,
        broker,
        task_authority,
        transaction["receipts"],
    )

    with pytest.raises(ValueError, match="cycle plus close tick"):
        command._resolve_pending_action_ball_tasks()


def test_table_hit_true_reset_replaces_birth_task_and_clears_swing_clock():
    command, runtime, broker, _provider, _domain = _motion_harness(1)
    task_authority = _bind_task_authority(command, runtime, broker)
    command._event_scheduler = None
    command.canonical_ready_mode = True
    command._multiseg = True
    env_ids = torch.tensor([0], dtype=torch.long)

    command._resample_command(env_ids)
    first_slot = int(command.clip_id[0])
    first_birth = broker.pending_receipt(
        env_id=0,
        reset_generation=1,
        action_uid=command._action_ball_action_uids[first_slot],
        action_slot=first_slot,
        reset_kind="true_reset",
    )
    _install_current_tasks(
        command,
        runtime,
        broker,
        task_authority,
        (first_birth,),
    )
    command._resolve_pending_action_ball_tasks()
    first_ref = command._action_ball_active_task_refs[0]

    # A table-hit termination is a normal manager true reset to Motion: it
    # must mint a new physical birth and discard every old per-swing clock.
    command._resample_command(env_ids)
    second_slot = int(command.clip_id[0])
    second_birth = broker.pending_receipt(
        env_id=0,
        reset_generation=2,
        action_uid=command._action_ball_action_uids[second_slot],
        action_slot=second_slot,
        reset_kind="true_reset",
    )
    assert int(command._action_ball_reset_generation[0]) == 2
    assert int(command._action_ball_swing_generation[0]) == 0
    assert command._action_ball_active_task_refs[0] is None
    assert not bool(command._action_ball_task_timing_active[0])
    assert float(command._action_ball_task_age_s[0]) == 0.0
    assert second_birth.canonical_sha256 != first_birth.canonical_sha256
    assert first_birth.canonical_sha256 in (
        command._action_ball_seen_birth_receipts
    )

    second_tasks = _install_current_tasks(
        command,
        runtime,
        broker,
        task_authority,
        (second_birth,),
    )
    held, due = command._advance_action_ball_task_timing()
    second_ref = command._action_ball_active_task_refs[0]
    assert bool(held[0])
    assert not bool(due[0])
    assert float(command._action_ball_task_age_s[0]) == 0.0
    assert second_ref == second_tasks[0].task_ref()
    assert second_ref != first_ref
    assert second_ref.reset_generation == 2
    assert second_ref.swing_generation == 0


def test_motion_rejects_mutated_exact_receipt_instead_of_clipping_rate():
    command, runtime, broker, _provider, _domain = _motion_harness(1)
    task_authority = _bind_task_authority(command, runtime, broker)
    env_ids = torch.tensor([0], dtype=torch.long)
    transaction, _rollback = _reserve_write_commit(command, env_ids)
    consumed = _consume_committed(
        runtime, broker, transaction["receipts"]
    )
    task = _task_receipt(runtime, consumed[0], swing_generation=0)
    # Frozen dataclasses are an API guard, not a trust boundary. Simulate
    # corrupted in-process memory while preserving the exact runtime class.
    object.__setattr__(task, "teacher_rate", 1.21)
    task_authority.install((task,))
    command._begin_action_ball_task_pending(env_ids, elapsed_s=0.0)

    with pytest.raises(ValueError, match="outside its certified range"):
        command._resolve_pending_action_ball_tasks()
    assert float(command.speed_scale[0]) == 0.0


def test_missing_contact_geometry_preserves_the_root_attribute_error(
    monkeypatch,
):
    command, runtime, broker, _provider, _domain = _motion_harness(1)
    task_authority = _bind_task_authority(command, runtime, broker)
    env_ids = torch.tensor([0], dtype=torch.long)
    transaction, _rollback = _reserve_write_commit(command, env_ids)
    consumed = _consume_committed(
        runtime, broker, transaction["receipts"]
    )
    task_authority.install(
        (_task_receipt(runtime, consumed[0], swing_generation=0),)
    )
    command._begin_action_ball_task_pending(env_ids, elapsed_s=0.0)
    monkeypatch.delattr(runtime, "_contact_geometry")

    with pytest.raises(AttributeError, match="_contact_geometry"):
        command._resolve_pending_action_ball_tasks()


def test_motion_accepts_canonical_float32_teacher_rate_boundary_seam():
    command, runtime, broker, _provider, _domain = _motion_harness(1)
    task_authority = _bind_task_authority(command, runtime, broker)
    env_ids = torch.tensor([0], dtype=torch.long)
    transaction, _rollback = _reserve_write_commit(command, env_ids)
    consumed = _consume_committed(
        runtime, broker, transaction["receipts"]
    )
    task = _task_receipt(runtime, consumed[0], swing_generation=0)

    rate = float(task.teacher_rate_min) - 2.0e-7
    required_speed = float(task.reference_racket_site_speed_mps) * rate
    scaled_t_hit = float(task.reference_t_hit_s) / rate
    scaled_t_cycle = float(task.reference_t_cycle_s) / rate
    object.__setattr__(task, "teacher_rate", rate)
    object.__setattr__(task, "required_racket_site_speed_mps", required_speed)
    object.__setattr__(
        task, "racket_site_velocity_w_mps", (required_speed, 0.0, 0.0)
    )
    object.__setattr__(task, "scaled_t_hit_s", scaled_t_hit)
    object.__setattr__(task, "scaled_t_cycle_s", scaled_t_cycle)
    object.__setattr__(
        task,
        "pre_swing_wait_s",
        float(task.time_to_contact_s) - scaled_t_hit,
    )
    task_authority.install((task,))
    command._begin_action_ball_task_pending(env_ids, elapsed_s=0.0)

    command._resolve_pending_action_ball_tasks()
    assert bool(command.action_ball_task_timing_active[0])
    assert math.isclose(
        float(command._action_ball_teacher_rate[0]), rate, abs_tol=1.0e-7
    )


def test_schema4_exact_resume_handoff_restores_local_refs_without_shared_io():
    command, runtime, broker, provider, authority = _motion_harness(8)
    task_authority = _bind_task_authority(command, runtime, broker)
    env_ids = torch.arange(8, dtype=torch.long)
    first_transaction, _rollback = _reserve_write_commit(
        command, env_ids
    )
    _install_current_tasks(
        command,
        runtime,
        broker,
        task_authority,
        first_transaction["receipts"],
    )
    command._resolve_pending_action_ball_tasks()
    saved_shared_racket = deepcopy(task_authority.state_dict())
    saved = command.exact_resume_state_dict()
    assert saved["schema_version"] == 4
    assert "broker_state" not in saved["action_ball_birth"]
    assert (
        saved["action_ball_birth"]["shared_racket_state_sha256"]
        == runtime._sha256_json(saved_shared_racket)
    )
    assert (
        saved_shared_racket["broker"]["schema_version"]
        == runtime.BROKER_STATE_SCHEMA_VERSION
    )
    assert all(
        row is not None
        for row in saved["action_ball_birth"]["active_task_refs"]
    )

    # Move every behaviorally relevant tape/state away from the checkpoint.
    command._balanced_clip_sampler.sample(7)
    second_transaction = command._reserve_action_ball_true_reset(env_ids)
    command._commit_action_ball_true_reset(env_ids, second_transaction)
    _install_current_tasks(
        command,
        runtime,
        broker,
        task_authority,
        second_transaction["receipts"],
    )
    command._resolve_pending_action_ball_tasks()
    command.bin_failed_count[:] = 11.0
    command._current_bin_failed[:] = 13.0
    assert task_authority.state_dict() != saved_shared_racket

    # Racket owns the complete shared graph (including broker/tasks) and
    # restores it first. Motion receives only its digest and local refs.
    task_authority.load_state_dict(saved_shared_racket)
    broker_load_calls = 0
    original_broker_load = broker.load_state_dict

    def counted_broker_load(state):
        nonlocal broker_load_calls
        broker_load_calls += 1
        return original_broker_load(state)

    broker.load_state_dict = counted_broker_load

    provider_issue_before = provider.issue_invocations
    authority_claim_before = authority.claim_invocations
    sampler_calls_before = (
        command._balanced_clip_sampler.sample_invocations
    )
    root_writes_before = command.robot.root_write_calls
    joint_writes_before = command.robot.joint_write_calls
    ref_calls_before = task_authority.ref_calls
    resolve_calls_before = task_authority.resolve_calls
    digest_calls_before = task_authority.digest_calls
    expected_clip_ids = torch.tensor(
        [
            row["action_slot"]
            for row in saved["action_ball_birth"]["active_task_refs"]
        ],
        dtype=torch.long,
    )
    command.clip_id.copy_(
        (expected_clip_ids + 1) % _ACTION_COUNT
    )
    assert not torch.equal(command.clip_id, expected_clip_ids)

    command.load_exact_resume_state_dict(saved, strict=True)
    assert broker_load_calls == 0
    assert task_authority.ref_calls == ref_calls_before
    assert task_authority.resolve_calls == resolve_calls_before
    assert task_authority.digest_calls == digest_calls_before
    assert torch.equal(command.clip_id, expected_clip_ids)

    task_authority._nonce += 1
    with pytest.raises(RuntimeError, match="live Racket state differs"):
        command.finalize_action_ball_exact_resume()
    task_authority.load_state_dict(saved_shared_racket)
    command.finalize_action_ball_exact_resume()

    assert provider.issue_invocations == provider_issue_before
    assert authority.claim_invocations == authority_claim_before
    assert (
        command._balanced_clip_sampler.sample_invocations
        == sampler_calls_before
    )
    assert command.robot.root_write_calls == root_writes_before
    assert command.robot.joint_write_calls == joint_writes_before
    assert task_authority.state_dict() == saved_shared_racket
    _assert_nested_equal(command.exact_resume_state_dict(), saved)
