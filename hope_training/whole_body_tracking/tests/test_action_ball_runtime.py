from copy import deepcopy
from dataclasses import (
    FrozenInstanceError,
    fields as dataclass_fields,
    is_dataclass,
    replace,
)
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import pickle
import sys
import weakref

import pytest


PATH = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
    / "action_ball_runtime.py"
)
SPEC = importlib.util.spec_from_file_location("action_ball_runtime", PATH)
R = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = R
SPEC.loader.exec_module(R)

SAMPLING_PATH = PATH.with_name("action_ball_sampling.py")
SAMPLING_SPEC = importlib.util.spec_from_file_location(
    "action_ball_sampling_for_runtime_test", SAMPLING_PATH
)
S = importlib.util.module_from_spec(SAMPLING_SPEC)
assert SAMPLING_SPEC.loader is not None
sys.modules[SAMPLING_SPEC.name] = S
SAMPLING_SPEC.loader.exec_module(S)


def _digest(label):
    return hashlib.sha256(str(label).encode("utf-8")).hexdigest()


def _counter_rally_task_identity(objective_profile_sha256):
    incoming = (-4.0, 0.1, -0.2)
    horizontal_norm = math.hypot(incoming[0], incoming[1])
    return R.CounterRallyTaskIdentity(
        objective_profile_sha256=objective_profile_sha256,
        return_direction_env_xy=(
            -incoming[0] / horizontal_norm,
            -incoming[1] / horizontal_norm,
        ),
        target_baseline_speed_mps=math.sqrt(
            sum(component * component for component in incoming)
        ),
    )


def _pins(counter_rally_objective_profile_sha256=None):
    return R.RuntimePins(
        manifest_sha256=_digest("manifest"),
        sampler_sha256=_digest("sampler"),
        domain_authority_sha256=_digest("domain-authority"),
        physics_sha256=_digest("physics"),
        solver_sha256=_digest("solver"),
        counter_rally_objective_profile_sha256=(
            counter_rally_objective_profile_sha256
        ),
    )


def _bindings(count):
    return tuple(
        R.ActionBinding(
            action_uid=10_000 + slot * 17,
            action_slot=slot,
            motion_path=f"vendor_assets/motions/action_{slot:03d}.npz",
            motion_sha256=_digest(f"motion-{slot}"),
            profile_sha256=_digest(f"profile-{slot}"),
        )
        for slot in range(count)
    )


def _yaw_quat(yaw):
    return (math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw))


def _levels():
    return R.ActionDomainLevels()


def _sampling_profile(action_uid):
    return S.SamplingProfile(
        action_uid=action_uid,
        contact_offset_center_b_yaw_m=(0.55, 0.12, 0.82),
        contact_offset_std_lower_initial_m=(0.005, 0.01, 0.01),
        contact_offset_std_lower_max_m=(0.02, 0.12, 0.16),
        contact_offset_std_upper_initial_m=(0.004, 0.02, 0.01),
        contact_offset_std_upper_max_m=(0.02, 0.10, 0.15),
        contact_offset_min_b_yaw_m=(0.45, -0.20, 0.55),
        contact_offset_max_b_yaw_m=(0.65, 0.35, 1.10),
        time_to_contact_center_s=1.20,
        time_to_contact_std_lower_initial_s=0.01,
        time_to_contact_std_lower_max_s=0.15,
        time_to_contact_std_upper_initial_s=0.02,
        time_to_contact_std_upper_max_s=0.30,
        time_to_contact_min_s=1.05,
        time_to_contact_max_s=1.60,
        incoming_direction_center_b_yaw=(-1.0, 0.0, 0.0),
        incoming_direction_tangent_u_b_yaw=(0.0, 1.0, 0.0),
        incoming_direction_tangent_v_b_yaw=(0.0, 0.0, -1.0),
        incoming_direction_tangent_u_neg_initial_deg=0.5,
        incoming_direction_tangent_u_neg_max_deg=8.0,
        incoming_direction_tangent_u_pos_initial_deg=0.6,
        incoming_direction_tangent_u_pos_max_deg=7.0,
        incoming_direction_tangent_v_neg_initial_deg=0.7,
        incoming_direction_tangent_v_neg_max_deg=6.0,
        incoming_direction_tangent_v_pos_initial_deg=0.8,
        incoming_direction_tangent_v_pos_max_deg=5.0,
        incoming_inbound_axis_b_yaw=(-1.0, 0.0, 0.0),
        incoming_inbound_min_cosine=0.8,
        incoming_speed_center_mps=4.0,
        incoming_speed_std_lower_initial_mps=0.05,
        incoming_speed_std_lower_max_mps=1.2,
        incoming_speed_std_upper_initial_mps=0.06,
        incoming_speed_std_upper_max_mps=1.0,
        incoming_speed_min_mps=1.6,
        incoming_speed_max_mps=7.0,
        spin_direction_center_b_yaw=(0.0, 1.0, 0.0),
        spin_direction_tangent_u_b_yaw=(0.0, 0.0, 1.0),
        spin_direction_tangent_v_b_yaw=(1.0, 0.0, 0.0),
        spin_direction_tangent_u_neg_initial_deg=0.0,
        spin_direction_tangent_u_neg_max_deg=35.0,
        spin_direction_tangent_u_pos_initial_deg=0.0,
        spin_direction_tangent_u_pos_max_deg=30.0,
        spin_direction_tangent_v_neg_initial_deg=0.0,
        spin_direction_tangent_v_neg_max_deg=25.0,
        spin_direction_tangent_v_pos_initial_deg=0.0,
        spin_direction_tangent_v_pos_max_deg=20.0,
        spin_magnitude_center_radps=15.0,
        spin_magnitude_std_lower_initial_radps=0.2,
        spin_magnitude_std_lower_max_radps=8.0,
        spin_magnitude_std_upper_initial_radps=0.3,
        spin_magnitude_std_upper_max_radps=9.0,
        spin_magnitude_min_radps=0.0,
        spin_magnitude_max_radps=40.0,
        base_spawn_center_w_m=(-0.10, 0.05, 0.0),
        base_spawn_std_lower_initial_m=(0.005, 0.005, 0.0),
        base_spawn_std_lower_max_m=(0.15, 0.20, 0.0),
        base_spawn_std_upper_initial_m=(0.006, 0.007, 0.0),
        base_spawn_std_upper_max_m=(0.12, 0.18, 0.0),
        base_spawn_min_w_m=(-0.50, -0.40, 0.0),
        base_spawn_max_w_m=(0.30, 0.50, 0.0),
        base_travel_center_b_yaw_m=(0.20, -0.05, 0.0),
        base_travel_std_lower_initial_m=(0.01, 0.01, 0.0),
        base_travel_std_lower_max_m=(0.25, 0.25, 0.0),
        base_travel_std_upper_initial_m=(0.02, 0.01, 0.0),
        base_travel_std_upper_max_m=(0.25, 0.25, 0.0),
        base_travel_min_b_yaw_m=(-0.40, -0.40, 0.0),
        base_travel_max_b_yaw_m=(0.50, 0.40, 0.0),
        landing_aim_center_w_xy_m=(2.55, 0.0),
        landing_aim_std_lower_initial_m=(0.01, 0.01),
        landing_aim_std_lower_max_m=(0.25, 0.35),
        landing_aim_std_upper_initial_m=(0.02, 0.01),
        landing_aim_std_upper_max_m=(0.20, 0.30),
        landing_aim_min_w_xy_m=(2.20, -0.55),
        landing_aim_max_w_xy_m=(2.90, 0.55),
        reference_t_hit_s=0.80,
        reference_t_cycle_s=1.60,
        reference_racket_site_speed_mps=6.0,
        reaction_margin_s=0.05,
        teacher_rate_min=0.80,
        teacher_rate_max=1.20,
        mobility_mode="no_move",
    )


def _birth_from_request(
    request,
    *,
    sampler_birth_index,
    sampler_draw_start,
):
    yaw = 0.01 * (request.env_id + request.reset_generation)
    spawn = (
        -0.1 + 0.001 * request.env_id,
        0.02 * request.action_slot,
        0.0,
    )
    claim = request.domain_claim
    birth_identity = {
        "schema_version": R.SAMPLER_SCHEMA_VERSION,
        "kind": "base_birth",
        "sampler_contract_sha256": request.pins.sampler_sha256,
        "arm_catalog_sha256": R.ARM_CATALOG_SHA256,
        "action_uid": request.action_uid,
        "domain_epoch": claim.domain_epoch,
        "levels_sha256": claim.levels_sha256,
        "profile_sha256": request.binding.profile_sha256,
        "birth_index": sampler_birth_index,
        "draw_start": sampler_draw_start,
        "draw_end": sampler_draw_start + R.SAMPLER_BIRTH_DRAW_COUNT,
        "mobility_mode": request.mobility_mode,
        "base_yaw_rad": yaw,
        "base_start_w_m": spawn,
    }
    return R.ActionBirthReceipt(
        registry_sha256=request.registry_sha256,
        env_id=request.env_id,
        reset_generation=request.reset_generation,
        action_uid=request.action_uid,
        action_slot=request.action_slot,
        domain_epoch=claim.domain_epoch,
        domain_claim_sha256=claim.canonical_sha256,
        domain_authority_sha256=claim.authority_contract_sha256,
        domain_levels=claim.domain_levels,
        arm_catalog_sha256=claim.arm_catalog_sha256,
        levels_sha256=claim.levels_sha256,
        sampler_birth_sha256=R._sha256_json(birth_identity),
        sampler_birth_index=sampler_birth_index,
        sampler_draw_start=sampler_draw_start,
        sampler_draw_end=(
            sampler_draw_start + R.SAMPLER_BIRTH_DRAW_COUNT
        ),
        mobility_mode=request.mobility_mode,
        base_yaw_rad=yaw,
        base_quat_wxyz=_yaw_quat(yaw),
        base_spawn_w_m=spawn,
        manifest_sha256=request.pins.manifest_sha256,
        sampler_sha256=request.pins.sampler_sha256,
        profile_sha256=request.binding.profile_sha256,
        motion_sha256=request.binding.motion_sha256,
        physics_sha256=request.pins.physics_sha256,
        solver_sha256=request.pins.solver_sha256,
    )


class BirthProvider:
    def __init__(self):
        self.sampler_contract_sha256 = _digest("sampler")
        self.state_owner_sha256 = _digest("provider-state-owner")
        self.requests = []
        self.birth_index_by_uid = {}
        self.draw_count_by_uid = {}

    def state_dict(self):
        return {
            "birth_indices": [
                [uid, value]
                for uid, value in sorted(
                    self.birth_index_by_uid.items()
                )
            ],
            "draw_counts": [
                [uid, value]
                for uid, value in sorted(self.draw_count_by_uid.items())
            ],
        }

    def load_state_dict(self, state):
        if (
            not isinstance(state, dict)
            or set(state) != {"birth_indices", "draw_counts"}
        ):
            raise ValueError("invalid birth provider state")
        birth_indices = dict(state["birth_indices"])
        draw_counts = dict(state["draw_counts"])
        if set(birth_indices) != set(draw_counts):
            raise ValueError("provider counters disagree")
        self.birth_index_by_uid = birth_indices
        self.draw_count_by_uid = draw_counts

    def assert_issued_birth(self, receipt):
        uid = receipt.action_uid
        if (
            receipt.sampler_birth_index
            >= self.birth_index_by_uid.get(uid, 0)
            or receipt.sampler_draw_end
            > self.draw_count_by_uid.get(uid, 0)
        ):
            raise ValueError("birth is absent from provider state")

    def birth_highwater_for(self, action_uid):
        count = self.birth_index_by_uid.get(action_uid, 0)
        if count == 0:
            return (-1, 0)
        return (
            count - 1,
            self.draw_count_by_uid[action_uid],
        )

    def __call__(self, request):
        self.requests.append(request)
        uid = request.action_uid
        birth_index = self.birth_index_by_uid.get(uid, 0)
        draw_start = self.draw_count_by_uid.get(uid, 0)
        receipt = _birth_from_request(
            request,
            sampler_birth_index=birth_index,
            sampler_draw_start=draw_start,
        )
        self.birth_index_by_uid[uid] = birth_index + 1
        self.draw_count_by_uid[
            uid
        ] = draw_start + R.SAMPLER_BIRTH_DRAW_COUNT
        return receipt


class DomainAuthority:
    def __init__(self, bindings, mode):
        self.domain_authority_contract_sha256 = _digest(
            "domain-authority"
        )
        self.state_owner_sha256 = _digest("domain-state-owner")
        self.bindings = {
            binding.action_uid: binding for binding in bindings
        }
        self.mode = mode
        self.cursors = {}

    def state_dict(self):
        return {
            "cursors": [
                [uid, value]
                for uid, value in sorted(self.cursors.items())
            ]
        }

    def load_state_dict(self, state):
        if not isinstance(state, dict) or set(state) != {"cursors"}:
            raise ValueError("invalid domain authority state")
        self.cursors = dict(state["cursors"])

    def claim_for_action(self, action_uid):
        binding = self.bindings[action_uid]
        epoch = self.cursors.get(action_uid, 0)
        levels = _levels()
        claim = R.ActionDomainClaim(
            authority_contract_sha256=(
                self.domain_authority_contract_sha256
            ),
            arm_catalog_sha256=R.ARM_CATALOG_SHA256,
            action_uid=action_uid,
            domain_epoch=epoch,
            domain_levels=levels,
            levels_sha256=levels.canonical_sha256,
            profile_sha256=binding.profile_sha256,
            mobility_mode=self.mode,
        )
        self.cursors[action_uid] = epoch + 1
        return claim

    def domain_cursor_for(self, action_uid):
        return self.cursors.get(action_uid, 0)


class BatchedBirthProvider(BirthProvider):
    def __init__(self):
        super().__init__()
        self.scalar_calls = 0
        self.batch_calls = 0

    def __call__(self, request):
        self.scalar_calls += 1
        return super().__call__(request)

    def provide_many(self, requests):
        self.batch_calls += 1
        return tuple(
            BirthProvider.__call__(self, request)
            for request in requests
        )


class BatchedDomainAuthority(DomainAuthority):
    def __init__(self, bindings, mode):
        super().__init__(bindings, mode)
        self.scalar_calls = 0
        self.batch_calls = 0

    def claim_for_action(self, action_uid):
        self.scalar_calls += 1
        return super().claim_for_action(action_uid)

    def claim_many_for_actions(self, action_uids):
        self.batch_calls += 1
        return tuple(
            DomainAuthority.claim_for_action(self, action_uid)
            for action_uid in action_uids
        )


class CrossActionBirthProvider(BirthProvider):
    def __init__(self, untouched_uid):
        super().__init__()
        self.untouched_uid = untouched_uid

    def __call__(self, request):
        receipt = super().__call__(request)
        hidden_index = self.birth_index_by_uid.get(
            self.untouched_uid, 0
        )
        hidden_draw = self.draw_count_by_uid.get(
            self.untouched_uid, 0
        )
        self.birth_index_by_uid[self.untouched_uid] = hidden_index + 1
        self.draw_count_by_uid[self.untouched_uid] = (
            hidden_draw + R.SAMPLER_BIRTH_DRAW_COUNT
        )
        return receipt


class CrossActionDomainAuthority(DomainAuthority):
    def __init__(self, bindings, mode, untouched_uid):
        super().__init__(bindings, mode)
        self.untouched_uid = untouched_uid

    def claim_for_action(self, action_uid):
        claim = super().claim_for_action(action_uid)
        self.cursors[self.untouched_uid] = (
            self.cursors.get(self.untouched_uid, 0) + 1
        )
        return claim


class MutatingBirthHighwaterProvider(BirthProvider):
    def __init__(self, *, raise_after_mutation):
        super().__init__()
        self.raise_after_mutation = raise_after_mutation

    def birth_highwater_for(self, action_uid):
        prior = super().birth_highwater_for(action_uid)
        self.birth_index_by_uid[action_uid] = 1
        self.draw_count_by_uid[action_uid] = R.SAMPLER_BIRTH_DRAW_COUNT
        if self.raise_after_mutation:
            raise RuntimeError("mutating birth high-water hook failed")
        return prior


class MutatingDomainCursorAuthority(DomainAuthority):
    def __init__(self, bindings, mode, *, raise_after_mutation):
        super().__init__(bindings, mode)
        self.raise_after_mutation = raise_after_mutation

    def domain_cursor_for(self, action_uid):
        prior = super().domain_cursor_for(action_uid)
        self.cursors[action_uid] = prior + 1
        if self.raise_after_mutation:
            raise RuntimeError("mutating domain cursor hook failed")
        return prior


def _task(
    birth,
    sequence,
    *,
    swing_generation=None,
    base_goal_w_m=None,
    counter_rally_task=None,
    diagnostic_prevalidated=False,
    task_kwargs_transform=None,
):
    def rotate_inverse(value, yaw):
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        return (
            cosine * value[0] + sine * value[1],
            -sine * value[0] + cosine * value[1],
            value[2],
        )

    goal = (
        birth.base_spawn_w_m
        if base_goal_w_m is None
        else tuple(base_goal_w_m)
    )
    contact = (0.55, 0.01 * sequence, 0.9)
    contact_offset = rotate_inverse(
        tuple(contact[index] - goal[index] for index in range(3)),
        birth.base_yaw_rad,
    )
    travel = rotate_inverse(
        tuple(
            goal[index] - birth.base_spawn_w_m[index]
            for index in range(3)
        ),
        birth.base_yaw_rad,
    )
    incoming = (-4.0, 0.1, -0.2)
    speed = math.sqrt(sum(component * component for component in incoming))
    incoming_direction = rotate_inverse(
        tuple(component / speed for component in incoming),
        birth.base_yaw_rad,
    )
    spin = (0.0, 12.0, 1.0)
    spin_magnitude = math.sqrt(
        sum(component * component for component in spin)
    )
    spin_direction = rotate_inverse(
        tuple(component / spin_magnitude for component in spin),
        birth.base_yaw_rad,
    )
    time_to_contact_s = 1.2
    racket_velocity = (3.0, 0.2, 0.4)
    geometry = R._contact_geometry.solve_exact_face_contact(
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
    timing = R.derive_action_teacher_site_timing(
        racket_site_velocity_w_mps=(
            geometry.racket_site_velocity_w_mps
        ),
        time_to_contact_s=time_to_contact_s,
        reference_t_hit_s=0.42,
        reference_t_cycle_s=1.2,
        reference_racket_site_speed_mps=3.0,
        reaction_margin_s=0.05,
        teacher_rate_min=0.8,
        teacher_rate_max=1.2,
    )
    sample_index = sequence
    draw_start = 1_000 + sample_index * R.SAMPLER_SAMPLE_DRAW_COUNT
    sample_identity = {
        "schema_version": R.SAMPLER_SCHEMA_VERSION,
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
        "draw_end": draw_start + R.SAMPLER_SAMPLE_DRAW_COUNT,
        "mobility_mode": birth.mobility_mode,
        "base_yaw_rad": birth.base_yaw_rad,
        "base_start_w_m": birth.base_spawn_w_m,
        "base_spawn_latent_w_m": birth.base_spawn_w_m,
        "base_travel_latent_b_yaw_m": travel,
        "base_goal_w_m": goal,
        "contact_offset_from_base_goal_b_yaw_m": contact_offset,
        "contact_w_m": contact,
        "time_to_contact_s": time_to_contact_s,
        "incoming_speed_mps": speed,
        "incoming_direction_b_yaw": incoming_direction,
        "incoming_direction_w": R._rotate_yaw(
            incoming_direction, birth.base_yaw_rad
        ),
        "incoming_velocity_w_mps": incoming,
        "spin_magnitude_radps": spin_magnitude,
        "spin_direction_b_yaw": spin_direction,
        "spin_direction_w": R._rotate_yaw(
            spin_direction, birth.base_yaw_rad
        ),
        "spin_w_radps": spin,
        "landing_aim_w_xy_m": (2.5, -0.1),
    }
    task_kwargs = dict(
        sample_sha256=R._sha256_json(sample_identity),
        sample_index=sample_index,
        sample_draw_start=draw_start,
        sample_draw_end=draw_start + R.SAMPLER_SAMPLE_DRAW_COUNT,
        swing_generation=(
            sequence if swing_generation is None else swing_generation
        ),
        base_goal_w_m=base_goal_w_m,
        base_spawn_latent_w_m=birth.base_spawn_w_m,
        base_travel_latent_b_yaw_m=travel,
        contact_offset_from_base_goal_b_yaw_m=contact_offset,
        ball_contact_w_m=contact,
        time_to_contact_s=time_to_contact_s,
        incoming_speed_mps=speed,
        incoming_direction_b_yaw=incoming_direction,
        incoming_velocity_w_mps=incoming,
        spin_magnitude_radps=spin_magnitude,
        spin_direction_b_yaw=spin_direction,
        incoming_spin_w_radps=spin,
        landing_aim_w_xy_m=(2.5, -0.1),
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
        reference_t_hit_s=0.42,
        reference_t_cycle_s=1.2,
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
        contact_time_step_s=None,
        time_to_contact_tick=None,
        birth_index=-1,
        birth_sampling_stratum="domain",
        birth_sampling_levels=None,
        birth_frontier_arm=None,
        sampling_mixture=None,
        sampling_stratum="domain",
        sampling_levels=None,
        frontier_arm=None,
        counter_rally_task=counter_rally_task,
    )
    if task_kwargs_transform is not None:
        task_kwargs = task_kwargs_transform(task_kwargs)
    if diagnostic_prevalidated:
        return R._diagnostic_prevalidated_task_receipt_from_birth(
            birth,
            **task_kwargs,
        )
    return R.ActionBallTaskReceipt.from_birth(birth, **task_kwargs)


class Solver:
    def __init__(self):
        self.solver_contract_sha256 = _digest("solver")
        self.state_owner_sha256 = _digest("solver-state-owner")
        self.requests = []
        self.sequence = 0
        self.batch_calls = 0
        self.emitted_samples = set()
        self.emitted_tasks = []
        self.highwater_by_uid = {}
        self.proposal_assignments = {}

    def state_dict(self):
        return {
            "sequence": self.sequence,
            "emitted_samples": sorted(self.emitted_samples),
            "emitted_tasks": [
                receipt.to_dict() for receipt in self.emitted_tasks
            ],
            "proposal_assignments": [
                [uid, sample_index, birth_sha256, refill_index]
                for (
                    uid,
                    sample_index,
                ), (
                    birth_sha256,
                    refill_index,
                ) in sorted(self.proposal_assignments.items())
            ],
            "highwaters": [
                [uid, index, draw_end]
                for uid, (index, draw_end) in sorted(
                    self.highwater_by_uid.items()
                )
            ],
        }

    def load_state_dict(self, state):
        if (
            not isinstance(state, dict)
            or set(state)
            != {
                "sequence",
                "emitted_samples",
                "emitted_tasks",
                "proposal_assignments",
                "highwaters",
            }
            or type(state["sequence"]) is not int
            or state["sequence"] < 0
            or not isinstance(state["emitted_samples"], list)
            or state["emitted_samples"]
            != sorted(set(state["emitted_samples"]))
            or not isinstance(state["emitted_tasks"], list)
            or not isinstance(state["proposal_assignments"], list)
            or not isinstance(state["highwaters"], list)
        ):
            raise ValueError("invalid solver state")
        highwaters = {}
        for row in state["highwaters"]:
            if (
                not isinstance(row, list)
                or len(row) != 3
                or any(type(value) is not int for value in row)
                or row[0] in highwaters
                or row[1] < 0
                or row[2] < 1
            ):
                raise ValueError("invalid solver high-water state")
            highwaters[row[0]] = (row[1], row[2])
        if list(highwaters) != sorted(highwaters):
            raise ValueError("solver high-water state must be sorted")
        self.sequence = state["sequence"]
        self.emitted_samples = set(state["emitted_samples"])
        self.emitted_tasks = [
            R.ActionBallTaskReceipt.from_dict(receipt)
            for receipt in state["emitted_tasks"]
        ]
        task_digests = [
            receipt.canonical_sha256 for receipt in self.emitted_tasks
        ]
        if len(task_digests) != len(set(task_digests)):
            raise ValueError("solver emitted task transcript repeats")
        proposal_assignments = {}
        for row in state["proposal_assignments"]:
            if (
                not isinstance(row, list)
                or len(row) != 4
                or type(row[0]) is not int
                or type(row[1]) is not int
                or not isinstance(row[2], str)
                or type(row[3]) is not int
                or row[0] < 1
                or row[1] < 0
                or row[3] < 1
                or (row[0], row[1]) in proposal_assignments
            ):
                raise ValueError("invalid solver proposal assignment")
            proposal_assignments[(row[0], row[1])] = (
                row[2],
                row[3],
            )
        if list(proposal_assignments) != sorted(proposal_assignments):
            raise ValueError(
                "solver proposal assignments must be sorted"
            )
        self.proposal_assignments = proposal_assignments
        self.highwater_by_uid = highwaters

    def assert_emitted_sample(self, receipt):
        if receipt.sample_sha256 not in self.emitted_samples:
            raise ValueError("sample was not emitted by solver sampler")

    def sample_highwater_for(self, action_uid):
        return self.highwater_by_uid.get(action_uid, (-1, 0))

    def assert_emitted_tasks(self, receipts):
        authority = {
            receipt.canonical_sha256: receipt
            for receipt in self.emitted_tasks
        }
        for receipt in receipts:
            if authority.get(receipt.canonical_sha256) != receipt:
                raise ValueError("task was not emitted by exact solver")

    def emitted_task_count_for(self, action_uid):
        return sum(
            receipt.action_uid == action_uid
            for receipt in self.emitted_tasks
        )

    def task_transcript_for_birth(self, birth_sha256):
        digests = [
            receipt.canonical_sha256
            for receipt in self.emitted_tasks
            if receipt.birth_sha256 == birth_sha256
        ]
        return (
            len(digests),
            R.task_transcript_sha256(birth_sha256, digests),
        )

    def assert_proposal_assignments(self, assignments):
        for assignment in assignments:
            expected = (
                assignment.birth.canonical_sha256,
                assignment.refill_index,
            )
            for sample_index in assignment.proposal_sample_indices:
                if self.proposal_assignments.get(
                    (assignment.birth.action_uid, sample_index)
                ) != expected:
                    raise ValueError(
                        "proposal sample was not assigned to exact "
                        "birth/refill"
                    )

    def _record_assignment(self, request, sample_indices):
        value = (
            request.birth.canonical_sha256,
            request.refill_index,
        )
        for sample_index in sample_indices:
            key = (request.action_uid, sample_index)
            if key in self.proposal_assignments:
                raise ValueError("proposal sample assignment replayed")
            self.proposal_assignments[key] = value

    def _record_task(self, receipt):
        self.emitted_samples.add(receipt.sample_sha256)
        self.emitted_tasks.append(receipt)
        self.highwater_by_uid[receipt.action_uid] = (
            receipt.sample_index,
            receipt.sample_draw_end,
        )

    def _make_task(
        self,
        request,
        sample_index,
        swing_generation,
    ):
        return _task(
            request.birth,
            sample_index,
            swing_generation=swing_generation,
        )

    def __call__(self, request):
        self.requests.append(request)
        receipts = []
        for offset in range(request.minimum_receipts):
            sample_index = (
                self.highwater_by_uid.get(
                    request.action_uid, (-1, 0)
                )[0]
                + 1
            )
            receipts.append(
                self._make_task(
                    request,
                    sample_index,
                    request.swing_generation_start + offset,
                )
            )
            self._record_task(receipts[-1])
            self.sequence += 1
        proposal_indices = tuple(
            receipt.sample_index for receipt in receipts
        )
        self._record_assignment(request, proposal_indices)
        return R.ActionPoolRefillBatch(
            action_uid=request.action_uid,
            proposed_count=len(receipts),
            proposal_sample_indices=proposal_indices,
            receipts=tuple(receipts),
        )

    def solve_many(self, requests):
        self.batch_calls += 1
        return tuple(self(request) for request in requests)


class CounterRallySolver(Solver):
    def __init__(self, objective_profile_sha256):
        super().__init__()
        self.objective_profile_sha256 = objective_profile_sha256

    def _make_task(
        self,
        request,
        sample_index,
        swing_generation,
    ):
        return _task(
            request.birth,
            sample_index,
            swing_generation=swing_generation,
            counter_rally_task=_counter_rally_task_identity(
                self.objective_profile_sha256
            ),
        )


class RoundInterleavedSolver(Solver):
    """Return per-birth batches after sampling in cross-birth rounds."""

    def solve_many(self, requests):
        self.batch_calls += 1
        per_request = [[] for _ in requests]
        for offset in range(requests[0].minimum_receipts):
            for request_index, request in enumerate(requests):
                sample_index = (
                    self.highwater_by_uid.get(
                        request.action_uid, (-1, 0)
                    )[0]
                    + 1
                )
                receipt = _task(
                    request.birth,
                    sample_index,
                    swing_generation=(
                        request.swing_generation_start + offset
                    ),
                )
                per_request[request_index].append(receipt)
                self._record_task(receipt)
                self.sequence += 1
        batches = []
        for request, receipts in zip(requests, per_request):
            proposal_indices = tuple(
                receipt.sample_index for receipt in receipts
            )
            self._record_assignment(request, proposal_indices)
            batches.append(
                R.ActionPoolRefillBatch(
                    action_uid=request.action_uid,
                    proposed_count=len(receipts),
                    proposal_sample_indices=proposal_indices,
                    receipts=tuple(receipts),
                )
            )
        return tuple(batches)


class MutatingAssertionSolver(Solver):
    def assert_emitted_sample(self, receipt):
        super().assert_emitted_sample(receipt)
        self.sequence += 1


class MutatingHighwaterSolver(Solver):
    def __init__(self, *, raise_after_mutation):
        super().__init__()
        self.raise_after_mutation = raise_after_mutation

    def sample_highwater_for(self, action_uid):
        self.sequence += 1
        if self.raise_after_mutation:
            raise RuntimeError("mutating high-water hook failed")
        return self.highwater_by_uid.get(action_uid, (-1, 0))


class TranscriptProbeSolver(Solver):
    """Count snapshots and optionally corrupt a mid-batch transcript read."""

    def __init__(self):
        super().__init__()
        self.state_calls = 0
        self.transcript_calls = 0
        self.transcript_fault_at = None
        self.raise_after_transcript_mutation = False

    def state_dict(self):
        self.state_calls += 1
        return super().state_dict()

    def reset_state_calls(self):
        self.state_calls = 0

    def task_transcript_for_birth(self, birth_sha256):
        result = super().task_transcript_for_birth(birth_sha256)
        self.transcript_calls += 1
        if self.transcript_calls == self.transcript_fault_at:
            self.sequence += 1
            if self.raise_after_transcript_mutation:
                raise RuntimeError("mid-batch transcript read failed")
        return result


class CrossActionMutationSolver(Solver):
    def __init__(self, untouched_uid):
        super().__init__()
        self.untouched_uid = untouched_uid

    def __call__(self, request):
        batch = super().__call__(request)
        self.highwater_by_uid[self.untouched_uid] = (
            0,
            1_000 + R.SAMPLER_SAMPLE_DRAW_COUNT,
        )
        return batch


class RejectedTailReplaySolver(Solver):
    """Expose an issued-but-rejected tail, then try to admit it later."""

    def __call__(self, request):
        if self.sequence == 0:
            admitted = _task(
                request.birth,
                0,
                swing_generation=request.swing_generation_start,
            )
            rejected_tail = _task(
                request.birth,
                1,
                swing_generation=request.swing_generation_start + 1,
            )
            self._record_task(admitted)
            self.emitted_samples.add(rejected_tail.sample_sha256)
            self.highwater_by_uid[request.action_uid] = (
                rejected_tail.sample_index,
                rejected_tail.sample_draw_end,
            )
            self._record_assignment(request, (0, 1))
            self.sequence = 2
            return R.ActionPoolRefillBatch(
                action_uid=request.action_uid,
                proposed_count=2,
                proposal_sample_indices=(0, 1),
                receipts=(admitted,),
            )
        replay = _task(
            request.birth,
            1,
            swing_generation=request.swing_generation_start,
        )
        return R.ActionPoolRefillBatch(
            action_uid=request.action_uid,
            proposed_count=1,
            proposal_sample_indices=(1,),
            receipts=(replay,),
        )


class ReplayedSampleSolver(Solver):
    def __call__(self, request):
        first = _task(
            request.birth,
            self.sequence,
            swing_generation=request.swing_generation_start,
        )
        second = replace(
            _task(
                request.birth,
                self.sequence,
                swing_generation=request.swing_generation_start + 1,
            ),
        )
        self.sequence += 2
        self._record_task(first)
        self._record_task(second)
        return R.ActionPoolRefillBatch(
            action_uid=request.action_uid,
            proposed_count=2,
            proposal_sample_indices=(
                first.sample_index,
                second.sample_index,
            ),
            receipts=(first, second),
        )


class CrossBirthReplayedSampleSolver(Solver):
    def solve_many(self, requests):
        self.batch_calls += 1
        batches = list(super().solve_many(requests))
        repeated_sample = batches[0].receipts[0].sample_sha256
        second = batches[1]
        batches[1] = R.ActionPoolRefillBatch(
            action_uid=second.action_uid,
            proposed_count=second.proposed_count,
            proposal_sample_indices=second.proposal_sample_indices,
            receipts=(
                replace(
                    second.receipts[0],
                    sample_sha256=repeated_sample,
                ),
            ),
        )
        return tuple(batches)


def _broker(
    count=5,
    mode="no_move",
    pins=None,
    *,
    diagnostic_unauthorized=False,
):
    bindings = _bindings(count)
    broker = R.ActionBirthBroker(
        bindings,
        _pins() if pins is None else pins,
        mode,
        diagnostic_unauthorized=diagnostic_unauthorized,
    )
    authority = DomainAuthority(bindings, mode)
    provider = BirthProvider()
    broker.bind_domain_claim_authority(authority)
    broker.bind_provider(provider)
    return broker, provider


def _diagnostic_broker_with_counted_producers():
    bindings = _bindings(1)
    broker = R.ActionBirthBroker(
        bindings,
        _pins(),
        "no_move",
        diagnostic_unauthorized=True,
    )
    authority = BatchedDomainAuthority(bindings, "no_move")
    provider = BatchedBirthProvider()
    broker.bind_domain_claim_authority(authority)
    broker.bind_provider(provider)
    return broker, provider, authority


def _reserve(
    broker,
    *,
    env_id=0,
    generation=1,
    slot=0,
):
    uid = broker.ordered_action_uids[slot]
    return broker.reserve_true_reset(
        env_id=env_id,
        reset_generation=generation,
        action_uid=uid,
        action_slot=slot,
    )


def _commit(broker, birth):
    broker.commit_true_reset(
        env_id=birth.env_id,
        reset_generation=birth.reset_generation,
        receipt_sha256=birth.canonical_sha256,
    )


def _consume(broker, birth):
    _commit(broker, birth)
    return broker.consume_many_true_reset((_claim(birth),))[0]


def _claim(birth, **overrides):
    values = dict(
        env_id=birth.env_id,
        reset_generation=birth.reset_generation,
        action_uid=birth.action_uid,
        action_slot=birth.action_slot,
        receipt_sha256=birth.canonical_sha256,
    )
    values.update(overrides)
    return R.BirthConsumeRequest(**values)


def _reserve_claim(broker, *, env_id, generation=1, slot=0):
    return R.BirthReserveRequest(
        env_id=env_id,
        reset_generation=generation,
        action_uid=broker.ordered_action_uids[slot],
        action_slot=slot,
    )


def _formal_pool_batch(count, solver=None):
    broker, _provider = _broker(1)
    births = tuple(
        _reserve(broker, env_id=env_id) for env_id in range(count)
    )
    for birth in births:
        _commit(broker, birth)
    broker.consume_many_true_reset(
        tuple(_claim(birth) for birth in births)
    )
    pool = R.LazyActionTaskPool(
        _bindings(1), _pins(), "no_move", refill_size=1
    )
    bound_solver = TranscriptProbeSolver() if solver is None else solver
    pool.bind_solver(bound_solver)
    pool.bind_birth_authority(broker)
    return pool, bound_solver, births


def _integrity(payload):
    body = {
        key: value
        for key, value in payload.items()
        if key != "integrity_sha256"
    }
    return hashlib.sha256(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _canonical_json_bytes(payload):
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _consume_same_env_birth_history(broker, *, generations=3):
    history = []
    for generation in range(1, generations + 1):
        birth = _reserve(broker, env_id=0, generation=generation)
        history.append(_consume(broker, birth))
    return tuple(history)


def _mutate_consumed_birth_history(history, case):
    if case == "missing":
        return (history[0], *history[2:])
    if case == "extra":
        return (
            *history,
            replace(history[-1], reset_generation=len(history) + 1),
        )
    if case == "out_of_order":
        return (history[1], history[0], *history[2:])
    if case == "identity_drift":
        return (
            replace(history[0], motion_sha256=_digest("history-drift")),
            *history[1:],
        )
    raise AssertionError(f"unknown history mutation case {case!r}")


def test_receipts_are_immutable_canonical_strict_and_no_move_is_physical():
    broker, _ = _broker(1)
    birth = _reserve(broker)
    assert len(birth.base_quat_wxyz) == 4
    assert R.ActionBirthReceipt.from_dict(birth.to_dict()) == birth
    assert len(birth.canonical_sha256) == 64
    with pytest.raises(FrozenInstanceError):
        birth.action_uid = 4

    task = _task(birth, 0)
    assert task.base_goal_w_m == birth.base_spawn_w_m
    task.assert_birth(birth)
    assert R.ActionBallTaskReceipt.from_dict(task.to_dict()) == task
    with pytest.raises(FrozenInstanceError):
        task.scaled_t_hit_s = 2.0

    with pytest.raises(R.ActionBallContractError, match="no_move"):
        replace(task, base_goal_w_m=(0.0, 0.0, 0.0))
    with pytest.raises(R.ActionBallContractError, match="yaw-only"):
        replace(birth, base_quat_wxyz=(1.0, 0.0, 0.0, 0.0))
    with pytest.raises(R.ActionBallContractError, match="length-3"):
        replace(task, ball_contact_w_m=(0.0, 1.0))
    with pytest.raises(R.ActionBallContractError, match="finite"):
        replace(task, solver_residual_m=float("nan"))
    with pytest.raises(R.ActionBallContractError, match="unit length"):
        replace(task, racket_normal_w=(2.0, 0.0, 0.0))

    tampered = deepcopy(task.to_dict())
    tampered["ball_contact_w_m"][0] += 0.01
    with pytest.raises(
        R.ActionBallContractError,
        match="contact|exact-face geometry|canonical SHA",
    ):
        R.ActionBallTaskReceipt.from_dict(tampered)


def test_diagnostic_prevalidated_receipt_is_exact_base_wire_parity():
    broker, _ = _broker(1, diagnostic_unauthorized=True)
    birth = _reserve(broker)
    formal = _task(birth, 0)
    diagnostic = _task(
        birth,
        0,
        diagnostic_prevalidated=True,
    )

    assert type(diagnostic) is R.ActionBallTaskReceipt
    assert diagnostic == formal
    assert hash(diagnostic) == hash(formal)
    assert diagnostic.payload_dict() == formal.payload_dict()
    assert diagnostic.to_dict() == formal.to_dict()
    assert diagnostic.canonical_sha256 == formal.canonical_sha256
    assert diagnostic.sampler_identity_receipt() == (
        formal.sampler_identity_receipt()
    )
    assert diagnostic.task_ref() == formal.task_ref()
    assert "_validation_mode" not in {
        field.name for field in dataclass_fields(diagnostic)
    }
    assert "_validation_mode" not in vars(diagnostic)
    diagnostic.assert_birth(birth)
    diagnostic.assert_contract(
        binding=_bindings(1)[0],
        pins=_pins(),
        mobility_mode="no_move",
        registry_sha256=broker.registry_sha256,
    )
    assert (
        R.ActionBallTaskReceipt.from_dict(diagnostic.to_dict())
        == formal
    )


def test_diagnostic_prevalidated_move_and_counter_rally_wire_parity():
    broker, _ = _broker(
        1,
        mode="move",
        diagnostic_unauthorized=True,
    )
    birth = _reserve(broker)
    goal = (
        birth.base_spawn_w_m[0] + 0.03,
        birth.base_spawn_w_m[1] - 0.02,
        birth.base_spawn_w_m[2],
    )
    counter_rally = _counter_rally_task_identity(
        _digest("diagnostic-counter-rally")
    )
    formal = _task(
        birth,
        0,
        base_goal_w_m=goal,
        counter_rally_task=counter_rally,
    )
    diagnostic = _task(
        birth,
        0,
        base_goal_w_m=goal,
        counter_rally_task=counter_rally,
        diagnostic_prevalidated=True,
    )

    assert type(diagnostic) is R.ActionBallTaskReceipt
    assert diagnostic == formal
    assert diagnostic.to_dict() == formal.to_dict()
    assert diagnostic.canonical_sha256 == formal.canonical_sha256
    diagnostic.assert_birth(birth)
    diagnostic.assert_contract(
        binding=_bindings(1)[0],
        pins=_pins(
            counter_rally_objective_profile_sha256=(
                counter_rally.objective_profile_sha256
            )
        ),
        mobility_mode="move",
        registry_sha256=broker.registry_sha256,
    )


def test_diagnostic_prevalidated_receipt_bypasses_post_init(monkeypatch):
    broker, _ = _broker(1, diagnostic_unauthorized=True)
    birth = _reserve(broker)
    formal = _task(birth, 0)
    mutable_contact = []

    def expose_mutable_contact(kwargs):
        contact = list(kwargs["ball_contact_w_m"])
        mutable_contact.append(contact)
        kwargs["ball_contact_w_m"] = contact
        return kwargs

    def reject_post_init(self, _validation_mode=None):
        raise AssertionError("diagnostic constructor re-entered post-init")

    monkeypatch.setattr(
        R.ActionBallTaskReceipt,
        "__post_init__",
        reject_post_init,
    )
    diagnostic = _task(
        birth,
        0,
        diagnostic_prevalidated=True,
        task_kwargs_transform=expose_mutable_contact,
    )
    mutable_contact[0][0] += 1.0
    assert diagnostic == formal
    assert hash(diagnostic) == hash(formal)
    assert diagnostic.to_dict() == formal.to_dict()


def test_diagnostic_prevalidated_receipt_rejects_keyword_drift():
    broker, _ = _broker(1, diagnostic_unauthorized=True)
    birth = _reserve(broker)

    def without_sample_sha256(kwargs):
        del kwargs["sample_sha256"]
        return kwargs

    with pytest.raises(
        R.ActionBallContractError,
        match=r"exact producer keyword set.*sample_sha256",
    ):
        _task(
            birth,
            0,
            diagnostic_prevalidated=True,
            task_kwargs_transform=without_sample_sha256,
        )

    def with_unknown_keyword(kwargs):
        kwargs["unknown_producer_field"] = "forbidden"
        return kwargs

    with pytest.raises(
        R.ActionBallContractError,
        match=r"exact producer keyword set.*unknown_producer_field",
    ):
        _task(
            birth,
            0,
            diagnostic_prevalidated=True,
            task_kwargs_transform=with_unknown_keyword,
        )


def test_task_receipt_rejects_forged_validation_authority():
    class EqualToEverything:
        def __eq__(self, other):
            return True

    broker, _ = _broker(1, diagnostic_unauthorized=True)
    birth = _reserve(broker)
    task = _task(birth, 0)
    for value in (
        False,
        True,
        1,
        None,
        object(),
        EqualToEverything(),
    ):
        with pytest.raises(
            R.ActionBallContractError,
            match="validation mode is not an internal authority",
        ):
            replace(task, _validation_mode=value)


def test_formal_task_constructor_still_replays_geometry_proof():
    broker, _ = _broker(1)
    birth = _reserve(broker)
    task = _task(birth, 0)
    with pytest.raises(
        R.ActionBallContractError,
        match="exact-face geometry",
    ):
        replace(
            task,
            racket_site_target_w_m=(
                task.racket_site_target_w_m[0] + 0.01,
                task.racket_site_target_w_m[1],
                task.racket_site_target_w_m[2],
            ),
        )


def test_frozen_canonical_sha_is_cached_without_changing_contract(
    monkeypatch,
):
    broker, provider = _broker(1)
    birth = _reserve(broker)
    claim = provider.requests[-1].domain_claim
    contract = R.BasePreparationContract(
        max_planar_speed_mps=1.0,
        max_planar_acceleration_mps2=4.0,
        settle_margin_s=0.05,
    )
    preparation = R.BasePreparationReceipt.evaluate(
        proposal_sample_sha256=_digest("cached-preparation"),
        proposal_sample_index=0,
        mobility_mode="move",
        base_travel_b_yaw_m=(0.08, 0.0, 0.0),
        reaction_margin_s=0.05,
        available_preparation_s=0.70,
        contract=contract,
    )
    templates = (
        _counter_rally_task_identity(_digest("cached-counter-rally")),
        _levels(),
        claim,
        _pins(),
        birth,
        contract,
        preparation,
        _task(birth, 0),
    )

    def assert_deeply_immutable(value):
        if value is None or isinstance(
            value, (bool, int, float, str, bytes)
        ):
            return
        if isinstance(value, tuple):
            for item in value:
                assert_deeply_immutable(item)
            return
        assert is_dataclass(value)
        assert value.__dataclass_params__.frozen
        for dataclass_field in dataclass_fields(value):
            assert_deeply_immutable(
                getattr(value, dataclass_field.name)
            )

    original_sha256_json = R._sha256_json
    hashed_payloads = []

    def counted_sha256_json(value):
        hashed_payloads.append(value)
        return original_sha256_json(value)

    monkeypatch.setattr(R, "_sha256_json", counted_sha256_json)
    for template in templates:
        # replace() copies only declared dataclass fields, never an entry
        # from the weak external cache.
        instance = replace(template)
        hashed_payloads.clear()
        payload_method = getattr(instance, "payload_dict", None)
        payload_before = (
            payload_method()
            if payload_method is not None
            else instance.to_dict()
        )
        expected_sha256 = original_sha256_json(payload_before)
        twin = replace(instance)
        repr_before = repr(instance)
        vars_before = dict(vars(instance))
        pickle_before = pickle.dumps(
            instance, protocol=pickle.HIGHEST_PROTOCOL
        )
        deepcopy_before = deepcopy(instance)
        field_names = tuple(
            dataclass_field.name
            for dataclass_field in dataclass_fields(instance)
        )
        for dataclass_field in dataclass_fields(instance):
            assert_deeply_immutable(
                getattr(instance, dataclass_field.name)
            )

        # Construction of the equality twin may validate nested canonical
        # identities; only accesses on this instance belong to the cache
        # assertion below.
        hashed_payloads.clear()
        assert instance.canonical_sha256 == expected_sha256
        assert instance.canonical_sha256 == expected_sha256
        assert len(hashed_payloads) == 1
        assert "canonical_sha256" not in field_names
        with pytest.raises(AttributeError, match="read-only"):
            object.__setattr__(
                instance, "canonical_sha256", _digest("shadow-attempt")
            )
        # A data descriptor also wins over a hostile same-name ``__dict__`` entry, matching the
        # precedence of the former property.  Remove the synthetic entry before wire/pickle checks.
        vars(instance)["canonical_sha256"] = _digest("dict-shadow-attempt")
        assert instance.canonical_sha256 == expected_sha256
        vars(instance).pop("canonical_sha256")
        assert instance == twin
        assert repr(instance) == repr_before
        assert vars(instance) == vars_before
        assert pickle.dumps(
            instance, protocol=pickle.HIGHEST_PROTOCOL
        ) == pickle_before
        restored = pickle.loads(pickle_before)
        assert restored == instance
        assert vars(restored) == vars_before
        assert deepcopy(instance) == deepcopy_before
        assert vars(deepcopy(instance)) == vars_before
        payload_after = (
            payload_method()
            if payload_method is not None
            else instance.to_dict()
        )
        assert payload_after == payload_before

        to_dict = getattr(instance, "to_dict", None)
        if to_dict is None:
            wire_after = payload_after
            expected_wire = payload_before
        else:
            wire_after = to_dict()
            if set(wire_after) == set(payload_before) | {
                "canonical_sha256"
            }:
                expected_wire = {
                    **payload_before,
                    "canonical_sha256": expected_sha256,
                }
            else:
                expected_wire = payload_before
        assert json.dumps(
            wire_after,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ) == json.dumps(
            expected_wire,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )


def test_frozen_canonical_sha_cache_releases_dead_receipts():
    descriptor = vars(R.ActionDomainLevels)["canonical_sha256"]
    receipt = R.ActionDomainLevels()
    receipt_id = id(receipt)
    receipt_ref = weakref.ref(receipt)
    entries_before = len(descriptor._entries)

    assert len(receipt.canonical_sha256) == 64
    assert receipt_id in descriptor._entries
    assert len(descriptor._entries) == entries_before + 1

    del receipt
    gc.collect()
    assert receipt_ref() is None
    assert receipt_id not in descriptor._entries
    assert len(descriptor._entries) == entries_before


def test_counter_rally_task_identity_is_optional_strict_and_sha_bound():
    broker, _ = _broker(1)
    birth = _reserve(broker)
    ordinary = _task(birth, 0)
    ordinary_payload = ordinary.payload_dict()
    assert set(ordinary_payload) == set(R._TASK_PAYLOAD_KEYS)
    assert "counter_rally_task" not in ordinary_payload
    assert R.ActionBallTaskReceipt.from_dict(
        ordinary.to_dict()
    ) == ordinary
    with pytest.raises(
        R.CounterRallyTaskIdentityError,
        match="missing",
    ):
        ordinary.require_counter_rally_task(
            expected_objective_profile_sha256=_digest(
                "counter-rally-objective"
            )
        )

    objective_sha256 = _digest("counter-rally-objective")
    identity = _counter_rally_task_identity(
        objective_sha256
    )
    assert R.CounterRallyTaskIdentity.from_dict(
        identity.to_dict()
    ) == identity
    counter = _task(
        birth,
        0,
        counter_rally_task=identity,
    )
    assert counter.canonical_sha256 != ordinary.canonical_sha256
    assert counter.payload_dict()["counter_rally_task"] == (
        identity.to_dict()
    )
    restored = R.ActionBallTaskReceipt.from_dict(counter.to_dict())
    assert restored == counter
    assert restored.require_counter_rally_task(
        expected_objective_profile_sha256=objective_sha256
    ) == identity

    tampered = deepcopy(counter.to_dict())
    tampered["counter_rally_task"][
        "target_baseline_speed_mps"
    ] += 0.25
    with pytest.raises(
        R.CounterRallyTaskIdentityError,
        match="canonical SHA mismatch",
    ):
        R.ActionBallTaskReceipt.from_dict(tampered)


def test_counter_rally_runtime_pin_is_exact_n1_and_all_or_none():
    objective_sha256 = _digest("counter-rally-objective")
    ordinary = _pins()
    assert "counter_rally_objective_profile_sha256" not in (
        ordinary.to_dict()
    )
    assert R.RuntimePins.from_dict(ordinary.to_dict()) == ordinary

    pinned = _pins(objective_sha256)
    assert R.RuntimePins.from_dict(pinned.to_dict()) == pinned
    assert pinned.counter_rally_objective_profile_sha256 == (
        objective_sha256
    )
    with pytest.raises(
        R.CounterRallyTaskIdentityError,
        match="exact N=1",
    ):
        R.ActionBirthBroker(_bindings(2), pinned, "no_move")
    with pytest.raises(
        R.CounterRallyTaskIdentityError,
        match="exact N=1",
    ):
        R.LazyActionTaskPool(_bindings(2), pinned, "no_move")

    ordinary_broker, _ = _broker(1)
    ordinary_birth = _reserve(ordinary_broker)
    counter_task = _task(
        ordinary_birth,
        0,
        counter_rally_task=_counter_rally_task_identity(
            objective_sha256
        ),
    )
    with pytest.raises(
        R.CounterRallyTaskIdentityError,
        match="ordinary task/run pins",
    ):
        counter_task.assert_contract(
            binding=_bindings(1)[0],
            pins=ordinary,
            mobility_mode="no_move",
            registry_sha256=ordinary_broker.registry_sha256,
        )

    pinned_broker, _ = _broker(1, pins=pinned)
    pinned_birth = _reserve(pinned_broker)
    missing_identity = _task(pinned_birth, 0)
    with pytest.raises(
        R.CounterRallyTaskIdentityError,
        match="missing",
    ):
        missing_identity.assert_contract(
            binding=_bindings(1)[0],
            pins=pinned,
            mobility_mode="no_move",
            registry_sha256=pinned_broker.registry_sha256,
        )


def test_counter_rally_return_direction_must_exactly_reverse_sampled_ball():
    broker, _ = _broker(1)
    birth = _reserve(broker)
    identity = _counter_rally_task_identity(
        _digest("counter-rally-objective")
    )
    wrong = replace(identity, return_direction_env_xy=(1.0, 0.0))
    with pytest.raises(
        R.CounterRallyTaskIdentityError,
        match="exact horizontal reverse",
    ):
        _task(birth, 0, counter_rally_task=wrong)


def test_counter_rally_objective_resign_still_hard_stops_against_launch_identity():
    broker, _ = _broker(1)
    birth = _reserve(broker)
    expected_sha256 = _digest("counter-rally-objective")
    task = _task(
        birth,
        0,
        counter_rally_task=_counter_rally_task_identity(
            expected_sha256
        ),
    )
    drifted = deepcopy(task.to_dict())
    nested = drifted["counter_rally_task"]
    nested["objective_profile_sha256"] = _digest(
        "different-counter-rally-objective"
    )
    nested["canonical_sha256"] = R._sha256_json(
        {
            key: value
            for key, value in nested.items()
            if key != "canonical_sha256"
        }
    )
    drifted["canonical_sha256"] = R._sha256_json(
        {
            key: value
            for key, value in drifted.items()
            if key != "canonical_sha256"
        }
    )
    restored = R.ActionBallTaskReceipt.from_dict(drifted)
    with pytest.raises(
        R.CounterRallyTaskIdentityError,
        match="objective profile SHA mismatch",
    ):
        restored.require_counter_rally_task(
            expected_objective_profile_sha256=expected_sha256
        )


def test_runtime_sampler_catalog_and_identity_schema_are_exactly_pinned():
    assert R.SAMPLER_SCHEMA_VERSION == S.SCHEMA_VERSION == 3
    assert R.SAMPLER_BIRTH_DRAW_COUNT == S.DRAWS_PER_BIRTH == 3
    assert R.SAMPLER_SAMPLE_DRAW_COUNT == S.DRAWS_PER_SAMPLE == 18
    assert R.ARM_KEYS == S.ARM_KEYS
    assert R.ARM_CATALOG_SHA256 == S.ARM_CATALOG_SHA256


def test_task_ref_and_teacher_timing_are_exact_and_unclipped():
    broker, _ = _broker(1)
    birth = _reserve(broker)
    task = _task(birth, 0)
    ref = task.task_ref()
    assert R.ActionTaskReceiptRef.from_dict(ref.to_dict()) == ref
    assert ref.task_sha256 == task.canonical_sha256
    assert ref.birth_sha256 == birth.canonical_sha256

    with pytest.raises(
        R.ActionBallContractError,
        match="exact unclipped formula|exact face/site angular solve",
    ):
        replace(task, teacher_rate=task.teacher_rate + 1.0e-12)
    with pytest.raises(
        R.ActionBallContractError,
        match="outside certified bounds|teacher_rate_out_of_bounds",
    ):
        replace(
            task,
            racket_face_center_velocity_w_mps=(4.0, 0.0, 0.0),
            racket_site_velocity_w_mps=(4.0, 0.0, 0.0),
            required_racket_site_speed_mps=4.0,
            teacher_rate=4.0 / task.reference_racket_site_speed_mps,
            scaled_t_hit_s=(
                task.reference_t_hit_s
                / (4.0 / task.reference_racket_site_speed_mps)
            ),
            scaled_t_cycle_s=(
                task.reference_t_cycle_s
                / (4.0 / task.reference_racket_site_speed_mps)
            ),
            pre_swing_wait_s=(
                task.time_to_contact_s
                - task.reference_t_hit_s
                / (4.0 / task.reference_racket_site_speed_mps)
            ),
        )
    with pytest.raises(
        R.ActionBallContractError,
        match="reaction-margin/1s",
    ):
        R.derive_action_teacher_timing(
            racket_velocity_w_mps=(3.0, 0.0, 0.0),
            time_to_contact_s=2.0,
            reference_t_hit_s=0.42,
            reference_t_cycle_s=1.2,
            reference_racket_site_speed_mps=3.0,
            reaction_margin_s=0.05,
            teacher_rate_min=0.8,
            teacher_rate_max=1.2,
        )


@pytest.mark.parametrize(
    "native_float32_rate",
    (
        1.00000006489,
        1.00000002471,
        1.00000000142,
        0.99999998217,
    ),
)
def test_float32_native_boundary_survives_geometry_timing_and_receipt(
    native_float32_rate,
):
    """The actual fivebind float32 seam must survive the complete receipt."""

    broker, _ = _broker(1)
    birth = _reserve(broker)
    baseline = _task(birth, 0)
    reference_speed = 3.0
    face_velocity = (
        reference_speed * native_float32_rate,
        0.0,
        0.0,
    )
    geometry = R._contact_geometry.solve_exact_face_contact(
        ball_contact_w_m=baseline.ball_contact_w_m,
        racket_face_center_velocity_w_mps=face_velocity,
        solved_raw_a_normal_w=(1.0, 0.0, 0.0),
        mount_normal_sign=1,
        reference_racket_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        reference_racket_angular_velocity_w_radps=(0.0, 0.0, 0.0),
        reference_racket_site_speed_mps=reference_speed,
        teacher_rate_min=0.6,
        teacher_rate_max=1.0,
    )
    timing = R.derive_action_teacher_site_timing(
        racket_site_velocity_w_mps=(
            geometry.racket_site_velocity_w_mps
        ),
        time_to_contact_s=baseline.time_to_contact_s,
        reference_t_hit_s=baseline.reference_t_hit_s,
        reference_t_cycle_s=baseline.reference_t_cycle_s,
        reference_racket_site_speed_mps=reference_speed,
        reaction_margin_s=baseline.reaction_margin_s,
        teacher_rate_min=0.6,
        teacher_rate_max=1.0,
    )
    assert math.isclose(
        timing.teacher_rate,
        geometry.teacher_rate,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    )

    receipt = replace(
        baseline,
        racket_site_target_w_m=geometry.racket_site_target_w_m,
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
        reference_racket_site_speed_mps=reference_speed,
        required_racket_site_speed_mps=(
            timing.required_racket_site_speed_mps
        ),
        teacher_rate_min=0.6,
        teacher_rate_max=1.0,
        teacher_rate=timing.teacher_rate,
        scaled_t_hit_s=timing.scaled_t_hit_s,
        scaled_t_cycle_s=timing.scaled_t_cycle_s,
        pre_swing_wait_s=timing.pre_swing_wait_s,
    )
    assert R.ActionBallTaskReceipt.from_dict(receipt.to_dict()) == receipt


def test_exact_face_task_receipt_roundtrip_rejects_geometry_tampering_and_v3():
    broker, _ = _broker(1)
    birth = _reserve(broker)
    task = _task(birth, 0)
    row = task.to_dict()

    assert task.racket_site_target_w_m != task.ball_contact_w_m
    assert (
        task.racket_face_center_velocity_w_mps
        == task.racket_site_velocity_w_mps
    )
    assert R.ActionBallTaskReceipt.from_dict(row) == task

    for field, delta in (
        ("racket_site_target_w_m", (1.0e-4, 0.0, 0.0)),
        ("racket_site_velocity_w_mps", (0.0, 1.0e-4, 0.0)),
        (
            "racket_command_angular_velocity_w_radps",
            (0.0, 0.0, 1.0e-4),
        ),
    ):
        tampered = deepcopy(row)
        tampered[field] = [
            float(value) + float(change)
            for value, change in zip(tampered[field], delta)
        ]
        with pytest.raises(
            R.ActionBallContractError,
            match="exact-face geometry|face/site angular solve",
        ):
            R.ActionBallTaskReceipt.from_dict(tampered)

    tampered_quat = deepcopy(row)
    tampered_quat["racket_command_quat_wxyz"] = [1.0, 0.0, 0.0, 0.0]
    with pytest.raises(
        R.ActionBallContractError,
        match="quaternion|exact-face geometry",
    ):
        R.ActionBallTaskReceipt.from_dict(tampered_quat)

    tampered_sha = deepcopy(row)
    tampered_sha["geometry_source_sha256"] = "0" * 64
    with pytest.raises(
        R.ActionBallContractError,
        match="geometry source SHA",
    ):
        R.ActionBallTaskReceipt.from_dict(tampered_sha)

    old_v3 = deepcopy(row)
    old_v3["schema_version"] = 3
    for field in (
        "racket_site_target_w_m",
        "mount_normal_sign",
        "reference_racket_quat_wxyz",
        "reference_racket_angular_velocity_w_radps",
        "racket_command_quat_wxyz",
        "racket_face_center_velocity_w_mps",
        "racket_site_velocity_w_mps",
        "racket_command_angular_velocity_w_radps",
        "geometry_source_sha256",
    ):
        del old_v3[field]
    old_v3["racket_velocity_w_mps"] = [3.0, 0.2, 0.4]
    with pytest.raises(
        R.ActionBallContractError,
        match="invalid keys|schema_version",
    ):
        R.ActionBallTaskReceipt.from_dict(old_v3)


def test_base_preparation_contract_and_receipt_are_strict_and_roundtrip():
    contract = R.BasePreparationContract(
        max_planar_speed_mps=1.0,
        max_planar_acceleration_mps2=4.0,
        settle_margin_s=0.05,
    )
    assert R.BasePreparationContract.from_dict(
        contract.to_dict()
    ) == contract
    with pytest.raises(
        R.ActionBallContractError,
        match="speed must be > 0",
    ):
        R.BasePreparationContract(
            max_planar_speed_mps=0.0,
            max_planar_acceleration_mps2=4.0,
            settle_margin_s=0.05,
        )

    # d_switch=v^2/a=0.25 m, so 0.40 m is trapezoidal:
    # 2*v/a + (d-d_switch)/v = 0.65 s, then 0.05 s settle.
    receipt = R.BasePreparationReceipt.evaluate(
        proposal_sample_sha256=_digest("move-sample"),
        proposal_sample_index=17,
        mobility_mode="move",
        base_travel_b_yaw_m=(0.40, 0.0, 0.0),
        reaction_margin_s=0.10,
        available_preparation_s=0.75,
        contract=contract,
    )
    assert receipt.admitted
    assert receipt.reject_reason == ""
    assert math.isclose(receipt.planar_travel_distance_m, 0.40)
    assert math.isclose(receipt.motion_time_required_s, 0.65)
    assert math.isclose(
        receipt.move_preparation_required_s,
        0.70,
    )
    assert math.isclose(receipt.required_preparation_s, 0.70)
    assert receipt.available_preparation_s == 0.75
    assert receipt.proposal_count_delta == 1
    assert receipt.policy_attempt_count_delta == 0
    assert receipt.solver_rejection_count_delta == 0
    assert R.BasePreparationReceipt.from_dict(
        receipt.to_dict()
    ) == receipt

    tampered = deepcopy(receipt.to_dict())
    tampered["required_preparation_s"] += 1.0e-12
    with pytest.raises(
        R.ActionBallContractError,
        match="motion envelope/admission formula",
    ):
        R.BasePreparationReceipt.from_dict(tampered)


def test_move_teacher_timing_rejects_unpreparable_sample_before_policy():
    contract = R.BasePreparationContract(
        max_planar_speed_mps=1.0,
        max_planar_acceleration_mps2=2.0,
        settle_margin_s=0.20,
    )
    kwargs = {
        "racket_velocity_w_mps": (3.0, 0.0, 0.0),
        "time_to_contact_s": 0.90,
        "reference_t_hit_s": 0.40,
        "reference_t_cycle_s": 1.20,
        "reference_racket_site_speed_mps": 3.0,
        "reaction_margin_s": 0.05,
        "teacher_rate_min": 0.8,
        "teacher_rate_max": 1.2,
        "proposal_sample_sha256": _digest("unpreparable-move"),
        "proposal_sample_index": 23,
        "mobility_mode": "move",
        "base_travel_b_yaw_m": (0.18, 0.0, 0.0),
    }
    # Available wait is 0.50 s.  Triangular travel needs 0.60 s plus
    # 0.20 s settle, so this remains a solver rejection, not an attempt.
    with pytest.raises(R.BasePreparationAdmissionError) as caught:
        R.derive_action_teacher_timing_with_base_preparation(
            **kwargs,
            base_preparation_contract=contract,
        )
    receipt = caught.value.receipt
    assert caught.value.reject_reason == (
        R.BASE_PREPARATION_REJECT_REASON
    )
    assert receipt.reject_reason == R.BASE_PREPARATION_REJECT_REASON
    assert math.isclose(receipt.planar_travel_distance_m, 0.18)
    assert math.isclose(receipt.motion_time_required_s, 0.60)
    assert math.isclose(receipt.required_preparation_s, 0.80)
    assert math.isclose(receipt.available_preparation_s, 0.50)
    assert receipt.proposal_count_delta == 1
    assert receipt.policy_attempt_count_delta == 0
    assert receipt.solver_rejection_count_delta == 1
    assert R.BasePreparationReceipt.from_dict(
        receipt.to_dict()
    ) == receipt

    with pytest.raises(
        R.ActionBallContractError,
        match="requires an exact pinned",
    ):
        R.derive_action_teacher_timing_with_base_preparation(
            **kwargs,
            base_preparation_contract=None,
        )


def test_move_teacher_timing_admits_exact_boundary_and_no_move_is_legacy():
    legacy = R.derive_action_teacher_timing(
        racket_velocity_w_mps=(3.0, 0.0, 0.0),
        time_to_contact_s=0.90,
        reference_t_hit_s=0.40,
        reference_t_cycle_s=1.20,
        reference_racket_site_speed_mps=3.0,
        reaction_margin_s=0.05,
        teacher_rate_min=0.8,
        teacher_rate_max=1.2,
    )
    no_move = R.derive_action_teacher_timing_with_base_preparation(
        racket_velocity_w_mps=(3.0, 0.0, 0.0),
        time_to_contact_s=0.90,
        reference_t_hit_s=0.40,
        reference_t_cycle_s=1.20,
        reference_racket_site_speed_mps=3.0,
        reaction_margin_s=0.05,
        teacher_rate_min=0.8,
        teacher_rate_max=1.2,
        proposal_sample_sha256=_digest("legacy-no-move"),
        proposal_sample_index=0,
        mobility_mode="no_move",
        # Latent travel still exists for RNG parity but is not executed.
        base_travel_b_yaw_m=(0.40, -0.30, 0.0),
        base_preparation_contract=None,
    )
    assert no_move.timing == legacy
    assert (
        tuple(legacy.__dict__)
        == (
            "required_racket_site_speed_mps",
            "teacher_rate",
            "scaled_t_hit_s",
            "scaled_t_cycle_s",
            "pre_swing_wait_s",
        )
    )
    assert no_move.base_preparation.planar_travel_distance_m == 0.0
    assert no_move.base_preparation.preparation_contract_sha256 is None
    assert no_move.base_preparation.admitted

    # With a=2, d=0.08 needs exactly 0.40 s; plus 0.10 s settle
    # exactly consumes the 0.50 s available wait and must be admitted.
    contract = R.BasePreparationContract(
        max_planar_speed_mps=1.0,
        max_planar_acceleration_mps2=2.0,
        settle_margin_s=0.10,
    )
    move = R.derive_action_teacher_timing_with_base_preparation(
        racket_velocity_w_mps=(3.0, 0.0, 0.0),
        time_to_contact_s=0.90,
        reference_t_hit_s=0.40,
        reference_t_cycle_s=1.20,
        reference_racket_site_speed_mps=3.0,
        reaction_margin_s=0.05,
        teacher_rate_min=0.8,
        teacher_rate_max=1.2,
        proposal_sample_sha256=_digest("boundary-move"),
        proposal_sample_index=1,
        mobility_mode="move",
        base_travel_b_yaw_m=(0.08, 0.0, 0.0),
        base_preparation_contract=contract,
    )
    assert move.timing == legacy
    assert move.base_preparation.admitted
    assert math.isclose(
        move.base_preparation.required_preparation_s,
        move.timing.pre_swing_wait_s,
    )


def test_real_sampler_identity_is_bit_exact_with_runtime_receipt():
    action_uid = 12_345
    profile = _sampling_profile(action_uid)
    mixture = S.SamplingMixture()
    policy_dt_s = 0.02
    sampler = S.ActionBallSampler(
        (profile,),
        seed=20260727,
        sampling_mixture=mixture,
        contact_time_step_s=policy_dt_s,
    )
    sampler_levels = S.DomainLevels(
        landing_aim_x_lower=0.2,
        contact_y_upper=0.3,
        incoming_speed_lower=0.4,
        spin_magnitude_upper=0.5,
        spin_direction_u_neg=0.6,
        base_spawn_y_upper=0.7,
    )
    levels = R.ActionDomainLevels.from_dict(
        sampler_levels.as_dict()
    )
    authority_sha = _digest("real-domain-authority")
    pins = R.RuntimePins(
        manifest_sha256=_digest("real-manifest"),
        sampler_sha256=sampler.sampler_contract_sha256,
        domain_authority_sha256=authority_sha,
        physics_sha256=_digest("real-physics"),
        solver_sha256=_digest("real-solver"),
    )
    binding = R.ActionBinding(
        action_uid=action_uid,
        action_slot=0,
        motion_path="vendor_assets/motions/real.npz",
        motion_sha256=_digest("real-motion"),
        profile_sha256=profile.sha256,
    )
    yaw = 0.31
    sampled_birth = sampler.reserve_birth(
        action_uid=action_uid,
        domain_epoch=7,
        levels=sampler_levels,
        base_yaw_rad=yaw,
    )
    claim = R.ActionDomainClaim(
        authority_contract_sha256=authority_sha,
        arm_catalog_sha256=R.ARM_CATALOG_SHA256,
        action_uid=action_uid,
        domain_epoch=7,
        domain_levels=levels,
        levels_sha256=levels.canonical_sha256,
        profile_sha256=profile.sha256,
        mobility_mode="no_move",
    )
    birth = R.ActionBirthReceipt(
        env_id=4,
        reset_generation=1,
        action_uid=action_uid,
        action_slot=0,
        domain_epoch=7,
        domain_claim_sha256=claim.canonical_sha256,
        domain_authority_sha256=authority_sha,
        domain_levels=levels,
        arm_catalog_sha256=R.ARM_CATALOG_SHA256,
        levels_sha256=levels.canonical_sha256,
        sampler_birth_sha256=sampled_birth.birth_id,
        sampler_birth_index=sampled_birth.birth_index,
        sampler_draw_start=sampled_birth.draw_start,
        sampler_draw_end=sampled_birth.draw_end,
        mobility_mode="no_move",
        base_yaw_rad=yaw,
        base_quat_wxyz=_yaw_quat(yaw),
        base_spawn_w_m=sampled_birth.base_start_w_m,
        manifest_sha256=pins.manifest_sha256,
        sampler_sha256=pins.sampler_sha256,
        profile_sha256=profile.sha256,
        motion_sha256=binding.motion_sha256,
        physics_sha256=pins.physics_sha256,
        solver_sha256=pins.solver_sha256,
        registry_sha256=R._registry_sha256(
            (binding,), pins, "no_move"
        ),
        sampling_mixture=R.ActionSamplingMixture.from_dict(
            sampled_birth.sampling_mixture.as_dict()
        ),
        sampling_stratum=sampled_birth.sampling_stratum,
        sampling_levels=R.ActionDomainLevels.from_dict(
            sampled_birth.sampling_levels.as_dict()
        ),
        frontier_arm=sampled_birth.frontier_arm,
    )
    sampled = sampler.sample(
        birth=sampled_birth,
        action_uid=action_uid,
        domain_epoch=7,
        levels=sampler_levels,
        base_yaw_rad=yaw,
    )
    racket_velocity = (6.0, 0.0, 0.0)
    reference_quat = R._contact_geometry.canonical_quat_wxyz(
        (
            0.6867758396936938,
            0.3442809801333191,
            -0.23836926079947673,
            0.6530397713504417,
        )
    )
    reference_normal = R._contact_geometry.quat_rotate_wxyz(
        reference_quat,
        (0.0, 1.0, 0.0),
    )
    geometry = R._contact_geometry.solve_exact_face_contact(
        ball_contact_w_m=sampled.contact_w_m,
        racket_face_center_velocity_w_mps=racket_velocity,
        solved_raw_a_normal_w=reference_normal,
        mount_normal_sign=1,
        reference_racket_quat_wxyz=reference_quat,
        reference_racket_angular_velocity_w_radps=(0.0, 0.0, 0.0),
        reference_racket_site_speed_mps=(
            profile.reference_racket_site_speed_mps
        ),
        teacher_rate_min=profile.teacher_rate_min,
        teacher_rate_max=profile.teacher_rate_max,
    )
    timing = R.derive_action_teacher_site_timing(
        racket_site_velocity_w_mps=(
            geometry.racket_site_velocity_w_mps
        ),
        time_to_contact_s=sampled.time_to_contact_s,
        reference_t_hit_s=profile.reference_t_hit_s,
        reference_t_cycle_s=profile.reference_t_cycle_s,
        reference_racket_site_speed_mps=(
            profile.reference_racket_site_speed_mps
        ),
        reaction_margin_s=profile.reaction_margin_s,
        teacher_rate_min=profile.teacher_rate_min,
        teacher_rate_max=profile.teacher_rate_max,
    )
    incoming_horizontal_norm = math.hypot(
        sampled.incoming_velocity_w_mps[0],
        sampled.incoming_velocity_w_mps[1],
    )
    counter_rally_task = R.CounterRallyTaskIdentity(
        objective_profile_sha256=_digest(
            "real-counter-rally-objective"
        ),
        return_direction_env_xy=(
            -sampled.incoming_velocity_w_mps[0]
            / incoming_horizontal_norm,
            -sampled.incoming_velocity_w_mps[1]
            / incoming_horizontal_norm,
        ),
        target_baseline_speed_mps=sampled.incoming_speed_mps,
    )
    task = R.ActionBallTaskReceipt.from_birth(
        birth,
        sample_sha256=sampled.sample_id,
        sample_index=sampled.sample_index,
        sample_draw_start=sampled.draw_start,
        sample_draw_end=sampled.draw_end,
        swing_generation=0,
        base_goal_w_m=sampled.base_goal_w_m,
        base_spawn_latent_w_m=sampled.base_spawn_latent_w_m,
        base_travel_latent_b_yaw_m=(
            sampled.base_travel_latent_b_yaw_m
        ),
        contact_offset_from_base_goal_b_yaw_m=(
            sampled.contact_offset_from_base_goal_b_yaw_m
        ),
        ball_contact_w_m=sampled.contact_w_m,
        time_to_contact_s=sampled.time_to_contact_s,
        incoming_speed_mps=sampled.incoming_speed_mps,
        incoming_direction_b_yaw=sampled.incoming_direction_b_yaw,
        incoming_velocity_w_mps=sampled.incoming_velocity_w_mps,
        spin_magnitude_radps=sampled.spin_magnitude_radps,
        spin_direction_b_yaw=sampled.spin_direction_b_yaw,
        incoming_spin_w_radps=sampled.spin_w_radps,
        landing_aim_w_xy_m=sampled.landing_aim_w_xy_m,
        racket_site_target_w_m=geometry.racket_site_target_w_m,
        mount_normal_sign=1,
        racket_normal_w=reference_normal,
        reference_racket_quat_wxyz=reference_quat,
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
        reference_t_hit_s=profile.reference_t_hit_s,
        reference_t_cycle_s=profile.reference_t_cycle_s,
        reference_racket_site_speed_mps=(
            profile.reference_racket_site_speed_mps
        ),
        required_racket_site_speed_mps=(
            timing.required_racket_site_speed_mps
        ),
        reaction_margin_s=profile.reaction_margin_s,
        teacher_rate_min=profile.teacher_rate_min,
        teacher_rate_max=profile.teacher_rate_max,
        teacher_rate=timing.teacher_rate,
        scaled_t_hit_s=timing.scaled_t_hit_s,
        scaled_t_cycle_s=timing.scaled_t_cycle_s,
        pre_swing_wait_s=timing.pre_swing_wait_s,
        solver_residual_m=0.004,
        contact_time_step_s=sampled.contact_time_step_s,
        time_to_contact_tick=sampled.time_to_contact_tick,
        birth_index=sampled.birth_index,
        birth_sampling_stratum=sampled.birth_sampling_stratum,
        birth_sampling_levels=R.ActionDomainLevels.from_dict(
            sampled.birth_sampling_levels.as_dict()
        ),
        birth_frontier_arm=sampled.birth_frontier_arm,
        sampling_mixture=R.ActionSamplingMixture.from_dict(
            sampled.sampling_mixture.as_dict()
        ),
        sampling_stratum=sampled.sampling_stratum,
        sampling_levels=R.ActionDomainLevels.from_dict(
            sampled.sampling_levels.as_dict()
        ),
        frontier_arm=sampled.frontier_arm,
        counter_rally_task=counter_rally_task,
    )
    diagnostic_task = replace(
        task,
        _validation_mode=(
            R._DIAGNOSTIC_PREVALIDATED_TASK_RECEIPT
        ),
    )
    assert type(diagnostic_task) is R.ActionBallTaskReceipt
    assert diagnostic_task == task
    assert diagnostic_task.to_dict() == task.to_dict()
    assert diagnostic_task.canonical_sha256 == task.canonical_sha256
    assert diagnostic_task.sampler_identity_receipt() == (
        sampled.to_identity_receipt()
    )
    assert task.sample_sha256 == sampled.sample_id
    assert task.sampler_identity_receipt() == (
        sampled.to_identity_receipt()
    )
    before = deepcopy(sampler.state_dict())
    sampler.assert_issued_sample(task.sampler_identity_receipt())
    assert sampler.state_dict() == before
    assert task.time_to_contact_s == (
        task.time_to_contact_tick * task.contact_time_step_s
    )
    assert R.ActionBallTaskReceipt.from_dict(task.to_dict()) == task
    for field, replacement in (
        ("contact_time_step_s", policy_dt_s * 2.0),
        ("time_to_contact_tick", task.time_to_contact_tick + 1),
        ("sampling_stratum", "center"),
    ):
        tampered = task.to_dict()
        tampered[field] = replacement
        with pytest.raises(R.ActionBallContractError):
            R.ActionBallTaskReceipt.from_dict(tampered)


@pytest.mark.parametrize("count", [1, 5, 93])
def test_broker_registry_and_protocol_work_for_n1_n5_n93(count):
    broker, provider = _broker(count)
    assert broker.action_count == count
    assert broker.ordered_action_uids == tuple(
        binding.action_uid for binding in _bindings(count)
    )
    for slot in (0, count - 1):
        binding = broker.binding_for_slot(slot)
        assert broker.binding_for_uid(binding.action_uid) == binding
        assert binding.motion_path.endswith(f"action_{slot:03d}.npz")

    birth = _reserve(broker, env_id=7, slot=count - 1)
    assert provider.requests[0].__dict__.keys() == {
        "env_id",
        "reset_generation",
        "action_uid",
        "action_slot",
        "domain_claim",
        "registry_sha256",
        "mobility_mode",
        "binding",
        "pins",
    }
    assert broker.pending_receipt(
        env_id=7,
        reset_generation=1,
        action_uid=birth.action_uid,
        action_slot=birth.action_slot,
    ) is birth
    _commit(broker, birth)
    consumed = broker.consume_true_reset(
        env_id=7,
        reset_generation=1,
        action_uid=birth.action_uid,
        action_slot=birth.action_slot,
        receipt_sha256=birth.canonical_sha256,
    )
    assert consumed is birth


def test_broker_rejects_wrap_wrong_slot_stale_and_replay():
    broker, _ = _broker(5)
    with pytest.raises(R.BirthProtocolError, match="true_reset"):
        broker.reserve_true_reset(
            env_id=0,
            reset_generation=1,
            action_uid=broker.ordered_action_uids[0],
            action_slot=0,
            reset_kind="wrap",
        )
    with pytest.raises(R.ActionBallContractError, match="not bound"):
        broker.reserve_true_reset(
            env_id=0,
            reset_generation=1,
            action_uid=broker.ordered_action_uids[0],
            action_slot=1,
        )
    with pytest.raises(R.BirthProtocolError, match="exactly 1"):
        broker.reserve_true_reset(
            env_id=0,
            reset_generation=2,
            action_uid=broker.ordered_action_uids[0],
            action_slot=0,
        )

    birth = _reserve(broker)
    with pytest.raises(R.BirthProtocolError, match="before Motion"):
        broker.consume_true_reset(
            env_id=0,
            reset_generation=1,
            action_uid=birth.action_uid,
            action_slot=birth.action_slot,
            receipt_sha256=birth.canonical_sha256,
        )
    _commit(broker, birth)
    broker.consume_many_true_reset((_claim(birth),))
    with pytest.raises(R.BirthProtocolError, match="replay/stale"):
        broker.consume_many_true_reset((_claim(birth),))


def test_batch_consume_is_atomic_on_late_bad_claim():
    broker, _ = _broker(5)
    first = _reserve(broker, env_id=0, slot=0)
    second = _reserve(broker, env_id=1, slot=1)
    _commit(broker, first)
    _commit(broker, second)

    bad = _claim(second, receipt_sha256=_digest("wrong"))
    with pytest.raises(R.BirthProtocolError, match="mismatch"):
        broker.consume_many_true_reset((_claim(first), bad))
    # First was not consumed as a prefix side effect.
    assert broker.pending_receipt(
        env_id=first.env_id,
        reset_generation=first.reset_generation,
        action_uid=first.action_uid,
        action_slot=first.action_slot,
    ) == first
    assert broker.consume_many_true_reset(
        (_claim(first), _claim(second))
    ) == (first, second)


def test_batch_reserve_and_commit_are_atomic_on_late_bad_row():
    bindings = _bindings(5)
    pins = _pins()
    broker = R.ActionBirthBroker(bindings, pins, "no_move")

    class BadSecondProvider(BirthProvider):
        def __call__(self, request):
            receipt = super().__call__(request)
            if request.env_id == 1:
                return replace(
                    receipt, motion_sha256=_digest("wrong-motion")
                )
            return receipt

    broker.bind_domain_claim_authority(
        DomainAuthority(bindings, "no_move")
    )
    broker.bind_provider(BadSecondProvider())
    before = deepcopy(broker.state_dict())
    with pytest.raises(R.ActionBallContractError, match="action binding"):
        broker.reserve_many_true_reset(
            (
                _reserve_claim(broker, env_id=0, slot=0),
                _reserve_claim(broker, env_id=1, slot=1),
            )
        )
    assert broker.state_dict() == before

    good, _ = _broker(5)
    first, second = good.reserve_many_true_reset(
        (
            _reserve_claim(good, env_id=0, slot=0),
            _reserve_claim(good, env_id=1, slot=1),
        )
    )
    with pytest.raises(R.BirthProtocolError, match="does not match"):
        good.commit_many_true_reset(
            (
                R.BirthCommitRequest(
                    first.env_id,
                    first.reset_generation,
                    first.canonical_sha256,
                ),
                R.BirthCommitRequest(
                    second.env_id,
                    second.reset_generation,
                    _digest("wrong"),
                ),
            )
        )
    # A bad second row did not commit the valid prefix.
    with pytest.raises(R.BirthProtocolError, match="before Motion"):
        good.consume_many_true_reset((_claim(first),))
    good.commit_many_true_reset(
        (
            R.BirthCommitRequest(
                first.env_id,
                first.reset_generation,
                first.canonical_sha256,
            ),
            R.BirthCommitRequest(
                second.env_id,
                second.reset_generation,
                second.canonical_sha256,
            ),
        )
    )


def test_diagnostic_batch_birth_callbacks_match_scalar_fixed_tape_n5():
    bindings = _bindings(5)
    pins = _pins()
    scalar = R.ActionBirthBroker(
        bindings, pins, "no_move", diagnostic_unauthorized=True
    )
    scalar_authority = DomainAuthority(bindings, "no_move")
    scalar_provider = BirthProvider()
    scalar.bind_domain_claim_authority(scalar_authority)
    scalar.bind_provider(scalar_provider)

    batched = R.ActionBirthBroker(
        bindings, pins, "no_move", diagnostic_unauthorized=True
    )
    batched_authority = BatchedDomainAuthority(
        bindings, "no_move"
    )
    batched_provider = BatchedBirthProvider()
    batched.bind_domain_claim_authority(batched_authority)
    batched.bind_provider(batched_provider)

    slots = (4, 0, 4, 2, 1, 0, 3, 2)
    scalar_requests = tuple(
        _reserve_claim(
            scalar,
            env_id=100 + index,
            slot=slot,
        )
        for index, slot in enumerate(slots)
    )
    batched_requests = tuple(
        _reserve_claim(
            batched,
            env_id=100 + index,
            slot=slot,
        )
        for index, slot in enumerate(slots)
    )
    scalar_births = scalar.reserve_many_true_reset(scalar_requests)
    batched_births = batched.reserve_many_true_reset(batched_requests)

    assert [birth.to_dict() for birth in batched_births] == [
        birth.to_dict() for birth in scalar_births
    ]
    assert [birth.canonical_sha256 for birth in batched_births] == [
        birth.canonical_sha256 for birth in scalar_births
    ]
    assert batched_authority.batch_calls == 1
    assert batched_authority.scalar_calls == 0
    assert batched_provider.batch_calls == 1
    assert batched_provider.scalar_calls == 0
    assert batched_provider.state_dict() == scalar_provider.state_dict()
    assert batched_authority.state_dict() == scalar_authority.state_dict()
    assert batched.state_dict() == scalar.state_dict()

    for broker, births in (
        (scalar, scalar_births),
        (batched, batched_births),
    ):
        broker.commit_many_true_reset(
            tuple(
                R.BirthCommitRequest(
                    birth.env_id,
                    birth.reset_generation,
                    birth.canonical_sha256,
                )
                for birth in births
            )
        )
        broker.consume_many_true_reset(
            tuple(_claim(birth) for birth in births)
        )

    scalar_pool = R.LazyActionTaskPool(
        bindings, pins, "no_move", diagnostic_unauthorized=True
    )
    batched_pool = R.LazyActionTaskPool(
        bindings, pins, "no_move", diagnostic_unauthorized=True
    )
    scalar_solver = Solver()
    batched_solver = Solver()
    scalar_pool.bind_solver(scalar_solver)
    batched_pool.bind_solver(batched_solver)
    scalar_pool.bind_birth_authority(scalar)
    batched_pool.bind_birth_authority(batched)
    scalar_tasks = scalar_pool.request_many(
        tuple(
            R.ActionTaskIssueRequest(birth, 0)
            for birth in scalar_births
        )
    )
    batched_tasks = batched_pool.request_many(
        tuple(
            R.ActionTaskIssueRequest(birth, 0)
            for birth in batched_births
        )
    )
    assert [task.to_dict() for task in batched_tasks] == [
        task.to_dict() for task in scalar_tasks
    ]
    assert batched_solver.state_dict() == scalar_solver.state_dict()
    # Diagnostic pools intentionally omit the formal compact lifecycle and
    # cannot produce an exact-resume state_dict.  Assert the boundary on both
    # sides instead of mixing a formal pool with diagnostic birth authority.
    for pool in (scalar_pool, batched_pool):
        with pytest.raises(
            R.ActionBallContractError,
            match="task lifecycle must cover sample indices",
        ):
            pool.state_dict()
    for name in (
        "_births",
        "_pending",
        "_issued_task_transcript_sha256",
        "_cursor",
        "_refill_index",
        "_proposed_by_birth",
        "_sample_assignments",
        "_ledger",
        "_seen_sha256",
        "_seen_sample_sha256",
        "_retired_births",
        "_task_lifecycle",
        "_last_sample_index",
        "_last_sample_draw_end",
        "_retired_generation",
        "_diagnostic_birth_by_env",
        "_diagnostic_active_sample_sha256",
    ):
        assert getattr(batched_pool, name) == getattr(scalar_pool, name)
    for pool in (scalar_pool, batched_pool):
        assert pool._issued_task_transcript_sha256 == {}
        assert pool._proposed_by_birth == {}
        assert pool._sample_assignments == {}
        expected_issued = {}
        for birth in scalar_births:
            expected_issued[birth.action_uid] = (
                expected_issued.get(birth.action_uid, 0) + 1
            )
        for action_uid in batched.ordered_action_uids:
            count = expected_issued.get(action_uid, 0)
            assert pool.ledger(action_uid) == R.PoolLedger(
                requests=count,
                refill_calls=count,
                proposed=count,
                admitted=count,
                issued=count,
                discarded=0,
            )
    assert {
        uid: batched_pool.ledger(uid).to_dict()
        for uid in batched.ordered_action_uids
    } == {
        uid: scalar_pool.ledger(uid).to_dict()
        for uid in scalar.ordered_action_uids
    }


def test_diagnostic_pool_preserves_rejected_proposal_ledger_without_formal_maps():
    broker, _provider = _broker(1, diagnostic_unauthorized=True)
    birth = _reserve(broker)
    _consume(broker, birth)
    pool = R.LazyActionTaskPool(
        _bindings(1),
        _pins(),
        "no_move",
        refill_size=1,
        diagnostic_unauthorized=True,
    )
    solver = RejectedTailReplaySolver()
    pool.bind_solver(solver)
    pool.bind_birth_authority(broker)

    task = pool.request(birth, swing_generation=0)

    assert task.sample_index == 0
    assert solver.sample_highwater_for(birth.action_uid)[0] == 1
    assert pool.ledger(birth.action_uid) == R.PoolLedger(
        requests=1,
        refill_calls=1,
        proposed=2,
        admitted=1,
        issued=1,
        discarded=0,
    )
    assert pool._issued_task_transcript_sha256 == {}
    assert pool._proposed_by_birth == {}
    assert pool._sample_assignments == {}
    assert pool.retire_birth(birth) == 0
    assert pool.ledger(birth.action_uid) == R.PoolLedger(
        requests=1,
        refill_calls=1,
        proposed=2,
        admitted=1,
        issued=1,
        discarded=0,
    )
    assert pool.materialized_action_uids == ()


def test_diagnostic_batch_late_bad_receipt_has_no_broker_prefix():
    bindings = _bindings(5)
    broker = R.ActionBirthBroker(
        bindings,
        _pins(),
        "no_move",
        diagnostic_unauthorized=True,
    )
    authority = BatchedDomainAuthority(bindings, "no_move")

    class BadTailBatchProvider(BatchedBirthProvider):
        def provide_many(self, requests):
            receipts = list(super().provide_many(requests))
            receipts[-1] = replace(
                receipts[-1], motion_sha256=_digest("wrong-motion")
            )
            return tuple(receipts)

    provider = BadTailBatchProvider()
    broker.bind_domain_claim_authority(authority)
    broker.bind_provider(provider)
    requests = tuple(
        _reserve_claim(
            broker,
            env_id=index,
            slot=index,
        )
        for index in range(5)
    )
    with pytest.raises(R.ActionBallContractError, match="action binding"):
        broker.reserve_many_true_reset(requests)
    assert broker._pending == {}
    assert broker._last_generation == {}
    assert broker._last_sampler_birth_index == {}
    assert broker._last_sampler_draw_end == {}
    assert broker._domain_claim_count == {}
    assert provider.batch_calls == 1
    for request in requests:
        with pytest.raises(R.BirthProtocolError, match="no pending"):
            broker.consume_many_true_reset(
                (
                    R.BirthConsumeRequest(
                        env_id=request.env_id,
                        reset_generation=request.reset_generation,
                        action_uid=request.action_uid,
                        action_slot=request.action_slot,
                        receipt_sha256=_digest("unpublished"),
                    ),
                )
            )


def test_formal_broker_keeps_scalar_callbacks_when_batch_api_exists():
    bindings = _bindings(5)
    broker = R.ActionBirthBroker(bindings, _pins(), "no_move")
    authority = BatchedDomainAuthority(bindings, "no_move")
    provider = BatchedBirthProvider()
    broker.bind_domain_claim_authority(authority)
    broker.bind_provider(provider)
    requests = tuple(
        _reserve_claim(
            broker,
            env_id=index,
            slot=slot,
        )
        for index, slot in enumerate((4, 0, 3, 1, 2))
    )
    births = broker.reserve_many_true_reset(requests)
    assert len(births) == len(requests)
    assert authority.batch_calls == 0
    assert authority.scalar_calls == len(requests)
    assert provider.batch_calls == 0
    assert provider.scalar_calls == len(requests)


@pytest.mark.parametrize(
    "mutate_provider,birth_count",
    [(True, 1), (True, 2), (False, 1), (False, 2)],
)
def test_broker_rejects_hidden_cross_action_tape_advance_atomically(
    mutate_provider,
    birth_count,
):
    bindings = _bindings(2)
    broker = R.ActionBirthBroker(bindings, _pins(), "no_move")
    untouched_uid = bindings[1].action_uid
    authority = (
        DomainAuthority(bindings, "no_move")
        if mutate_provider
        else CrossActionDomainAuthority(
            bindings, "no_move", untouched_uid
        )
    )
    provider = (
        CrossActionBirthProvider(untouched_uid)
        if mutate_provider
        else BirthProvider()
    )
    broker.bind_domain_claim_authority(authority)
    broker.bind_provider(provider)
    before_broker = deepcopy(broker.state_dict())
    before_provider = deepcopy(provider.state_dict())
    before_authority = deepcopy(authority.state_dict())

    with pytest.raises(
        R.ActionBallContractError,
        match="advanced an unstaged action tape",
    ):
        broker.reserve_many_true_reset(
            tuple(
                _reserve_claim(broker, env_id=env_id, slot=0)
                for env_id in range(birth_count)
            )
        )
    assert broker.state_dict() == before_broker
    assert provider.state_dict() == before_provider
    assert authority.state_dict() == before_authority


@pytest.mark.parametrize(
    "mutate_provider,raise_after_mutation",
    [(True, False), (True, True), (False, False), (False, True)],
)
def test_broker_state_read_rolls_back_mutating_tape_authority(
    mutate_provider,
    raise_after_mutation,
):
    bindings = _bindings(1)
    broker = R.ActionBirthBroker(bindings, _pins(), "no_move")
    authority = (
        DomainAuthority(bindings, "no_move")
        if mutate_provider
        else MutatingDomainCursorAuthority(
            bindings,
            "no_move",
            raise_after_mutation=raise_after_mutation,
        )
    )
    provider = (
        MutatingBirthHighwaterProvider(
            raise_after_mutation=raise_after_mutation
        )
        if mutate_provider
        else BirthProvider()
    )
    broker.bind_domain_claim_authority(authority)
    broker.bind_provider(provider)
    before_provider = deepcopy(provider.state_dict())
    before_authority = deepcopy(authority.state_dict())
    expected_error = (
        RuntimeError
        if raise_after_mutation
        else R.ActionBallContractError
    )

    with pytest.raises(expected_error):
        broker.state_dict()
    assert provider.state_dict() == before_provider
    assert authority.state_dict() == before_authority


def test_broker_exact_resume_and_atomic_tamper_rejection():
    source, _ = _broker(5)
    consumed = _reserve(source, env_id=0, slot=0)
    _commit(source, consumed)
    source.consume_many_true_reset((_claim(consumed),))
    reserved = _reserve(source, env_id=0, generation=2, slot=1)
    committed = _reserve(source, env_id=1, generation=1, slot=2)
    _commit(source, committed)
    saved = deepcopy(source.state_dict())
    json.dumps(saved, allow_nan=False)

    restored, _ = _broker(5)
    restored.load_state_dict(saved)
    assert restored.state_dict() == saved
    assert restored.pending_receipt(
        env_id=reserved.env_id,
        reset_generation=reserved.reset_generation,
        action_uid=reserved.action_uid,
        action_slot=reserved.action_slot,
    ) == reserved
    assert restored.consume_many_true_reset((_claim(committed),)) == (
        committed,
    )

    before = deepcopy(restored.state_dict())
    corrupt = deepcopy(saved)
    corrupt["pending"][0]["receipt"]["base_spawn_w_m"][0] += 0.5
    with pytest.raises(R.ActionBallContractError, match="integrity"):
        restored.load_state_dict(corrupt)
    assert restored.state_dict() == before

    forged = deepcopy(saved)
    forged["pending"][0]["receipt"]["base_spawn_w_m"][0] += 0.5
    forged["integrity_sha256"] = _integrity(forged)
    with pytest.raises(
        R.ActionBallContractError,
        match="sampler birth SHA|canonical SHA",
    ):
        restored.load_state_dict(forged)
    assert restored.state_dict() == before


def test_diagnostic_consumed_history_keeps_legacy_state_bytes_and_values():
    broker, _provider = _broker(1, diagnostic_unauthorized=True)
    history = _consume_same_env_birth_history(broker, generations=1)
    legacy_before = deepcopy(broker.state_dict())
    legacy_bytes_before = _canonical_json_bytes(legacy_before)

    full_history = broker.diagnostic_state_dict_with_consumed_history(history)
    legacy_after = broker.state_dict()

    assert full_history == legacy_before
    assert legacy_after == legacy_before
    assert _canonical_json_bytes(legacy_after) == legacy_bytes_before


def test_diagnostic_consumed_history_exact_resume_without_producer_calls():
    source, source_provider, source_authority = (
        _diagnostic_broker_with_counted_producers()
    )
    history = _consume_same_env_birth_history(source)
    source_producer_calls = (
        source_provider.scalar_calls,
        source_provider.batch_calls,
        source_authority.scalar_calls,
        source_authority.batch_calls,
    )
    source_callback_state = (
        deepcopy(source_provider.state_dict()),
        deepcopy(source_authority.state_dict()),
    )

    saved = source.diagnostic_state_dict_with_consumed_history(history)
    saved_bytes = _canonical_json_bytes(saved)
    assert len(saved["consumed_receipts"]) == len(history) == 3
    assert (
        source_provider.scalar_calls,
        source_provider.batch_calls,
        source_authority.scalar_calls,
        source_authority.batch_calls,
    ) == source_producer_calls
    assert (
        source_provider.state_dict(),
        source_authority.state_dict(),
    ) == source_callback_state

    restored, restored_provider, restored_authority = (
        _diagnostic_broker_with_counted_producers()
    )
    assert (
        restored_provider.scalar_calls,
        restored_provider.batch_calls,
        restored_authority.scalar_calls,
        restored_authority.batch_calls,
    ) == (0, 0, 0, 0)
    restored.load_state_dict(saved)
    restored_history = tuple(
        R.ActionBirthReceipt.from_dict(row)
        for row in saved["consumed_receipts"]
    )
    restored_callback_state = (
        deepcopy(restored_provider.state_dict()),
        deepcopy(restored_authority.state_dict()),
    )
    resaved = restored.diagnostic_state_dict_with_consumed_history(
        restored_history
    )

    assert resaved == saved
    assert _canonical_json_bytes(resaved) == saved_bytes
    assert (
        restored_provider.scalar_calls,
        restored_provider.batch_calls,
        restored_authority.scalar_calls,
        restored_authority.batch_calls,
    ) == (0, 0, 0, 0)
    assert (
        restored_provider.state_dict(),
        restored_authority.state_dict(),
    ) == restored_callback_state


@pytest.mark.parametrize(
    ("case", "error_match"),
    (
        ("missing", "missing or adds"),
        ("extra", "missing or adds"),
        ("out_of_order", "unique and sorted"),
        ("identity_drift", "action binding|issued birth|live env receipt"),
    ),
)
def test_diagnostic_consumed_history_save_fails_closed_atomically(
    case, error_match
):
    broker, provider, authority = _diagnostic_broker_with_counted_producers()
    history = _consume_same_env_birth_history(broker)
    before_broker = broker.diagnostic_state_dict_with_consumed_history(history)
    before_provider = deepcopy(provider.state_dict())
    before_authority = deepcopy(authority.state_dict())
    before_producer_calls = (
        provider.scalar_calls,
        provider.batch_calls,
        authority.scalar_calls,
        authority.batch_calls,
    )

    with pytest.raises(R.ActionBallContractError, match=error_match):
        broker.diagnostic_state_dict_with_consumed_history(
            _mutate_consumed_birth_history(history, case)
        )

    assert broker.diagnostic_state_dict_with_consumed_history(history) == (
        before_broker
    )
    assert provider.state_dict() == before_provider
    assert authority.state_dict() == before_authority
    assert (
        provider.scalar_calls,
        provider.batch_calls,
        authority.scalar_calls,
        authority.batch_calls,
    ) == before_producer_calls


@pytest.mark.parametrize(
    ("case", "error_match"),
    (
        ("missing", "contiguous"),
        ("extra", "exceeds generation ledger"),
        ("out_of_order", "sorted by env/generation"),
        ("identity_drift", "action binding|canonical SHA"),
    ),
)
def test_diagnostic_consumed_history_load_fails_closed_atomically(
    case, error_match
):
    source, _provider, _authority = (
        _diagnostic_broker_with_counted_producers()
    )
    source_history = _consume_same_env_birth_history(source)
    saved = source.diagnostic_state_dict_with_consumed_history(source_history)
    forged = deepcopy(saved)
    forged["consumed_receipts"] = [
        receipt.to_dict()
        for receipt in _mutate_consumed_birth_history(source_history, case)
    ]
    forged["integrity_sha256"] = _integrity(forged)

    target, provider, authority = _diagnostic_broker_with_counted_producers()
    target_history = _consume_same_env_birth_history(target, generations=1)
    before_broker = target.diagnostic_state_dict_with_consumed_history(
        target_history
    )
    before_provider = deepcopy(provider.state_dict())
    before_authority = deepcopy(authority.state_dict())
    before_producer_calls = (
        provider.scalar_calls,
        provider.batch_calls,
        authority.scalar_calls,
        authority.batch_calls,
    )

    with pytest.raises(R.ActionBallContractError, match=error_match):
        target.load_state_dict(forged)

    assert target.diagnostic_state_dict_with_consumed_history(
        target_history
    ) == before_broker
    assert provider.state_dict() == before_provider
    assert authority.state_dict() == before_authority
    assert (
        provider.scalar_calls,
        provider.batch_calls,
        authority.scalar_calls,
        authority.batch_calls,
    ) == before_producer_calls


def test_broker_load_rejects_pending_sampler_transcript_forgery():
    source, _ = _broker(1)
    first = _reserve(source, env_id=0)
    _consume(source, first)
    second = _reserve(source, env_id=0, generation=2)
    saved = deepcopy(source.state_dict())

    behind = deepcopy(saved)
    behind["last_sampler_birth_indices"][0][1] = 0
    behind["last_sampler_draw_ends"][0][1] = (
        R.SAMPLER_BIRTH_DRAW_COUNT
    )
    behind["integrity_sha256"] = _integrity(behind)
    target, _ = _broker(1)
    before = deepcopy(target.state_dict())
    with pytest.raises(
        R.ActionBallContractError,
        match="counters disagree|exceeds broker high-water",
    ):
        target.load_state_dict(behind)
    assert target.state_dict() == before

    replay = deepcopy(saved)
    forged = replace(second, env_id=1, reset_generation=1)
    replay["last_generations"].append([1, 1])
    replay["last_generations"].sort()
    replay["pending"].append(
        {
            "env_id": 1,
            "status": "reserved",
            "receipt": forged.to_dict(),
        }
    )
    replay["pending"].sort(key=lambda row: row["env_id"])
    replay["integrity_sha256"] = _integrity(replay)
    with pytest.raises(
        R.ActionBallContractError, match="replays one sampler birth"
    ):
        target.load_state_dict(replay)
    assert target.state_dict() == before


def test_broker_load_rejects_deleted_consumed_birth_history():
    source, _ = _broker(1)
    first = _reserve(source, env_id=0)
    second = _reserve(source, env_id=1)
    _commit(source, first)
    _commit(source, second)
    source.consume_many_true_reset((_claim(first), _claim(second)))
    saved = deepcopy(source.state_dict())

    # Keep the provider/domain state and their final high-water at birth 1,
    # but delete env 1's complete runtime assignment and generation ledgers.
    # All remaining JSON is self-consistent and is re-signed by the attacker.
    forged = deepcopy(saved)
    forged["last_generations"] = [
        row for row in forged["last_generations"] if row[0] != 1
    ]
    forged["consumed_generations"] = [
        row for row in forged["consumed_generations"] if row[0] != 1
    ]
    forged["consumed_receipts"] = [
        row for row in forged["consumed_receipts"] if row["env_id"] != 1
    ]
    forged["integrity_sha256"] = _integrity(forged)

    target, _ = _broker(1)
    before = deepcopy(target.state_dict())
    with pytest.raises(
        R.ActionBallContractError,
        match="does not exhaust provider-issued birth indices",
    ):
        target.load_state_dict(forged)
    assert target.state_dict() == before


@pytest.mark.parametrize("count", [1, 5, 93])
def test_lazy_pool_materializes_only_the_requested_action(count):
    bindings = _bindings(count)
    pool = R.LazyActionTaskPool(
        bindings, _pins(), "no_move", refill_size=3
    )
    solver = Solver()
    pool.bind_solver(solver)
    assert pool.materialized_action_uids == ()
    assert pool.state_dict()["actions"] == []

    broker, _ = _broker(count)
    fake_authority = type(
        "FakeAuthority",
        (),
        {
            "registry_sha256": broker.registry_sha256,
            "assert_consumed_birth": lambda self, birth: None,
        },
    )()
    with pytest.raises(R.ActionBallContractError, match="exact"):
        pool.bind_birth_authority(fake_authority)
    pool.bind_birth_authority(broker)
    birth = _reserve(broker, env_id=9, slot=count - 1)
    with pytest.raises(R.BirthProtocolError, match="not the env's exact"):
        pool.request(birth, swing_generation=0)
    _consume(broker, birth)
    task = pool.request(birth, swing_generation=0)
    task.assert_birth(birth)
    with pytest.raises(R.PoolProtocolError, match="exactly 1"):
        pool.request(birth, swing_generation=0)
    assert pool.materialized_action_uids == (birth.action_uid,)
    assert len(solver.requests) == 1
    assert solver.requests[0].action_uid == birth.action_uid
    assert solver.requests[0].birth == birth
    assert pool.pending_count(
        birth.action_uid, birth_sha256=birth.canonical_sha256
    ) == 2
    assert len(pool.state_dict()["actions"]) == 1


def test_request_many_uses_one_vectorized_callback_for_n93_births():
    count = 93
    broker, _ = _broker(count)
    reserve_requests = tuple(
        _reserve_claim(
            broker,
            env_id=env_id,
            slot=env_id,
        )
        for env_id in range(count)
    )
    births = broker.reserve_many_true_reset(reserve_requests)
    broker.commit_many_true_reset(
        tuple(
            R.BirthCommitRequest(
                birth.env_id,
                birth.reset_generation,
                birth.canonical_sha256,
            )
            for birth in births
        )
    )
    broker.consume_many_true_reset(tuple(_claim(birth) for birth in births))

    pool = R.LazyActionTaskPool(
        _bindings(count), _pins(), "no_move"
    )
    solver = Solver()
    pool.bind_solver(solver)
    pool.bind_birth_authority(broker)
    tasks = pool.request_many(
        tuple(
            R.ActionTaskIssueRequest(birth, 0) for birth in births
        )
    )
    assert len(tasks) == count
    assert solver.batch_calls == 1
    assert pool.materialized_action_uids == tuple(
        sorted(birth.action_uid for birth in births)
    )
    assert all(task.swing_generation == 0 for task in tasks)


def test_formal_batch_solver_snapshots_do_not_scale_with_birth_count():
    request_state_calls = []
    retire_state_calls = []
    for birth_count in (1, 8, 32):
        pool, solver, births = _formal_pool_batch(birth_count)
        solver.reset_state_calls()
        tasks = pool.request_many(
            tuple(
                R.ActionTaskIssueRequest(birth, 0)
                for birth in births
            )
        )
        assert len(tasks) == birth_count
        request_state_calls.append(solver.state_calls)

        solver.reset_state_calls()
        assert pool.retire_many(births) == (0,) * birth_count
        retire_state_calls.append(solver.state_calls)

    assert len(set(request_state_calls)) == 1
    assert request_state_calls[0] > 0
    assert retire_state_calls == [2, 2, 2]


@pytest.mark.parametrize("raise_after_mutation", [False, True])
def test_request_many_rolls_back_mid_batch_transcript_fault(
    raise_after_mutation,
):
    pool, solver, births = _formal_pool_batch(4)
    before_solver = deepcopy(solver.state_dict())
    solver.transcript_fault_at = 2
    solver.raise_after_transcript_mutation = raise_after_mutation
    expected_error = (
        RuntimeError
        if raise_after_mutation
        else R.ActionBallContractError
    )

    with pytest.raises(expected_error):
        pool.request_many(
            tuple(
                R.ActionTaskIssueRequest(birth, 0)
                for birth in births
            )
        )

    solver.transcript_fault_at = None
    assert solver.state_dict() == before_solver
    assert pool.materialized_action_uids == ()
    assert pool.state_dict()["actions"] == []


@pytest.mark.parametrize("raise_after_mutation", [False, True])
def test_retire_many_rolls_back_mid_batch_transcript_fault(
    raise_after_mutation,
):
    pool, solver, births = _formal_pool_batch(4)
    pool.request_many(
        tuple(
            R.ActionTaskIssueRequest(birth, 0)
            for birth in births
        )
    )
    before_pool = deepcopy(pool.state_dict())
    before_solver = deepcopy(solver.state_dict())
    solver.transcript_fault_at = solver.transcript_calls + 2
    solver.raise_after_transcript_mutation = raise_after_mutation
    expected_error = (
        RuntimeError
        if raise_after_mutation
        else R.ActionBallContractError
    )

    with pytest.raises(expected_error):
        pool.retire_many(births)

    solver.transcript_fault_at = None
    assert solver.state_dict() == before_solver
    assert pool.state_dict() == before_pool


def test_diagnostic_broker_skips_formal_state_and_replay_hooks():
    bindings = _bindings(1)

    class CountingProvider(BirthProvider):
        def __init__(self):
            super().__init__()
            self.state_calls = 0
            self.assert_calls = 0
            self.highwater_calls = 0

        def state_dict(self):
            self.state_calls += 1
            return super().state_dict()

        def assert_issued_birth(self, receipt):
            self.assert_calls += 1
            return super().assert_issued_birth(receipt)

        def birth_highwater_for(self, action_uid):
            self.highwater_calls += 1
            return super().birth_highwater_for(action_uid)

    class CountingAuthority(DomainAuthority):
        def __init__(self):
            super().__init__(bindings, "no_move")
            self.state_calls = 0
            self.cursor_calls = 0

        def state_dict(self):
            self.state_calls += 1
            return super().state_dict()

        def domain_cursor_for(self, action_uid):
            self.cursor_calls += 1
            return super().domain_cursor_for(action_uid)

    counts = {}
    births = {}
    for diagnostic in (False, True):
        broker = R.ActionBirthBroker(
            bindings,
            _pins(),
            "no_move",
            diagnostic_unauthorized=diagnostic,
        )
        authority = CountingAuthority()
        provider = CountingProvider()
        broker.bind_domain_claim_authority(authority)
        broker.bind_provider(provider)
        provider.state_calls = 0
        provider.assert_calls = 0
        provider.highwater_calls = 0
        authority.state_calls = 0
        authority.cursor_calls = 0

        births[diagnostic] = _reserve(broker)
        counts[diagnostic] = (
            provider.state_calls,
            provider.assert_calls,
            provider.highwater_calls,
            authority.state_calls,
            authority.cursor_calls,
        )

    assert births[False] == births[True]
    assert all(value > 0 for value in counts[False])
    assert counts[True] == (0, 0, 0, 0, 0)


def test_diagnostic_pool_matches_formal_task_without_proof_hooks():
    class CountingSolver(Solver):
        def __init__(self):
            super().__init__()
            self.state_calls = 0
            self.sample_assert_calls = 0
            self.task_assert_calls = 0
            self.assignment_assert_calls = 0

        def state_dict(self):
            self.state_calls += 1
            return super().state_dict()

        def assert_emitted_sample(self, receipt):
            self.sample_assert_calls += 1
            return super().assert_emitted_sample(receipt)

        def assert_emitted_tasks(self, receipts):
            self.task_assert_calls += 1
            return super().assert_emitted_tasks(receipts)

        def assert_proposal_assignments(self, assignments):
            self.assignment_assert_calls += 1
            return super().assert_proposal_assignments(assignments)

        def reset_proof_counts(self):
            self.state_calls = 0
            self.sample_assert_calls = 0
            self.task_assert_calls = 0
            self.assignment_assert_calls = 0

    tasks = {}
    proof_counts = {}
    fast_pool = None
    fast_birth = None
    for diagnostic in (False, True):
        broker, _provider = _broker(
            1, diagnostic_unauthorized=diagnostic
        )
        birth = _reserve(broker)
        _consume(broker, birth)
        pool = R.LazyActionTaskPool(
            _bindings(1),
            _pins(),
            "no_move",
            refill_size=1,
            diagnostic_unauthorized=diagnostic,
        )
        solver = CountingSolver()
        pool.bind_solver(solver)
        pool.bind_birth_authority(broker)
        solver.reset_proof_counts()
        tasks[diagnostic] = pool.request(
            birth, swing_generation=0
        )
        proof_counts[diagnostic] = (
            solver.state_calls,
            solver.sample_assert_calls,
            solver.task_assert_calls,
            solver.assignment_assert_calls,
        )
        if diagnostic:
            fast_pool = pool
            fast_birth = birth

    assert tasks[False] == tasks[True]
    assert all(value > 0 for value in proof_counts[False])
    assert proof_counts[True] == (0, 0, 0, 0)
    assert fast_pool.retire_birth(fast_birth) == 0
    assert fast_pool._retired_births == {}
    assert fast_pool._task_lifecycle == {}
    assert fast_pool._diagnostic_birth_by_env == {}


def test_diagnostic_pool_still_rejects_invalid_task_receipt():
    class WrongMotionSolver(Solver):
        def __call__(self, request):
            batch = super().__call__(request)
            forged = replace(
                batch.receipts[0],
                motion_sha256=_digest("wrong-motion"),
            )
            return R.ActionPoolRefillBatch(
                action_uid=batch.action_uid,
                proposed_count=batch.proposed_count,
                proposal_sample_indices=batch.proposal_sample_indices,
                receipts=(forged,),
            )

    broker, _provider = _broker(
        1, diagnostic_unauthorized=True
    )
    birth = _reserve(broker)
    _consume(broker, birth)
    pool = R.LazyActionTaskPool(
        _bindings(1),
        _pins(),
        "no_move",
        refill_size=1,
        diagnostic_unauthorized=True,
    )
    pool.bind_solver(WrongMotionSolver())
    pool.bind_birth_authority(broker)

    with pytest.raises(R.ActionBallContractError, match="action binding"):
        pool.request(birth, swing_generation=0)
    assert pool.materialized_action_uids == ()
    assert pool._diagnostic_birth_by_env == {}


def test_diagnostic_pool_fails_after_zero_support_redraw_cap():
    class ZeroSupportSolver(Solver):
        def __call__(self, request):
            return R.ActionPoolRefillBatch(
                action_uid=request.action_uid,
                proposed_count=64,
                proposal_sample_indices=tuple(range(64)),
                receipts=(),
            )

    broker, _provider = _broker(
        1, diagnostic_unauthorized=True
    )
    birth = _reserve(broker)
    _consume(broker, birth)
    pool = R.LazyActionTaskPool(
        _bindings(1),
        _pins(),
        "no_move",
        refill_size=1,
        diagnostic_unauthorized=True,
    )
    pool.bind_solver(ZeroSupportSolver())
    pool.bind_birth_authority(broker)

    with pytest.raises(
        R.PoolProtocolError,
        match="admitted no receipts",
    ):
        pool.request(birth, swing_generation=0)
    assert pool.materialized_action_uids == ()


def test_diagnostic_pool_requires_single_row_refill_and_matching_broker():
    with pytest.raises(
        R.ActionBallContractError,
        match="require refill_size=1",
    ):
        R.LazyActionTaskPool(
            _bindings(1),
            _pins(),
            "no_move",
            refill_size=2,
            diagnostic_unauthorized=True,
        )

    formal_broker, _provider = _broker(1)
    fast_pool = R.LazyActionTaskPool(
        _bindings(1),
        _pins(),
        "no_move",
        diagnostic_unauthorized=True,
    )
    fast_pool.bind_solver(Solver())
    with pytest.raises(
        R.ActionBallContractError,
        match="diagnostic modes differ",
    ):
        fast_pool.bind_birth_authority(formal_broker)


def test_diagnostic_async_resets_keep_only_live_environment_state():
    live_envs = 8
    broker, _provider = _broker(
        1, diagnostic_unauthorized=True
    )
    births = [
        _reserve(broker, env_id=env_id)
        for env_id in range(live_envs)
    ]
    for birth in births:
        _commit(broker, birth)
    broker.consume_many_true_reset(
        tuple(_claim(birth) for birth in births)
    )

    pool = R.LazyActionTaskPool(
        _bindings(1),
        _pins(),
        "no_move",
        refill_size=1,
        diagnostic_unauthorized=True,
    )
    pool.bind_solver(Solver())
    pool.bind_birth_authority(broker)
    pool.request_many(
        tuple(
            R.ActionTaskIssueRequest(birth, 0)
            for birth in births
        )
    )

    assert (
        len(broker._diagnostic_consumed_receipt_by_env)
        == live_envs
    )
    assert len(pool._diagnostic_birth_by_env) == live_envs
    assert len(pool._diagnostic_active_sample_sha256) == live_envs

    for generation in range(2, 7):
        retired = births[0]
        assert pool.retire_birth(retired) == 0
        replacement = _reserve(
            broker,
            env_id=retired.env_id,
            generation=generation,
        )
        _consume(broker, replacement)
        pool.request(replacement, swing_generation=0)
        births[0] = replacement

        assert (
            len(broker._diagnostic_consumed_receipt_by_env)
            == live_envs
        )
        assert len(pool._diagnostic_birth_by_env) == live_envs
        assert len(pool._diagnostic_active_sample_sha256) == live_envs
        assert pool._retired_births == {}
        assert pool._task_lifecycle == {}


def test_same_action_concurrent_births_have_independent_subqueues():
    broker, _ = _broker(1)
    first = _reserve(broker, env_id=0)
    second = _reserve(broker, env_id=1)
    _commit(broker, first)
    _commit(broker, second)
    broker.consume_many_true_reset((_claim(first), _claim(second)))
    pool = R.LazyActionTaskPool(
        _bindings(1), _pins(), "no_move", refill_size=2
    )
    solver = Solver()
    pool.bind_solver(solver)
    pool.bind_birth_authority(broker)

    first_task = pool.request(first, swing_generation=0)
    second_task = pool.request(second, swing_generation=0)
    first_task.assert_birth(first)
    second_task.assert_birth(second)
    assert first_task.birth_sha256 != second_task.birth_sha256
    assert len(solver.requests) == 2
    assert {
        request.birth.canonical_sha256 for request in solver.requests
    } == {first.canonical_sha256, second.canonical_sha256}
    assert pool.pending_count(
        first.action_uid, birth_sha256=first.canonical_sha256
    ) == 1
    assert pool.pending_count(
        second.action_uid, birth_sha256=second.canonical_sha256
    ) == 1

    assert pool.retire_birth(first) == 1
    assert pool.pending_count(first.action_uid) == 1
    ledger = pool.ledger(first.action_uid)
    assert ledger.issued == 2
    assert ledger.discarded == 1
    with pytest.raises(R.PoolProtocolError, match="already retired"):
        pool.retire_birth(first)
    with pytest.raises(R.PoolProtocolError, match="retired/stale"):
        pool.request(first, swing_generation=0)


def test_same_action_round_interleaved_vectorized_samples_are_valid():
    broker, _ = _broker(1)
    first = _reserve(broker, env_id=0)
    second = _reserve(broker, env_id=1)
    for birth in (first, second):
        _commit(broker, birth)
    broker.consume_many_true_reset((_claim(first), _claim(second)))
    pool = R.LazyActionTaskPool(
        _bindings(1), _pins(), "no_move", refill_size=2
    )
    solver = RoundInterleavedSolver()
    pool.bind_solver(solver)
    pool.bind_birth_authority(broker)

    tasks = pool.request_many(
        (
            R.ActionTaskIssueRequest(first, 0),
            R.ActionTaskIssueRequest(second, 0),
        )
    )
    assert tuple(task.sample_index for task in tasks) == (0, 1)
    assert solver.batch_calls == 1
    action_state = pool.state_dict()["actions"][0]
    assert action_state["last_sample_index"] == 3
    assert action_state["last_sample_draw_end"] == (
        1_000 + 4 * R.SAMPLER_SAMPLE_DRAW_COUNT
    )


def test_retire_many_is_atomic_and_old_birth_can_retire_after_new_consume():
    broker, _ = _broker(1)
    first = _reserve(broker, env_id=0)
    second = _reserve(broker, env_id=1)
    third = _reserve(broker, env_id=2)
    for birth in (first, second, third):
        _commit(broker, birth)
    broker.consume_many_true_reset(
        (_claim(first), _claim(second), _claim(third))
    )
    pool = R.LazyActionTaskPool(
        _bindings(1), _pins(), "no_move", refill_size=2
    )
    pool.bind_solver(Solver())
    pool.bind_birth_authority(broker)
    pool.request(first, swing_generation=0)
    pool.request(second, swing_generation=0)

    before = deepcopy(pool.state_dict())
    with pytest.raises(R.PoolProtocolError, match="unknown"):
        pool.retire_many((first, third))
    assert pool.state_dict() == before

    next_first = _reserve(broker, env_id=0, generation=2)
    _consume(broker, next_first)
    assert pool.retire_birth(first) == 1
    pool.request(next_first, swing_generation=0).assert_birth(next_first)
    assert pool.retire_many((second, next_first)) == (1, 1)


def test_pool_resume_allows_active_n_after_broker_consumed_n_plus_one():
    broker, _ = _broker(1)
    first = _reserve(broker, env_id=0)
    _consume(broker, first)
    source = R.LazyActionTaskPool(
        _bindings(1), _pins(), "no_move", refill_size=2
    )
    source.bind_solver(Solver())
    source.bind_birth_authority(broker)
    source.request(first, swing_generation=0)
    second = _reserve(broker, env_id=0, generation=2)
    _consume(broker, second)
    broker_state = deepcopy(broker.state_dict())
    pool_state = deepcopy(source.state_dict())

    restored_broker, _ = _broker(1)
    restored_broker.load_state_dict(broker_state)
    restored = R.LazyActionTaskPool(
        _bindings(1), _pins(), "no_move", refill_size=2
    )
    restored.bind_solver(Solver())
    restored.bind_birth_authority(restored_broker)
    restored.load_state_dict(pool_state)
    assert restored.retire_birth(first) == 1
    restored.request(second, swing_generation=0).assert_birth(second)


def test_lazy_pool_exact_resume_preserves_birth_order_cursor_and_ledger():
    broker, _ = _broker(5)
    birth_a = _reserve(broker, env_id=0, slot=3)
    birth_b = _reserve(broker, env_id=1, slot=3)
    _commit(broker, birth_a)
    _commit(broker, birth_b)
    broker.consume_many_true_reset((_claim(birth_a), _claim(birth_b)))

    source = R.LazyActionTaskPool(
        _bindings(5), _pins(), "no_move", refill_size=4
    )
    source_solver = Solver()
    source.bind_solver(source_solver)
    source.bind_birth_authority(broker)
    source.request(birth_a, swing_generation=0)
    source.request(birth_b, swing_generation=0)
    source.request(birth_a, swing_generation=1)
    saved = deepcopy(source.state_dict())
    json.dumps(saved, allow_nan=False)

    restored = R.LazyActionTaskPool(
        _bindings(5), _pins(), "no_move", refill_size=4
    )
    restored.bind_solver(Solver())
    restored.bind_birth_authority(broker)
    restored.load_state_dict(saved)
    assert restored.state_dict() == saved
    assert restored.request(
        birth_b, swing_generation=1
    ) == source.request(birth_b, swing_generation=1)
    assert restored.request(
        birth_a, swing_generation=2
    ) == source.request(birth_a, swing_generation=2)
    assert restored.request(
        birth_a, swing_generation=3
    ) == source.request(birth_a, swing_generation=3)
    # The next issue crosses the saved pending boundary and refills.  Exact
    # solver state in the pool envelope must reproduce it, not merely the
    # already-buffered prefix.
    assert restored.request(
        birth_a, swing_generation=4
    ) == source.request(birth_a, swing_generation=4)
    assert restored.state_dict() == source.state_dict()


def test_counter_rally_task_identity_survives_exact_pool_resume():
    objective_sha256 = _digest("counter-rally-objective")
    pins = _pins(objective_sha256)
    broker, _ = _broker(1, pins=pins)
    birth = _reserve(broker)
    _consume(broker, birth)

    source = R.LazyActionTaskPool(
        _bindings(1), pins, "no_move", refill_size=2
    )
    source.bind_solver(CounterRallySolver(objective_sha256))
    source.bind_birth_authority(broker)
    first = source.request(birth, swing_generation=0)
    first.require_counter_rally_task(
        expected_objective_profile_sha256=objective_sha256
    )
    saved = deepcopy(source.state_dict())
    json.dumps(saved, allow_nan=False)

    restored = R.LazyActionTaskPool(
        _bindings(1), pins, "no_move", refill_size=2
    )
    restored.bind_solver(CounterRallySolver(objective_sha256))
    restored.bind_birth_authority(broker)
    restored.load_state_dict(saved)
    assert restored.state_dict() == saved
    actual = restored.request(birth, swing_generation=1)
    expected = source.request(birth, swing_generation=1)
    assert actual == expected
    identity = actual.require_counter_rally_task(
        expected_objective_profile_sha256=objective_sha256
    )
    assert identity == _counter_rally_task_identity(objective_sha256)
    assert restored.state_dict() == source.state_dict()


def test_counter_rally_exact_resume_rejects_solver_objective_drift_atomically():
    objective_a = _digest("counter-rally-objective-a")
    objective_b = _digest("counter-rally-objective-b")
    pins = _pins(objective_a)
    broker, _ = _broker(1, pins=pins)
    birth = _reserve(broker)
    _consume(broker, birth)

    source = R.LazyActionTaskPool(
        _bindings(1), pins, "no_move", refill_size=1
    )
    source.bind_solver(CounterRallySolver(objective_a))
    source.bind_birth_authority(broker)
    source.request(birth, swing_generation=0)
    saved = deepcopy(source.state_dict())

    restored = R.LazyActionTaskPool(
        _bindings(1), pins, "no_move", refill_size=1
    )
    restored_solver = CounterRallySolver(objective_b)
    restored.bind_solver(restored_solver)
    restored.bind_birth_authority(broker)
    restored.load_state_dict(saved)
    before_pool = deepcopy(restored.state_dict())
    before_solver = deepcopy(restored_solver.state_dict())
    with pytest.raises(
        R.CounterRallyTaskIdentityError,
        match="objective profile SHA mismatch",
    ):
        restored.request(birth, swing_generation=1)
    assert restored.state_dict() == before_pool
    assert restored_solver.state_dict() == before_solver


def test_pool_load_rejects_cross_birth_proposal_segment_reclassification():
    broker, _ = _broker(1)
    first = _reserve(broker, env_id=0)
    second = _reserve(broker, env_id=1)
    _commit(broker, first)
    _commit(broker, second)
    broker.consume_many_true_reset((_claim(first), _claim(second)))
    source = R.LazyActionTaskPool(
        _bindings(1), _pins(), "no_move", refill_size=2
    )
    source.bind_solver(RoundInterleavedSolver())
    source.bind_birth_authority(broker)
    source.request_many(
        (
            R.ActionTaskIssueRequest(first, 0),
            R.ActionTaskIssueRequest(second, 0),
        )
    )
    forged = deepcopy(source.state_dict())
    births = forged["actions"][0]["births"]
    first_segments = births[0]["sample_assignments"][0][
        "proposal_index_segments"
    ]
    births[0]["sample_assignments"][0][
        "proposal_index_segments"
    ] = births[1]["sample_assignments"][0][
        "proposal_index_segments"
    ]
    births[1]["sample_assignments"][0][
        "proposal_index_segments"
    ] = first_segments
    forged["integrity_sha256"] = _integrity(forged)

    restored = R.LazyActionTaskPool(
        _bindings(1), _pins(), "no_move", refill_size=2
    )
    restored.bind_solver(RoundInterleavedSolver())
    restored.bind_birth_authority(broker)
    before = restored.state_dict()
    with pytest.raises(
        (R.ActionBallContractError, ValueError),
        match="assignment|pending|birth",
    ):
        restored.load_state_dict(forged)
    assert restored.state_dict() == before


def test_pool_load_rejects_single_sided_solver_assignment_reclassification():
    broker, _ = _broker(1)
    birth = _reserve(broker)
    _consume(broker, birth)
    source = R.LazyActionTaskPool(
        _bindings(1), _pins(), "no_move", refill_size=2
    )
    source.bind_solver(Solver())
    source.bind_birth_authority(broker)
    source.request(birth, swing_generation=0)
    forged = deepcopy(source.state_dict())
    assignment = forged["solver_state"]["proposal_assignments"][0]
    assignment[2] = _digest("wrong-birth")
    forged["solver_state_sha256"] = R._sha256_json(
        forged["solver_state"]
    )
    forged["integrity_sha256"] = _integrity(forged)

    restored = R.LazyActionTaskPool(
        _bindings(1), _pins(), "no_move", refill_size=2
    )
    restored.bind_solver(Solver())
    restored.bind_birth_authority(broker)
    before = restored.state_dict()
    with pytest.raises(
        (R.ActionBallContractError, ValueError),
        match="assignment|birth",
    ):
        restored.load_state_dict(forged)
    assert restored.state_dict() == before


def test_lazy_pool_load_rejects_cross_bound_birth_authority_atomically():
    source_broker, _ = _broker(1)
    birth = _reserve(source_broker)
    _consume(source_broker, birth)
    source = R.LazyActionTaskPool(
        _bindings(1), _pins(), "no_move", refill_size=2
    )
    source.bind_solver(Solver())
    source.bind_birth_authority(source_broker)
    source.request(birth, swing_generation=0)
    saved = deepcopy(source.state_dict())

    fresh_broker, _ = _broker(1)
    restored = R.LazyActionTaskPool(
        _bindings(1), _pins(), "no_move", refill_size=2
    )
    restored_solver = Solver()
    restored.bind_solver(restored_solver)
    restored.bind_birth_authority(fresh_broker)
    before = deepcopy(restored.state_dict())
    with pytest.raises(
        R.ActionBallContractError,
        match="birth authority state mismatch",
    ):
        restored.load_state_dict(saved)
    assert restored.state_dict() == before
    assert restored_solver.state_dict() == {
        "sequence": 0,
        "emitted_samples": [],
        "emitted_tasks": [],
        "proposal_assignments": [],
        "highwaters": [],
    }

    newer = _reserve(fresh_broker, generation=1)
    _consume(fresh_broker, newer)
    newer = _reserve(fresh_broker, generation=2)
    _consume(fresh_broker, newer)
    before = deepcopy(restored.state_dict())
    with pytest.raises(
        R.ActionBallContractError,
        match="birth authority state mismatch",
    ):
        restored.load_state_dict(saved)
    assert restored.state_dict() == before
    assert restored_solver.state_dict() == {
        "sequence": 0,
        "emitted_samples": [],
        "emitted_tasks": [],
        "proposal_assignments": [],
        "highwaters": [],
    }


def test_empty_pool_checkpoint_still_pins_full_birth_authority_state():
    broker, _ = _broker(1)
    birth = _reserve(broker)
    _consume(broker, birth)
    source = R.LazyActionTaskPool(
        _bindings(1), _pins(), "no_move"
    )
    source.bind_solver(Solver())
    source.bind_birth_authority(broker)
    source.request(birth, swing_generation=0)
    source.retire_birth(birth)
    saved = deepcopy(source.state_dict())
    assert source.materialized_action_uids == ()

    unrelated = _reserve(broker, env_id=1)
    _consume(broker, unrelated)
    restored = R.LazyActionTaskPool(
        _bindings(1), _pins(), "no_move"
    )
    restored_solver = Solver()
    restored.bind_solver(restored_solver)
    restored.bind_birth_authority(broker)
    before = deepcopy(restored.state_dict())
    with pytest.raises(
        R.ActionBallContractError,
        match="birth authority state mismatch",
    ):
        restored.load_state_dict(saved)
    assert restored.state_dict() == before
    assert restored_solver.state_dict() == {
        "sequence": 0,
        "emitted_samples": [],
        "emitted_tasks": [],
        "proposal_assignments": [],
        "highwaters": [],
    }


def test_retired_pool_load_rejects_forged_sample_highwater():
    broker, _ = _broker(1)
    birth = _reserve(broker)
    _consume(broker, birth)
    source = R.LazyActionTaskPool(
        _bindings(1), _pins(), "no_move", refill_size=2
    )
    source.bind_solver(Solver())
    source.bind_birth_authority(broker)
    source.request(birth, swing_generation=0)
    source.retire_birth(birth)
    forged = deepcopy(source.state_dict())
    forged["actions"][0]["last_sample_index"] = 999
    forged["actions"][0]["last_sample_draw_end"] = 1_000_000
    forged["integrity_sha256"] = _integrity(forged)

    restored = R.LazyActionTaskPool(
        _bindings(1), _pins(), "no_move", refill_size=2
    )
    restored.bind_solver(Solver())
    restored.bind_birth_authority(broker)
    before = deepcopy(restored.state_dict())
    with pytest.raises(
        R.ActionBallContractError,
        match="lifecycle|differs from solver authority",
    ):
        restored.load_state_dict(forged)
    assert restored.state_dict() == before


def test_lazy_pool_load_is_atomic_and_rejects_nested_tampering():
    broker, _ = _broker(1)
    birth = _reserve(broker)
    _consume(broker, birth)
    pool = R.LazyActionTaskPool(
        _bindings(1), _pins(), "no_move", refill_size=2
    )
    pool.bind_solver(Solver())
    pool.bind_birth_authority(broker)
    pool.request(birth, swing_generation=0)
    saved = deepcopy(pool.state_dict())
    before = deepcopy(pool.state_dict())

    corrupt = deepcopy(saved)
    corrupt["actions"][0]["births"][0]["cursor"] += 1
    with pytest.raises(R.ActionBallContractError, match="integrity"):
        pool.load_state_dict(corrupt)
    assert pool.state_dict() == before

    forged = deepcopy(saved)
    forged["actions"][0]["births"][0]["pending_receipts"][0][
        "incoming_velocity_w_mps"
    ][0] = -99.0
    forged["integrity_sha256"] = _integrity(forged)
    with pytest.raises(
        R.ActionBallContractError,
        match="velocity|canonical SHA",
    ):
        pool.load_state_dict(forged)
    assert pool.state_dict() == before

    forged_samples = deepcopy(saved)
    birth_row = forged_samples["actions"][0]["births"][0]
    pending_sample = birth_row["pending_receipts"][0]["sample_sha256"]
    birth_row["seen_sample_sha256"].remove(pending_sample)
    birth_row["seen_sample_sha256"].append(_digest("forged-sample"))
    birth_row["seen_sample_sha256"].sort()
    forged_samples["integrity_sha256"] = _integrity(forged_samples)
    with pytest.raises(
        R.ActionBallContractError,
        match="pending receipt sample is missing",
    ):
        pool.load_state_dict(forged_samples)
    assert pool.state_dict() == before


def test_lazy_pool_rejects_replayed_sampler_sample_atomically():
    broker, _ = _broker(1)
    birth = _reserve(broker)
    _consume(broker, birth)
    pool = R.LazyActionTaskPool(
        _bindings(1), _pins(), "no_move", refill_size=2
    )
    solver = ReplayedSampleSolver()
    pool.bind_solver(solver)
    pool.bind_birth_authority(broker)

    with pytest.raises(
        R.ActionBallContractError,
        match=(
            "sample index|sample SHA|reused one sampler sample|"
            "proposal sample indices"
        ),
    ):
        pool.request(birth, swing_generation=0)
    assert solver.state_dict() == {
        "sequence": 0,
        "emitted_samples": [],
        "emitted_tasks": [],
        "proposal_assignments": [],
        "highwaters": [],
    }
    assert pool.materialized_action_uids == ()
    assert pool.state_dict()["actions"] == []


def test_solver_authority_assertion_must_be_pure_and_rolls_back():
    broker, _ = _broker(1)
    birth = _reserve(broker)
    _consume(broker, birth)
    pool = R.LazyActionTaskPool(
        _bindings(1), _pins(), "no_move", refill_size=2
    )
    solver = MutatingAssertionSolver()
    pool.bind_solver(solver)
    pool.bind_birth_authority(broker)

    with pytest.raises(
        R.ActionBallContractError, match="authority assertion must be pure"
    ):
        pool.request(birth, swing_generation=0)
    assert solver.state_dict() == {
        "sequence": 0,
        "emitted_samples": [],
        "emitted_tasks": [],
        "proposal_assignments": [],
        "highwaters": [],
    }
    assert pool.materialized_action_uids == ()


@pytest.mark.parametrize("raise_after_mutation", [False, True])
def test_pool_state_read_rolls_back_mutating_highwater_authority(
    raise_after_mutation,
):
    pool = R.LazyActionTaskPool(
        _bindings(1), _pins(), "no_move", refill_size=1
    )
    solver = MutatingHighwaterSolver(
        raise_after_mutation=raise_after_mutation
    )
    pool.bind_solver(solver)
    before = deepcopy(solver.state_dict())
    expected_error = (
        RuntimeError
        if raise_after_mutation
        else R.ActionBallContractError
    )
    with pytest.raises(expected_error):
        pool.state_dict()
    assert solver.state_dict() == before


@pytest.mark.parametrize("birth_count", [1, 2])
def test_refill_rejects_cross_action_sample_tape_side_effect_atomically(
    birth_count,
):
    broker, _ = _broker(2)
    births = tuple(
        _reserve(broker, env_id=env_id, slot=0)
        for env_id in range(birth_count)
    )
    for birth in births:
        _commit(broker, birth)
    broker.consume_many_true_reset(
        tuple(_claim(birth) for birth in births)
    )
    pool = R.LazyActionTaskPool(
        _bindings(2), _pins(), "no_move", refill_size=1
    )
    solver = CrossActionMutationSolver(
        untouched_uid=broker.ordered_action_uids[1]
    )
    pool.bind_solver(solver)
    pool.bind_birth_authority(broker)
    before_pool = deepcopy(pool.state_dict())
    before_solver = deepcopy(solver.state_dict())

    with pytest.raises(
        R.ActionBallContractError,
        match="unstaged action sample tape",
    ):
        pool.request_many(
            tuple(
                R.ActionTaskIssueRequest(birth, 0)
                for birth in births
            )
        )
    assert pool.state_dict() == before_pool
    assert solver.state_dict() == before_solver


def test_issued_but_rejected_tail_cannot_be_admitted_later():
    broker, _ = _broker(1)
    birth = _reserve(broker)
    _consume(broker, birth)
    pool = R.LazyActionTaskPool(
        _bindings(1), _pins(), "no_move", refill_size=1
    )
    solver = RejectedTailReplaySolver()
    pool.bind_solver(solver)
    pool.bind_birth_authority(broker)
    pool.request(birth, swing_generation=0)
    before_pool = deepcopy(pool.state_dict())
    before_solver = deepcopy(solver.state_dict())

    with pytest.raises(
        R.ActionBallContractError, match="index replayed/went backwards"
    ):
        pool.request(birth, swing_generation=1)
    assert pool.state_dict() == before_pool
    assert solver.state_dict() == before_solver


def test_request_many_rejects_cross_birth_sample_replay_atomically():
    broker, _ = _broker(2)
    first = _reserve(broker, env_id=0, slot=0)
    second = _reserve(broker, env_id=1, slot=1)
    _commit(broker, first)
    _commit(broker, second)
    broker.consume_many_true_reset((_claim(first), _claim(second)))
    pool = R.LazyActionTaskPool(_bindings(2), _pins(), "no_move")
    solver = CrossBirthReplayedSampleSolver()
    pool.bind_solver(solver)
    pool.bind_birth_authority(broker)

    with pytest.raises(
        R.ActionBallContractError,
        match="sample SHA|reused one sampler sample",
    ):
        pool.request_many(
            (
                R.ActionTaskIssueRequest(first, 0),
                R.ActionTaskIssueRequest(second, 0),
            )
        )
    assert solver.state_dict() == {
        "sequence": 0,
        "emitted_samples": [],
        "emitted_tasks": [],
        "proposal_assignments": [],
        "highwaters": [],
    }
    assert pool.materialized_action_uids == ()
    assert pool.state_dict()["actions"] == []


def test_move_mode_is_frozen_and_no_move_task_cannot_smuggle_travel():
    broker, _ = _broker(1, mode="move")
    moving = _reserve(broker)
    with pytest.raises(R.ActionBallContractError, match="explicit per-swing"):
        _task(moving, 99)
    moving_a = _task(
        moving,
        0,
        base_goal_w_m=(
            moving.base_spawn_w_m[0],
            moving.base_spawn_w_m[1] + 0.1,
            moving.base_spawn_w_m[2],
        ),
    )
    moving_b = _task(
        moving,
        1,
        base_goal_w_m=(
            moving.base_spawn_w_m[0],
            moving.base_spawn_w_m[1] - 0.2,
            moving.base_spawn_w_m[2],
        ),
    )
    moving_a.assert_birth(moving)
    moving_b.assert_birth(moving)
    assert moving_a.base_goal_w_m != moving_b.base_goal_w_m
    drifted_goal = deepcopy(moving_a.to_dict())
    drifted_goal["base_goal_w_m"][1] += 0.25
    with pytest.raises(
        R.ActionBallContractError,
        match="base goal|canonical SHA",
    ):
        R.ActionBallTaskReceipt.from_dict(drifted_goal)
    with pytest.raises(
        R.ActionBallContractError,
        match="domain claim|disagree",
    ):
        replace(moving_a, action_uid=moving_a.action_uid + 1).assert_birth(
            moving
        )

    other = _reserve(broker, env_id=1)
    with pytest.raises(R.ActionBallContractError, match="birth SHA"):
        moving_a.assert_birth(other)

    no_move_broker, _ = _broker(1, mode="no_move")
    frozen = _reserve(no_move_broker)
    with pytest.raises(R.ActionBallContractError, match="no_move"):
        replace(
            _task(frozen, 0),
            base_goal_w_m=(frozen.base_spawn_w_m[0] + 0.1, 0.0, 0.0),
        )
    move_pool = R.LazyActionTaskPool(
        _bindings(1), _pins(), "move", refill_size=1
    )
    move_pool.bind_solver(Solver())
    move_pool.bind_birth_authority(broker)
    with pytest.raises(R.ActionBallContractError, match="frozen run mode"):
        move_pool.request(frozen, swing_generation=0)


# --- initial-center single-question collapse -------------------------------
#
# ``ActionBallSampler`` collapses the whole sampling plan to the literal profile
# centre while ``initial_center_single_question`` is on and all 32 curriculum
# arms are exactly zero.  The quota schedule no longer picks the stratum in that
# regime, so the receipt gates judge the row against the collapse law instead.
# These tests pin both directions: the collapse is accepted, and every gate that
# guards the ordinary quota regime still refuses a mislabelled row.

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CENTER_TASK_RECEIPT = _REPO_ROOT / (
    "configs/action_ball_n1_measured_20260803/"
    "fresh_tape_seed0_20260803_take061_robust20n_r9_center/"
    "current_lm.target.task_receipt.v5.f9e0ddf178aa.json"
)
_INTERIOR_TASK_RECEIPT = _REPO_ROOT / (
    "configs/action_ball_n1_measured_20260803/"
    "fresh_tape_seed0_20260803_take061_robust20n_r4_splitready/"
    "current_lm.target.task_receipt.v5.f64f52137ad8.json"
)


def _initial_center_birth_kwargs(*, initial_center_single_question):
    """Reserve one real level-zero birth from an initial-center sampler."""

    action_uid = 907_311
    profile = _sampling_profile(action_uid)
    mixture = S.SamplingMixture()
    sampler = S.ActionBallSampler(
        (profile,),
        seed=20260805,
        sampling_mixture=mixture,
        contact_time_step_s=0.02,
        initial_center_single_question=initial_center_single_question,
    )
    zero_levels = S.DomainLevels()
    levels = R.ActionDomainLevels.from_dict(zero_levels.as_dict())
    authority_sha = _digest("initial-center-authority")
    pins = R.RuntimePins(
        manifest_sha256=_digest("initial-center-manifest"),
        sampler_sha256=sampler.sampler_contract_sha256,
        domain_authority_sha256=authority_sha,
        physics_sha256=_digest("initial-center-physics"),
        solver_sha256=_digest("initial-center-solver"),
    )
    binding = R.ActionBinding(
        action_uid=action_uid,
        action_slot=0,
        motion_path="vendor_assets/motions/initial_center.npz",
        motion_sha256=_digest("initial-center-motion"),
        profile_sha256=profile.sha256,
    )
    yaw = 0.0
    sampled_birth = sampler.reserve_birth(
        action_uid=action_uid,
        domain_epoch=0,
        levels=zero_levels,
        base_yaw_rad=yaw,
    )
    claim = R.ActionDomainClaim(
        authority_contract_sha256=authority_sha,
        arm_catalog_sha256=R.ARM_CATALOG_SHA256,
        action_uid=action_uid,
        domain_epoch=0,
        domain_levels=levels,
        levels_sha256=levels.canonical_sha256,
        profile_sha256=profile.sha256,
        mobility_mode="no_move",
    )
    kwargs = dict(
        env_id=0,
        reset_generation=1,
        action_uid=action_uid,
        action_slot=0,
        domain_epoch=0,
        domain_claim_sha256=claim.canonical_sha256,
        domain_authority_sha256=authority_sha,
        domain_levels=levels,
        arm_catalog_sha256=R.ARM_CATALOG_SHA256,
        levels_sha256=levels.canonical_sha256,
        sampler_birth_sha256=sampled_birth.birth_id,
        sampler_birth_index=sampled_birth.birth_index,
        sampler_draw_start=sampled_birth.draw_start,
        sampler_draw_end=sampled_birth.draw_end,
        mobility_mode="no_move",
        base_yaw_rad=yaw,
        base_quat_wxyz=_yaw_quat(yaw),
        base_spawn_w_m=sampled_birth.base_start_w_m,
        manifest_sha256=pins.manifest_sha256,
        sampler_sha256=pins.sampler_sha256,
        profile_sha256=profile.sha256,
        motion_sha256=binding.motion_sha256,
        physics_sha256=pins.physics_sha256,
        solver_sha256=pins.solver_sha256,
        registry_sha256=R._registry_sha256((binding,), pins, "no_move"),
        sampling_mixture=R.ActionSamplingMixture.from_dict(
            sampled_birth.sampling_mixture.as_dict()
        ),
        sampling_stratum=sampled_birth.sampling_stratum,
        sampling_levels=R.ActionDomainLevels.from_dict(
            sampled_birth.sampling_levels.as_dict()
        ),
        frontier_arm=sampled_birth.frontier_arm,
        initial_center_single_question=initial_center_single_question,
    )
    return mixture, sampled_birth, kwargs


def test_initial_center_birth_receipt_accepts_the_collapse_the_quota_forbids():
    mixture, sampled_birth, kwargs = _initial_center_birth_kwargs(
        initial_center_single_question=True
    )
    # The quota slot this birth landed on is *not* centre.  Without the
    # collapse being written down, the schedule comparison is the wrong law.
    assert mixture.schedule[0] == "interior"
    assert sampled_birth.birth_index == 0
    assert sampled_birth.sampling_stratum == "center"
    assert sampled_birth.frontier_arm is None

    receipt = R.ActionBirthReceipt(**kwargs)
    assert receipt.initial_center_single_question is True
    payload = receipt.payload_dict()
    assert payload["initial_center_single_question"] is True
    assert R.ActionBirthReceipt.from_dict(receipt.to_dict()) == receipt


def test_initial_center_birth_receipt_refuses_any_non_center_row():
    _mixture, _sampled_birth, kwargs = _initial_center_birth_kwargs(
        initial_center_single_question=True
    )
    for override in (
        {"sampling_stratum": "interior"},
        {"sampling_stratum": "frontier", "frontier_arm": "base_spawn_x_lower"},
        {
            "sampling_levels": R.ActionDomainLevels.from_dict(
                {
                    **S.DomainLevels().as_dict(),
                    "contact_x_lower": 0.25,
                }
            )
        },
    ):
        with pytest.raises(
            R.ActionBallContractError,
            match="birth initial-center collapse is not the literal",
        ):
            R.ActionBirthReceipt(**{**kwargs, **override})


def test_quota_schedule_birth_gate_survives_without_the_initial_center_law():
    # Mutation control: strip the recorded collapse and the ordinary quota gate
    # must still refuse the very same centre row.
    _mixture, _sampled_birth, kwargs = _initial_center_birth_kwargs(
        initial_center_single_question=True
    )
    with pytest.raises(
        R.ActionBallContractError,
        match="birth sampling stratum differs from quota schedule",
    ):
        R.ActionBirthReceipt(
            **{**kwargs, "initial_center_single_question": False}
        )
    # A legacy row may not carry the flag at all.
    with pytest.raises(
        R.ActionBallContractError,
        match="legacy birth cannot carry mixture sampling metadata",
    ):
        R.ActionBirthReceipt(
            **{
                **kwargs,
                "sampling_mixture": None,
                "sampling_stratum": "domain",
                "sampling_levels": None,
                "frontier_arm": None,
            }
        )


def test_center_task_receipt_round_trips_and_is_the_literal_center_row():
    row = json.loads(_CENTER_TASK_RECEIPT.read_text(encoding="utf-8"))
    receipt = R.ActionBallTaskReceipt.from_dict(row)
    assert receipt.initial_center_single_question is True
    assert receipt.sampling_stratum == "center"
    assert receipt.birth_sampling_stratum == "center"
    assert receipt.frontier_arm is None and receipt.birth_frontier_arm is None
    assert receipt.time_to_contact_tick == 91
    assert receipt.time_to_contact_s == 1.82
    assert receipt.sampling_mixture.schedule[0] == "interior"
    assert all(
        getattr(receipt.domain_levels, arm) == 0.0 for arm in R.ARM_KEYS
    )
    assert receipt.to_dict() == row


def test_center_task_receipt_refuses_a_non_center_stratum():
    row = json.loads(_CENTER_TASK_RECEIPT.read_text(encoding="utf-8"))
    for key in ("sampling_stratum", "birth_sampling_stratum"):
        mutated = {**row, key: "interior"}
        with pytest.raises(
            R.ActionBallContractError,
            match="task initial-center collapse is not the literal",
        ):
            R.ActionBallTaskReceipt.from_dict(mutated)


def test_quota_schedule_task_gate_survives_without_the_initial_center_law():
    # Mutation control 1: drop the recorded collapse from the centre row.
    row = json.loads(_CENTER_TASK_RECEIPT.read_text(encoding="utf-8"))
    stripped = {
        key: value
        for key, value in row.items()
        if key != "initial_center_single_question"
    }
    with pytest.raises(
        R.ActionBallContractError,
        match="task sampling stratum differs from quota schedule",
    ):
        R.ActionBallTaskReceipt.from_dict(stripped)

    # Mutation control 2: the tracked interior receipt carries no collapse, so
    # relabelling its stratum must still be refused by the quota gate.
    interior = json.loads(_INTERIOR_TASK_RECEIPT.read_text(encoding="utf-8"))
    assert "initial_center_single_question" not in interior
    assert interior["sampling_stratum"] == "interior"
    with pytest.raises(
        R.ActionBallContractError,
        match="task sampling stratum differs from quota schedule",
    ):
        R.ActionBallTaskReceipt.from_dict(
            {**interior, "sampling_stratum": "center"}
        )
