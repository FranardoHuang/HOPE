from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

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


def _pins():
    return R.RuntimePins(
        manifest_sha256=_digest("manifest"),
        sampler_sha256=_digest("sampler"),
        domain_authority_sha256=_digest("domain-authority"),
        physics_sha256=_digest("physics"),
        solver_sha256=_digest("solver"),
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
    timing = R.derive_action_teacher_timing(
        racket_velocity_w_mps=racket_velocity,
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
    return R.ActionBallTaskReceipt.from_birth(
        birth,
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
        racket_velocity_w_mps=racket_velocity,
        racket_normal_w=(1.0, 0.0, 0.0),
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
    )


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
                _task(
                    request.birth,
                    sample_index,
                    swing_generation=(
                        request.swing_generation_start + offset
                    ),
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


def _broker(count=5, mode="no_move"):
    bindings = _bindings(count)
    broker = R.ActionBirthBroker(bindings, _pins(), mode)
    authority = DomainAuthority(bindings, mode)
    provider = BirthProvider()
    broker.bind_domain_claim_authority(authority)
    broker.bind_provider(provider)
    return broker, provider


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
        match="contact|canonical SHA",
    ):
        R.ActionBallTaskReceipt.from_dict(tampered)


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
        R.ActionBallContractError, match="exact unclipped formula"
    ):
        replace(task, teacher_rate=task.teacher_rate + 1.0e-12)
    with pytest.raises(
        R.ActionBallContractError, match="outside certified bounds"
    ):
        replace(
            task,
            racket_velocity_w_mps=(4.0, 0.0, 0.0),
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


def test_real_sampler_identity_is_bit_exact_with_runtime_receipt():
    action_uid = 12_345
    profile = _sampling_profile(action_uid)
    sampler = S.ActionBallSampler((profile,), seed=20260727)
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
    )
    sampled = sampler.sample(
        birth=sampled_birth,
        action_uid=action_uid,
        domain_epoch=7,
        levels=sampler_levels,
        base_yaw_rad=yaw,
    )
    racket_velocity = (6.0, 0.0, 0.0)
    timing = R.derive_action_teacher_timing(
        racket_velocity_w_mps=racket_velocity,
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
        racket_velocity_w_mps=racket_velocity,
        racket_normal_w=(1.0, 0.0, 0.0),
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
    )
    assert task.sample_sha256 == sampled.sample_id
    assert task.sampler_identity_receipt() == (
        sampled.to_identity_receipt()
    )
    before = deepcopy(sampler.state_dict())
    sampler.assert_issued_sample(task.sampler_identity_receipt())
    assert sampler.state_dict() == before


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
