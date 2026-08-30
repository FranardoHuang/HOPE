"""Focused CPU tests for the compact ActionEpoch observation manager ABI."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
MDP = ROOT / "source" / "whole_body_tracking" / "whole_body_tracking" / "tasks" / "tracking" / "mdp"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


E = _load("action_ball_full_mdp_epoch", MDP / "action_ball_full_mdp_epoch.py")
R = _load("action_ball_full_mdp_lean_rewards", MDP / "action_ball_full_mdp_lean_rewards.py")
L = _load("action_ball_full_mdp_lean_runtime", MDP / "action_ball_full_mdp_lean_runtime.py")
P = _load(
    "action_ball_full_mdp_portable_observation",
    MDP / "action_ball_full_mdp_portable_observation.py",
)
O = _load("action_ball_full_mdp_lean_observation_cfg", MDP / "action_ball_full_mdp_lean_observation_cfg.py")


class _Env:
    def __init__(self):
        self._action_ball_full_mdp_manager_construction_state = "runtime_graph_ready"
        self.common_step_counter = 10
        self.step_dt = 0.02

    def _action_ball_full_mdp_lean_observe_term(self, *, group):
        source = getattr(self, "_installed_lean_observation_source", None)
        if (
            type(source) is not O.LeanActionEpochObservationSource
            or source._env is not self
        ):
            raise O.LeanObservationConstructionHold(
                "installed lean observation source identity differs"
            )
        return source.observe(group)


class _R06:
    num_envs = 2
    flight_slot_capacity = 2
    device = torch.device("cpu")
    dtype = torch.float32


def _prepared_epoch():
    owner = E.ActionEpochOwner(num_envs=2, device="cpu", shot_slot_capacity=2)
    owner.activate_reset_genesis(
        selected_mask=torch.ones(2, dtype=torch.bool),
        reset_generation=torch.zeros(2, dtype=torch.int64),
    )
    return owner


def _runtime(env, epoch):
    parts = {
        "r05_runtime": object(),
        "motion": types.SimpleNamespace(
            _action_ball_continuous_motion_mutation_version=0
        ),
        "racket": object(),
        "physical_ball": object(),
        "r06_landing_outcome": _R06(),
        "r03_strike_fact": object(),
        "r07_recovery": object(),
    }
    owner = L.ActionBallFullMdpLeanRuntimeOwner(
        env=env,
        runtime_lease=object(),
        epoch_owner=epoch,
        reward_graph=R.LeanActionEpochRewardGraph(epoch_owner=epoch),
        **parts,
    )
    return owner, parts


def _direct_view(env, record, parts):
    value = 1.0
    actor_rows = {}
    for name, width in O.ACTOR_LAYOUT_V3:
        if name in (
            "motion_phase_one_hot",
            "epoch_learning_phase_one_hot",
            "task_valid",
        ):
            continue
        actor_rows[name] = torch.full((2, width), value)
        value += 1.0
    critic_rows = {}
    for name, width in O.CRITIC_EXTENSION_LAYOUT_V3:
        critic_rows[name] = torch.full((2, width), value)
        value += 1.0
    task_valid = O._gather_selected(
        record.task.task_valid, record.current_task_slot
    )
    return O.DirectActionEpochObservationFacts(
        actor_rows=actor_rows,
        critic_rows=critic_rows,
        motion_phase_code=torch.tensor([1, 4], dtype=torch.int64),
        task_valid=task_valid,
        transaction_epoch=record.epoch,
        transaction_version=record.version,
        common_step=env.common_step_counter,
    )


def _hard_coded_actor_scale_v2():
    groups = (
        (1.0, 3),
        (0.25, 3),
        (1.0, 3),
        (1.0, 2),
        (0.5, 3),
        (1.0, 31),
        (0.05, 31),
        (1.0, 31),
        (1.0, 31),
        (0.05, 31),
        (10.0 / 3.0, 3),
        (1.0, 6),
        (1.0, 5),
        (5.0, 3),
        (1.0, 3),
        (2.0, 3),
        (5.0, 2),
        (1.0 / 2.42, 1),
        (1.0, 1),
        (1.0 / 5.86, 1),
        (1.0, 5),
        (1.0, 1),
    )
    return tuple(scale for scale, width in groups for _ in range(width))


def _hard_coded_actor_scale_v3():
    groups = (
        (1.0, 3),
        (0.25, 3),
        (1.0, 3),
        (1.0, 2),
        (0.5, 3),
        (1.0, 31),
        (0.05, 31),
        (1.0, 31),
        (1.0, 31),
        (0.05, 31),
        (10.0 / 3.0, 3),
        (1.0, 6),
        (1.0, 5),
        (5.0, 3),
        (1.0, 3),
        (2.0, 3),
        (2.0, 3),
        (5.0, 3),
        (1.0, 3),
        (2.0, 3),
        (5.0, 2),
        (1.0 / 2.42, 1),
        (1.0, 1),
        (1.0 / 5.86, 1),
        (1.0, 5),
        (1.0, 1),
    )
    return tuple(scale for scale, width in groups for _ in range(width))


def _hard_coded_critic_extension_scale_v2():
    groups = (
        (1.0 / 30.0, 1),
        (1.0, 3),
        (0.1, 3),
        (1.0 / 60.0, 3),
        (1.0, 1),
        (1.0, 1),
        (1.0, 1),
        (1.0, 2),
        (1.0, 1),
    )
    return tuple(scale for scale, width in groups for _ in range(width))


def test_v2_history_is_preserved_and_live_v3_is_semantic_215_231():
    assert P.COMMON_ACTOR_WIDTH_V2 == 183
    assert O.ACTOR_WIDTH_V2 == 203
    assert O.CRITIC_WIDTH_V2 == 219
    assert O.ACTOR_LAYOUT_V2 is P.ACTOR_LAYOUT_V2
    assert O.CRITIC_EXTENSION_LAYOUT_V2 is P.CRITIC_EXTENSION_LAYOUT_V2
    assert [name for name, _ in O.ACTOR_LAYOUT_V2][-9:] == [
        "racket_target_pos_error_heading",
        "racket_target_vel_error_heading",
        "racket_target_normal_error_heading",
        "base_goal_error_heading_xy",
        "time_to_contact_s",
        "time_to_teacher_start_s",
        "time_to_next_opportunity_s",
        "epoch_learning_phase_one_hot",
        "task_valid",
    ]
    assert sum(width for _, width in O.CRITIC_EXTENSION_LAYOUT_V2) == 16
    assert P.COMMON_ACTOR_WIDTH_V3 == 195
    assert O.ACTOR_WIDTH_V3 == 215
    assert O.CRITIC_WIDTH_V3 == 231
    assert O.ACTOR_CONTRACT_V3 == "action_ball_full_mdp_semantic_actor_v3"
    assert O.CRITIC_CONTRACT_V3 == "action_ball_full_mdp_semantic_critic_v3"
    assert O.OBSERVATION_KIND_V3 == "action_ball_full_mdp_semantic_observation_v3"
    assert O.ACTOR_LAYOUT_V3 is P.ACTOR_LAYOUT_V3
    assert O.CRITIC_EXTENSION_LAYOUT_V3 is P.CRITIC_EXTENSION_LAYOUT_V3
    assert [name for name, _ in P.COMMON_ACTOR_LAYOUT_V3][-4:] == [
        "motion_racket_pos_error_heading",
        "motion_racket_vel_error_heading",
        "motion_racket_signed_normal_error_heading",
        "motion_racket_long_axis_error_heading",
    ]
    assert P.ACTOR_LAYOUT_V3[-9:] == P.ACTOR_TASK_LAYOUT_V2
    assert O.DIRECT_VIEW_METHOD == "semantic_action_epoch_observation_v3"
    assert callable(getattr(L.ActionBallFullMdpLeanRuntimeOwner, O.DIRECT_VIEW_METHOD))
    assert not hasattr(
        L.ActionBallFullMdpLeanRuntimeOwner,
        "semantic_action_epoch_observation_v2",
    )
    assert sum(width for _, width in O.CRITIC_EXTENSION_LAYOUT_V3) == 16


def test_v2_static_scale_keys_and_flat_values_match_independent_golden():
    assert P.ACTOR_SCALE_BY_FIELD_V2 == (
        ("projected_gravity_b", 1.0),
        ("base_ang_vel_b", 0.25),
        ("base_position_table", 1.0),
        ("base_heading_table_xy", 1.0),
        ("base_com_lin_vel_heading", 0.5),
        ("joint_pos_rel", 1.0),
        ("joint_vel", 0.05),
        ("last_action", 1.0),
        ("teacher_joint_pos_rel", 1.0),
        ("teacher_joint_vel", 0.05),
        ("motion_anchor_pos_b", 10.0 / 3.0),
        ("motion_anchor_ori_b6", 1.0),
        ("motion_phase_one_hot", 1.0),
        ("racket_target_pos_error_heading", 5.0),
        ("racket_target_vel_error_heading", 1.0),
        ("racket_target_normal_error_heading", 2.0),
        ("base_goal_error_heading_xy", 5.0),
        ("time_to_contact_s", 1.0 / 2.42),
        ("time_to_teacher_start_s", 1.0),
        ("time_to_next_opportunity_s", 1.0 / 5.86),
        ("epoch_learning_phase_one_hot", 1.0),
        ("task_valid", 1.0),
    )
    assert P.CRITIC_EXTENSION_SCALE_BY_FIELD_V2 == (
        ("episode_time_remaining_s", 1.0 / 30.0),
        ("live_ball_center_rel_root_heading", 1.0),
        ("live_ball_lin_vel_heading", 0.1),
        ("live_ball_ang_vel_heading", 1.0 / 60.0),
        ("selected_rubber_contact_latched", 1.0),
        ("net_crossed_latched", 1.0),
        ("net_clear_latched", 1.0),
        ("foot_supported_lr", 1.0),
        ("cadence_ready_dwell_fraction", 1.0),
    )
    assert P.ACTOR_SCALE_FLAT_V2 == _hard_coded_actor_scale_v2()
    assert (
        P.CRITIC_EXTENSION_SCALE_FLAT_V2
        == _hard_coded_critic_extension_scale_v2()
    )
    assert len(P.ACTOR_SCALE_FLAT_V2) == 203
    assert len(P.CRITIC_EXTENSION_SCALE_FLAT_V2) == 16


def test_v3_static_scale_and_critic_prefix_match_independent_golden():
    assert P.MOTION_RACKET_RESIDUAL_SCALE_BY_FIELD_V3 == (
        ("motion_racket_pos_error_heading", 5.0),
        ("motion_racket_vel_error_heading", 1.0),
        ("motion_racket_signed_normal_error_heading", 2.0),
        ("motion_racket_long_axis_error_heading", 2.0),
    )
    assert P.ACTOR_SCALE_FLAT_V3 == _hard_coded_actor_scale_v3()
    assert (
        P.CRITIC_EXTENSION_SCALE_FLAT_V3
        == _hard_coded_critic_extension_scale_v2()
    )
    assert len(P.ACTOR_SCALE_FLAT_V3) == 215
    assert len(P.CRITIC_EXTENSION_SCALE_FLAT_V3) == 16


def test_portable_keeps_complete_legacy_v1_surface_until_atomic_cutover():
    direct = (
        ("projected_gravity_b", 3),
        ("base_ang_vel_b", 3),
        ("joint_pos_rel", 31),
        ("joint_vel_rel", 31),
        ("last_action", 31),
        ("teacher_joint_pos_rel", 31),
        ("teacher_joint_vel_rel", 31),
    )
    actor = direct + (
        ("motion_phase_one_hot", 5),
        ("epoch_task_f32", 45),
        ("epoch_clock_remaining_s", 5),
        ("epoch_phase_one_hot", 10),
        ("epoch_task_valid", 1),
        ("epoch_selected", 1),
        ("epoch_launch_succeeded", 1),
    )
    critic = (
        ("physical_r03_r06_r07_fact_present", 4),
        ("physical_r03_r06_r07_fact_age_s", 4),
        ("physical_r03_r06_r07_fact_f32", 128),
        ("physical_r03_r06_r07_fault_present", 4),
        ("reward_cycle_open", 1),
        ("reward_cycle_fault_present", 1),
        ("reward_due", 14),
        ("reward_paid", 14),
    )
    assert P.TASK_F32_WIDTH == 45
    assert P.OWNER_FACT_F32_WIDTH == 32
    assert P.REWARD_CONSUMER_COUNT == 14
    assert P.EPOCH_IDLE_PHASE_INDEX == 0
    assert P.ACTOR_CONTRACT_V1 == "action_ball_full_mdp_action_epoch_v1"
    assert P.CRITIC_CONTRACT_V1 == "action_ball_full_mdp_action_epoch_critic_v1"
    assert P.OBSERVATION_KIND_V1 == "action_ball_full_mdp_action_epoch_observation_v1"
    assert P.DIRECT_FIELD_LAYOUT_V1 == direct
    assert P.ACTOR_LAYOUT_V1 == actor
    assert P.CRITIC_EXTENSION_LAYOUT_V1 == critic
    assert P.ACTOR_WIDTH_V1 == 229
    assert P.CRITIC_WIDTH_V1 == 399
    raw_rows = {
        name: torch.full((1, width), float(index + 2))
        for index, (name, width) in enumerate(direct)
    }
    torch.testing.assert_close(
        P.concatenate_layout_rows(P.DIRECT_FIELD_LAYOUT_V1, raw_rows),
        torch.cat(
            (
                raw_rows["projected_gravity_b"],
                raw_rows["base_ang_vel_b"],
                raw_rows["joint_pos_rel"],
                raw_rows["joint_vel_rel"],
                raw_rows["last_action"],
                raw_rows["teacher_joint_pos_rel"],
                raw_rows["teacher_joint_vel_rel"],
            ),
            dim=1,
        ),
    )
    assert {
        "ACTOR_LAYOUT_V1",
        "CRITIC_EXTENSION_LAYOUT_V1",
        "ACTOR_WIDTH_V1",
        "CRITIC_WIDTH_V1",
        "concatenate_layout_rows",
    }.issubset(P.__all__)


def test_portable_frame_transforms_match_independent_ninety_degree_golden():
    half = 2.0 ** -0.5
    yaw_90 = torch.tensor([[half, 0.0, 0.0, half]])
    torch.testing.assert_close(
        P.heading_xy_from_quat_wxyz(yaw_90),
        torch.tensor([[0.0, 1.0]]),
        atol=1.0e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        P.rotate_world_to_heading(yaw_90, torch.tensor([[1.0, 0.0, 3.0]])),
        torch.tensor([[0.0, -1.0, 3.0]]),
        atol=1.0e-6,
        rtol=0.0,
    )
    position, orientation = P.relative_pose_6d(
        torch.tensor([[1.0, 2.0, 0.0]]),
        yaw_90,
        torch.tensor([[2.0, 2.0, 0.0]]),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    )
    torch.testing.assert_close(
        position, torch.tensor([[0.0, -1.0, 0.0]]), atol=1.0e-6, rtol=0.0
    )
    torch.testing.assert_close(
        orientation,
        torch.tensor([[0.0, 1.0, -1.0, 0.0, 0.0, 0.0]]),
        atol=1.0e-6,
        rtol=0.0,
    )


def test_tilted_heading_is_unit_and_near_vertical_heading_is_canonical():
    yaw_45_pitch_60 = torch.tensor(
        [[0.80010315, -0.19134172, 0.46193977, 0.33141357]]
    )
    expected_heading = torch.tensor([[0.70710678, 0.70710678]])
    torch.testing.assert_close(
        P.heading_xy_from_quat_wxyz(yaw_45_pitch_60),
        expected_heading,
        atol=1.0e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        P.rotate_world_to_heading(
            yaw_45_pitch_60, torch.tensor([[1.0, 0.0, 2.0]])
        ),
        torch.tensor([[0.70710678, -0.70710678, 2.0]]),
        atol=1.0e-6,
        rtol=0.0,
    )
    near_vertical = torch.tensor([[0.70710678, 0.0, 0.70710678, 0.0]])
    mixed = torch.cat((yaw_45_pitch_60, near_vertical), dim=0)
    heading = P.heading_xy_from_quat_wxyz(mixed)
    torch.testing.assert_close(
        heading,
        torch.cat((expected_heading, torch.tensor([[1.0, 0.0]])), dim=0),
        atol=1.0e-6,
        rtol=0.0,
    )
    assert torch.isfinite(heading).all()
    torch.testing.assert_close(
        torch.linalg.vector_norm(heading, dim=-1),
        torch.ones(2),
        atol=1.0e-6,
        rtol=0.0,
    )


def test_new_root_translation_yaw_and_com_velocity_break_old_229_aliases():
    env = _Env()
    epoch = _prepared_epoch()
    runtime, parts = _runtime(env, epoch)
    source = O.LeanActionEpochObservationSource(env=env, runtime_owner=runtime)
    record = epoch.current()
    baseline_view = _direct_view(env, record, parts)
    baseline, _ = source._pack(baseline_view, record, common_step=10)

    def mutate(name, value):
        return O.DirectActionEpochObservationFacts(
            **{
                **baseline_view.__dict__,
                "actor_rows": {**baseline_view.actor_rows, name: value},
            }
        )

    translated, _ = source._pack(
        mutate(
            "base_position_table",
            torch.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]),
        ),
        record,
        common_step=10,
    )
    torch.testing.assert_close(
        translated[:, 6:9],
        torch.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]),
    )
    torch.testing.assert_close(translated[:, :6], baseline[:, :6])
    torch.testing.assert_close(translated[:, 9:], baseline[:, 9:])

    yawed, _ = source._pack(
        mutate("base_heading_table_xy", torch.tensor([[0.0, 1.0], [1.0, 0.0]])),
        record,
        common_step=10,
    )
    torch.testing.assert_close(
        yawed[:, 9:11], torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    )
    torch.testing.assert_close(yawed[:, :9], baseline[:, :9])
    torch.testing.assert_close(yawed[:, 11:], baseline[:, 11:])

    moving, _ = source._pack(
        mutate(
            "base_com_lin_vel_heading",
            torch.tensor([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        ),
        record,
        common_step=10,
    )
    torch.testing.assert_close(
        moving[:, 11:14],
        torch.tensor([[2.0, 2.5, 3.0], [3.5, 4.0, 4.5]]),
    )
    torch.testing.assert_close(moving[:, :11], baseline[:, :11])
    torch.testing.assert_close(moving[:, 14:], baseline[:, 14:])


def test_direct_builder_reads_live_ball_support_and_dwell_without_old_facts(
    monkeypatch,
):
    env = _Env()
    env.episode_length_buf = torch.tensor([10, 20], dtype=torch.int64)
    env.max_episode_length = 100
    env.scene = types.SimpleNamespace(env_origins=torch.zeros((2, 3)))
    env.action_manager = types.SimpleNamespace(action=torch.zeros((2, 31)))
    epoch = _prepared_epoch()
    record = epoch.current()
    record.phase[0, 0] = E.PHASE_LAUNCH_SETTLED
    record.task.task_valid[0, 0] = True
    # Epoch retains the completed payload until the next ACCEPT.  Motion has
    # already hidden it during the inter-reveal recovery interval below.
    record.phase[1, 0] = E.PHASE_RETIRED
    record.task.task_valid[1, 0] = True
    current_key_values = {
        "reset_generation": 1,
        "ball_generation": 7,
        "action_uid": 101,
        "action_slot": 0,
        "shot_index": 1,
        "task_identity": 11,
        "outcome_identity": 12,
        "ball_identity": 13,
    }
    for name, value in current_key_values.items():
        getattr(record.identity.shot_key, name)[0, 0] = value
    record.publication_ordinal[0, 0] = 17

    class Motion:
        pass

    commands = types.ModuleType("commands")
    commands.MotionCommand = Motion
    table_tennis = types.ModuleType("whole_body_tracking.tasks.table_tennis")
    table_tennis.geometry = types.SimpleNamespace(TABLE_LENGTH=2.74)
    monkeypatch.setitem(sys.modules, "commands", commands)
    monkeypatch.setitem(
        sys.modules, "whole_body_tracking.tasks.table_tennis", table_tennis
    )

    data = types.SimpleNamespace(
        projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]).repeat(2, 1),
        root_ang_vel_b=torch.zeros((2, 3)),
        root_pos_w=torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        root_quat_w=torch.tensor(
            [[2.0 ** -0.5, 0.0, 0.0, 2.0 ** -0.5], [1.0, 0.0, 0.0, 0.0]]
        ),
        root_lin_vel_w=torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
        joint_pos=torch.zeros((2, 31)),
        default_joint_pos=torch.zeros((2, 31)),
        joint_vel=torch.zeros((2, 31)),
    )
    motion = Motion()
    motion._action_ball_continuous_motion_mutation_version = 0
    motion.robot = types.SimpleNamespace(data=data)
    motion.joint_pos = torch.zeros((2, 31))
    motion.joint_vel = torch.zeros((2, 31))
    motion.robot_anchor_pos_w = data.root_pos_w
    motion.robot_anchor_quat_w = data.root_quat_w
    motion.anchor_pos_w = data.root_pos_w
    motion.anchor_quat_w = data.root_quat_w
    motion_view = types.SimpleNamespace(
        motion_owner=motion,
        common_step=10,
        phase=torch.tensor([0, 4], dtype=torch.int64),
        task_valid=torch.tensor([True, False]),
        # Motion's analytic task clocks are intentionally float64.  The semantic
        # observation boundary, not Motion state, owns the policy ABI cast.
        time_to_contact_remaining_s=torch.tensor(
            [0.242, 0.3], dtype=torch.float64
        ),
        time_to_teacher_start_remaining_s=torch.tensor(
            [0.125, 0.2], dtype=torch.float64
        ),
        time_to_next_reveal_s=torch.tensor(
            [0.586, 1.172], dtype=torch.float64
        ),
        control_tick=torch.tensor([10, 10], dtype=torch.int64),
        reset_generation=torch.tensor([1, 1], dtype=torch.int64),
        action_uid=torch.tensor([101, -1], dtype=torch.int64),
    )
    token = object()
    motion.action_ball_continuous_motion_observation_projection = lambda: token
    motion.require_owned_action_ball_continuous_motion_observation = (
        lambda value: motion_view if value is token else None
    )

    actor_target_pos_w = torch.tensor(
        [[2.0, 0.0, 1.0], [3.0, 4.0, 5.0]]
    )
    actor_target_vel_w = torch.tensor(
        [[0.0, 2.0, 0.0], [6.0, 7.0, 8.0]]
    )
    actor_target_normal_w = torch.tensor(
        [[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0]]
    )
    motion_target_pos_w = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    )
    motion_target_signed_normal_w = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    motion_target_vel_w = torch.tensor(
        [[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]
    )
    motion_target_long_axis_w = torch.tensor(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    hope_rewards = types.ModuleType("hope_rewards")
    hope_rewards.stage1_aligned_clip_racket_target_now = lambda _cmd: (
        motion_target_pos_w,
        motion_target_signed_normal_w,
        motion_target_vel_w,
        motion_target_long_axis_w,
    )
    hope_rewards.stage1_actual_racket_face_normal_now = (
        lambda cmd: cmd.racket_normal_w
    )
    monkeypatch.setitem(sys.modules, "hope_rewards", hope_rewards)
    racket = types.SimpleNamespace(
        cfg=types.SimpleNamespace(
            vb_table_near_x=0.0,
            vb_table_surface_z=0.76,
            face_command=False,
        ),
        # The live controller target deliberately differs from the delayed
        # actor-visible command.  V3 must consume the latter while keeping the
        # measured racket pose/velocity/normal live.
        racket_target_pos_w=torch.ones((2, 3)),
        racket_pos_w=torch.zeros((2, 3)),
        racket_target_vel_w=torch.ones((2, 3)),
        racket_lin_vel_w=torch.zeros((2, 3)),
        racket_normal_raw_w=torch.tensor([[0.0, 1.0, 0.0]]).repeat(2, 1),
        # The signed normal deliberately points the other way.  The full-phase
        # mimic residual uses this physical face, while the ball-task question
        # stays aligned with the raw A/+Y question and R03 reward.
        racket_normal_w=torch.tensor([[0.0, -1.0, 0.0]]).repeat(2, 1),
        racket_long_axis_w=torch.tensor([[1.0, 0.0, 0.0]]).repeat(2, 1),
        target_normal_cmd=torch.tensor([[1.0, 0.0, 0.0]]).repeat(2, 1),
        base_target_pos_w=torch.ones((2, 2)),
        actor_racket_target_pos_w=lambda: actor_target_pos_w,
        actor_racket_target_vel_w=lambda: actor_target_vel_w,
        actor_target_normal_cmd=lambda: actor_target_normal_w,
    )
    r06_token = object()
    r06 = _R06()
    flight = types.SimpleNamespace(
        r06_owner=r06,
        publication_identity=r06_token,
        # The R06 semantic projection has already selected the current typed
        # ActionEpoch row.  Slot 1 is only a Physical scene locator.
        flight_slot=torch.tensor([1, -1], dtype=torch.int64),
        contact_valid=torch.tensor([True, False]),
        net_crossed=torch.tensor([True, False]),
        net_clear=torch.tensor([True, False]),
    )
    r06.action_ball_full_mdp_observation_projection = lambda: r06_token

    def current_flight(
        value, *, current_shot_key, current_publication_ordinal
    ):
        assert value is r06_token
        for name in current_key_values:
            torch.testing.assert_close(
                getattr(current_shot_key, name),
                O._gather_selected(
                    getattr(record.identity.shot_key, name),
                    record.current_task_slot,
                ),
            )
        torch.testing.assert_close(
            current_publication_ordinal,
            O._gather_selected(
                record.publication_ordinal, record.current_task_slot
            ),
        )
        return flight

    r06.require_owned_action_epoch_current_flight_observation = current_flight
    state = torch.zeros((2, 2, 13))
    state[0, 1, :3] = torch.tensor([1.0, 0.0, 1.0])
    state[0, 1, 7:10] = torch.tensor([1.0, 0.0, 0.0])
    state[0, 1, 10:13] = torch.tensor([0.0, 1.0, 0.0])
    physical = types.SimpleNamespace(
        scene_port=types.SimpleNamespace(read_state_env=lambda: state)
    )
    ready = types.SimpleNamespace(
        postphysics_valid=torch.ones(2, dtype=torch.bool),
        source_step=torch.tensor([9, 9], dtype=torch.int64),
        reset_generation=torch.tensor([1, 1], dtype=torch.int64),
        control_tick=torch.tensor([10, 10], dtype=torch.int64),
        ready_streak=torch.tensor([1, 3], dtype=torch.int64),
        required_dwell=2,
        foot_supported_lr=torch.tensor(
            [[True, False], [True, True]], dtype=torch.bool
        ),
    )
    r07 = types.SimpleNamespace(
        plant_fact_adapter=types.SimpleNamespace(
            read=lambda: (_ for _ in ()).throw(
                AssertionError("Observation V3 reread the R07 plant adapter")
            )
        ),
        action_epoch_observation_state=lambda: ready,
    )
    runtime = L.ActionBallFullMdpLeanRuntimeOwner(
        env=env,
        runtime_lease=object(),
        epoch_owner=epoch,
        reward_graph=R.LeanActionEpochRewardGraph(epoch_owner=epoch),
        r05_runtime=object(),
        motion=motion,
        racket=racket,
        physical_ball=physical,
        r06_landing_outcome=r06,
        r03_strike_fact=object(),
        r07_recovery=r07,
    )
    view = O.build_direct_action_epoch_observation_facts(
        runtime_owner=runtime, record=record
    )
    torch.testing.assert_close(
        view.actor_rows["motion_racket_pos_error_heading"],
        torch.tensor([[2.0, -1.0, 3.0], [4.0, 5.0, 6.0]]),
        atol=1.0e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        view.actor_rows["motion_racket_vel_error_heading"],
        torch.tensor([[4.0, -3.0, 5.0], [6.0, 7.0, 8.0]]),
        atol=1.0e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        view.actor_rows["motion_racket_signed_normal_error_heading"],
        torch.tensor([[1.0, -1.0, 0.0], [0.0, 1.0, 1.0]]),
        atol=1.0e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        view.actor_rows["motion_racket_long_axis_error_heading"],
        torch.tensor([[1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]]),
        atol=1.0e-6,
        rtol=0.0,
    )
    # The hidden-task row still observes its full-phase mimic error.  These
    # blocks are not task leakage and must never be zeroed by task visibility.
    for name, _ in P.MOTION_RACKET_RESIDUAL_LAYOUT_V3:
        assert view.actor_rows[name][1].ne(0).any(), name
    torch.testing.assert_close(
        view.actor_rows["racket_target_pos_error_heading"],
        torch.tensor([[0.0, -2.0, 1.0], [0.0, 0.0, 0.0]]),
        atol=1.0e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        view.actor_rows["racket_target_vel_error_heading"],
        torch.tensor([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        atol=1.0e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        view.actor_rows["racket_target_normal_error_heading"],
        torch.tensor([[-2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        atol=1.0e-6,
        rtol=0.0,
    )
    assert not torch.equal(
        view.actor_rows["motion_racket_signed_normal_error_heading"][0],
        view.actor_rows["racket_target_normal_error_heading"][0],
    )
    torch.testing.assert_close(
        view.critic_rows["live_ball_center_rel_root_heading"],
        torch.tensor([[0.0, -1.0, 0.0], [0.0, 0.0, 0.0]]),
        atol=1.0e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        view.critic_rows["live_ball_lin_vel_heading"],
        torch.tensor([[0.0, -1.0, 0.0], [0.0, 0.0, 0.0]]),
        atol=1.0e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        view.critic_rows["live_ball_ang_vel_heading"],
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        atol=1.0e-6,
        rtol=0.0,
    )
    assert view.critic_rows["foot_supported_lr"].tolist() == [[1.0, 0.0], [1.0, 1.0]]
    assert view.critic_rows["cadence_ready_dwell_fraction"].tolist() == [[0.5], [1.0]]
    for name in (
        "racket_target_pos_error_heading",
        "racket_target_vel_error_heading",
        "racket_target_normal_error_heading",
        "base_goal_error_heading_xy",
        "time_to_contact_s",
        "time_to_teacher_start_s",
    ):
        assert view.actor_rows[name][1].eq(0).all()
    for name in (
        "time_to_contact_s",
        "time_to_teacher_start_s",
        "time_to_next_opportunity_s",
    ):
        assert view.actor_rows[name].dtype == torch.float32
    torch.testing.assert_close(
        view.actor_rows["time_to_contact_s"],
        torch.tensor([[0.242], [0.0]], dtype=torch.float32),
    )
    torch.testing.assert_close(
        view.actor_rows["time_to_teacher_start_s"],
        torch.tensor([[0.125], [0.0]], dtype=torch.float32),
    )
    torch.testing.assert_close(
        view.actor_rows["time_to_next_opportunity_s"],
        torch.tensor([[0.586], [1.172]], dtype=torch.float32),
    )
    source = O.LeanActionEpochObservationSource(env=env, runtime_owner=runtime)
    policy, critic = source._pack(view, record, common_step=10)
    assert policy.shape == (2, 215) and policy.dtype == torch.float32
    assert critic.shape == (2, 231) and critic.dtype == torch.float32
    assert torch.isfinite(policy).all() and torch.isfinite(critic).all()
    torch.testing.assert_close(
        policy[:, 206:209],
        torch.tensor([[0.1, 0.125, 0.1], [0.0, 0.0, 0.2]]),
    )
    torch.testing.assert_close(critic[:, :215], policy)
    # Task-dependent fields are masked, while next-opportunity and RETIRED
    # phase remain observable and the actor-visible validity bit is false.
    assert policy[1, 183:195].ne(0).any()
    assert policy[1, 195:208].eq(0).all()
    assert policy[1, 208].item() == pytest.approx(1.172 / 5.86)
    torch.testing.assert_close(
        policy[1, 209:214], torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0])
    )
    assert policy[1, 214].item() == 0.0

    # R06 owns its latch hierarchy and scene-slot range.  A malformed row is
    # consumed as no live flight, while the independent peer remains exact.
    r06_row_names = (
        "live_ball_center_rel_root_heading",
        "live_ball_lin_vel_heading",
        "live_ball_ang_vel_heading",
        "selected_rubber_contact_latched",
        "net_crossed_latched",
        "net_clear_latched",
    )
    for case, slot_value, contact_value, crossed_value, clear_value in (
        ("absent-retained", -1, True, True, True),
        ("slot-outside", 2, True, True, True),
        ("clear-before-crossing", 1, True, False, True),
        ("crossing-before-contact", 1, False, True, False),
    ):
        flight.flight_slot[0] = slot_value
        flight.contact_valid[0] = contact_value
        flight.net_crossed[0] = crossed_value
        flight.net_clear[0] = clear_value
        faulty_view = O.build_direct_action_epoch_observation_facts(
            runtime_owner=runtime, record=record
        )
        for name in r06_row_names:
            assert faulty_view.critic_rows[name][0].eq(0).all(), (case, name)
        faulty_policy, faulty_critic = source._pack(
            faulty_view, record, common_step=10
        )
        assert torch.equal(faulty_policy[1], policy[1]), case
        assert torch.equal(faulty_critic[1], critic[1]), case
    flight.flight_slot[0] = 1
    flight.contact_valid[0] = True
    flight.net_crossed[0] = True
    flight.net_clear[0] = True

    # Motion/Epoch UID mismatch invalidates only the task-conditioned actor
    # question.  Robot state and the healthy peer are not collateral damage.
    motion_view.action_uid[0] = 102
    uid_fault_view = O.build_direct_action_epoch_observation_facts(
        runtime_owner=runtime, record=record
    )
    assert not uid_fault_view.task_valid[0]
    for name in (
        "racket_target_pos_error_heading",
        "racket_target_vel_error_heading",
        "racket_target_normal_error_heading",
        "base_goal_error_heading_xy",
        "time_to_contact_s",
        "time_to_teacher_start_s",
    ):
        assert uid_fault_view.actor_rows[name][0].eq(0).all(), name
    uid_fault_policy, uid_fault_critic = source._pack(
        uid_fault_view, record, common_step=10
    )
    assert torch.equal(uid_fault_policy[1], policy[1])
    assert torch.equal(uid_fault_critic[1], critic[1])
    assert torch.equal(
        uid_fault_view.actor_rows["time_to_next_opportunity_s"][0],
        view.actor_rows["time_to_next_opportunity_s"][0],
    )
    motion_view.action_uid[0] = 101

    # A true selected reset advances only row 0's Motion generation after the
    # real post-physics publication above.  Its first returned observation is
    # finite and reset-safe; the untouched peer remains byte-identical.
    peer_policy = policy[1].clone()
    peer_critic = critic[1].clone()
    motion_view.reset_generation[0] = 2
    motion_view.control_tick[0] = 0
    env.episode_length_buf[0] = 0
    ready.ready_streak[0] = 2
    ready.foot_supported_lr[0].fill_(True)
    reset_view = O.build_direct_action_epoch_observation_facts(
        runtime_owner=runtime, record=record
    )
    reset_policy, reset_critic = source._pack(
        reset_view, record, common_step=10
    )
    assert torch.isfinite(reset_policy).all()
    assert torch.isfinite(reset_critic).all()
    assert reset_critic[0, 228:230].eq(0).all()
    assert reset_critic[0, 230].item() == 0.0
    assert torch.equal(reset_policy[1], peer_policy)
    assert torch.equal(reset_critic[1], peer_critic)

    # The next real post-physics publication carries the new generation and
    # restores row 0 without disturbing its peer.
    ready.reset_generation[0] = 2
    ready.control_tick[0] = 0
    ready.source_step[0] = 10
    ready.ready_streak[0] = 1
    ready.foot_supported_lr[0] = torch.tensor([True, False])
    resumed_view = O.build_direct_action_epoch_observation_facts(
        runtime_owner=runtime, record=record
    )
    assert resumed_view.critic_rows["foot_supported_lr"][0].tolist() == [1.0, 0.0]
    assert resumed_view.critic_rows["cadence_ready_dwell_fraction"][0].item() == 0.5
    resumed_policy, resumed_critic = source._pack(
        resumed_view, record, common_step=10
    )

    # R07 only stamps the pre-reset cadence it observed.  Motion is the
    # independent chronology authority, so a replayed same-generation stamp
    # contributes no support/dwell instead of poisoning a later CUDA launch.
    ready.control_tick[0] -= 1
    replayed_view = O.build_direct_action_epoch_observation_facts(
        runtime_owner=runtime, record=record
    )
    assert replayed_view.critic_rows["foot_supported_lr"][0].eq(0).all()
    assert replayed_view.critic_rows["cadence_ready_dwell_fraction"][0].eq(0).all()
    replayed_policy, replayed_critic = source._pack(
        replayed_view, record, common_step=10
    )
    assert torch.equal(replayed_policy[1], resumed_policy[1])
    assert torch.equal(replayed_critic[1], resumed_critic[1])
    ready.control_tick[0] += 1

    # Cold genesis has no post-physics capability.  Its explicit invalid/zero
    # R07 state is the one allowed no-publication case and remains finite.
    ready.postphysics_valid.zero_()
    ready.source_step.fill_(-1)
    ready.reset_generation.fill_(-1)
    ready.control_tick.fill_(-1)
    ready.ready_streak.zero_()
    ready.foot_supported_lr.zero_()
    motion_view.reset_generation.zero_()
    motion_view.control_tick.zero_()
    genesis_view = O.build_direct_action_epoch_observation_facts(
        runtime_owner=runtime, record=record
    )
    genesis_policy, genesis_critic = source._pack(
        genesis_view, record, common_step=10
    )
    assert torch.isfinite(genesis_policy).all()
    assert torch.isfinite(genesis_critic).all()
    assert genesis_critic[:, 228:231].eq(0).all()

    # MAX may not wrap into a reset boundary and hide stale chronology.  The
    # malformed row is zeroed and the valid peer remains byte-identical.
    ready.postphysics_valid.fill_(True)
    ready.source_step.fill_(10)
    ready.reset_generation[:] = torch.tensor(
        [torch.iinfo(torch.int64).max, 1], dtype=torch.int64
    )
    ready.control_tick.fill_(0)
    motion_view.reset_generation[:] = torch.tensor(
        [torch.iinfo(torch.int64).min, 1], dtype=torch.int64
    )
    wrapped_view = O.build_direct_action_epoch_observation_facts(
        runtime_owner=runtime, record=record
    )
    assert wrapped_view.critic_rows["foot_supported_lr"][0].eq(0).all()
    assert wrapped_view.critic_rows["cadence_ready_dwell_fraction"][0].eq(0).all()
    wrapped_policy, wrapped_critic = source._pack(
        wrapped_view, record, common_step=10
    )
    assert torch.equal(wrapped_policy[1], genesis_policy[1])
    assert torch.equal(wrapped_critic[1], genesis_critic[1])

    ready.reset_generation.fill_(1)
    motion_view.reset_generation.fill_(1)

    r06.flight_slot_capacity = 1
    with pytest.raises(O.LeanObservationError, match="scene ABI differs"):
        O.build_direct_action_epoch_observation_facts(
            runtime_owner=runtime, record=record
        )


def test_missing_exact_direct_runtime_method_holds_before_manager_import(monkeypatch):
    env = _Env()
    runtime, _ = _runtime(env, _prepared_epoch())
    monkeypatch.delattr(
        L.ActionBallFullMdpLeanRuntimeOwner,
        O.DIRECT_VIEW_METHOD,
        raising=True,
    )
    monkeypatch.setattr(O.importlib, "import_module", lambda _name: (_ for _ in ()).throw(AssertionError("must hold first")))
    with pytest.raises(O.LeanObservationConstructionHold, match=O.DIRECT_VIEW_METHOD):
        O.materialize_observation_manager_cfg(env=env, runtime_owner=runtime)


def test_shape_probe_then_semantic_pack_reads_current_public_epoch(monkeypatch):
    env = _Env()
    epoch = _prepared_epoch()
    runtime, parts = _runtime(env, epoch)
    calls = []
    current_calls = []
    original_current = E.ActionEpochOwner.current

    def counted_current(owner):
        assert owner is epoch
        current_calls.append(owner.commit_head)
        return original_current(owner)

    monkeypatch.setattr(E.ActionEpochOwner, "current", counted_current)

    def direct(self, record):
        calls.append(record.version)
        return _direct_view(env, record, parts)

    monkeypatch.setattr(L.ActionBallFullMdpLeanRuntimeOwner, O.DIRECT_VIEW_METHOD, direct, raising=False)

    class Group:
        pass

    class Term:
        def __init__(self, *, func, params):
            self.func, self.params = func, params

    monkeypatch.setattr(
        O.importlib,
        "import_module",
        lambda name: types.SimpleNamespace(ObservationGroupCfg=Group, ObservationTermCfg=Term) if name == "isaaclab.managers" else None,
    )
    bundle = O.materialize_observation_manager_cfg(
        env=env, runtime_owner=runtime
    )
    assert type(bundle) is O.DiagnosticN2ObservationManagerBundle
    actor_scale_ptr = bundle.source._actor_scale_v3.data_ptr()
    critic_scale_ptr = bundle.source._critic_extension_scale_v3.data_ptr()
    torch.testing.assert_close(
        bundle.source._actor_scale_v3,
        torch.tensor(_hard_coded_actor_scale_v3())[None, :],
    )
    torch.testing.assert_close(
        bundle.source._critic_extension_scale_v3,
        torch.tensor(_hard_coded_critic_extension_scale_v2())[None, :],
    )
    env._installed_lean_observation_source = bundle.source
    cfg = bundle.manager_cfg
    policy_term = cfg["policy"].action_epoch
    critic_term = cfg["critic"].action_epoch
    assert policy_term.params == {"group": "policy"}
    assert critic_term.params == {"group": "critic"}
    copied = copy.deepcopy(cfg)
    assert copied["policy"].action_epoch.params == {"group": "policy"}
    assert copied["critic"].action_epoch.params == {"group": "critic"}
    assert policy_term.func(env, **policy_term.params).shape == (2, 215)
    assert critic_term.func(env, **critic_term.params).shape == (2, 231)
    assert calls == []

    env._action_ball_full_mdp_manager_construction_state = "base_managers_complete"
    policy = policy_term.func(env, **policy_term.params)
    critic = critic_term.func(env, **critic_term.params)
    assert bundle.source._actor_scale_v3.data_ptr() == actor_scale_ptr
    assert bundle.source._critic_extension_scale_v3.data_ptr() == critic_scale_ptr
    assert calls == [0]
    assert current_calls == [1]
    assert policy.shape == (2, 215) and torch.all(torch.isfinite(policy))
    assert critic.shape == (2, 231) and torch.all(torch.isfinite(critic))

    # A real same-step Epoch commit invalidates the cache even though this
    # zero fault row does not change any packed semantic value.
    r03_owner = parts["r03_strike_fact"]
    epoch.bind_fact_owner("r03_strike_fact", r03_owner)
    epoch.merge_runtime_owner_fault(
        "r03_strike_fact", torch.zeros((2, 2), dtype=torch.int64), owner=r03_owner
    )
    same_step_policy = policy_term.func(env, **policy_term.params)
    torch.testing.assert_close(same_step_policy, policy)
    assert calls == [0, 1]
    assert current_calls == [1, 2]

    # RecorderManager may request an observation before Motion advances in the
    # same environment step.  Motion generation, not instrumentation order,
    # must invalidate that earlier whole-observation cache.
    parts["motion"]._action_ball_continuous_motion_mutation_version += 1
    same_step_after_motion = policy_term.func(env, **policy_term.params)
    torch.testing.assert_close(same_step_after_motion, policy)
    assert calls == [0, 1, 1]
    assert current_calls == [1, 2, 2]

    # A new control step rebuilds once even if Epoch itself has not mutated.
    env.common_step_counter += 1
    next_step_critic = critic_term.func(env, **critic_term.params)
    torch.testing.assert_close(next_step_critic, critic)
    assert calls == [0, 1, 1, 1]
    assert current_calls == [1, 2, 2, 2]

    # Independent hard-coded golden: expected order is written here rather
    # than generated from the shared layout under test.
    view = _direct_view(env, original_current(epoch), parts)
    actor = view.actor_rows
    expected_policy_raw = torch.cat(
        (
            actor["projected_gravity_b"],
            actor["base_ang_vel_b"],
            actor["base_position_table"],
            actor["base_heading_table_xy"],
            actor["base_com_lin_vel_heading"],
            actor["joint_pos_rel"],
            actor["joint_vel"],
            actor["last_action"],
            actor["teacher_joint_pos_rel"],
            actor["teacher_joint_vel"],
            actor["motion_anchor_pos_b"],
            actor["motion_anchor_ori_b6"],
            torch.nn.functional.one_hot(
                torch.tensor([1, 4]), num_classes=5
            ).float(),
            actor["motion_racket_pos_error_heading"],
            actor["motion_racket_vel_error_heading"],
            actor["motion_racket_signed_normal_error_heading"],
            actor["motion_racket_long_axis_error_heading"],
            actor["racket_target_pos_error_heading"],
            actor["racket_target_vel_error_heading"],
            actor["racket_target_normal_error_heading"],
            actor["base_goal_error_heading_xy"],
            actor["time_to_contact_s"],
            actor["time_to_teacher_start_s"],
            actor["time_to_next_opportunity_s"],
            torch.tensor([[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]]).float(),
            torch.zeros((2, 1)),
        ),
        dim=1,
    )
    expected_policy = expected_policy_raw * torch.tensor(
        _hard_coded_actor_scale_v3()
    )[None, :]
    torch.testing.assert_close(policy, expected_policy)
    extension = view.critic_rows
    expected_extension_raw = torch.cat(
        (
            extension["episode_time_remaining_s"],
            extension["live_ball_center_rel_root_heading"],
            extension["live_ball_lin_vel_heading"],
            extension["live_ball_ang_vel_heading"],
            extension["selected_rubber_contact_latched"],
            extension["net_crossed_latched"],
            extension["net_clear_latched"],
            extension["foot_supported_lr"],
            extension["cadence_ready_dwell_fraction"],
        ),
        dim=1,
    )
    expected_extension = expected_extension_raw * torch.tensor(
        _hard_coded_critic_extension_scale_v2()
    )[None, :]
    expected_critic = torch.cat((expected_policy, expected_extension), dim=1)
    torch.testing.assert_close(critic, expected_critic)
    torch.testing.assert_close(critic[:, :215], policy)
    assert policy.abs().max().item() > 1.0
    assert not hasattr(bundle.source, "semantic_publication_count")

    # Poison is checked before the hot cache and cannot return a previously
    # finite observation from the same step.
    epoch.poison_owner_write("r03_strike_fact", 1, owner=r03_owner)
    with pytest.raises(O.LeanObservationError, match="poisoned"):
        policy_term.func(env, **policy_term.params)
    assert calls == [0, 1, 1, 1]


def test_packer_ignores_raw_task_owner_facts_and_reward_ledger():
    env = _Env()
    epoch = _prepared_epoch()
    runtime, parts = _runtime(env, epoch)
    source = O.LeanActionEpochObservationSource(env=env, runtime_owner=runtime)
    record = epoch.current()
    view = _direct_view(env, record, parts)
    actor, critic = source._pack(view, record, common_step=10)

    legacy_mutated = record.clone()
    legacy_mutated.task.task_f32.fill_(123.0)
    legacy_mutated.fact_valid_bits.fill_(7)
    legacy_mutated.fact_source_step.fill_(999)
    legacy_mutated.fact_f32.fill_(-456.0)
    legacy_mutated.owner_fault_bits.fill_(31)
    legacy_mutated.reward_cycle_open.fill_(True)
    legacy_mutated.reward_cycle_fault.fill_(1)
    legacy_mutated.reward_due.fill_(True)
    legacy_mutated.reward_paid.fill_(True)
    actor_mutated, critic_mutated = source._pack(
        view, legacy_mutated, common_step=1_000_000
    )
    torch.testing.assert_close(actor, actor_mutated)
    torch.testing.assert_close(critic, critic_mutated)

    phase_changed = record.clone()
    phase_changed.phase[0, 0] = E.PHASE_REVEAL_COMMITTED
    phase_changed.phase[1, 0] = E.PHASE_RETIRED
    phased, _ = source._pack(view, phase_changed, common_step=10)
    torch.testing.assert_close(
        phased[:, 209:214],
        torch.tensor([[0, 1, 0, 0, 0], [0, 0, 0, 0, 1]]).float(),
    )

    # Motion and Epoch own their enum production invariants.  The observation
    # consumer cannot let one malformed row poison a CUDA context or alias a
    # real phase, so it emits an all-zero phase slice and preserves its peer.
    motion_phase_bad = view.motion_phase_code.clone()
    motion_phase_bad[0] = 9
    bad_motion_view = O.DirectActionEpochObservationFacts(
        **{**view.__dict__, "motion_phase_code": motion_phase_bad}
    )
    bad_motion_actor, bad_motion_critic = source._pack(
        bad_motion_view, record, common_step=10
    )
    assert bad_motion_actor[0, 178:183].eq(0).all()
    assert torch.equal(bad_motion_actor[1], actor[1])
    assert torch.equal(bad_motion_critic[1], critic[1])

    epoch_phase_bad = record.clone()
    epoch_phase_bad.phase[0, 0] = 999
    bad_epoch_actor, bad_epoch_critic = source._pack(
        view, epoch_phase_bad, common_step=10
    )
    assert bad_epoch_actor[0, 209:214].eq(0).all()
    assert torch.equal(bad_epoch_actor[1], actor[1])
    assert torch.equal(bad_epoch_critic[1], critic[1])


def test_term_rejects_invalid_group_instance_shadow_and_caller_source(monkeypatch):
    env = _Env()
    epoch = _prepared_epoch()
    runtime, _ = _runtime(env, epoch)

    class Group:
        pass

    class Term:
        def __init__(self, *, func, params):
            self.func, self.params = func, params

    monkeypatch.setattr(
        O.importlib,
        "import_module",
        lambda name: types.SimpleNamespace(
            ObservationGroupCfg=Group, ObservationTermCfg=Term
        )
        if name == "isaaclab.managers"
        else None,
    )
    bundle = O.materialize_observation_manager_cfg(
        env=env, runtime_owner=runtime
    )
    env._installed_lean_observation_source = bundle.source

    with pytest.raises(O.LeanObservationError, match="policy or critic"):
        O._term(env, group="foreign")
    with pytest.raises(TypeError, match="unexpected keyword argument 'source'"):
        O._term(env, group="policy", source=bundle.source)

    env.__dict__[O.ENV_TERM_METHOD] = lambda **_kwargs: torch.zeros((2, 215))
    with pytest.raises(O.LeanObservationConstructionHold, match="shadowed"):
        O._term(env, group="policy")


def test_term_rejects_foreign_source_retained_by_exact_env_resolver(monkeypatch):
    env = _Env()
    foreign_env = _Env()
    runtime, _ = _runtime(foreign_env, _prepared_epoch())

    class Group:
        pass

    class Term:
        def __init__(self, *, func, params):
            self.func, self.params = func, params

    monkeypatch.setattr(
        O.importlib,
        "import_module",
        lambda name: types.SimpleNamespace(
            ObservationGroupCfg=Group, ObservationTermCfg=Term
        )
        if name == "isaaclab.managers"
        else None,
    )
    foreign = O.materialize_observation_manager_cfg(
        env=foreign_env, runtime_owner=runtime
    )
    env._installed_lean_observation_source = foreign.source
    with pytest.raises(
        O.LeanObservationConstructionHold,
        match="source identity differs",
    ):
        O._term(env, group="policy")


def test_nonfinite_direct_fact_reaches_existing_optimizer_finite_boundary(
    monkeypatch,
):
    env = _Env()
    epoch = _prepared_epoch()
    runtime, parts = _runtime(env, epoch)

    def bad(self, record):
        del self
        view = _direct_view(env, record, parts)
        nonfinite = torch.zeros((2, 31), dtype=torch.float32)
        nonfinite[0, 0] = float("nan")
        nonfinite[1, 0] = float("inf")
        return O.DirectActionEpochObservationFacts(
            **{
                **view.__dict__,
                "actor_rows": {
                    **view.actor_rows,
                    "joint_pos_rel": nonfinite,
                },
            }
        )

    monkeypatch.setattr(
        L.ActionBallFullMdpLeanRuntimeOwner,
        O.DIRECT_VIEW_METHOD,
        bad,
        raising=False,
    )
    source = O.LeanActionEpochObservationSource(env=env, runtime_owner=runtime)
    source.observe("policy")
    source.observe("critic")
    env._action_ball_full_mdp_manager_construction_state = "base_managers_complete"
    policy = source.observe("policy")
    critic = source.observe("critic")
    assert torch.isnan(policy[0, 14])
    assert torch.isinf(policy[1, 14])
    assert torch.isnan(critic[0, 14])
    assert torch.isinf(critic[1, 14])
    assert not hasattr(source, "semantic_publication_count")


def test_source_has_no_superseded_observation_or_zero_prefix_adapter():
    source = (MDP / "action_ball_full_mdp_lean_observation_cfg.py").read_text(encoding="utf-8")
    assert "_assert_async_all" not in source
    assert "torch._assert_async" not in source
    for marker in (
        "FreshFullMdpObservationOwner",
        "ACTOR_FIXED_WIDTH",
        "CRITIC_FIXED_WIDTH",
        "actor_prefix",
        "critic_prefix",
        "publish_shadow_from_epoch_sources",
        "publish_from_bound_providers",
        "receipt_sha256",
        "source_stamp",
        "numeric_authority",
        "mailbox_capacity",
        "epoch_task_f32",
        "physical_r03_r06_r07_fact_f32",
        '"reward_due"',
        '"reward_paid"',
        "plant_fact_adapter.read",
        'owner_kind="motion"',
    ):
        assert marker not in source
    # The one retained capacity spelling is an exact R06/Physical K-axis ABI
    # check, not capacity padding or an observation feature.
    assert source.count("flight_slot_capacity") == 1
