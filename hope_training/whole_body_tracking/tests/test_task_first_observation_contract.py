from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


_MODULE = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "actor_observation_contract.py"
)
_SPEC = importlib.util.spec_from_file_location("task_first_actor_contract", _MODULE)
assert _SPEC is not None and _SPEC.loader is not None
contract_mod = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = contract_mod
_SPEC.loader.exec_module(contract_mod)

_OBS_MODULE = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
    / "hope_observations.py"
)


def test_stage1_natural_clip_site_contract_is_exact_and_task_only():
    contract = contract_mod.resolve_actor_observation_contract(
        "stage1_natural_clip_site_v1"
    )
    assert contract.name == "stage1_natural_clip_site_v1"
    assert contract.obs_mode == "stage1_natural_clip"
    assert contract.total_dim == 170
    assert contract.layout == (
        ("command", 62),
        ("motion_anchor_pos_b", 3),
        ("motion_anchor_ori_b", 6),
        ("base_ang_vel", 3),
        ("joint_pos", 31),
        ("joint_vel", 31),
        ("actions", 31),
        ("projected_gravity", 3),
    )
    names = {name for name, _dim in contract.layout}
    assert "base_lin_vel" not in names
    assert "action_one_hot" not in names
    assert "racket_target_normal_cmd" not in names
    assert "time_to_strike" not in names


def test_stage1_natural_clip_paddle_world_v2_contract_is_exact_and_paired():
    contract = contract_mod.resolve_actor_observation_contract(
        "stage1_natural_clip_paddle_world_v2"
    )
    assert contract.name == "stage1_natural_clip_paddle_world_v2"
    assert contract.obs_mode == "stage1_natural_clip_paddle_world"
    assert contract.total_dim == 225
    assert contract.layout == (
        ("actual_base_now_world", 15),
        ("teacher_base_now_world", 15),
        ("joint_pos", 31),
        ("teacher_joint_pos", 31),
        ("joint_vel", 31),
        ("teacher_joint_vel", 31),
        ("actions", 31),
        ("racket_site_achieved_now_heading", 9),
        ("racket_site_teacher_now_heading", 9),
        ("racket_site_teacher_at_reference_hit_heading", 9),
        ("racket_contact_desired_at_t_hit_heading", 9),
        ("desired_base_xy_world", 2),
        ("time_to_contact", 1),
        ("time_to_teacher_start", 1),
    )

    offsets = {}
    offset = 0
    for term in contract.terms:
        offsets[term.name] = (offset, offset + term.dim)
        offset += term.dim
    assert offset == 225
    assert offsets == {
        "actual_base_now_world": (0, 15),
        "teacher_base_now_world": (15, 30),
        "joint_pos": (30, 61),
        "teacher_joint_pos": (61, 92),
        "joint_vel": (92, 123),
        "teacher_joint_vel": (123, 154),
        "actions": (154, 185),
        "racket_site_achieved_now_heading": (185, 194),
        "racket_site_teacher_now_heading": (194, 203),
        "racket_site_teacher_at_reference_hit_heading": (203, 212),
        "racket_contact_desired_at_t_hit_heading": (212, 221),
        "desired_base_xy_world": (221, 223),
        "time_to_contact": (223, 224),
        "time_to_teacher_start": (224, 225),
    }

    names = set(offsets)
    assert names.isdisjoint(
        {
            "action_one_hot",
            "swing_type",
            "racket_target_normal_cmd_heading",
            "rho",
            "projected_gravity",
            "motion_anchor_pos_b",
            "motion_anchor_ori_b",
            "command",
        }
    )
    descriptions = {term.name: term.description for term in contract.terms}
    assert "canonical HOPE world" in descriptions["actual_base_now_world"]
    assert "same current base yaw-heading frame" in descriptions[
        "racket_site_teacher_at_reference_hit_heading"
    ]
    assert "Stage-1 copies the teacher hit tuple" in descriptions[
        "racket_contact_desired_at_t_hit_heading"
    ]


def test_infer_actor_observation_contract_recognizes_stage1_paddle_world_v2():
    expected = contract_mod.resolve_actor_observation_contract(
        "stage1_natural_clip_paddle_world_v2"
    )
    observation_manager = SimpleNamespace(
        active_terms={"policy": [term.name for term in expected.terms]},
        group_obs_term_dim={
            "policy": [(term.dim,) for term in expected.terms]
        },
        group_obs_dim={"policy": (expected.total_dim,)},
    )
    env = SimpleNamespace(observation_manager=observation_manager)
    assert contract_mod.infer_actor_observation_contract(env) == expected


def test_stage1_paddle_world_v2_producers_are_explicit_and_share_geometry():
    source = _OBS_MODULE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {
        node.name: ast.get_source_segment(source, node) or ""
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    required = {
        "_stage1_env_position_to_hope_world",
        "stage1_base_state_world",
        "stage1_teacher_base_state_now_world",
        "stage1_joint_pos_rel",
        "stage1_teacher_joint_pos_rel",
        "stage1_joint_vel",
        "stage1_teacher_joint_vel",
        "stage1_actions",
        "stage1_racket_site_achieved_now_heading",
        "stage1_racket_site_teacher_now_heading",
        "stage1_racket_site_teacher_at_reference_hit_heading",
        "stage1_racket_contact_desired_at_t_hit_heading",
        "stage1_base_target_position_world_xy",
        "stage1_time_to_contact_s",
    }
    assert required <= set(functions)

    bridge = functions["_stage1_env_position_to_hope_world"]
    assert "vb_table_near_x" in bridge
    assert "_vb_half_w" in bridge
    assert "vb_table_surface_z" in bridge
    assert "env_origins_w" in bridge
    assert (
        "result = position_w - env_origins_w[..., :width] - translation"
        in bridge
    )

    assert "default_joint_pos" in functions["stage1_joint_pos_rel"]
    assert "default_joint_pos" in functions["stage1_teacher_joint_pos_rel"]
    assert "stage1_aligned_clip_site_target_now" in functions[
        "stage1_racket_site_teacher_now_heading"
    ]
    for name in (
        "stage1_racket_site_teacher_at_reference_hit_heading",
        "stage1_racket_contact_desired_at_t_hit_heading",
    ):
        assert "stage1_aligned_clip_site_target_at_reference_hit" in functions[name]
        assert "racket_target_pos_w" not in functions[name]
        assert "racket_target_vel_w" not in functions[name]
    for name in (
        "stage1_racket_site_achieved_now_heading",
        "stage1_racket_site_teacher_now_heading",
        "stage1_racket_site_teacher_at_reference_hit_heading",
        "stage1_racket_contact_desired_at_t_hit_heading",
    ):
        assert "_stage1_pack_racket_state_heading" in functions[name]

    # Simulation env origins are clone offsets.  The producer must additionally apply the
    # authoritative robot/floor -> near-left-table-surface HOPE translation.
    assert "env.scene.env_origins" in functions["stage1_base_state_world"]
    assert "env.scene.env_origins" in functions[
        "stage1_teacher_base_state_now_world"
    ]
    assert "env.scene.env_origins" in functions[
        "stage1_base_target_position_world_xy"
    ]


@pytest.mark.parametrize("action_count", [1, 2, 5, 6, 93])
def test_task_first_contract_is_explicitly_sized(action_count):
    contract = contract_mod.resolve_actor_observation_contract(
        f"task_first_n{action_count}"
    )
    assert contract.name == f"task_first_n{action_count}"
    assert contract.obs_mode == "hitter_footwork"
    assert contract.total_dim == 181 + action_count
    assert contract.layout[-2:] == (
        ("racket_target_normal_cmd", 4),
        ("action_one_hot", action_count),
    )


@pytest.mark.parametrize("action_count", [1, 5, 93])
def test_action_ball_contract_is_explicitly_sized(action_count):
    contract = contract_mod.resolve_actor_observation_contract(
        f"action_ball_n{action_count}"
    )
    assert contract.name == f"action_ball_n{action_count}"
    assert contract.obs_mode == "hitter_footwork"
    assert contract.total_dim == 181 + action_count
    assert sum(term.dim for term in contract.terms) == contract.total_dim
    assert contract.layout[-2:] == (
        ("racket_target_normal_cmd", 4),
        ("action_one_hot", action_count),
    )


@pytest.mark.parametrize("action_count", [1, 5, 73, 93])
def test_table_pose_action_ball_contract_is_explicitly_sized(action_count):
    contract = contract_mod.resolve_actor_observation_contract(
        f"action_ball_table_pose_n{action_count}"
    )
    assert contract.name == f"action_ball_table_pose_n{action_count}"
    assert contract.obs_mode == "hitter_footwork"
    assert contract.total_dim == 190 + action_count
    assert sum(term.dim for term in contract.terms) == contract.total_dim
    prefix_count = len(contract_mod.HITTER_FOOTWORK.terms)
    assert contract.layout[:prefix_count] == (
        contract_mod.HITTER_FOOTWORK.layout
    )
    assert contract.layout[-4:] == (
        ("base_position_table", 3),
        ("base_orientation_table_6d", 6),
        ("racket_target_normal_cmd", 4),
        ("action_one_hot", action_count),
    )


def test_n1_table_pose_action_ball_contract_is_191d():
    contract = contract_mod.resolve_actor_observation_contract(
        "action_ball_table_pose_n1"
    )
    assert contract.total_dim == 191
    sources = {term.name: term.deploy_source for term in contract.terms}
    assert sources["motion_anchor_ori_b"].startswith("mocap_")
    assert sources["base_ang_vel"] == "mocap_pose_history"
    assert sources["projected_gravity"] == "mocap"
    assert sources["racket_target_pos_b"] == (
        "planner_plus_mocap_plus_racket_fk"
    )
    assert all("imu" not in term.deploy_source for term in contract.terms)


@pytest.mark.parametrize("action_count", [1, 5, 73, 93])
def test_table_pose_twist_action_ball_contract_is_explicitly_sized(action_count):
    contract = contract_mod.resolve_actor_observation_contract(
        f"action_ball_table_pose_twist_n{action_count}"
    )
    assert contract.name == f"action_ball_table_pose_twist_n{action_count}"
    assert contract.obs_mode == "hitter_footwork"
    assert contract.total_dim == 193 + action_count
    assert sum(term.dim for term in contract.terms) == contract.total_dim
    prefix_count = len(contract_mod.HITTER_FOOTWORK.terms)
    assert tuple(term.name for term in contract.terms[:prefix_count]) == tuple(
        term.name for term in contract_mod.HITTER_FOOTWORK.terms
    )
    assert contract.layout[-5:] == (
        ("base_position_table", 3),
        ("base_orientation_table_6d", 6),
        ("base_lin_vel_heading", 3),
        ("racket_target_normal_cmd", 4),
        ("action_one_hot", action_count),
    )


def test_n1_table_pose_twist_action_ball_contract_is_194d_and_sensor_bound():
    contract = contract_mod.resolve_actor_observation_contract(
        "action_ball_table_pose_twist_n1"
    )
    assert contract.total_dim == 194
    sources = {term.name: term.deploy_source for term in contract.terms}
    assert sources["motion_anchor_ori_b"].startswith("optitrack_")
    assert sources["base_ang_vel"] == "imu_gyro"
    assert sources["projected_gravity"] == "optitrack"
    assert sources["base_target_pos_b"] == "planner_plus_optitrack"
    assert sources["racket_target_pos_b"] == (
        "planner_plus_optitrack_plus_racket_fk"
    )
    assert sources["base_position_table"] == (
        "optitrack_plus_table_calibration"
    )
    assert sources["base_orientation_table_6d"] == (
        "optitrack_plus_table_calibration"
    )
    assert sources["base_lin_vel_heading"] == "fused_root_com_velocity_estimator"

    offset = 0
    offsets = {}
    for term in contract.terms:
        offsets[term.name] = (offset, offset + term.dim)
        offset += term.dim
    assert offset == 194
    assert offsets["base_position_table"] == (177, 180)
    assert offsets["base_orientation_table_6d"] == (180, 186)
    assert offsets["base_lin_vel_heading"] == (186, 189)
    assert offsets["racket_target_normal_cmd"] == (189, 193)
    assert offsets["action_one_hot"] == (193, 194)


@pytest.mark.parametrize("action_count", [1, 5, 73, 93])
def test_heading_task_action_ball_contract_is_frame_consistent_and_sized(
    action_count,
):
    contract = contract_mod.resolve_actor_observation_contract(
        "action_ball_table_pose_twist_heading_task_n"
        f"{action_count}"
    )
    assert contract.name == (
        "action_ball_table_pose_twist_heading_task_n"
        f"{action_count}"
    )
    assert contract.obs_mode == "hitter_footwork"
    assert contract.total_dim == 193 + action_count
    assert sum(term.dim for term in contract.terms) == contract.total_dim
    assert ("racket_target_vel_w", 3) not in contract.layout
    assert ("racket_target_vel_heading", 3) in contract.layout
    assert contract.layout[-5:] == (
        ("base_position_table", 3),
        ("base_orientation_table_6d", 6),
        ("base_lin_vel_heading", 3),
        ("racket_target_normal_cmd_heading", 4),
        ("action_one_hot", action_count),
    )


def test_n1_heading_task_action_ball_offsets_are_exactly_194d():
    contract = contract_mod.resolve_actor_observation_contract(
        "action_ball_table_pose_twist_heading_task_n1"
    )
    offsets = {}
    offset = 0
    for term in contract.terms:
        offsets[term.name] = (offset, offset + term.dim)
        offset += term.dim
    assert offset == 194
    assert offsets["racket_target_pos_b"] == (169, 172)
    assert offsets["racket_target_vel_heading"] == (172, 175)
    assert offsets["base_position_table"] == (177, 180)
    assert offsets["base_orientation_table_6d"] == (180, 186)
    assert offsets["base_lin_vel_heading"] == (186, 189)
    assert offsets["racket_target_normal_cmd_heading"] == (189, 193)
    assert offsets["action_one_hot"] == (193, 194)


@pytest.mark.parametrize("action_count", [1, 5, 73, 93])
def test_teacher_start_action_ball_contract_is_explicit_and_sized(
    action_count,
):
    contract = contract_mod.resolve_actor_observation_contract(
        "action_ball_table_pose_twist_heading_task_teacher_start_n"
        f"{action_count}"
    )
    assert contract.name == (
        "action_ball_table_pose_twist_heading_task_teacher_start_n"
        f"{action_count}"
    )
    assert contract.total_dim == 194 + action_count
    assert sum(term.dim for term in contract.terms) == contract.total_dim
    assert contract.layout[-3:] == (
        ("racket_target_normal_cmd_heading", 4),
        ("time_to_teacher_start_s", 1),
        ("action_one_hot", action_count),
    )
    sources = {term.name: term.deploy_source for term in contract.terms}
    assert sources["time_to_teacher_start_s"] == (
        "action_ball_motion_phase_governor"
    )


def test_n1_teacher_start_action_ball_offsets_are_exactly_195d():
    contract = contract_mod.resolve_actor_observation_contract(
        "action_ball_table_pose_twist_heading_task_teacher_start_n1"
    )
    offsets = {}
    offset = 0
    for term in contract.terms:
        offsets[term.name] = (offset, offset + term.dim)
        offset += term.dim
    assert offset == 195
    assert offsets["racket_target_normal_cmd_heading"] == (189, 193)
    assert offsets["time_to_teacher_start_s"] == (193, 194)
    assert offsets["action_one_hot"] == (194, 195)


def test_teacher_start_v2_is_fixed_194d_and_omits_action_one_hot():
    contract = contract_mod.resolve_actor_observation_contract(
        "action_ball_table_pose_twist_heading_task_teacher_start_v2"
    )
    assert contract.name == (
        "action_ball_table_pose_twist_heading_task_teacher_start_v2"
    )
    assert contract.obs_mode == "hitter_footwork"
    assert contract.total_dim == 194
    assert sum(term.dim for term in contract.terms) == 194
    assert contract.layout[-2:] == (
        ("racket_target_normal_cmd_heading", 4),
        ("time_to_teacher_start_s", 1),
    )
    assert all(term.name != "action_one_hot" for term in contract.terms)

    offsets = {}
    offset = 0
    for term in contract.terms:
        offsets[term.name] = (offset, offset + term.dim)
        offset += term.dim
    assert offset == 194
    assert offsets["racket_target_normal_cmd_heading"] == (189, 193)
    assert offsets["time_to_teacher_start_s"] == (193, 194)


def test_infer_actor_observation_contract_recognizes_teacher_start_v2():
    expected = contract_mod.resolve_actor_observation_contract(
        "action_ball_table_pose_twist_heading_task_teacher_start_v2"
    )
    observation_manager = SimpleNamespace(
        active_terms={
            "policy": [term.name for term in expected.terms],
        },
        group_obs_term_dim={
            "policy": [(term.dim,) for term in expected.terms],
        },
        group_obs_dim={"policy": (expected.total_dim,)},
    )
    env = SimpleNamespace(observation_manager=observation_manager)
    assert contract_mod.infer_actor_observation_contract(env) == expected


@pytest.mark.parametrize("action_count", [1, 5, 93])
def test_task_first_and_action_ball_share_columns_but_not_identity(action_count):
    task_first = contract_mod.resolve_actor_observation_contract(
        f"task_first_n{action_count}"
    )
    action_ball = contract_mod.resolve_actor_observation_contract(
        f"action_ball_n{action_count}"
    )
    assert action_ball.name == f"action_ball_n{action_count}"
    assert task_first.name == f"task_first_n{action_count}"
    assert action_ball.name != task_first.name
    assert action_ball.obs_mode == task_first.obs_mode
    assert action_ball.total_dim == task_first.total_dim
    assert action_ball.terms == task_first.terms


@pytest.mark.parametrize("action_count", [0, 1025, True, 1.0, "5", None])
def test_action_ball_constructor_rejects_non_plain_or_out_of_range_counts(action_count):
    with pytest.raises(
        ValueError,
        match=r"^action-ball action_count must be a plain integer in \[1,1024\]",
    ):
        contract_mod.action_ball_n_contract(action_count)


def test_action_ball_constructor_accepts_upper_bound():
    contract = contract_mod.action_ball_n_contract(1024)
    assert contract.name == "action_ball_n1024"
    assert contract.total_dim == 1205
    assert contract.layout[-1] == ("action_one_hot", 1024)


@pytest.mark.parametrize("action_count", [0, 1025, True, 1.0, "5", None])
def test_table_pose_action_ball_constructor_rejects_invalid_counts(action_count):
    with pytest.raises(
        ValueError,
        match=(
            r"^action-ball table-pose action_count must be a plain integer "
            r"in \[1,1024\]"
        ),
    ):
        contract_mod.action_ball_table_pose_n_contract(action_count)


@pytest.mark.parametrize("action_count", [0, 1025, True, 1.0, "5", None])
def test_table_pose_twist_action_ball_constructor_rejects_invalid_counts(
    action_count,
):
    with pytest.raises(
        ValueError,
        match=(
            r"^action-ball table-pose-twist action_count must be a plain integer "
            r"in \[1,1024\]"
        ),
    ):
        contract_mod.action_ball_table_pose_twist_n_contract(action_count)


@pytest.mark.parametrize("action_count", [0, 1025, True, 1.0, "5", None])
def test_heading_task_action_ball_constructor_rejects_invalid_counts(
    action_count,
):
    with pytest.raises(
        ValueError,
        match=(
            r"^action-ball table-pose-twist-heading-task action_count must "
            r"be a plain integer in \[1,1024\]"
        ),
    ):
        contract_mod.action_ball_table_pose_twist_heading_task_n_contract(
            action_count
        )


@pytest.mark.parametrize("action_count", [0, 1025, True, 1.0, "5", None])
def test_teacher_start_action_ball_constructor_rejects_invalid_counts(
    action_count,
):
    with pytest.raises(
        ValueError,
        match=(
            r"^action-ball table-pose-twist-heading-task action_count must "
            r"be a plain integer in \[1,1024\]"
        ),
    ):
        (
            contract_mod
            .action_ball_table_pose_twist_heading_task_teacher_start_n_contract(
                action_count
            )
        )


def test_action_bank_size_changes_the_contract_and_shape():
    five = contract_mod.resolve_actor_observation_contract("task_first_n5")
    six = contract_mod.resolve_actor_observation_contract("task_first_n6")
    assert five.name != six.name
    assert five.total_dim + 1 == six.total_dim
    assert five.layout[-1] == ("action_one_hot", 5)
    assert six.layout[-1] == ("action_one_hot", 6)


@pytest.mark.parametrize(
    "name",
    ["task_first_n0", "task_first_n-1", "task_first_n", "task_first_n1.5"],
)
def test_invalid_dynamic_contract_names_fail_closed(name):
    with pytest.raises(ValueError, match="Unknown actor observation contract"):
        contract_mod.resolve_actor_observation_contract(name)


@pytest.mark.parametrize(
    "name",
    [
        "action_ball_n0",
        "action_ball_n-1",
        "action_ball_n",
        "action_ball_n1.5",
        "action_ball_n05",
    ],
)
def test_invalid_action_ball_dynamic_contract_names_fail_closed(name):
    with pytest.raises(
        ValueError,
        match=(
            r"^Invalid action-ball actor observation contract .*; expected "
            r"action_ball_n<N> with a base-10 N in \[1,1024\] and no leading zeros$"
        ),
    ):
        contract_mod.resolve_actor_observation_contract(name)


@pytest.mark.parametrize(
    "name",
    [
        "action_ball_table_pose_n0",
        "action_ball_table_pose_n-1",
        "action_ball_table_pose_n",
        "action_ball_table_pose_n1.5",
        "action_ball_table_pose_n05",
    ],
)
def test_invalid_table_pose_action_ball_contract_names_fail_closed(name):
    with pytest.raises(
        ValueError,
        match=(
            r"^Invalid table-pose action-ball actor observation contract .*; "
            r"expected action_ball_table_pose_n<N> with a base-10 N in "
            r"\[1,1024\] and no leading zeros$"
        ),
    ):
        contract_mod.resolve_actor_observation_contract(name)


@pytest.mark.parametrize(
    "name",
    [
        "action_ball_table_pose_twist_n0",
        "action_ball_table_pose_twist_n-1",
        "action_ball_table_pose_twist_n",
        "action_ball_table_pose_twist_n1.5",
        "action_ball_table_pose_twist_n05",
    ],
)
def test_invalid_table_pose_twist_action_ball_contract_names_fail_closed(name):
    with pytest.raises(
        ValueError,
        match=(
            r"^Invalid table-pose-twist action-ball actor observation contract "
            r".*; expected action_ball_table_pose_twist_n<N> with a base-10 N "
            r"in \[1,1024\] and no leading zeros$"
        ),
    ):
        contract_mod.resolve_actor_observation_contract(name)


@pytest.mark.parametrize(
    "name",
    [
        "action_ball_table_pose_twist_heading_task_n0",
        "action_ball_table_pose_twist_heading_task_n-1",
        "action_ball_table_pose_twist_heading_task_n",
        "action_ball_table_pose_twist_heading_task_n1.5",
        "action_ball_table_pose_twist_heading_task_n05",
    ],
)
def test_invalid_heading_task_action_ball_contract_names_fail_closed(name):
    with pytest.raises(
        ValueError,
        match=(
            r"^Invalid frame-consistent table-pose-twist action-ball actor "
            r"observation contract .*; expected "
            r"action_ball_table_pose_twist_heading_task_n<N> with a base-10 N "
            r"in \[1,1024\] and no leading zeros$"
        ),
    ):
        contract_mod.resolve_actor_observation_contract(name)


@pytest.mark.parametrize(
    "name",
    [
        "action_ball_table_pose_twist_heading_task_teacher_start_n0",
        "action_ball_table_pose_twist_heading_task_teacher_start_n-1",
        "action_ball_table_pose_twist_heading_task_teacher_start_n",
        "action_ball_table_pose_twist_heading_task_teacher_start_n1.5",
        "action_ball_table_pose_twist_heading_task_teacher_start_n05",
    ],
)
def test_invalid_teacher_start_action_ball_contract_names_fail_closed(name):
    with pytest.raises(
        ValueError,
        match=(
            r"^Invalid teacher-start table-pose-twist action-ball actor "
            r"observation contract .*; expected "
            r"action_ball_table_pose_twist_heading_task_teacher_start_n<N> "
            r"with a base-10 N in \[1,1024\] and no leading zeros$"
        ),
    ):
        contract_mod.resolve_actor_observation_contract(name)


def test_action_ball_resolver_rejects_count_above_upper_bound():
    with pytest.raises(
        ValueError,
        match=r"^action-ball action_count must be a plain integer in \[1,1024\], got 1025$",
    ):
        contract_mod.resolve_actor_observation_contract("action_ball_n1025")


@pytest.mark.parametrize(
    ("name", "expected_name", "expected_dim"),
    [
        ("full", "full", 180),
        ("deploy_parity", "deploy_parity", 175),
        ("real_sensor_only", "deploy_parity", 175),
        ("deploy_parity_face179", "deploy_parity_face179", 179),
        ("deploy_parity_station181", "deploy_parity_station181", 181),
        ("hitter_footwork", "hitter_footwork", 177),
        (
            "stage1_natural_clip_site_v1",
            "stage1_natural_clip_site_v1",
            170,
        ),
        (
            "stage1_natural_clip_paddle_world_v2",
            "stage1_natural_clip_paddle_world_v2",
            225,
        ),
        ("hitter_pure", "hitter_pure", 110),
        ("task_first_n5", "task_first_n5", 186),
        (
            "action_ball_table_pose_n1",
            "action_ball_table_pose_n1",
            191,
        ),
        (
            "action_ball_table_pose_twist_n1",
            "action_ball_table_pose_twist_n1",
            194,
        ),
        (
            "action_ball_table_pose_twist_heading_task_teacher_start_v2",
            "action_ball_table_pose_twist_heading_task_teacher_start_v2",
            194,
        ),
    ],
)
def test_existing_contract_resolution_regression(name, expected_name, expected_dim):
    contract = contract_mod.resolve_actor_observation_contract(name)
    assert contract.name == expected_name
    assert contract.total_dim == expected_dim
    assert sum(term.dim for term in contract.terms) == expected_dim
