from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

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
        ("hitter_pure", "hitter_pure", 110),
        ("task_first_n5", "task_first_n5", 186),
        (
            "action_ball_table_pose_n1",
            "action_ball_table_pose_n1",
            191,
        ),
    ],
)
def test_existing_contract_resolution_regression(name, expected_name, expected_dim):
    contract = contract_mod.resolve_actor_observation_contract(name)
    assert contract.name == expected_name
    assert contract.total_dim == expected_dim
    assert sum(term.dim for term in contract.terms) == expected_dim
