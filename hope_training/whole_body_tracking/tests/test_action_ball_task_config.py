"""Host-only contract tests for the dedicated action-ball task composition.

These tests do not import Isaac Lab.  They verify the source inheritance that supplies the
runtime managers, compose the real Hydra task YAML, and resolve the pure actor-observation
contract for more than one action-bank size.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
CFG_DIR = ROOT / "cfg"
ENV_CFG_PATH = (
    ROOT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "config"
    / "agibot_a3"
    / "hope_env_cfg.py"
)
REGISTRY_PATH = ENV_CFG_PATH.with_name("__init__.py")
ACTOR_CONTRACT_PATH = (
    ROOT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "actor_observation_contract.py"
)
TRAIN_PATH = ROOT / "scripts" / "train.py"

ENV_SOURCE = ENV_CFG_PATH.read_text(encoding="utf-8")
ENV_TREE = ast.parse(ENV_SOURCE, filename=str(ENV_CFG_PATH))
ENV_CLASSES = {
    node.name: node
    for node in ENV_TREE.body
    if isinstance(node, ast.ClassDef)
}


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_dotted_name(node.value)}.{node.attr}"
    raise AssertionError(f"expected a dotted name AST, got {type(node).__name__}")


def _base_names(node: ast.ClassDef) -> tuple[str, ...]:
    return tuple(_dotted_name(base) for base in node.bases)


def _class_fields(name: str) -> set[str]:
    """Collect named class attributes over the local source inheritance chain."""

    node = ENV_CLASSES[name]
    fields: set[str] = set()
    for base in _base_names(node):
        if base in ENV_CLASSES:
            fields.update(_class_fields(base))
    for child in node.body:
        if isinstance(child, ast.Assign):
            fields.update(
                target.id
                for target in child.targets
                if isinstance(target, ast.Name)
            )
        elif isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
            fields.add(child.target.id)
    return fields


def _assigned_call(node: ast.ClassDef, field: str) -> ast.Call:
    for child in node.body:
        if not isinstance(child, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == field
            for target in child.targets
        ):
            assert isinstance(child.value, ast.Call)
            return child.value
    raise AssertionError(f"{node.name}.{field} is not assigned")


def _load_actor_contract_module():
    spec = importlib.util.spec_from_file_location(
        "action_ball_task_config_actor_contract",
        ACTOR_CONTRACT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _compose_task(*overrides: str, task_name: str = "HOPEPingPongActionBall"):
    hydra = pytest.importorskip("hydra")
    with hydra.initialize_config_dir(
        version_base=None,
        config_dir=str(CFG_DIR.resolve()),
    ):
        return hydra.compose(
            config_name="train",
            overrides=[f"task={task_name}", *overrides],
        ).task


def test_action_ball_source_combines_hitter_observations_and_virtualball_v2_terms():
    reward_node = ENV_CLASSES["HOPEActionBallRewardsCfg"]
    assert _base_names(reward_node) == ("HOPEVirtualBallRewardsCfg",)
    reward_fields = _class_fields("HOPEActionBallRewardsCfg")
    # Every direct v2 term declared in the local task lineage must exist before train.py applies
    # the pack.  ``action_rate_l2`` lives in Isaac Lab's external base class and
    # ``table_hit_penalty`` is attached by apply_table_obstacle(), both covered by their existing
    # runtime/source tests.
    assert {
        "base_position",
        "strike_upright",
        "strike_ang_vel",
        "strike_foot_vel",
        "strike_vbob",
        "arm_overreach",
        "racket_strike_success",
        "virtual_pass_net",
        "virtual_landing",
        "virtual_landing_dense",
        "virtual_spin",
        "action_acc_l2",
        "action_rate_clamped",
        "death_penalty",
        "hit_unstable_support",
        "upright_exp",
        "racket_position",
        "racket_velocity",
        "racket_normal",
        "qdes_limit_barrier",
        "qdes_limit_barrier_probe",
        "joint_limit",
        "actual_joint_limit_barrier_probe",
    } <= reward_fields
    assert "joint_safety_terminal" not in reward_fields

    env_node = ENV_CLASSES["HOPEPingPongActionBallAgibotA3EnvCfg"]
    assert _base_names(env_node) == ("HOPEPingPongHitterAgibotA3EnvCfg",)
    env_fields = {
        child.target.id
        for child in env_node.body
        if isinstance(child, ast.AnnAssign)
        and isinstance(child.target, ast.Name)
    }
    assert {"obs_mode", "observations", "rewards"} <= env_fields
    env_segment = ast.get_source_segment(ENV_SOURCE, env_node)
    assert "HOPEObservationsHitterCfg()" in env_segment
    assert "HOPEActionBallRewardsCfg()" in env_segment
    assert "self.commands.racket_target.virtual_ball = True" in env_segment

    # Hitter inherits DeployParity's absolute fall/table safety lineage; action-ball does not
    # substitute a lighter terminations class.
    deploy_fields = _class_fields("HOPEDeployParityTerminationsCfg")
    assert {"base_fell_tilt", "base_too_low", "robot_hit_table"} <= deploy_fields


def test_action_ball_source_adds_exact_preclamp_and_actual_joint_safety_terms():
    terminations_node = ENV_CLASSES["HOPEActionBallTerminationsCfg"]
    assert _base_names(terminations_node) == ("HOPEDeployParityTerminationsCfg",)

    expected = {
        "joint_qdes_forbidden": (
            "mdp.pre_clamp_qdes_forbidden_zone",
            {
                "action_name": "joint_pos",
                "limit_source": "joint_pos_limits",
                "margin_rad": 0.0,
                "margin_fraction": 0.02,
            },
        ),
        "joint_actual_forbidden": (
            "mdp.actual_joint_position_forbidden_zone",
            {
                "asset_cfg": "SceneEntityCfg(robot,all_joints)",
                "limit_source": "joint_pos_limits",
                "margin_rad": 0.0,
                "margin_fraction": 0.02,
            },
        ),
    }
    for field, (expected_func, expected_params) in expected.items():
        call = _assigned_call(terminations_node, field)
        assert _dotted_name(call.func) == "DoneTerm"
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        assert _dotted_name(keywords["func"]) == expected_func
        assert isinstance(keywords["time_out"], ast.Constant)
        assert keywords["time_out"].value is False
        assert isinstance(keywords["params"], ast.Dict)
        params = {}
        for key, value in zip(
            keywords["params"].keys, keywords["params"].values
        ):
            assert isinstance(key, ast.Constant) and isinstance(key.value, str)
            if key.value != "asset_cfg":
                params[key.value] = ast.literal_eval(value)
                continue
            assert isinstance(value, ast.Call)
            assert _dotted_name(value.func) == "SceneEntityCfg"
            assert ast.literal_eval(value.args[0]) == "robot"
            asset_keywords = {
                keyword.arg: ast.literal_eval(keyword.value)
                for keyword in value.keywords
            }
            assert asset_keywords == {"joint_names": [".*"]}
            params[key.value] = "SceneEntityCfg(robot,all_joints)"
        assert params == expected_params

    env_node = ENV_CLASSES["HOPEPingPongActionBallAgibotA3EnvCfg"]
    env_segment = ast.get_source_segment(ENV_SOURCE, env_node)
    assert (
        "terminations: HOPEActionBallTerminationsCfg = "
        "HOPEActionBallTerminationsCfg()"
    ) in env_segment
    assert "self.actions.joint_pos.pre_apply_limit_guard = True" in env_segment
    assert "self.actions.joint_pos.pre_apply_guard_policy_dt_s = 0.02" in env_segment
    assert (
        "self.actions.joint_pos.pre_apply_guard_expected_decimation = 4"
        in env_segment
    )
    assert (
        "self.actions.joint_pos.pre_apply_guard_terminal_archive_capacity = 4096"
        in env_segment
    )
    assert "self.actions.joint_pos.pre_apply_guard_margin_rad = 0.0" in env_segment
    assert (
        "self.actions.joint_pos.pre_apply_guard_margin_fraction = 0.05"
        in env_segment
    )
    assert "table_robot_keepout: bool = True" in env_segment
    assert (
        "self.actions.joint_pos.table_contact_substep_guard = True"
        in env_segment
    )
    assert (
        "self.actions.joint_pos.table_contact_guard_expected_decimation = 4"
        in env_segment
    )
    assert (
        'self.terminations.robot_hit_table.params["require_substep_latch"] = True'
        in env_segment
    )


def test_action_ball_gym_id_points_only_to_the_new_leaf_class():
    source = REGISTRY_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(REGISTRY_PATH))
    registrations = {}
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "register"
        ):
            continue
        keywords = {keyword.arg: keyword.value for keyword in node.keywords}
        task_id_node = keywords.get("id")
        kwargs_node = keywords.get("kwargs")
        if not (
            isinstance(task_id_node, ast.Constant)
            and isinstance(task_id_node.value, str)
            and isinstance(kwargs_node, ast.Dict)
        ):
            continue
        kwargs = {
            key.value: value
            for key, value in zip(kwargs_node.keys, kwargs_node.values)
            if isinstance(key, ast.Constant)
        }
        registrations[task_id_node.value] = _dotted_name(
            kwargs["env_cfg_entry_point"]
        )

    assert registrations["HOPE-PingPong-ActionBall-AgibotA3-v0"] == (
        "hope_env_cfg.HOPEPingPongActionBallAgibotA3EnvCfg"
    )
    assert registrations["HOPE-PingPong-Hitter-AgibotA3-v0"] == (
        "hope_env_cfg.HOPEPingPongHitterAgibotA3EnvCfg"
    )
    assert registrations["HOPE-PingPong-VirtualBall-AgibotA3-v0"] == (
        "hope_env_cfg.HOPEPingPongVirtualBallAgibotA3EnvCfg"
    )


def test_action_ball_yaml_composes_a_fail_closed_preflight_surface():
    task = _compose_task()
    assert task.name == "HOPEPingPongActionBall"
    assert task.gym_task == "HOPE-PingPong-ActionBall-AgibotA3-v0"

    # The shared task never guesses N or a manifest/order/policy identity.
    assert task.actor_obs_contract is None
    assert list(task.racket.clip_names) == []
    assert task.racket.action_ball_manifest_path == ""
    assert task.racket.action_ball_manifest_sha256 == ""
    assert task.racket.action_ball_policy_contract_sha256 == ""
    assert task.racket.action_ball_diagnostic_unauthorized is False

    assert task.racket.target_mode == "action_ball"
    assert task.racket.virtual_ball is True
    assert task.racket.vb_metrics_only is False
    assert task.racket.face_command is True
    assert task.racket.face_command_pairing == "shared_plus_y"
    assert task.racket.face_command_obs is True
    assert task.racket.station_obs is False
    assert task.physical_ball is False
    assert task.table_obstacle is True

    # Raw Hydra composition must already carry the adopted low-tracking recipe.  A separate test
    # exercises train.py's pack-first / explicit-values-last translation into effective weights.
    assert task.rewards.reward_pack == "v2"
    assert task.rewards.reward_pack_strict is False
    assert task.rewards.full_body_mimic is False
    assert task.rewards.free_wrist_ori_mimic is True
    assert task.rewards.free_wrist_vel_mimic is True
    assert task.rewards.racket_position_weight == pytest.approx(4.0)
    assert task.rewards.racket_position_std == pytest.approx(0.075)
    assert "racket_position_coarse_weight" not in task.rewards
    assert "racket_position_coarse_std" not in task.rewards
    coarse_cfg = _assigned_call(
        ENV_CLASSES["HOPERewardsCfg"], "racket_position_coarse"
    )
    coarse_kwargs = {
        keyword.arg: ast.literal_eval(keyword.value)
        for keyword in coarse_cfg.keywords
        if keyword.arg in {"weight", "params"}
    }
    assert coarse_kwargs["weight"] == pytest.approx(0.0)
    assert coarse_kwargs["params"]["std"] == pytest.approx(0.30)
    assert task.rewards.racket_velocity_weight == pytest.approx(0.5)
    assert task.rewards.racket_normal_weight == pytest.approx(0.5)
    assert task.rewards.virtual_landing_weight == pytest.approx(500.0)
    assert task.rewards.death_penalty_weight == pytest.approx(-300.0)
    assert task.rewards.table_hit_penalty_weight == pytest.approx(0.0)
    assert task.rewards.qdes_limit_barrier_weight == pytest.approx(-5.0)
    assert task.rewards.qdes_limit_barrier_margin_frac == pytest.approx(0.08)
    assert task.rewards.joint_limit_weight == pytest.approx(-5.0)
    assert task.actions.qdes_clamp is True

    assert task.motion.canonical_ready_mode is True
    assert task.motion.balanced_clip_sampling is True
    assert list(task.motion.hold_steps_range) == [0, 0]
    assert task.motion.stand_start_min_hold == 0
    assert task.motion.post_swing_start_prob == pytest.approx(0.0)
    assert task.motion.post_swing_min_hold == 0
    assert task.motion.stagger_initial_clock is False
    assert task.motion.stagger_hold_max_steps == 0
    assert list(task.motion.speed_scale_range) == [1.0, 1.0]
    assert task.motion.speed_scale_per_clip is None
    assert list(task.motion.joint_position_range) == [0.0, 0.0]
    assert all(
        list(pair) == [0.0, 0.0]
        for pair in task.motion.pose_range.values()
    )
    assert all(
        list(pair) == [0.0, 0.0]
        for pair in task.motion.velocity_range.values()
    )
    assert task.racket.shadow_ball is False
    assert task.racket.shadow_table is False
    assert task.racket.pos_range_per_clip is None
    assert task.racket.vel_range_per_clip is None


def test_a3_vendor_v1_task_profile_composes_exact_push_and_control_step_delay():
    task = _compose_task(task_name="HOPEPingPongActionBallA3VendorV1")
    assert task.name == "HOPEPingPongActionBallA3VendorV1"
    assert task.gym_task == "HOPE-PingPong-ActionBall-AgibotA3-v0"
    assert task.actions.qdes_clamp is True
    assert task.actions.control_step_action_delay_min == 0
    assert task.actions.control_step_action_delay_max == 2
    assert (
        task.actions.pre_apply_guard_brake_mode
        == "max_inward_until_nonoutward_v1"
    )
    assert task.actions.pre_apply_guard_margin_fraction == pytest.approx(0.06)
    assert (
        task.actions.physx_control_position_limit_inset_fraction
        == pytest.approx(0.02)
    )
    assert task.rewards.racket_position_weight == pytest.approx(4.0)
    assert task.rewards.racket_position_std == pytest.approx(0.075)
    assert task.rewards.action_acc_weight == pytest.approx(0.0)
    assert task.rewards.racket_position_coarse_weight == pytest.approx(1.0)
    assert task.rewards.racket_position_coarse_std == pytest.approx(0.30)
    assert task.push.enable is True
    assert task.push.recipe == "axis_box_6d_v2"
    assert set(task.push.keys()) == {
        "enable", "recipe", "interval_range_s", "combined_exclusive",
        "velocity_range",
    }
    assert list(task.push.interval_range_s) == [1.0, 3.0]
    assert task.push.combined_exclusive is False
    assert {
        axis: list(task.push.velocity_range[axis])
        for axis in ("x", "y", "z", "roll", "pitch", "yaw")
    } == {
        "x": [-0.25, 0.25],
        "y": [-0.25, 0.25],
        "z": [-0.1, 0.1],
        "roll": [-0.26, 0.26],
        "pitch": [-0.26, 0.26],
        "yaw": [-0.39, 0.39],
    }
    assert set(task.force_push.keys()) == {"enable"}
    assert task.force_push.enable is False


def test_vendor_v2_n1_fixed_question_leaf_cannot_fall_back_to_online_solver():
    task = _compose_task(
        task_name="HOPEPingPongActionBallA3VendorV2N1Diagnostic"
    )
    assert task.actor_obs_contract == (
        "action_ball_table_pose_twist_heading_task_teacher_start_v2"
    )
    assert task.motion.action_ball_diagnostic_split_ready_teacher is True
    assert task.racket.action_ball_target_source == "immutable_tape"
    assert task.racket.action_ball_immutable_tape_path == ""
    assert task.racket.action_ball_immutable_tape_sha256 == ""
    assert task.racket.action_ball_target_recipe == "current_lm"
    assert list(task.racket.action_ball_target_validity_mask) == [
        True,
        True,
        True,
    ]
    assert task.racket.action_ball_target_observation_noise is False
    assert task.actions.control_step_action_delay_min == 0
    assert task.actions.control_step_action_delay_max == 0
    assert task.push.enable is False
    assert task.rewards.strike_capture_bonus_weight == pytest.approx(25.0)
    assert task.rewards.virtual_pass_net_weight == pytest.approx(20.0)
    assert task.rewards.virtual_landing_dense_weight == pytest.approx(20.0)
    assert task.rewards.virtual_landing_weight == pytest.approx(500.0)


def test_ordinary_action_ball_does_not_override_code_owned_five_percent_guard():
    task = _compose_task(task_name="HOPEPingPongActionBall")
    assert "pre_apply_guard_margin_fraction" not in task.actions
    assert "physx_control_position_limit_inset_fraction" not in task.actions
    assert "push" not in task
    assert "force_push" not in task


@pytest.mark.parametrize("action_count", [4, 73])
def test_launch_can_bind_exact_n_without_changing_the_task_lineage(action_count):
    action_names = [f"action_{index:03d}" for index in range(action_count)]
    list_override = "[" + ",".join(action_names) + "]"
    task = _compose_task(
        f"task.actor_obs_contract=action_ball_n{action_count}",
        f"task.racket.clip_names={list_override}",
        "task.racket.action_ball_manifest_path=configs/exact_manifest.json",
        f"task.racket.action_ball_manifest_sha256={'a' * 64}",
        f"task.racket.action_ball_policy_contract_sha256={'b' * 64}",
    )

    assert task.gym_task == "HOPE-PingPong-ActionBall-AgibotA3-v0"
    assert task.actor_obs_contract == f"action_ball_n{action_count}"
    assert list(task.racket.clip_names) == action_names

    contract_mod = _load_actor_contract_module()
    contract = contract_mod.resolve_actor_observation_contract(
        task.actor_obs_contract
    )
    assert contract.obs_mode == "hitter_footwork"
    assert contract.total_dim == 181 + action_count
    assert contract.layout[-2:] == (
        ("racket_target_normal_cmd", 4),
        ("action_one_hot", action_count),
    )


@pytest.mark.parametrize("action_count", [1, 5, 73])
def test_new_launch_can_bind_exact_n_with_table_pose_contract(action_count):
    action_names = [f"action_{index:03d}" for index in range(action_count)]
    list_override = "[" + ",".join(action_names) + "]"
    task = _compose_task(
        (
            "task.actor_obs_contract="
            f"action_ball_table_pose_n{action_count}"
        ),
        f"task.racket.clip_names={list_override}",
        "task.racket.action_ball_manifest_path=configs/exact_manifest.json",
        f"task.racket.action_ball_manifest_sha256={'a' * 64}",
        f"task.racket.action_ball_policy_contract_sha256={'b' * 64}",
    )

    contract_mod = _load_actor_contract_module()
    contract = contract_mod.resolve_actor_observation_contract(
        task.actor_obs_contract
    )
    assert contract.total_dim == 190 + action_count
    assert contract.layout[-4:] == (
        ("base_position_table", 3),
        ("base_orientation_table_6d", 6),
        ("racket_target_normal_cmd", 4),
        ("action_one_hot", action_count),
    )


@pytest.mark.parametrize("action_count", [1, 5, 73])
def test_historical_launch_can_bind_exact_n_with_frame_consistent_contract(
    action_count,
):
    action_names = [f"action_{index:03d}" for index in range(action_count)]
    list_override = "[" + ",".join(action_names) + "]"
    task = _compose_task(
        (
            "task.actor_obs_contract="
            "action_ball_table_pose_twist_heading_task_teacher_start_n"
            f"{action_count}"
        ),
        f"task.racket.clip_names={list_override}",
        "task.racket.action_ball_manifest_path=configs/exact_manifest.json",
        f"task.racket.action_ball_manifest_sha256={'a' * 64}",
        f"task.racket.action_ball_policy_contract_sha256={'b' * 64}",
    )

    contract_mod = _load_actor_contract_module()
    contract = contract_mod.resolve_actor_observation_contract(
        task.actor_obs_contract
    )
    assert contract.total_dim == 194 + action_count
    assert contract.layout[-6:] == (
        ("base_position_table", 3),
        ("base_orientation_table_6d", 6),
        ("base_lin_vel_heading", 3),
        ("racket_target_normal_cmd_heading", 4),
        ("time_to_teacher_start_s", 1),
        ("action_one_hot", action_count),
    )


@pytest.mark.parametrize("action_count", [1, 5, 73])
def test_v2_layout_is_fixed_even_when_task_yaml_names_a_larger_bank(
    action_count,
):
    action_names = [f"action_{index:03d}" for index in range(action_count)]
    list_override = "[" + ",".join(action_names) + "]"
    task = _compose_task(
        (
            "task.actor_obs_contract="
            "action_ball_table_pose_twist_heading_task_teacher_start_v2"
        ),
        f"task.racket.clip_names={list_override}",
        "task.racket.action_ball_manifest_path=configs/exact_manifest.json",
        f"task.racket.action_ball_manifest_sha256={'a' * 64}",
        f"task.racket.action_ball_policy_contract_sha256={'b' * 64}",
    )

    contract_mod = _load_actor_contract_module()
    contract = contract_mod.resolve_actor_observation_contract(
        task.actor_obs_contract
    )
    assert list(task.racket.clip_names) == action_names
    assert contract.total_dim == 194
    assert contract.layout[-2:] == (
        ("racket_target_normal_cmd_heading", 4),
        ("time_to_teacher_start_s", 1),
    )
    assert all(term.name != "action_one_hot" for term in contract.terms)


def test_train_finalizer_rejects_v2_for_multi_action_banks():
    source = TRAIN_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(TRAIN_PATH))
    finalizer = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_finalize_action_ball_training_cfg"
    )
    segment = ast.get_source_segment(source, finalizer)
    assert segment is not None
    assert (
        "configured_actor_contract == fixed_teacher_start_actor_contract"
        in segment
    )
    assert "and action_count != 1" in segment
    assert "fixed-194 ActionBall v2 is N=1-only" in segment
    assert "teacher-trajectory/ball/task ABI" in segment
    assert "synthetic intent code" in segment


def test_train_finalizer_does_not_instantiate_historical_one_hot_layouts():
    source = TRAIN_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(TRAIN_PATH))
    finalizer = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_finalize_action_ball_training_cfg"
    )
    segment = ast.get_source_segment(source, finalizer)
    assert segment is not None
    assert "if configured_actor_contract not in (" in segment
    assert "a225_actor_contract" in segment
    assert "c225_actor_contract" in segment
    assert "cannot be instantiated by the current trainer" in segment
    assert "policy.action_one_hot =" not in segment
