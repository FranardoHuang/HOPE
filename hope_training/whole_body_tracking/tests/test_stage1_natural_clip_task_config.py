"""Host-only contracts for the ball-free natural-clip Stage-1 task.

Runtime tests belong on the exact Pod checkout.  This module intentionally uses source/AST and
Hydra composition only: it proves the new leaf is isolated from historical ActionBall defaults,
keeps the vendor plant/safety recipe, and cannot silently arm a ball or inverse producer.
"""

from __future__ import annotations

import ast
import hashlib
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
TASK_PATH = CFG_DIR / "task" / "HOPEPingPongStage1NaturalClipA3VendorV2.yaml"
TRAIN_PATH = ROOT / "scripts" / "train.py"
LANE_CONTRACT_PATH = (
    ROOT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "stage1_natural_clip_contract.py"
)

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
    raise AssertionError(f"expected dotted name, got {type(node).__name__}")


def _base_names(node: ast.ClassDef) -> tuple[str, ...]:
    return tuple(_dotted_name(base) for base in node.bases)


def _assigned_call(node: ast.ClassDef, name: str) -> ast.Call:
    for child in node.body:
        if not isinstance(child, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in child.targets):
            assert isinstance(child.value, ast.Call)
            return child.value
    raise AssertionError(f"{node.name}.{name} is not assigned a call")


def _call_keywords(call: ast.Call) -> dict[str, ast.AST]:
    return {keyword.arg: keyword.value for keyword in call.keywords if keyword.arg is not None}


def _compose_task():
    hydra = pytest.importorskip("hydra")
    with hydra.initialize_config_dir(version_base=None, config_dir=str(CFG_DIR.resolve())):
        return hydra.compose(
            config_name="train",
            overrides=["task=HOPEPingPongStage1NaturalClipA3VendorV2"],
        ).task


def test_stage1_reward_leaf_uses_clip_site_terms_with_reviewed_starting_economy():
    node = ENV_CLASSES["HOPEStage1NaturalClipRewardsV2Cfg"]
    assert _base_names(node) == ("HOPEActionBallRewardsCfg",)

    expected = {
        "racket_position": (
            "mdp.stage1_clip_racket_position_tracking_exp",
            0.90,
            0.50,
        ),
        "racket_velocity": (
            "mdp.stage1_clip_racket_velocity_tracking_exp",
            0.45,
            3.0,
        ),
        "racket_normal": (
            "mdp.stage1_clip_racket_normal_tracking_exp",
            0.90,
            2.10,
        ),
        "racket_position_coarse": (
            "mdp.stage1_clip_racket_position_coarse_tracking_exp",
            0.30,
            0.70,
        ),
        "racket_velocity_coarse": (
            "mdp.stage1_clip_racket_velocity_coarse_tracking_exp",
            0.15,
            4.0,
        ),
        "racket_normal_coarse": (
            "mdp.stage1_clip_racket_normal_coarse_tracking_exp",
            0.30,
            "math.pi",
        ),
        "racket_position_precision": (
            "mdp.stage1_clip_racket_position_precision_tracking_exp",
            0.50,
            0.075,
        ),
        "racket_velocity_precision": (
            "mdp.stage1_clip_racket_velocity_precision_tracking_exp",
            0.25,
            0.50,
        ),
        "racket_normal_precision": (
            "mdp.stage1_clip_racket_normal_precision_tracking_exp",
            0.50,
            0.262,
        ),
    }
    for term_name, (function_name, weight, std) in expected.items():
        call = _assigned_call(node, term_name)
        assert _dotted_name(call.func) == "RewTerm"
        keywords = _call_keywords(call)
        assert _dotted_name(keywords["func"]) == function_name
        assert ast.literal_eval(keywords["weight"]) == pytest.approx(weight)
        params = keywords["params"]
        assert isinstance(params, ast.Dict)
        params_by_name = {
            ast.literal_eval(key): value
            for key, value in zip(params.keys, params.values)
        }
        assert ast.literal_eval(params_by_name["command_name"]) == "racket_target"
        if std == "math.pi":
            assert _dotted_name(params_by_name["std"]) == std
        else:
            assert ast.literal_eval(params_by_name["std"]) == pytest.approx(std)

    death_call = _assigned_call(node, "death_penalty")
    death_keywords = _call_keywords(death_call)
    assert (
        _dotted_name(death_keywords["func"])
        == "mdp.stage1_object_free_safety_terminated"
    )
    assert ast.literal_eval(death_keywords["weight"]) == 0.0
    assert ast.literal_eval(death_keywords["params"]) == {
        "term_names": (
            "base_fell_tilt",
            "base_too_low",
            "joint_actual_forbidden",
            "joint_qdes_forbidden",
        )
    }

    segment = ast.get_source_segment(ENV_SOURCE, node)
    assert segment is not None
    assert "base_position = None" in segment
    assert "racket_progress = None" in segment
    for term_name in (
        "racket_strike_success",
        "strike_capture_bonus",
        "virtual_pass_net",
        "virtual_landing",
        "virtual_spin",
    ):
        call = _assigned_call(node, term_name)
        assert ast.literal_eval(_call_keywords(call)["weight"]) == 0.0


def test_stage1_env_reuses_joint_safety_but_removes_task_ball_and_table():
    historical = ENV_CLASSES["HOPEPingPongStage1NaturalClipAgibotA3EnvCfg"]
    node = ENV_CLASSES["HOPEPingPongStage1NaturalClipV2AgibotA3EnvCfg"]
    assert _base_names(node) == ("HOPEPingPongStage1NaturalClipAgibotA3EnvCfg",)
    historical_segment = ast.get_source_segment(ENV_SOURCE, historical)
    segment = ast.get_source_segment(ENV_SOURCE, node)
    assert historical_segment is not None and segment is not None

    # The versioned natural-clip group has no one-hot and no ball/planner/demanded-face/reserved
    # scalar tail.
    assert (
        "observations: HOPEStage1NaturalClipObservationsV2Cfg = "
        "HOPEStage1NaturalClipObservationsV2Cfg()"
    ) in segment
    assert 'obs_mode: str = "stage1_natural_clip_paddle_world"' in segment
    assert "action_one_hot" not in segment

    # The parent safety guard is retained, then every ball/task producer is explicitly disabled.
    for statement in (
        'command.target_mode = "reference_perturbed"',
        "command.virtual_ball = False",
        "command.vb_metrics_only = False",
        "command.shadow_ball = False",
        "command.shadow_table = False",
        "command.face_command = False",
        "self.physical_ball = False",
        "self.face_command_obs = False",
        "self.table_obstacle = False",
        "apply_table_obstacle(self)",
        "self.actions.joint_pos.pre_apply_guard_diagnostic_compact_evidence = True",
    ):
        assert statement in historical_segment


def test_stage1_observation_leaf_pins_exact_paddle_world_v2_order_and_producers():
    node = ENV_CLASSES["HOPEStage1NaturalClipObservationsV2Cfg"]
    assert _base_names(node) == ("ObservationsCfg",)
    policy = next(
        child
        for child in node.body
        if isinstance(child, ast.ClassDef) and child.name == "Stage1PolicyCfg"
    )
    assert _base_names(policy) == ("ObsGroup",)
    expected_terms = (
        ("actual_base_now_world", "mdp.stage1_base_state_world"),
        ("teacher_base_now_world", "mdp.stage1_teacher_base_state_now_world"),
        ("joint_pos", "mdp.stage1_joint_pos_rel"),
        ("teacher_joint_pos", "mdp.stage1_teacher_joint_pos_rel"),
        ("joint_vel", "mdp.stage1_joint_vel"),
        ("teacher_joint_vel", "mdp.stage1_teacher_joint_vel"),
        ("actions", "mdp.stage1_actions"),
        (
            "racket_site_achieved_now_heading",
            "mdp.stage1_racket_site_achieved_now_heading",
        ),
        (
            "racket_site_teacher_now_heading",
            "mdp.stage1_racket_site_teacher_now_heading",
        ),
        (
            "racket_site_teacher_at_reference_hit_heading",
            "mdp.stage1_racket_site_teacher_at_reference_hit_heading",
        ),
        (
            "racket_contact_desired_at_t_hit_heading",
            "mdp.stage1_racket_contact_desired_at_t_hit_heading",
        ),
        ("desired_base_xy_world", "mdp.stage1_base_target_position_world_xy"),
        ("time_to_contact", "mdp.stage1_time_to_contact_s"),
        ("time_to_teacher_start", "mdp.time_to_teacher_start_s"),
    )
    assigned_terms = tuple(
        target.id
        for child in policy.body
        if isinstance(child, ast.Assign)
        for target in child.targets
        if isinstance(target, ast.Name) and isinstance(child.value, ast.Call)
    )
    assert assigned_terms == tuple(name for name, _func in expected_terms)
    for term_name, function_name in expected_terms:
        call = _assigned_call(policy, term_name)
        assert _dotted_name(call.func) == "ObsTerm"
        assert _dotted_name(_call_keywords(call)["func"]) == function_name

    observation_segment = ast.get_source_segment(ENV_SOURCE, node)
    assert observation_segment is not None
    assert "critic: Stage1CriticCfg = Stage1CriticCfg()" in observation_segment
    critic = next(
        child
        for child in node.body
        if isinstance(child, ast.ClassDef) and child.name == "Stage1CriticCfg"
    )
    assert _base_names(critic) == ("ObservationsCfg.PrivilegedCfg",)
    critic_tail = tuple(
        target.id
        for child in critic.body
        if isinstance(child, ast.Assign)
        for target in child.targets
        if isinstance(target, ast.Name) and isinstance(child.value, ast.Call)
    )
    assert critic_tail == (
        "racket_site_teacher_at_reference_hit_heading",
        "racket_contact_desired_at_t_hit_heading",
        "desired_base_xy_world",
        "time_to_contact",
        "time_to_teacher_start",
    )


def test_stage1_env_pins_split_windows_and_window_aware_sigma_contract():
    node = ENV_CLASSES["HOPEPingPongStage1NaturalClipV2AgibotA3EnvCfg"]
    segment = ast.get_source_segment(ENV_SOURCE, node)
    assert segment is not None

    expected_literals = {
        "command.sigma_pos_min": 0.075,
        "command.sigma_pos_max": 0.50,
        "command.sigma_vel_min": 0.50,
        "command.sigma_vel_max": 3.0,
        "command.sigma_normal_min": 0.262,
        "command.sigma_normal_max": 2.10,
    }
    method = next(
        child
        for child in node.body
        if isinstance(child, ast.FunctionDef) and child.name == "__post_init__"
    )
    assignments = {}
    for child in ast.walk(method):
        if not isinstance(child, ast.Assign) or len(child.targets) != 1:
            continue
        try:
            target = _dotted_name(child.targets[0])
        except AssertionError:
            continue
        assignments[target] = child.value
    for field, value in expected_literals.items():
        assert ast.literal_eval(assignments[field]) == pytest.approx(value)

    assert ast.literal_eval(assignments["command.adaptive_sigma"]) is True
    assert ast.literal_eval(assignments["command.adaptive_sigma_monotonic"]) is True
    assert ast.literal_eval(assignments["command.adaptive_sigma_normal"]) is True
    assert (
        ast.literal_eval(assignments["command.adaptive_sigma_source"])
        == "stage1_clip_site_full_phase_rms"
    )

    historical = ENV_CLASSES["HOPEPingPongStage1NaturalClipAgibotA3EnvCfg"]
    historical_method = next(
        child
        for child in historical.body
        if isinstance(child, ast.FunctionDef) and child.name == "__post_init__"
    )
    historical_assignments = {}
    for child in ast.walk(historical_method):
        if not isinstance(child, ast.Assign) or len(child.targets) != 1:
            continue
        try:
            target = _dotted_name(child.targets[0])
        except AssertionError:
            continue
        historical_assignments[target] = child.value
    assert ast.literal_eval(
        historical_assignments["command.strike_window_pos_s"]
    ) == pytest.approx(0.02)
    assert ast.literal_eval(
        historical_assignments["command.strike_window_wide_s"]
    ) == pytest.approx(0.10)


def test_stage1_hydra_leaf_keeps_vendor_plant_delay_push_and_natural_timing():
    task = _compose_task()
    assert task.name == "HOPEPingPongStage1NaturalClipA3VendorV2"
    assert task.gym_task == "HOPE-PingPong-Stage1NaturalClip-AgibotA3-v1"
    assert task.actor_obs_contract == "stage1_natural_clip_paddle_world_v2"
    assert task.registry_name is None
    assert task.registry_name_2 is None
    assert task.physical_ball is False
    assert task.table_obstacle is False

    assert list(task.domain_rand.kp_gain_range) == [0.8, 1.2]
    assert list(task.domain_rand.kd_gain_range) == [0.7, 1.3]
    assert list(task.domain_rand.link_mass_range) == [0.85, 1.15]
    assert task.actions.control_step_action_delay_min == 0
    assert task.actions.control_step_action_delay_max == 2
    assert task.actions.diagnostic_compact_evidence is True
    assert task.push.enable is True
    assert task.push.recipe == "axis_box_6d_v2"
    assert list(task.push.interval_range_s) == [1.0, 3.0]
    assert task.force_push.enable is False

    assert task.motion.canonical_ready_mode is False
    assert task.motion.stand_start_prob == pytest.approx(0.25)
    assert list(task.motion.speed_scale_range) == [1.0, 1.0]
    assert task.motion.speed_scale_per_clip is None
    assert task.motion.wrap_teleport is False
    assert task.motion.clip_switch_prob == 0.0


def test_stage1_hydra_leaf_is_full_body_wrist_free_and_has_no_ball_income():
    task = _compose_task()
    reward = task.rewards
    assert reward.reward_pack == "v2"
    assert reward.full_body_mimic is True
    assert reward.free_wrist_ori_mimic is True
    assert reward.free_wrist_vel_mimic is True
    expected_paddle_terms = {
        "racket_position": (0.90, 0.50),
        "racket_velocity": (0.45, 3.0),
        "racket_normal": (0.90, 2.10),
        "racket_position_coarse": (0.30, 0.70),
        "racket_velocity_coarse": (0.15, 4.0),
        "racket_normal_coarse": (0.30, 3.141592653589793),
        "racket_position_precision": (0.50, 0.075),
        "racket_velocity_precision": (0.25, 0.50),
        "racket_normal_precision": (0.50, 0.262),
    }
    for term_name, (weight, std) in expected_paddle_terms.items():
        assert getattr(reward, f"{term_name}_weight") == pytest.approx(weight)
        assert getattr(reward, f"{term_name}_std") == pytest.approx(std)
    assert reward.base_position_weight is None
    assert reward.base_position_std is None
    assert reward.virtual_landing_weight == 0.0
    assert reward.death_penalty_weight == pytest.approx(-300.0)
    assert reward.table_hit_penalty_weight is None
    assert reward.qdes_limit_barrier_weight == pytest.approx(-5.0)
    assert reward.joint_limit_weight == pytest.approx(-5.0)
    assert reward.action_acc_weight == 0.0

    racket = task.racket
    assert racket.target_mode == "reference_perturbed"
    assert racket.clip_names is None
    assert racket.virtual_ball is False
    assert racket.vb_metrics_only is False
    assert racket.shadow_ball is False
    assert racket.shadow_table is False
    assert racket.face_command is False
    assert racket.face_command_obs is False
    assert racket.question_bank == ""
    assert racket.cq_anchor_bank == ""
    assert racket.exam_bank == ""
    assert list(racket.ref_perturb_pos) == [0.0, 0.0, 0.0]
    assert list(racket.ref_perturb_vel) == [0.0, 0.0, 0.0]
    assert racket.ref_perturb_normal == 0.0
    assert racket.ref_perturb_curriculum_steps == 0
    assert racket.ref_perturb_success_gated is False
    assert racket.strike_window_pos_s == pytest.approx(0.02)
    assert racket.strike_window_wide_s == pytest.approx(0.10)
    assert racket.adaptive_sigma is True
    assert racket.adaptive_sigma_monotonic is True
    assert racket.adaptive_sigma_normal is True
    assert racket.adaptive_sigma_source == "stage1_clip_site_full_phase_rms"
    assert racket.sigma_pos_min == pytest.approx(0.075)
    assert racket.sigma_pos_max == pytest.approx(0.50)
    assert racket.sigma_vel_min == pytest.approx(0.50)
    assert racket.sigma_vel_max == pytest.approx(3.0)
    assert racket.sigma_normal_min == pytest.approx(0.262)
    assert racket.sigma_normal_max == pytest.approx(2.10)


def test_stage1_trainer_fail_closed_hooks_and_code_owned_lane_binding_exist():
    source = TRAIN_PATH.read_text(encoding="utf-8")
    assert "def _finalize_stage1_natural_clip_training_cfg(" in source
    assert "_finalize_stage1_natural_clip_training_cfg(env_cfg, task, applied)" in source
    assert "def _validate_stage1_natural_clip_motion_sources(" in source
    assert "_validate_stage1_natural_clip_motion_sources(env_cfg, motion_files)" in source
    for literal in (
        '"adaptive_sigma_source"',
        '"sigma_normal_min"',
        '"sigma_normal_max"',
        '"stage1_clip_site_full_phase_rms"',
        '"stage1_natural_clip_paddle_world_v2"',
        '"[train.py] Stage-1 natural clip requires table_obstacle=false"',
        '"[train.py] Stage-1 object-free motion prior forbids robot_hit_table"',
        '"[train.py] Stage-1 object-free motion prior forbids table_hit_penalty"',
        '"[train.py] Stage-1 object-free motion prior forbids the table substep guard"',
        '"[train.py] Stage-1 death_penalty must use the exact object-free "',
    ):
        assert literal in source


def test_stage1_code_owned_lane_bytes_match_repo_assets():
    spec = importlib.util.spec_from_file_location(
        "stage1_lane_contract_test", LANE_CONTRACT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    lanes = module.STAGE1_NATURAL_CLIP_LANES
    assert len(lanes) == 3
    assert [lane.side for lane in lanes].count("BH") == 2
    for lane in lanes:
        path = ROOT.parents[1] / lane.motion_path
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == lane.motion_sha256
        assert 0 < lane.strike_frame < lane.frame_count - 1
        assert 0.0 < lane.strike_phase < 1.0
