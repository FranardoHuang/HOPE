"""CPU-only contract tests for the A3 vendor Play/ranking profiles."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "hope_training/whole_body_tracking/scripts"
PROFILE_PATH = SCRIPTS / "vendor_a3_eval_profile.py"
PLAY_PATH = SCRIPTS / "play.py"
EVAL_PATH = SCRIPTS / "eval_deterministic.py"


def _load_profile_module():
    spec = importlib.util.spec_from_file_location("vendor_a3_eval_profile_test", PROFILE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _term(**params):
    return NS(params=dict(params))


def _reset_joints_by_offset():
    """Name-only stand-in for the vendor additive joint reset event."""


def _cfg():
    events = NS(
        physics_material=_term(),
        add_joint_default_pos=_term(pos_distribution_params=(-0.01, 0.01)),
        base_com=_term(),
        randomize_link_mass=_term(),
        randomize_pd_gains=_term(),
        push_robot=_term(),
        force_push=None,
        force_push_sweep=None,
        combined_push=None,
        combined_push_sweep=None,
        reset_base=_term(
            pose_range={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            velocity_range={"x": (-0.2, 0.2), "yaw": (-0.2, 0.2)},
        ),
        reset_robot_joints=NS(
            func=_reset_joints_by_offset,
            params={
                "position_range": (-0.15, 0.15),
                "velocity_range": (-0.2, 0.2),
            },
        ),
    )
    motion = NS(
        pose_range={"x": (-0.1, 0.1), "yaw": (-0.1, 0.1)},
        velocity_range={"x": (-0.2, 0.2), "yaw": (-0.2, 0.2)},
        joint_position_range=(-0.15, 0.15),
        stand_start_yaw_range=(-0.1, 0.1),
    )
    return NS(
        events=events,
        commands=NS(motion=motion),
        actions=NS(
            joint_pos=NS(
                control_step_action_delay_min=0,
                control_step_action_delay_max=2,
            )
        ),
        observations=NS(policy=NS(enable_corruption=True)),
    )


def _vendor_task():
    return {"name": "HOPEPingPongActionBallA3VendorV1"}


def test_vendor_play_disables_train_only_events_but_keeps_vendor_play_noise_and_delay():
    module = _load_profile_module()
    cfg = _cfg()
    original_reset = cfg.events.reset_base

    receipt = module.apply_vendor_a3_eval_profile(
        cfg, _vendor_task(), profile=module.VENDOR_PLAY_PROFILE
    )

    assert receipt["profile"] == "vendor_play_v1"
    assert receipt["observation_corruption_enabled"] is True
    assert receipt["control_step_action_delay"] == [0, 2]
    assert receipt["root_reset_semantics"] == "vendor_play_retained"
    assert receipt["joint_reset_semantics"] == "vendor_play_nominal_if_present"
    assert cfg.observations.policy.enable_corruption is True
    assert cfg.actions.joint_pos.control_step_action_delay_max == 2
    assert cfg.events.reset_base is original_reset
    assert cfg.events.reset_robot_joints.params == {
        "position_range": (0.0, 0.0),
        "velocity_range": (0.0, 0.0),
    }
    assert receipt["nominalized_reset_events"] == {
        "reset_robot_joints": {
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        }
    }
    for name in (
        "physics_material",
        "add_joint_default_pos",
        "base_com",
        "randomize_link_mass",
        "randomize_pd_gains",
        "push_robot",
    ):
        assert getattr(cfg.events, name) is None
        assert name in receipt["disabled_train_only_events"]


def test_deterministic_ranking_additionally_removes_obs_delay_and_reset_noise():
    module = _load_profile_module()
    cfg = _cfg()

    receipt = module.apply_vendor_a3_eval_profile(
        cfg, _vendor_task(), profile=module.DETERMINISTIC_RANKING_PROFILE
    )

    assert receipt["profile"] == "deterministic_ranking_v1"
    assert receipt["observation_corruption_enabled"] is False
    assert receipt["control_step_action_delay"] == [0, 0]
    assert receipt["root_reset_semantics"] == "nominal_deterministic"
    assert cfg.observations.policy.enable_corruption is False
    assert cfg.actions.joint_pos.control_step_action_delay_min == 0
    assert cfg.actions.joint_pos.control_step_action_delay_max == 0
    assert cfg.events.reset_robot_joints.params == {
        "position_range": (0.0, 0.0),
        "velocity_range": (0.0, 0.0),
    }
    assert cfg.events.reset_base.params["pose_range"] == {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
    }
    assert cfg.events.reset_base.params["velocity_range"] == {
        "x": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    assert cfg.commands.motion.joint_position_range == (0.0, 0.0)
    assert cfg.commands.motion.stand_start_yaw_range == (0.0, 0.0)
    assert all(
        bounds == (0.0, 0.0)
        for bounds in cfg.commands.motion.pose_range.values()
    )
    assert all(
        bounds == (0.0, 0.0)
        for bounds in cfg.commands.motion.velocity_range.values()
    )


def test_non_vendor_task_is_untouched_and_unknown_vendor_profile_fails_closed():
    module = _load_profile_module()
    cfg = _cfg()
    event = cfg.events.physics_material
    assert (
        module.apply_vendor_a3_eval_profile(
            cfg, {"name": "HOPEPingPongActionBall"}, profile="unknown"
        )
        is None
    )
    assert cfg.events.physics_material is event
    with pytest.raises(module.VendorA3EvalProfileError, match="unsupported"):
        module.apply_vendor_a3_eval_profile(
            cfg, _vendor_task(), profile="unknown"
        )


def test_exact_vendor_task_fails_closed_when_expected_surface_is_missing():
    module = _load_profile_module()
    cfg = _cfg()
    del cfg.events.randomize_pd_gains
    with pytest.raises(module.VendorA3EvalProfileError, match="missing required event slots"):
        module.apply_vendor_a3_eval_profile(
            cfg, _vendor_task(), profile=module.VENDOR_PLAY_PROFILE
        )


@pytest.mark.parametrize(
    ("source_path", "runner_name", "profile_name"),
    (
        (PLAY_PATH, "_run_play", "VENDOR_PLAY_PROFILE"),
        (EVAL_PATH, "_run", "DETERMINISTIC_RANKING_PROFILE"),
    ),
)
def test_runner_applies_explicit_vendor_profile_after_task_compose_before_gym_make(
    source_path, runner_name, profile_name
):
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    runner = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == runner_name
    )
    calls = [node for node in ast.walk(runner) if isinstance(node, ast.Call)]
    apply_task = next(
        node for node in calls if ast.unparse(node.func) == "_apply_task_overrides"
    )
    apply_profile = next(
        node for node in calls if ast.unparse(node.func) == "apply_vendor_a3_eval_profile"
    )
    gym_make = next(node for node in calls if ast.unparse(node.func) == "gym.make")
    assert ast.unparse(apply_profile.keywords[0].value) == profile_name
    assert apply_task.lineno < apply_profile.lineno < gym_make.lineno
