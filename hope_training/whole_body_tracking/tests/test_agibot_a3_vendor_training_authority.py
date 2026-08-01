"""Fail-loud checks for the authoritative vendor A3 training nominal."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from types import SimpleNamespace


ROBOT_CFG = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "robots"
    / "agibot_a3.py"
)


class _Cfg:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _ArticulationCfg(_Cfg):
    class InitialStateCfg(_Cfg):
        pass


def _load_actuator_contract():
    """Execute only the robot cfg declaration and derived action-scale loop."""
    tree = ast.parse(ROBOT_CFG.read_text(encoding="utf-8"), filename=str(ROBOT_CFG))
    body = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "AGIBOT_A3_CFG"
            for target in node.targets
        ):
            body.append(node)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "AGIBOT_A3_ACTION_SCALE"
        ):
            body.append(node)
        elif isinstance(node, ast.For) and any(
            isinstance(child, ast.Name) and child.id == "AGIBOT_A3_ACTION_SCALE"
            for child in ast.walk(node)
        ):
            body.append(node)

    assert len(body) == 3, "robot cfg or derived action-scale declaration changed unexpectedly"
    module = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "ArticulationCfg": _ArticulationCfg,
        "ImplicitActuatorCfg": _Cfg,
        "_make_agibot_a3_spawn_cfg": lambda: SimpleNamespace(),
    }
    exec(compile(module, str(ROBOT_CFG), "exec"), namespace)
    return namespace["AGIBOT_A3_CFG"], namespace["AGIBOT_A3_ACTION_SCALE"]


def _resolve(value, joint_name: str) -> float:
    if not isinstance(value, dict):
        return value
    matches = [candidate for pattern, candidate in value.items() if re.fullmatch(pattern, joint_name)]
    assert len(matches) == 1, f"expected exactly one value for {joint_name}, got {len(matches)}"
    return matches[0]


def test_vendor_a3_training_nominal_actuator_values_are_exact():
    cfg, _ = _load_actuator_contract()
    legs = cfg.actuators["legs"]
    feet = cfg.actuators["feet"]
    waist = cfg.actuators["waist"]
    head = cfg.actuators["head"]
    arms = cfg.actuators["arms"]

    for side in ("left", "right"):
        expected_groups = (
            (
                legs,
                {
                    f"{side}_hip_pitch_joint": (80.0, 3.0, 220.0, 12.0, 0.066472),
                    f"{side}_hip_yaw_joint": (80.0, 3.0, 220.0, 12.0, 0.066472),
                    f"{side}_hip_roll_joint": (120.0, 4.0, 220.0, 12.0, 0.066472),
                    f"{side}_knee_joint": (250.0, 8.0, 320.0, 14.6, 0.120340),
                },
            ),
            (
                feet,
                {
                    f"{side}_ankle_pitch_joint": (50.0, 2.0, 118.2, 10.8, 0.064449),
                    f"{side}_ankle_roll_joint": (50.0, 2.0, 54.75, 19.3, 0.020129),
                },
            ),
            (
                arms,
                {
                    f"{side}_shoulder_pitch_joint": (40.0, 3.0, 60.0, 13.6, 0.012085),
                    f"{side}_shoulder_roll_joint": (40.0, 3.0, 60.0, 13.6, 0.012085),
                    f"{side}_shoulder_yaw_joint": (30.0, 2.0, 24.0, 15.7, 0.004968),
                    f"{side}_elbow_joint": (30.0, 2.0, 24.0, 15.7, 0.004968),
                    f"{side}_wrist_roll_joint": (30.0, 2.0, 24.0, 15.7, 0.004968),
                    f"{side}_wrist_pitch_joint": (20.0, 2.0, 6.0, 12.7, 0.0008100893338),
                    f"{side}_wrist_yaw_joint": (20.0, 2.0, 6.0, 12.7, 0.0008100893338),
                },
            ),
        )
        for actuator, expected in expected_groups:
            for joint, (kp, kd, effort, velocity, armature) in expected.items():
                assert _resolve(actuator.stiffness, joint) == kp
                assert _resolve(actuator.damping, joint) == kd
                assert _resolve(actuator.effort_limit_sim, joint) == effort
                assert _resolve(actuator.velocity_limit_sim, joint) == velocity
                assert _resolve(actuator.armature, joint) == armature

    waist_expected = {
        "waist_yaw_joint": (85.0, 3.0, 220.0, 12.0, 0.066472),
        "waist_roll_joint": (50.0, 2.0, 46.0, 22.7, 0.014623),
        "waist_pitch_joint": (50.0, 2.0, 118.0, 9.2, 0.088220),
    }
    for joint, (kp, kd, effort, velocity, armature) in waist_expected.items():
        assert _resolve(waist.stiffness, joint) == kp
        assert _resolve(waist.damping, joint) == kd
        assert _resolve(waist.effort_limit_sim, joint) == effort
        assert _resolve(waist.velocity_limit_sim, joint) == velocity
        assert _resolve(waist.armature, joint) == armature

    # The vendor 29-DoF table excludes head; this is the explicit HOPE deploy fallback.
    for joint in ("head_yaw_joint", "head_pitch_joint"):
        assert _resolve(head.stiffness, joint) == 40.0
        assert _resolve(head.damping, joint) == 2.0
        assert _resolve(head.effort_limit_sim, joint) == 6.0
        assert _resolve(head.velocity_limit_sim, joint) == 12.7
        assert _resolve(head.armature, joint) == 0.0008100893338


def test_vendor_nominal_armature_table_covers_all_29_body_dofs():
    cfg, _ = _load_actuator_contract()

    for side in ("left", "right"):
        legs = cfg.actuators["legs"]
        for axis in ("yaw", "roll", "pitch"):
            assert _resolve(legs.armature, f"{side}_hip_{axis}_joint") == 0.066472
        assert _resolve(legs.armature, f"{side}_knee_joint") == 0.120340

        feet = cfg.actuators["feet"]
        assert _resolve(feet.armature, f"{side}_ankle_pitch_joint") == 0.064449
        assert _resolve(feet.armature, f"{side}_ankle_roll_joint") == 0.020129

        arms = cfg.actuators["arms"]
        assert _resolve(arms.armature, f"{side}_shoulder_pitch_joint") == 0.012085
        assert _resolve(arms.armature, f"{side}_shoulder_roll_joint") == 0.012085
        for joint_type in (
            "shoulder_yaw",
            "elbow",
        ):
            assert _resolve(arms.armature, f"{side}_{joint_type}_joint") == 0.004968
        assert _resolve(arms.armature, f"{side}_wrist_roll_joint") == 0.004968
        for joint_type in ("wrist_pitch", "wrist_yaw"):
            assert _resolve(arms.armature, f"{side}_{joint_type}_joint") == 0.0008100893338

    waist = cfg.actuators["waist"]
    assert _resolve(waist.armature, "waist_yaw_joint") == 0.066472
    assert _resolve(waist.armature, "waist_roll_joint") == 0.014623
    assert _resolve(waist.armature, "waist_pitch_joint") == 0.088220

    # The vendor 29-DoF table excludes head; retain the explicit HOPE fallback.
    head = cfg.actuators["head"]
    assert _resolve(head.armature, "head_yaw_joint") == 0.0008100893338
    assert _resolve(head.armature, "head_pitch_joint") == 0.0008100893338


def test_vendor_training_nominal_action_scales_are_derived_from_kp_and_effort():
    cfg, action_scale = _load_actuator_contract()
    joint_groups = {
        "legs": [
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
        ],
        "feet": [
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
        ],
        "waist": ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
        "head": ["head_yaw_joint", "head_pitch_joint"],
        "arms": [
            f"{side}_{joint_type}_joint"
            for side in ("left", "right")
            for joint_type in (
                "shoulder_pitch",
                "shoulder_roll",
                "shoulder_yaw",
                "elbow",
                "wrist_roll",
                "wrist_pitch",
                "wrist_yaw",
            )
        ],
    }

    # Every active joint is covered: vendor 29-DoF body plus HOPE's 2-DoF head fallback.
    assert sum(len(joints) for joints in joint_groups.values()) == 31
    for actuator_name, joints in joint_groups.items():
        actuator = cfg.actuators[actuator_name]
        for joint in joints:
            expected = (
                0.25
                * _resolve(actuator.effort_limit_sim, joint)
                / _resolve(actuator.stiffness, joint)
            )
            assert _resolve(action_scale, joint) == expected
