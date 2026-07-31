"""Fail-loud host checks for the latest vendor A3 training actuator identity."""

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


def test_latest_vendor_training_actuator_values_are_exact():
    cfg, _ = _load_actuator_contract()
    waist = cfg.actuators["waist"]
    arms = cfg.actuators["arms"]

    assert _resolve(waist.stiffness, "waist_yaw_joint") == 80.0
    assert _resolve(waist.effort_limit_sim, "waist_pitch_joint") == 115.0
    assert _resolve(arms.stiffness, "left_wrist_pitch_joint") == 30.0
    assert _resolve(arms.effort_limit_sim, "left_wrist_pitch_joint") == 24.0
    assert _resolve(arms.armature, "left_wrist_pitch_joint") == 0.004968

    # Both pitch/yaw sides share the vendor group; fail if a regex silently misses one joint.
    for side in ("left", "right"):
        for axis in ("pitch", "yaw"):
            joint = f"{side}_wrist_{axis}_joint"
            assert _resolve(arms.stiffness, joint) == 30.0
            assert _resolve(arms.effort_limit_sim, joint) == 24.0
            assert _resolve(arms.armature, joint) == 0.004968


def test_latest_vendor_training_armature_table_covers_all_29_body_dofs():
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
            "wrist_roll",
            "wrist_pitch",
            "wrist_yaw",
        ):
            assert _resolve(arms.armature, f"{side}_{joint_type}_joint") == 0.004968

    waist = cfg.actuators["waist"]
    assert _resolve(waist.armature, "waist_yaw_joint") == 0.066472
    assert _resolve(waist.armature, "waist_roll_joint") == 0.014623
    assert _resolve(waist.armature, "waist_pitch_joint") == 0.088220

    # The vendor's 29-DoF table does not cover the two head joints.
    head = cfg.actuators["head"]
    assert _resolve(head.armature, "head_yaw_joint") == 0.0008100893338
    assert _resolve(head.armature, "head_pitch_joint") == 0.0008100893338


def test_all_four_vendor_wrist_action_scales_are_derived_from_kp_and_effort():
    cfg, action_scale = _load_actuator_contract()
    arms = cfg.actuators["arms"]
    waist = cfg.actuators["waist"]

    assert _resolve(action_scale, "waist_yaw_joint") == 0.6875
    assert _resolve(action_scale, "waist_pitch_joint") == 0.575
    assert _resolve(action_scale, "waist_yaw_joint") == (
        0.25
        * _resolve(waist.effort_limit_sim, "waist_yaw_joint")
        / _resolve(waist.stiffness, "waist_yaw_joint")
    )
    assert _resolve(action_scale, "waist_pitch_joint") == (
        0.25
        * _resolve(waist.effort_limit_sim, "waist_pitch_joint")
        / _resolve(waist.stiffness, "waist_pitch_joint")
    )

    for side in ("left", "right"):
        for axis in ("pitch", "yaw"):
            joint = f"{side}_wrist_{axis}_joint"
            kp = _resolve(arms.stiffness, joint)
            effort = _resolve(arms.effort_limit_sim, joint)
            expected = 0.25 * effort / kp
            assert expected == 0.2
            assert _resolve(action_scale, joint) == expected
