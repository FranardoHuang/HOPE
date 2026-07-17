"""Focused tests for bypassing A3's URDF importer with a pre-converted USD."""

from __future__ import annotations

import ast
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


class _UrdfFileCfg(_Cfg):
    pass


class _UsdFileCfg(_Cfg):
    pass


class _JointDriveCfg(_Cfg):
    class PDGainsCfg(_Cfg):
        pass


def _spawn_factory():
    tree = ast.parse(ROBOT_CFG.read_text(encoding="utf-8"), filename=str(ROBOT_CFG))
    matches = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_make_agibot_a3_spawn_cfg"]
    assert len(matches) == 1
    function_module = ast.Module(body=matches, type_ignores=[])
    ast.fix_missing_locations(function_module)
    sim_utils = SimpleNamespace(
        RigidBodyPropertiesCfg=_Cfg,
        ArticulationRootPropertiesCfg=_Cfg,
        UsdFileCfg=_UsdFileCfg,
        UrdfFileCfg=_UrdfFileCfg,
        UrdfConverterCfg=SimpleNamespace(JointDriveCfg=_JointDriveCfg),
    )
    namespace = {
        "os": __import__("os"),
        "sim_utils": sim_utils,
        "AGIBOT_A3_URDF_PATH": "/assets/agibot_a3/urdf/model.urdf",
    }
    exec(compile(function_module, str(ROBOT_CFG), "exec"), namespace)
    return namespace["_make_agibot_a3_spawn_cfg"]


def _assert_common_spawn_properties(spawn):
    assert spawn.activate_contact_sensors is True
    assert spawn.rigid_props.disable_gravity is False
    assert spawn.rigid_props.max_linear_velocity == 1000.0
    assert spawn.articulation_props.enabled_self_collisions is False
    assert spawn.articulation_props.solver_position_iteration_count == 8
    assert spawn.articulation_props.solver_velocity_iteration_count == 4


def test_default_uses_original_urdf_importer(monkeypatch):
    monkeypatch.delenv("HOPE_AGIBOT_A3_USD_PATH", raising=False)
    spawn = _spawn_factory()()
    assert isinstance(spawn, _UrdfFileCfg)
    assert spawn.asset_path == "/assets/agibot_a3/urdf/model.urdf"
    assert spawn.fix_base is False
    assert spawn.replace_cylinders_with_capsules is True
    assert isinstance(spawn.joint_drive, _JointDriveCfg)
    _assert_common_spawn_properties(spawn)


def test_preconverted_path_uses_usd_cfg_and_bypasses_urdf(monkeypatch):
    usd_path = "/workspace/codexschema/assets/a3_preconverted_usd/model.usd"
    monkeypatch.setenv("HOPE_AGIBOT_A3_USD_PATH", usd_path)
    spawn = _spawn_factory()()
    assert isinstance(spawn, _UsdFileCfg)
    assert spawn.usd_path == usd_path
    assert not hasattr(spawn, "asset_path")
    assert not hasattr(spawn, "joint_drive")
    _assert_common_spawn_properties(spawn)


def test_empty_preconverted_path_keeps_default(monkeypatch):
    monkeypatch.setenv("HOPE_AGIBOT_A3_USD_PATH", "")
    assert isinstance(_spawn_factory()(), _UrdfFileCfg)
