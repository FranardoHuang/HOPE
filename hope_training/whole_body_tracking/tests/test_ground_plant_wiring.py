"""地面摩擦 + 随机凹凸地形接线(task.plant ground/terrain keys,2026-07-22)— host-only tests.

Pinned here:

* train.py task.plant 白名单:新五键收编,未知键仍拒收;默认全缺席(或 rough 显式 null)=
  对 cfg 零改动 = 字节等价。
* 地面材质摩擦覆盖(scene.terrain.physics_material):单边/双边覆盖、动摩擦不得大于静摩擦的
  交叉检查(用"显式值或现值")、bool/NaN/负数拒收。
* 机器人 body 材质随机化范围覆盖(events.physics_material.params):写成 (lo, hi) tuple、
  长度/顺序/负数 fail-loud。
* 随机凹凸地形:plane -> generator 切换 + noise_range 传递(builder monkeypatch 检 wiring;
  真 isaaclab 构造用 importorskip);hi<=0 / hi>0.5 / 非 plane 起点拒收。
* schema-3 ground_plant 合同块(training_contract):默认配方 -> None(不落键,历史 checkpoint
  逐字节兼容);任何偏离 -> 完整块;validator 拒 null 拼写、拒"写着默认值的块"、拒篡改/缺键/
  多键;train.py _ground_plant_contract 从 POST-OVERRIDE cfg 读出正确块。

Run:  python -m pytest hope_training/whole_body_tracking/tests/test_ground_plant_wiring.py -q
"""

from __future__ import annotations

import importlib.util
import types
from pathlib import Path

import pytest

from test_reward_flags_overrides import _NS, _Term, _make_env_cfg, train_mod

ROOT = Path(__file__).resolve().parents[1]
TC_PATH = (
    ROOT / "source/whole_body_tracking/whole_body_tracking/utils/training_contract.py"
)
_SPEC = importlib.util.spec_from_file_location("tc_ground_plant_under_test", TC_PATH)
TC = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(TC)

_DEFAULT_STATIC_RANGE = (0.3, 1.6)
_DEFAULT_DYNAMIC_RANGE = (0.3, 1.2)


def _ground_env_cfg():
    """现役 cfg 形状:plane 地形 + 1.0/1.0 地面材质 + (0.3,1.6)/(0.3,1.2) 机器人材质随机化。"""
    cfg = _make_env_cfg()
    cfg.scene = _NS(
        terrain=_NS(
            terrain_type="plane",
            terrain_generator=None,
            physics_material=_NS(static_friction=1.0, dynamic_friction=1.0),
        )
    )
    cfg.events = _NS(
        physics_material=_Term(
            weight=None,
            params={
                "static_friction_range": _DEFAULT_STATIC_RANGE,
                "dynamic_friction_range": _DEFAULT_DYNAMIC_RANGE,
                "restitution_range": (0.0, 0.5),
                "num_buckets": 64,
            },
        )
    )
    return cfg


def _apply_plant(plant, cfg=None):
    cfg = cfg if cfg is not None else _ground_env_cfg()
    applied = train_mod._apply_task_overrides(cfg, {"plant": plant}, clip_name=None)
    return cfg, applied


# --------------------------------------------------------------------------------------------- #
# whitelist + default path
# --------------------------------------------------------------------------------------------- #
def test_plant_whitelist_accepts_new_keys_and_rejects_unknown():
    for key in (
        "ground_static_friction",
        "ground_dynamic_friction",
        "robot_material_static_friction_range",
        "robot_material_dynamic_friction_range",
        "terrain_rough_height_range",
    ):
        assert key in train_mod._PLANT_KEYS
    with pytest.raises(train_mod._OverrideError, match="does not consume"):
        _apply_plant({"ground_statik_friction": 0.8})


def test_absent_and_null_rough_are_byte_identical():
    for plant in (None, {}, {"terrain_rough_height_range": None}):
        cfg, applied = _apply_plant(plant)
        material = cfg.scene.terrain.physics_material
        assert (material.static_friction, material.dynamic_friction) == (1.0, 1.0)
        assert cfg.scene.terrain.terrain_type == "plane"
        params = cfg.events.physics_material.params
        assert params["static_friction_range"] == _DEFAULT_STATIC_RANGE
        assert params["dynamic_friction_range"] == _DEFAULT_DYNAMIC_RANGE
        assert not [line for line in applied if "terrain" in line or "friction" in line]


# --------------------------------------------------------------------------------------------- #
# ground material friction
# --------------------------------------------------------------------------------------------- #
def test_ground_friction_overrides_apply_and_log():
    cfg, applied = _apply_plant(
        {"ground_static_friction": 0.7, "ground_dynamic_friction": 0.55}
    )
    material = cfg.scene.terrain.physics_material
    assert (material.static_friction, material.dynamic_friction) == (0.7, 0.55)
    assert "scene.terrain.physics_material.static_friction=0.7" in applied
    assert "scene.terrain.physics_material.dynamic_friction=0.55" in applied


def test_ground_friction_one_sided_override_keeps_other_side():
    cfg, _ = _apply_plant({"ground_static_friction": 1.4})
    material = cfg.scene.terrain.physics_material
    assert (material.static_friction, material.dynamic_friction) == (1.4, 1.0)


def test_ground_dynamic_exceeding_static_is_refused_even_one_sided():
    # 显式动摩擦 1.2 > 现值静摩擦 1.0:单边覆盖也要交叉检查
    with pytest.raises(train_mod._OverrideError, match="must not exceed static"):
        _apply_plant({"ground_dynamic_friction": 1.2})
    with pytest.raises(train_mod._OverrideError, match="must not exceed static"):
        _apply_plant({"ground_static_friction": 0.5, "ground_dynamic_friction": 0.8})
    # 一起抬高且 static >= dynamic 合法
    cfg, _ = _apply_plant({"ground_static_friction": 1.5, "ground_dynamic_friction": 1.2})
    assert cfg.scene.terrain.physics_material.static_friction == 1.5


@pytest.mark.parametrize("bad", [True, float("nan"), float("inf"), -0.1, "x", [1.0]])
def test_ground_friction_rejects_bad_values(bad):
    with pytest.raises(train_mod._OverrideError, match="finite number >= 0"):
        _apply_plant({"ground_static_friction": bad})


def test_ground_friction_requires_terrain_material_node():
    cfg = _ground_env_cfg()
    cfg.scene.terrain.physics_material = None
    with pytest.raises(train_mod._OverrideError, match="physics_material"):
        _apply_plant({"ground_static_friction": 0.8}, cfg=cfg)


# --------------------------------------------------------------------------------------------- #
# robot-body material randomization ranges
# --------------------------------------------------------------------------------------------- #
def test_robot_material_ranges_apply_as_tuples():
    cfg, applied = _apply_plant({
        "robot_material_static_friction_range": [0.6, 1.6],
        "robot_material_dynamic_friction_range": [0.5, 1.2],
    })
    params = cfg.events.physics_material.params
    assert params["static_friction_range"] == (0.6, 1.6)
    assert params["dynamic_friction_range"] == (0.5, 1.2)
    assert params["restitution_range"] == (0.0, 0.5)  # 邻居参数不动
    assert "events.physics_material.params.static_friction_range=(0.6, 1.6)" in applied
    assert "events.physics_material.params.dynamic_friction_range=(0.5, 1.2)" in applied


@pytest.mark.parametrize(
    "bad",
    [[0.5], [0.5, 0.6, 0.7], [1.0, 0.5], [-0.1, 1.0], [True, 1.0], "0.5,1.0", 0.5],
)
def test_robot_material_range_shapes_are_fail_loud(bad):
    with pytest.raises(train_mod._OverrideError):
        _apply_plant({"robot_material_static_friction_range": bad})


def test_robot_material_range_requires_event_term():
    cfg = _ground_env_cfg()
    cfg.events.physics_material = None
    with pytest.raises(train_mod._OverrideError, match="events.physics_material"):
        _apply_plant({"robot_material_static_friction_range": [0.6, 1.6]}, cfg=cfg)


# --------------------------------------------------------------------------------------------- #
# rough terrain
# --------------------------------------------------------------------------------------------- #
def test_rough_terrain_switches_plane_to_generator(monkeypatch):
    sentinel = object()
    seen = []

    def fake_builder(height_range):
        seen.append(tuple(height_range))
        return sentinel

    monkeypatch.setattr(train_mod, "_build_rough_terrain_generator_cfg", fake_builder)
    cfg, applied = _apply_plant({"terrain_rough_height_range": [0.02, 0.06]})
    assert cfg.scene.terrain.terrain_type == "generator"
    assert cfg.scene.terrain.terrain_generator is sentinel
    assert seen == [(0.02, 0.06)]
    assert any("fresh-from-random" in line for line in applied)


def test_rough_terrain_composes_with_friction_overrides(monkeypatch):
    monkeypatch.setattr(
        train_mod, "_build_rough_terrain_generator_cfg", lambda height_range: "GEN"
    )
    cfg, _ = _apply_plant({
        "ground_static_friction": 0.9,
        "ground_dynamic_friction": 0.8,
        "terrain_rough_height_range": [0.02, 0.06],
    })
    material = cfg.scene.terrain.physics_material
    assert (material.static_friction, material.dynamic_friction) == (0.9, 0.8)
    assert cfg.scene.terrain.terrain_type == "generator"


@pytest.mark.parametrize(
    "bad,msg",
    [
        ([0.0, 0.0], "hi > 0"),
        ([0.02, 0.6], "0.5 m"),
        ([0.06, 0.02], "lo <= hi"),
        ([-0.01, 0.05], ">= 0"),
        ([0.02], "pair"),
    ],
)
def test_rough_terrain_range_fail_loud(bad, msg):
    with pytest.raises(train_mod._OverrideError, match=msg):
        _apply_plant({"terrain_rough_height_range": bad})


def test_rough_terrain_refuses_non_plane_start():
    cfg = _ground_env_cfg()
    cfg.scene.terrain.terrain_type = "generator"
    with pytest.raises(train_mod._OverrideError, match="terrain_type=='plane'"):
        _apply_plant({"terrain_rough_height_range": [0.02, 0.06]}, cfg=cfg)


def test_real_isaaclab_rough_generator_cfg():
    pytest.importorskip("isaaclab.terrains")
    gen = train_mod._build_rough_terrain_generator_cfg((0.02, 0.06))
    sub = gen.sub_terrains["random_rough"]
    assert sub.noise_range == (0.02, 0.06)
    assert sub.proportion == 1.0
    assert gen.use_cache is False


# --------------------------------------------------------------------------------------------- #
# schema-3 ground_plant contract block
# --------------------------------------------------------------------------------------------- #
_DEFAULT_BLOCK_KWARGS = dict(
    ground_static_friction=1.0,
    ground_dynamic_friction=1.0,
    robot_material_static_friction_range=[0.3, 1.6],
    robot_material_dynamic_friction_range=[0.3, 1.2],
    terrain_type=TC.GROUND_PLANT_TERRAIN_PLANE,
    terrain_rough_height_range_m=None,
)


def test_block_default_plant_is_spelled_by_none():
    assert TC.ground_plant_block(**_DEFAULT_BLOCK_KWARGS) is None


def test_block_any_deviation_yields_full_block():
    block = TC.ground_plant_block(
        **{**_DEFAULT_BLOCK_KWARGS, "ground_dynamic_friction": 0.7}
    )
    assert block == {
        "schema_version": 1,
        "ground_static_friction": 1.0,
        "ground_dynamic_friction": 0.7,
        "robot_material_static_friction_range": [0.3, 1.6],
        "robot_material_dynamic_friction_range": [0.3, 1.2],
        "terrain_type": "plane",
        "terrain_rough_height_range_m": None,
    }


def test_block_rough_terrain_requires_range_and_plane_refuses_it():
    block = TC.ground_plant_block(
        **{
            **_DEFAULT_BLOCK_KWARGS,
            "terrain_type": TC.GROUND_PLANT_TERRAIN_ROUGH,
            "terrain_rough_height_range_m": [0.02, 0.06],
        }
    )
    assert block["terrain_type"] == "random_rough_heightfield"
    assert block["terrain_rough_height_range_m"] == [0.02, 0.06]
    with pytest.raises(ValueError, match="must not carry"):
        TC.ground_plant_block(
            **{**_DEFAULT_BLOCK_KWARGS, "terrain_rough_height_range_m": [0.02, 0.06]}
        )
    with pytest.raises(ValueError, match="hi > 0"):
        TC.ground_plant_block(
            **{
                **_DEFAULT_BLOCK_KWARGS,
                "terrain_type": TC.GROUND_PLANT_TERRAIN_ROUGH,
                "terrain_rough_height_range_m": [0.0, 0.0],
            }
        )


def test_block_dynamic_over_static_and_bad_ranges_fail():
    with pytest.raises(ValueError, match="must not exceed static"):
        TC.ground_plant_block(
            **{**_DEFAULT_BLOCK_KWARGS, "ground_dynamic_friction": 1.3}
        )
    with pytest.raises(ValueError, match="lo <= hi"):
        TC.ground_plant_block(
            **{
                **_DEFAULT_BLOCK_KWARGS,
                "robot_material_static_friction_range": [1.6, 0.3],
            }
        )


def _valid_rough_block():
    return TC.ground_plant_block(
        **{
            **_DEFAULT_BLOCK_KWARGS,
            "terrain_type": TC.GROUND_PLANT_TERRAIN_ROUGH,
            "terrain_rough_height_range_m": [0.02, 0.06],
        }
    )


def test_validator_absent_ok_null_and_default_refused():
    TC._validate_ground_plant_contract({})  # absent = 默认 plant,合法
    with pytest.raises(ValueError, match="not null"):
        TC._validate_ground_plant_contract({"ground_plant": None})
    default_spelled_out = {
        "schema_version": 1,
        **{k: v for k, v in _DEFAULT_BLOCK_KWARGS.items()},
    }
    with pytest.raises(ValueError, match="omitting the block"):
        TC._validate_ground_plant_contract({"ground_plant": default_spelled_out})


def test_validator_accepts_canonical_and_rejects_tampering():
    block = _valid_rough_block()
    TC._validate_ground_plant_contract({"ground_plant": block})
    tampered = dict(block)
    tampered["ground_dynamic_friction"] = 1.4  # > static -> re-assembly 拒绝
    with pytest.raises(ValueError, match="ground_plant is invalid"):
        TC._validate_ground_plant_contract({"ground_plant": tampered})
    missing = {k: v for k, v in block.items() if k != "terrain_type"}
    with pytest.raises(ValueError, match="ground_plant"):
        TC._validate_ground_plant_contract({"ground_plant": missing})
    extra = dict(block, smuggled=1)
    with pytest.raises(ValueError, match="ground_plant"):
        TC._validate_ground_plant_contract({"ground_plant": extra})
    wrong_schema = dict(block, schema_version=2)
    with pytest.raises(ValueError, match="schema_version"):
        TC._validate_ground_plant_contract({"ground_plant": wrong_schema})


# --------------------------------------------------------------------------------------------- #
# train.py _ground_plant_contract reads the POST-OVERRIDE cfg
# --------------------------------------------------------------------------------------------- #
def test_contract_reader_default_cfg_yields_none():
    cfg = _ground_env_cfg()
    assert train_mod._ground_plant_contract(cfg) is None


def test_contract_reader_fingerprints_all_overrides():
    cfg, _ = _apply_plant({
        "ground_static_friction": 0.9,
        "ground_dynamic_friction": 0.8,
        "robot_material_static_friction_range": [0.6, 1.6],
    })
    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.terrain_generator = types.SimpleNamespace(
        sub_terrains={"random_rough": types.SimpleNamespace(noise_range=(0.02, 0.06))}
    )
    block = train_mod._ground_plant_contract(cfg)
    assert block == {
        "schema_version": 1,
        "ground_static_friction": 0.9,
        "ground_dynamic_friction": 0.8,
        "robot_material_static_friction_range": [0.6, 1.6],
        "robot_material_dynamic_friction_range": [0.3, 1.2],
        "terrain_type": "random_rough_heightfield",
        "terrain_rough_height_range_m": [0.02, 0.06],
    }
    # 老谱系 resume 对账:默认合同没有 ground_plant 键,出现即 diff 不匹配
    diffs = train_mod._contract_diff({}, {"ground_plant": block})
    assert diffs


def test_contract_reader_fail_loud_on_odd_terrain():
    cfg = _ground_env_cfg()
    cfg.scene.terrain.terrain_type = "usd"
    with pytest.raises(RuntimeError, match="cannot fingerprint"):
        train_mod._ground_plant_contract(cfg)
    cfg2 = _ground_env_cfg()
    cfg2.scene.terrain.terrain_type = "generator"
    cfg2.scene.terrain.terrain_generator = types.SimpleNamespace(
        sub_terrains={"pyramids": object()}
    )
    with pytest.raises(RuntimeError, match="random_rough"):
        train_mod._ground_plant_contract(cfg2)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
