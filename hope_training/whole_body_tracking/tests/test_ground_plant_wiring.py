"""地面摩擦 + 随机凹凸地形接线(task.plant ground/terrain keys,2026-07-22)— host-only tests.

Pinned here:

* train.py task.plant 白名单:新五键收编,未知键仍拒收;默认全缺席(或 rough 显式 null)=
  对 cfg 零改动 = 字节等价。
* 地面材质摩擦覆盖(scene.terrain.physics_material):单边/双边覆盖、动摩擦不得大于静摩擦的
  交叉检查(用"显式值或现值")、bool/NaN/负数拒收。
* 机器人 body 材质随机化范围覆盖(events.physics_material.params):写成 (lo, hi) tuple、
  长度/顺序/负数 fail-loud。
* 随机凹凸地垫(2026-07-29 抬脚地形修复):plane 起点 -> _attach_rough_ground_patch seam
  (monkeypatch 检 wiring;垫子几何的纯 numpy 性质在 test_terrain_patch.py);hi<=0 / hi>0.5 /
  带宽<0.01 / 非 plane 起点拒收;摩擦覆盖先落盘、垫子继承 POST-OVERRIDE 材质。
* schema-3 ground_plant 合同块(training_contract):默认配方 -> None(不落键,历史 checkpoint
  逐字节兼容);任何偏离 -> 完整块;validator 拒 null 拼写、拒"写着默认值的块"、拒篡改/缺键/
  多键;train.py _ground_plant_contract 从 POST-OVERRIDE cfg 读出正确块。

Run:  python -m pytest hope_training/whole_body_tracking/tests/test_ground_plant_wiring.py -q
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

from test_reward_flags_overrides import _NS, _Term, _apply_legacy_v1, _make_env_cfg, train_mod

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
    # 2026-07-25 默认翻转后本套件仍测 legacy 翻译行为:钉 v1 + 滤 v1 记账行,原断言原样成立。
    cfg = cfg if cfg is not None else _ground_env_cfg()
    applied = _apply_legacy_v1(cfg, {"plant": plant})
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
        "robot_material_make_consistent",
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


def test_robot_material_make_consistent_true_writes_param_false_is_noop():
    # true:逐桶 dynamic=min(static, dynamic) 落进 event params;false/null = 缺席,零改动
    cfg, applied = _apply_plant({"robot_material_make_consistent": True})
    assert cfg.events.physics_material.params["make_consistent"] is True
    assert any("make_consistent=True" in line for line in applied)
    for spelled_false in (False, None):
        cfg, applied = _apply_plant({"robot_material_make_consistent": spelled_false})
        assert "make_consistent" not in cfg.events.physics_material.params
        assert not [line for line in applied if "make_consistent" in line]


@pytest.mark.parametrize("bad", ["yes", 1, 0, [True]])
def test_robot_material_make_consistent_rejects_non_bool(bad):
    with pytest.raises(train_mod._OverrideError, match="must be a bool"):
        _apply_plant({"robot_material_make_consistent": bad})


# --------------------------------------------------------------------------------------------- #
# rough terrain (2026-07-29 per-env zero-mean pad; the pure geometry lives in
# test_terrain_patch.py — here we pin the train.py wiring through the _attach seam)
# --------------------------------------------------------------------------------------------- #
def test_rough_terrain_calls_attach_seam_with_authored_band(monkeypatch):
    seen = []

    def fake_attach(env_cfg, height_range):
        seen.append(tuple(height_range))
        env_cfg.scene.terrain = None  # 真身的行为:摘掉 plane importer
        env_cfg.scene.rough_ground_patch = "PATCH"
        return ["fake pad line (fresh-from-random)"]

    monkeypatch.setattr(train_mod, "_attach_rough_ground_patch", fake_attach)
    cfg, applied = _apply_plant({"terrain_rough_height_range": [0.02, 0.06]})
    assert seen == [(0.02, 0.06)]
    assert cfg.scene.terrain is None
    assert cfg.scene.rough_ground_patch == "PATCH"
    assert "fake pad line (fresh-from-random)" in applied


def test_rough_terrain_composes_with_friction_overrides(monkeypatch):
    # 摩擦覆盖在挂垫之前落盘:seam 被调用时垫子继承的是 POST-OVERRIDE 地面材质。
    seen_material = []

    def fake_attach(env_cfg, height_range):
        material = env_cfg.scene.terrain.physics_material
        seen_material.append((material.static_friction, material.dynamic_friction))
        return []

    monkeypatch.setattr(train_mod, "_attach_rough_ground_patch", fake_attach)
    _apply_plant({
        "ground_static_friction": 0.9,
        "ground_dynamic_friction": 0.8,
        "terrain_rough_height_range": [0.02, 0.06],
    })
    assert seen_material == [(0.9, 0.8)]


@pytest.mark.parametrize(
    "bad,msg",
    [
        ([0.0, 0.0], "hi > 0"),
        ([0.02, 0.6], "0.5 m"),
        ([0.06, 0.02], "lo <= hi"),
        ([-0.01, 0.05], ">= 0"),
        ([0.02], "pair"),
        ([0.02, 0.025], "0.01 m"),  # 居中后 ±0.0025 会量化成死平垫,拒绝
        ([0.0, 0.2], "0.15 m"),  # 斜坡竖墙修正会破坏桌侧平区边界,拒绝
        ([0.0, 0.015], "multiple"),  # 非 0.01 倍数会悄悄铺出别的幅度,拒绝
    ],
)
def test_rough_terrain_range_fail_loud(bad, msg):
    with pytest.raises(train_mod._OverrideError, match=msg):
        _apply_plant({"terrain_rough_height_range": bad})


def test_rough_terrain_nominal_1cm_band_survives_float_noise(monkeypatch):
    # 0.03-0.02 = 0.00999...:名义 1 cm 带不许被浮点噪声拒掉(train.py 与合同同 epsilon)
    seen = []
    monkeypatch.setattr(
        train_mod, "_attach_rough_ground_patch",
        lambda env_cfg, hr: (seen.append(tuple(hr)), [])[1],
    )
    _apply_plant({"terrain_rough_height_range": [0.02, 0.03]})
    assert seen == [(0.02, 0.03)]
    block = TC.ground_plant_block(
        **{
            **_DEFAULT_BLOCK_KWARGS,
            "terrain_type": TC.GROUND_PLANT_TERRAIN_ROUGH,
            "terrain_rough_height_range_m": [0.02, 0.03],
        }
    )
    assert block["terrain_rough_height_range_m"] == [0.02, 0.03]


def test_rough_terrain_refuses_non_plane_start():
    cfg = _ground_env_cfg()
    cfg.scene.terrain.terrain_type = "generator"
    with pytest.raises(train_mod._OverrideError, match="terrain_type=='plane'"):
        _apply_plant({"terrain_rough_height_range": [0.02, 0.06]}, cfg=cfg)


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
    assert block["terrain_type"] == "robot_side_correlated_spawn_flat_v2"
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
    # 带宽合法域与 train.py 同一套红线:<0.01 死平垫 / >0.15 竖墙修正破平区 / 非 0.01 倍数
    for bad_range, msg in (
        ([0.02, 0.025], "0.01 m"),
        ([0.0, 0.2], "0.15 m"),
        ([0.0, 0.015], "multiple"),
    ):
        with pytest.raises(ValueError, match=msg):
            TC.ground_plant_block(
                **{
                    **_DEFAULT_BLOCK_KWARGS,
                    "terrain_type": TC.GROUND_PLANT_TERRAIN_ROUGH,
                    "terrain_rough_height_range_m": bad_range,
                }
            )


def test_block_make_consistent_true_spawns_key_false_is_spelled_by_omission():
    # true 单独就是一套新 plant(独立采样 vs min 钳制是不同物理),块里长出第 7 键
    block = TC.ground_plant_block(
        **{**_DEFAULT_BLOCK_KWARGS}, robot_material_make_consistent=True
    )
    assert block["robot_material_make_consistent"] is True
    TC._validate_ground_plant_contract({"ground_plant": block})
    # false = 键缺席;显式写 false 的块被拒(一套 plant 只有一种拼写)
    assert (
        TC.ground_plant_block(
            **{**_DEFAULT_BLOCK_KWARGS}, robot_material_make_consistent=False
        )
        is None
    )
    spelled_false = {k: v for k, v in block.items()}
    spelled_false["robot_material_make_consistent"] = False
    with pytest.raises(ValueError, match="omitting the key"):
        TC._validate_ground_plant_contract({"ground_plant": spelled_false})
    with pytest.raises(ValueError, match="must be a bool"):
        TC.ground_plant_block(
            **{**_DEFAULT_BLOCK_KWARGS}, robot_material_make_consistent="yes"
        )


def test_contract_reader_fingerprints_make_consistent():
    cfg, _ = _apply_plant({"robot_material_make_consistent": True})
    block = train_mod._ground_plant_contract(cfg)
    assert block["robot_material_make_consistent"] is True
    # 其余键仍是历史默认拼写
    assert block["robot_material_static_friction_range"] == [0.3, 1.6]
    assert block["terrain_type"] == "plane"


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
    # 模拟 attach_rough_ground_patch 的真身效果:摘掉 plane importer,挂上零均值凹凸垫。
    # 指纹从垫子 spawn cfg 读 AUTHORED 带宽 + POST-OVERRIDE 地面材质。
    material = cfg.scene.terrain.physics_material
    cfg.scene.terrain = None
    cfg.scene.rough_ground_patch = _NS(
        spawn=_NS(height_range_m=(0.02, 0.06), physics_material=material)
    )
    block = train_mod._ground_plant_contract(cfg)
    assert block == {
        "schema_version": 1,
        "ground_static_friction": 0.9,
        "ground_dynamic_friction": 0.8,
        "robot_material_static_friction_range": [0.6, 1.6],
        "robot_material_dynamic_friction_range": [0.3, 1.2],
        "terrain_type": "robot_side_correlated_spawn_flat_v2",
        "terrain_rough_height_range_m": [0.02, 0.06],
    }
    # 老谱系 resume 对账:默认合同没有 ground_plant 键,出现即 diff 不匹配
    diffs = train_mod._contract_diff({}, {"ground_plant": block})
    assert diffs


def test_contract_reader_fail_loud_on_odd_terrain():
    # 非 plane 的 TerrainImporter(含旧 generator 血统)一律拒绝指纹:旧 broken-generator
    # checkpoint 不许被静默当成新垫子 plant。
    cfg = _ground_env_cfg()
    cfg.scene.terrain.terrain_type = "usd"
    with pytest.raises(RuntimeError, match="cannot fingerprint"):
        train_mod._ground_plant_contract(cfg)
    cfg2 = _ground_env_cfg()
    cfg2.scene.terrain.terrain_type = "generator"
    cfg2.scene.terrain.terrain_generator = types.SimpleNamespace(
        sub_terrains={"random_rough": types.SimpleNamespace(noise_range=(0.02, 0.06))}
    )
    with pytest.raises(RuntimeError, match="cannot fingerprint"):
        train_mod._ground_plant_contract(cfg2)
    # 垫子和 terrain 同时在 / 同时不在 -> 配置血统不对,拒绝
    cfg3 = _ground_env_cfg()
    cfg3.scene.rough_ground_patch = _NS(spawn=_NS(height_range_m=(0.02, 0.06)))
    with pytest.raises(RuntimeError, match="exactly one of"):
        train_mod._ground_plant_contract(cfg3)
    cfg4 = _ground_env_cfg()
    cfg4.scene.terrain = None
    with pytest.raises(RuntimeError, match="exactly one of"):
        train_mod._ground_plant_contract(cfg4)
    # 垫子在但 spawn 带宽缺失/畸形 -> fail-loud
    cfg5 = _ground_env_cfg()
    cfg5.scene.terrain = None
    cfg5.scene.rough_ground_patch = _NS(spawn=_NS(height_range_m=None))
    with pytest.raises(RuntimeError, match="height_range_m"):
        train_mod._ground_plant_contract(cfg5)


# --------------------------------------------------------------------------------------------- #
# 2026-08-06:DR-L0/DR-L0N 把 events.physics_material 整个删掉后,指纹怎么诚实地记这件事
# --------------------------------------------------------------------------------------------- #
_ABSENT_BLOCK_KWARGS = dict(
    ground_static_friction=1.0,
    ground_dynamic_friction=1.0,
    robot_material_static_friction_range=None,
    robot_material_dynamic_friction_range=None,
    terrain_type=TC.GROUND_PLANT_TERRAIN_PLANE,
    terrain_rough_height_range_m=None,
)


def _dr_l0_cfg(*, marker="l0"):
    """现役 cfg,但 events.physics_material 已被 DR-L0/DR-L0N finalizer 置空 + 落下 marker。"""
    cfg = _ground_env_cfg()
    cfg.events.physics_material = None
    if marker == "l0":
        setattr(
            cfg,
            train_mod._ACTION_BALL_DR_L0_RUNTIME_ATTR,
            train_mod._action_ball_dr_l0_contract_payload(),
        )
    elif marker == "l0n":
        setattr(
            cfg,
            train_mod._ACTION_BALL_DR_L0N_RUNTIME_ATTR,
            train_mod._training_contract_module().action_ball_dr_l0n_contract_payload(),
        )
    return cfg


def test_block_randomization_absent_spells_null_ranges_and_round_trips():
    block = TC.ground_plant_block(
        **_ABSENT_BLOCK_KWARGS, robot_material_randomization_absent=True
    )
    assert block["robot_material_randomization_absent"] is True
    assert block["robot_material_static_friction_range"] is None
    assert block["robot_material_dynamic_friction_range"] is None
    # 事件不在场必然偏离历史默认 -> 落完整块,不是 None
    TC._validate_ground_plant_contract({"ground_plant": block})
    # false 的唯一拼写仍是键缺席
    spelled_false = dict(block)
    spelled_false["robot_material_randomization_absent"] = False
    with pytest.raises(ValueError, match="omitting the key"):
        TC._validate_ground_plant_contract({"ground_plant": spelled_false})


def test_block_randomization_absent_refuses_numbers_and_make_consistent():
    # 填回 base cfg 的范围 = 在收据里谎称跑过随机化,拒收
    with pytest.raises(ValueError, match="ranges to be null"):
        TC.ground_plant_block(
            **{
                **_ABSENT_BLOCK_KWARGS,
                "robot_material_static_friction_range": [0.3, 1.6],
            },
            robot_material_randomization_absent=True,
        )
    # 事件都没了就没有 make_consistent 可言
    with pytest.raises(ValueError, match="property of the removed randomization event"):
        TC.ground_plant_block(
            **_ABSENT_BLOCK_KWARGS,
            robot_material_randomization_absent=True,
            robot_material_make_consistent=True,
        )
    with pytest.raises(ValueError, match="randomization_absent must be a bool"):
        TC.ground_plant_block(
            **_ABSENT_BLOCK_KWARGS, robot_material_randomization_absent="yes"
        )
    # 不声明 absent 却传 null 范围 -> 老信封原样 fail-loud
    with pytest.raises(ValueError, match=r"\[lo, hi\] pair"):
        TC.ground_plant_block(**_ABSENT_BLOCK_KWARGS)


@pytest.mark.parametrize("marker", ["l0", "l0n"])
def test_contract_reader_fingerprints_removed_robot_material_event(marker):
    block = train_mod._ground_plant_contract(_dr_l0_cfg(marker=marker))
    assert block == {
        "schema_version": 1,
        "ground_static_friction": 1.0,
        "ground_dynamic_friction": 1.0,
        "robot_material_static_friction_range": None,
        "robot_material_dynamic_friction_range": None,
        "terrain_type": "plane",
        "terrain_rough_height_range_m": None,
        "robot_material_randomization_absent": True,
    }
    TC._validate_ground_plant_contract({"ground_plant": block})
    # 全 DR 谱系(不落键)与 DR-L0 谱系(落这个块)在 resume 对账上互相拒绝
    assert train_mod._contract_diff({}, {"ground_plant": block})


def test_contract_reader_still_fails_closed_without_the_finalizer_proof():
    # marker 缺席 = 不是 DR-L0 的定义删的,门一点没松
    cfg = _ground_env_cfg()
    cfg.events.physics_material = None
    with pytest.raises(RuntimeError, match="requires events.physics_material.params"):
        train_mod._ground_plant_contract(cfg)
    # 槽整个消失(不是 finalizer 写的那种最终状态)也照旧拒收
    cfg2 = _dr_l0_cfg()
    del cfg2.events.physics_material
    with pytest.raises(RuntimeError, match="requires events.physics_material.params"):
        train_mod._ground_plant_contract(cfg2)
    # marker 被篡改 -> 拒绝拿它当指纹依据
    cfg3 = _dr_l0_cfg()
    tampered = train_mod._action_ball_dr_l0_contract_payload()
    tampered["policy_observation_corruption"] = True
    setattr(cfg3, train_mod._ACTION_BALL_DR_L0_RUNTIME_ATTR, tampered)
    with pytest.raises(RuntimeError, match="differs from its canonical payload"):
        train_mod._ground_plant_contract(cfg3)


def test_fresh_full_mdp_contract_uses_installed_reset_only_event_graph(monkeypatch):
    cfg = _ground_env_cfg()
    del cfg.events.physics_material
    expected_reset = object()
    mdp_module = types.ModuleType("whole_body_tracking.tasks.tracking.mdp")
    mdp_module.reset_action_ball_full_mdp_robot_to_default = expected_reset
    monkeypatch.setitem(
        sys.modules, "whole_body_tracking.tasks.tracking.mdp", mdp_module
    )

    class _EventManager:
        active_terms = {"reset": ["action_ball_full_mdp_robot_reset"]}

        def __init__(self, *, func=expected_reset, mode="reset", params=None):
            self.term_cfg = _NS(
                func=func,
                mode=mode,
                params={"asset_cfg": _NS(name="robot")} if params is None else params,
            )

        def get_term_cfg(self, name):
            assert name == "action_ball_full_mdp_robot_reset"
            return self.term_cfg

    block = train_mod._ground_plant_contract(
        cfg,
        fresh_full_mdp_event_manager=_EventManager(),
    )
    assert block["robot_material_randomization_absent"] is True
    assert block["robot_material_static_friction_range"] is None
    assert block["robot_material_dynamic_friction_range"] is None
    TC._validate_ground_plant_contract({"ground_plant": block})

    with pytest.raises(RuntimeError, match="installed reset-only Event graph"):
        manager = _EventManager()
        manager.active_terms = {
            "reset": [
                "action_ball_full_mdp_robot_reset",
                "physics_material",
            ]
        }
        train_mod._ground_plant_contract(
            cfg,
            fresh_full_mdp_event_manager=manager,
        )
    with pytest.raises(RuntimeError, match="exact reset-mode graph"):
        manager = _EventManager()
        manager.active_terms = {
            "startup": ["action_ball_full_mdp_robot_reset"]
        }
        train_mod._ground_plant_contract(
            cfg,
            fresh_full_mdp_event_manager=manager,
        )
    for manager in (
        _EventManager(func=object()),
        _EventManager(mode="startup"),
        _EventManager(params={"asset_cfg": _NS(name="table")}),
        _EventManager(
            params={
                "asset_cfg": _NS(name="robot"),
                "material_range": (0.3, 1.6),
            }
        ),
    ):
        with pytest.raises(RuntimeError, match="deterministic robot-reset contract"):
            train_mod._ground_plant_contract(
                cfg,
                fresh_full_mdp_event_manager=manager,
            )


def test_contract_reader_refuses_authored_robot_material_under_dr_l0():
    cfg, _ = _apply_plant({"robot_material_static_friction_range": [0.6, 1.6]})
    cfg.events.physics_material = None
    setattr(
        cfg,
        train_mod._ACTION_BALL_DR_L0_RUNTIME_ATTR,
        train_mod._action_ball_dr_l0_contract_payload(),
    )
    with pytest.raises(RuntimeError, match="task.plant explicitly authored"):
        train_mod._ground_plant_contract(cfg)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
