"""venue_profile 加载器单测(host-only,纯标准库,不需要 isaaclab/torch)。

覆盖:

* 现役档案 franco_rig_20260725 可加载,值与 phase1 wave yaml / env cfg 的现役标定逐字一致;
* 裸名 vs 绝对路径解析出同一份档案(同 path、同 sha、同 profile);
* meta.sha256 = 文件字节的 sha256,跨两次加载稳定;
* 严格拒载:未知 section/键、缺 section/键、错类型(含 bool 冒充数、float 冒充 int)、
  非有限值(NaN/Infinity)、越界值(rho/dropout/delay/摩擦/弹性/质量乘子)、非法 tts_mode、
  schema_version 不符、JSON 重复键、裸名不存在(报错信息带解析出的路径)。

Run:  python -m pytest hope_training/whole_body_tracking/tests/test_venue_profile.py -q
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[3]
UTILS = ROOT / "source/whole_body_tracking/whole_body_tracking/utils"
SHIPPED = REPO / "configs/venue_profiles/franco_rig_20260725.json"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "venue_profile_under_test", UTILS / "venue_profile.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


vp = _load_module()


# ---------------------------------------------------------------------------
# 现役档案 + 名字/路径解析 + sha 稳定性
# ---------------------------------------------------------------------------


def test_shipped_profile_loads_and_matches_live_calibration():
    profile, meta = vp.load_venue_profile("franco_rig_20260725")

    # mocap_noise:phase1_integrated_upgrade_wave_20260723.yaml 现役覆写。
    assert profile["mocap_noise"] == {
        "target_noise_white": 0.0019,
        "target_noise_ar1_sigma": 0.0052,
        "target_noise_ar1_rho": 0.717,
    }
    # transport:delay=0(planner-revision guard 遗留)、无 dropout、时戳补偿模式。
    assert profile["transport"] == {
        "target_delay_steps": 0,
        "target_dropout_prob": 0.0,
        "target_delay_tts_mode": "source_timestamp_compensated",
    }
    # physics:EventCfg.physics_material + HOPEEventCfg.randomize_link_mass 现役区间,
    # 区间必须已规范化成 tuple(可直接塞 EventCfg params)。
    assert profile["physics"] == {
        "static_friction_range": (0.3, 1.6),
        "dynamic_friction_range": (0.3, 1.2),
        "restitution_range": (0.0, 0.5),
        "mass_distribution_params": (0.85, 1.15),
    }
    for key in ("static_friction_range", "mass_distribution_params"):
        assert isinstance(profile["physics"][key], tuple)

    assert meta["name"] == "franco_rig_20260725"
    assert meta["schema_version"] == vp.SCHEMA_VERSION == "venue_profile_v1"
    assert Path(meta["path"]).is_absolute()
    assert Path(meta["path"]) == SHIPPED.resolve()


def test_name_and_path_resolution_agree():
    by_name = vp.load_venue_profile("franco_rig_20260725")
    by_path = vp.load_venue_profile(str(SHIPPED))
    by_pathobj = vp.load_venue_profile(SHIPPED)
    assert by_name == by_path == by_pathobj


def test_sha256_matches_file_bytes_and_is_stable():
    _, meta1 = vp.load_venue_profile("franco_rig_20260725")
    _, meta2 = vp.load_venue_profile("franco_rig_20260725")
    expected = hashlib.sha256(SHIPPED.read_bytes()).hexdigest()
    assert meta1["sha256"] == meta2["sha256"] == expected


def test_missing_bare_name_error_names_resolved_path():
    with pytest.raises(vp.VenueProfileError, match="no_such_venue_xyz"):
        vp.load_venue_profile("no_such_venue_xyz")
    # 报错信息必须带解析出的目录/路径,方便一眼看出去哪找了。
    with pytest.raises(vp.VenueProfileError, match="configs"):
        vp.load_venue_profile("no_such_venue_xyz")


def test_bare_name_charset_rejected():
    with pytest.raises(vp.VenueProfileError):
        vp.load_venue_profile("-leading-dash")
    with pytest.raises(vp.VenueProfileError):
        vp.load_venue_profile("")


def test_explicit_path_must_end_in_json(tmp_path):
    bogus = tmp_path / "profile.yaml"
    bogus.write_text("{}", encoding="utf-8")
    with pytest.raises(vp.VenueProfileError, match=r"\.json"):
        vp.load_venue_profile(str(bogus))


# ---------------------------------------------------------------------------
# 严格 schema 拒载:统一从现役档案出发做单点突变。
# ---------------------------------------------------------------------------


def _valid_dict():
    return json.loads(SHIPPED.read_text(encoding="utf-8"))


def _write(tmp_path, data, name="mutant.json"):
    path = tmp_path / name
    if isinstance(data, str):
        path.write_text(data, encoding="utf-8")
    else:
        path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def _expect_reject(tmp_path, data, match=None):
    with pytest.raises(vp.VenueProfileError, match=match):
        vp.load_venue_profile(_write(tmp_path, data))


def test_valid_mutant_roundtrip(tmp_path):
    # 夹具本身健康:未突变的副本必须能加载,且 profile 与现役档案一致(sha 不同,因为
    # json.dumps 字节不同 —— 这也是 sha 跟字节走、不跟语义走的直接验证)。
    base = _valid_dict()
    path = _write(tmp_path, base)
    profile, meta = vp.load_venue_profile(path)
    shipped_profile, shipped_meta = vp.load_venue_profile("franco_rig_20260725")
    assert profile == shipped_profile
    assert meta["sha256"] != shipped_meta["sha256"]
    assert meta["name"] == "mutant"


def test_unknown_top_level_section_rejected(tmp_path):
    base = _valid_dict()
    base["ball_physics"] = {"restitution": 0.9}
    _expect_reject(tmp_path, base, match="未知键")


def test_missing_section_rejected(tmp_path):
    for section in ("mocap_noise", "transport", "physics"):
        base = _valid_dict()
        del base[section]
        _expect_reject(tmp_path, base, match="缺少必需键")


def test_missing_schema_version_rejected(tmp_path):
    base = _valid_dict()
    del base["schema_version"]
    _expect_reject(tmp_path, base, match="schema_version")


def test_wrong_schema_version_rejected(tmp_path):
    base = _valid_dict()
    base["schema_version"] = "venue_profile_v0"
    _expect_reject(tmp_path, base, match="schema_version")


def test_unknown_key_in_section_rejected(tmp_path):
    base = _valid_dict()
    base["mocap_noise"]["target_noise_pink"] = 0.001
    _expect_reject(tmp_path, base, match="未知键")

    base = _valid_dict()
    base["physics"]["ground_friction"] = [0.5, 1.0]
    _expect_reject(tmp_path, base, match="未知键")


def test_missing_key_in_section_rejected(tmp_path):
    base = _valid_dict()
    del base["transport"]["target_delay_tts_mode"]
    _expect_reject(tmp_path, base, match="缺少必需键")


def test_wrong_types_rejected(tmp_path):
    # 字符串冒充 float。
    base = _valid_dict()
    base["mocap_noise"]["target_noise_white"] = "0.0019"
    _expect_reject(tmp_path, base)
    # bool 冒充数(Python 里 bool 是 int 子类,必须显式挡)。
    base = _valid_dict()
    base["transport"]["target_dropout_prob"] = False
    _expect_reject(tmp_path, base)
    # float 冒充整数步数。
    base = _valid_dict()
    base["transport"]["target_delay_steps"] = 2.0
    _expect_reject(tmp_path, base, match="整数")
    # 区间不是两元素。
    base = _valid_dict()
    base["physics"]["static_friction_range"] = [0.3, 1.6, 2.0]
    _expect_reject(tmp_path, base)
    # 区间元素是字符串。
    base = _valid_dict()
    base["physics"]["dynamic_friction_range"] = ["0.3", 1.2]
    _expect_reject(tmp_path, base)
    # section 不是 object。
    base = _valid_dict()
    base["mocap_noise"] = [0.0019, 0.0052, 0.717]
    _expect_reject(tmp_path, base)
    # 顶层不是 object。
    _expect_reject(tmp_path, "[1, 2, 3]")
    # _comment 不是字符串。
    base = _valid_dict()
    base["_comment"] = {"note": "x"}
    _expect_reject(tmp_path, base, match="_comment")


def test_non_finite_rejected(tmp_path):
    # json 标准库默认放行 NaN/Infinity token,加载器必须拦下。
    base_text = SHIPPED.read_text(encoding="utf-8")
    _expect_reject(tmp_path, base_text.replace("0.0019", "NaN"), match="NaN|有限")
    _expect_reject(tmp_path, base_text.replace("0.0052", "Infinity"), match="NaN|有限")


def test_range_violations_rejected(tmp_path):
    # 负 std。
    base = _valid_dict()
    base["mocap_noise"]["target_noise_white"] = -0.001
    _expect_reject(tmp_path, base)
    # rho 会被下游 clamp 的值。
    base = _valid_dict()
    base["mocap_noise"]["target_noise_ar1_rho"] = 1.0
    _expect_reject(tmp_path, base, match="rho")
    # 负延迟。
    base = _valid_dict()
    base["transport"]["target_delay_steps"] = -1
    _expect_reject(tmp_path, base)
    # dropout=1.0(链路全断)。
    base = _valid_dict()
    base["transport"]["target_dropout_prob"] = 1.0
    _expect_reject(tmp_path, base)
    # lo > hi。
    base = _valid_dict()
    base["physics"]["static_friction_range"] = [1.6, 0.3]
    _expect_reject(tmp_path, base, match="lo <= hi")
    # 恢复系数越出 [0, 1]。
    base = _valid_dict()
    base["physics"]["restitution_range"] = [0.0, 1.5]
    _expect_reject(tmp_path, base)
    # 质量 scale 乘子下界 <= 0。
    base = _valid_dict()
    base["physics"]["mass_distribution_params"] = [0.0, 1.15]
    _expect_reject(tmp_path, base)
    # 负摩擦。
    base = _valid_dict()
    base["physics"]["dynamic_friction_range"] = [-0.1, 1.2]
    _expect_reject(tmp_path, base)


def test_bad_tts_mode_rejected(tmp_path):
    base = _valid_dict()
    base["transport"]["target_delay_tts_mode"] = "timestamp-ish"
    _expect_reject(tmp_path, base, match="target_delay_tts_mode")


def test_all_three_tts_modes_accepted(tmp_path):
    for mode in ("live", "source_timestamp_compensated", "uncompensated"):
        base = _valid_dict()
        base["transport"]["target_delay_tts_mode"] = mode
        profile, _ = vp.load_venue_profile(_write(tmp_path, base, name=mode + ".json"))
        assert profile["transport"]["target_delay_tts_mode"] == mode


def test_duplicate_json_key_rejected(tmp_path):
    text = (
        '{"schema_version": "venue_profile_v1",'
        ' "mocap_noise": {"target_noise_white": 0.0, "target_noise_white": 0.1,'
        ' "target_noise_ar1_sigma": 0.0, "target_noise_ar1_rho": 0.0},'
        ' "transport": {"target_delay_steps": 0, "target_dropout_prob": 0.0,'
        ' "target_delay_tts_mode": "live"},'
        ' "physics": {"static_friction_range": [0.3, 1.6],'
        ' "dynamic_friction_range": [0.3, 1.2], "restitution_range": [0.0, 0.5],'
        ' "mass_distribution_params": [0.85, 1.15]}}'
    )
    _expect_reject(tmp_path, text, match="重复键")


def test_malformed_json_rejected(tmp_path):
    _expect_reject(tmp_path, "{not json", match="JSON")
