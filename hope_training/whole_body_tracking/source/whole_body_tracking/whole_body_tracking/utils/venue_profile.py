# Copyright (c) 2026, HOPE Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""venue_profile — 场地/环境模型档案(venue profile)严格加载器。

人话:换场地、换动捕系统、换机器人硬件时,"环境长什么样"的三类标定量会跟着变。Franco 的
要求是把这三类建模放进一个独立接口,单独调整或导入,而不是散落在各个 yaml 覆写里。一个
venue profile JSON 恰好有三个 section,每个键都一一对应仓库里【已经存在】的旋钮:

* ``mocap_noise``  动捕建模 —— actor 可见目标位置上的测量噪声(白噪声 + AR1 相关噪声)。
  对应 ``task.racket.target_noise_white`` / ``target_noise_ar1_sigma`` /
  ``target_noise_ar1_rho``(hope_commands.RacketTargetCommandCfg;白噪声/AR1 边缘 std 单位
  m,rho 是 50 Hz 每策略步的相关系数)。
* ``transport``  通讯/运行建模 —— 目标数据链路的整步延迟、每步丢帧概率、TTS 时戳约定。
  对应 ``task.racket.target_delay_steps``(50 Hz 整步数)/ ``target_dropout_prob`` /
  ``target_delay_tts_mode``("live" / "source_timestamp_compensated" / "uncompensated",
  与 hope_commands._TARGET_DELAY_TTS_MODES 逐字对齐)。
* ``physics``  物理建模 —— 机器人 body 材质与连杆质量的随机化区间。对应
  tracking_env_cfg.EventCfg.physics_material 的 ``static_friction_range`` /
  ``dynamic_friction_range`` / ``restitution_range``,以及 hope_env_cfg.HOPEEventCfg.
  randomize_link_mass 的 ``mass_distribution_params``(scale 乘子区间)。

这个模块只做一件事:把档案严格校验后读进来,返回 ``(profile, meta)``。它【不】改任何训练
配置 —— 往 train.py / env cfg 灌值由消费方接线(``task.venue_profile=<name>``,由另一个
agent 单独接),所以引入本模块对现有训练路径是逐字节 no-op。

设计原则(与仓库 house rules 对齐):

* fail-loud:未知 section/键、缺 section/键、错类型、非有限值(NaN/Infinity)、会被下游
  静默 clamp 的越界值、JSON 重复键 —— 全部当场 raise ``VenueProfileError``,绝不静默兜底。
* 契约钉死:档案必须自带 ``schema_version`` 且逐字等于 ``SCHEMA_VERSION``;以后 schema 变
  了要 writer/validator/测试夹具一起换,旧档案会响亮地拒载而不是被误读。
* 字节可追溯:meta 带档案文件字节的 sha256;进 run 记录后可以逐字节对账"这次训练用的是
  哪份场地标定"。
* 零重依赖:纯标准库,不 import torch / isaaclab,host 单测、pod、部署机都能加载。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

__all__ = [
    "SCHEMA_VERSION",
    "VENUE_PROFILE_DIR_RELATIVE",
    "VenueProfileError",
    "load_venue_profile",
]

# 契约字符串:档案文件里的 schema_version 必须逐字等于它。改 schema 时必须同步改:
# (1) 这里的常量;(2) configs/venue_profiles/ 下所有档案;(3) tests/test_venue_profile.py
# 的夹具 —— 三者不同步,旧档案/旧代码组合会 fail-loud,这是有意为之。
SCHEMA_VERSION = "venue_profile_v1"

# 裸名解析目录(相对仓库根):load_venue_profile("franco_rig_20260725") ->
# <repo>/configs/venue_profiles/franco_rig_20260725.json。
VENUE_PROFILE_DIR_RELATIVE = "configs/venue_profiles"

# 裸名只允许安全字符,防止 "..%2F" 之类的路径伎俩混进裸名通道。
_BARE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-.]*$")

# transport.target_delay_tts_mode 允许值 —— 与 hope_commands._TARGET_DELAY_TTS_MODES 逐字
# 对齐(那边收紧/放宽,这里必须跟着改,否则档案先在这里响亮拒载)。
_TTS_MODES = ("live", "source_timestamp_compensated", "uncompensated")

# 三个 section 各自的【精确】键集合:多一个未知键、少一个必需键都拒载。
_MOCAP_NOISE_KEYS = frozenset(
    {"target_noise_white", "target_noise_ar1_sigma", "target_noise_ar1_rho"}
)
_TRANSPORT_KEYS = frozenset(
    {"target_delay_steps", "target_dropout_prob", "target_delay_tts_mode"}
)
_PHYSICS_KEYS = frozenset(
    {
        "static_friction_range",
        "dynamic_friction_range",
        "restitution_range",
        "mass_distribution_params",
    }
)
_SECTION_KEYS = {
    "mocap_noise": _MOCAP_NOISE_KEYS,
    "transport": _TRANSPORT_KEYS,
    "physics": _PHYSICS_KEYS,
}
# 顶层:三个 section 必需 + schema_version 必需 + 可选的 "_comment"(纯文档字符串,不进
# profile dict)。
_TOP_LEVEL_REQUIRED = frozenset(set(_SECTION_KEYS) | {"schema_version"})
_TOP_LEVEL_OPTIONAL = frozenset({"_comment"})


class VenueProfileError(ValueError):
    """venue profile 档案不合法(schema/类型/取值/解析任一环节)。

    人话:凡是"这份场地标定不能被安全消费"的情况都抛它 —— 消费方拿到异常就该停,
    而不是带着半份标定继续训练。继承 ValueError,方便上游统一捕获。
    """


# ---------------------------------------------------------------------------
# JSON 解析:重复键、NaN/Infinity 都要响亮拒绝(json 标准库默认会静默放行这两样)。
# ---------------------------------------------------------------------------


def _reject_constant(token: str):
    raise VenueProfileError(
        f"venue profile 不允许非有限 JSON 常量 {token!r}(NaN/Infinity):"
        "所有标定值必须是有限数"
    )


def _pairs_hook(pairs):
    seen = set()
    for key, _ in pairs:
        if key in seen:
            raise VenueProfileError(
                f"venue profile JSON 出现重复键 {key!r}:json 标准库默认保留最后一个,"
                "这属于静默覆盖,拒载"
            )
        seen.add(key)
    return dict(pairs)


def _parse_json_bytes(raw: bytes, path: Path) -> dict:
    try:
        data = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_pairs_hook,
        )
    except VenueProfileError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VenueProfileError(f"venue profile {path} 不是合法 UTF-8 JSON:{exc}") from exc
    if not isinstance(data, dict):
        raise VenueProfileError(
            f"venue profile {path} 顶层必须是 JSON object,拿到 {type(data).__name__}"
        )
    return data


# ---------------------------------------------------------------------------
# 标量/区间校验小件:bool 在 Python 里是 int 的子类,必须先挡掉。
# ---------------------------------------------------------------------------


def _as_float(section: str, key: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VenueProfileError(
            f"{section}.{key} 必须是数(int/float),拿到 {type(value).__name__}: {value!r}"
        )
    out = float(value)
    if not math.isfinite(out):
        raise VenueProfileError(f"{section}.{key} 必须是有限数,拿到 {value!r}")
    return out


def _as_int(section: str, key: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise VenueProfileError(
            f"{section}.{key} 必须是整数(JSON int,不接受 float/bool),"
            f"拿到 {type(value).__name__}: {value!r}"
        )
    return int(value)


def _as_str(section: str, key: str, value) -> str:
    if not isinstance(value, str):
        raise VenueProfileError(
            f"{section}.{key} 必须是字符串,拿到 {type(value).__name__}: {value!r}"
        )
    return value


def _as_pair(section: str, key: str, value) -> tuple:
    """[lo, hi] 两元素数组 -> (float, float),要求 lo <= hi 且全有限。"""
    if not isinstance(value, list) or len(value) != 2:
        raise VenueProfileError(
            f"{section}.{key} 必须是两元素数组 [lo, hi],拿到 {value!r}"
        )
    lo = _as_float(section, key + "[0]", value[0])
    hi = _as_float(section, key + "[1]", value[1])
    if lo > hi:
        raise VenueProfileError(f"{section}.{key} 要求 lo <= hi,拿到 [{lo}, {hi}]")
    return (lo, hi)


def _check_exact_keys(section: str, block, allowed: frozenset) -> None:
    if not isinstance(block, dict):
        raise VenueProfileError(
            f"section {section!r} 必须是 JSON object,拿到 {type(block).__name__}"
        )
    unknown = sorted(set(block) - allowed)
    missing = sorted(allowed - set(block))
    if unknown:
        raise VenueProfileError(
            f"section {section!r} 出现未知键 {unknown}(允许且必需的键:{sorted(allowed)})。"
            "这些键必须一一对应仓库里已存在的旋钮,不接受自创键名"
        )
    if missing:
        raise VenueProfileError(
            f"section {section!r} 缺少必需键 {missing}:场地标定必须完整,不接受半份档案"
        )


# ---------------------------------------------------------------------------
# 三个 section 的语义校验:范围界限都取自下游消费点,防止"这里放行、下游静默 clamp"。
# ---------------------------------------------------------------------------


def _validate_mocap_noise(block: dict) -> dict:
    _check_exact_keys("mocap_noise", block, _MOCAP_NOISE_KEYS)
    white = _as_float("mocap_noise", "target_noise_white", block["target_noise_white"])
    sigma = _as_float("mocap_noise", "target_noise_ar1_sigma", block["target_noise_ar1_sigma"])
    rho = _as_float("mocap_noise", "target_noise_ar1_rho", block["target_noise_ar1_rho"])
    if white < 0.0:
        raise VenueProfileError(f"mocap_noise.target_noise_white 必须 >= 0(std,单位 m),拿到 {white}")
    if sigma < 0.0:
        raise VenueProfileError(f"mocap_noise.target_noise_ar1_sigma 必须 >= 0(std,单位 m),拿到 {sigma}")
    # hope_commands 侧会把 rho 静默 clamp 到 [0, 0.9999];档案里出现会被 clamp 的值说明
    # 标定本身就错了,这里直接拒载而不是让下游悄悄改数。
    if not (0.0 <= rho <= 0.9999):
        raise VenueProfileError(
            f"mocap_noise.target_noise_ar1_rho 必须在 [0, 0.9999](每 50 Hz 步相关系数,"
            f"下游会 clamp 到该区间,越界即标定错误),拿到 {rho}"
        )
    return {
        "target_noise_white": white,
        "target_noise_ar1_sigma": sigma,
        "target_noise_ar1_rho": rho,
    }


def _validate_transport(block: dict) -> dict:
    _check_exact_keys("transport", block, _TRANSPORT_KEYS)
    delay = _as_int("transport", "target_delay_steps", block["target_delay_steps"])
    drop = _as_float("transport", "target_dropout_prob", block["target_dropout_prob"])
    mode = _as_str("transport", "target_delay_tts_mode", block["target_delay_tts_mode"])
    if delay < 0:
        raise VenueProfileError(f"transport.target_delay_steps 必须 >= 0(50 Hz 整步数),拿到 {delay}")
    # dropout=1.0 意味着 actor 永远收不到新帧,链路等于断线,不是一份可训练的场地标定。
    if not (0.0 <= drop < 1.0):
        raise VenueProfileError(
            f"transport.target_dropout_prob 必须在 [0, 1)(每步丢帧概率;1.0 = 链路全断,"
            f"不是合法标定),拿到 {drop}"
        )
    if mode not in _TTS_MODES:
        raise VenueProfileError(
            f"transport.target_delay_tts_mode 必须是 {_TTS_MODES} 之一"
            f"(与 hope_commands 逐字对齐),拿到 {mode!r}"
        )
    return {
        "target_delay_steps": delay,
        "target_dropout_prob": drop,
        "target_delay_tts_mode": mode,
    }


def _validate_physics(block: dict) -> dict:
    _check_exact_keys("physics", block, _PHYSICS_KEYS)
    static = _as_pair("physics", "static_friction_range", block["static_friction_range"])
    dynamic = _as_pair("physics", "dynamic_friction_range", block["dynamic_friction_range"])
    restitution = _as_pair("physics", "restitution_range", block["restitution_range"])
    mass = _as_pair("physics", "mass_distribution_params", block["mass_distribution_params"])
    if static[0] < 0.0:
        raise VenueProfileError(f"physics.static_friction_range 下界必须 >= 0,拿到 {static}")
    if dynamic[0] < 0.0:
        raise VenueProfileError(f"physics.dynamic_friction_range 下界必须 >= 0,拿到 {dynamic}")
    if not (0.0 <= restitution[0] and restitution[1] <= 1.0):
        raise VenueProfileError(
            f"physics.restitution_range 必须落在 [0, 1](恢复系数物理范围),拿到 {restitution}"
        )
    # randomize_link_mass 用 operation="scale":乘子必须 > 0,否则连杆质量会被抹成 0/负数。
    if mass[0] <= 0.0:
        raise VenueProfileError(
            f"physics.mass_distribution_params 是质量 scale 乘子区间,下界必须 > 0,拿到 {mass}"
        )
    return {
        "static_friction_range": static,
        "dynamic_friction_range": dynamic,
        "restitution_range": restitution,
        "mass_distribution_params": mass,
    }


_SECTION_VALIDATORS = {
    "mocap_noise": _validate_mocap_noise,
    "transport": _validate_transport,
    "physics": _validate_physics,
}


# ---------------------------------------------------------------------------
# 名字/路径解析。
# ---------------------------------------------------------------------------


def _default_profile_dir() -> Path:
    """仓库根 /configs/venue_profiles(仓库根 = 本文件向上 6 级)。

    人话:本文件钉死在
    hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/utils/
    下,仓库根就是固定的相对位置;目录不存在说明仓库布局变了,响亮报错而不是猜。
    """
    repo_root = Path(__file__).resolve().parents[6]
    profile_dir = repo_root / VENUE_PROFILE_DIR_RELATIVE
    if not profile_dir.is_dir():
        raise VenueProfileError(
            f"裸名解析目录不存在:{profile_dir}(期望 <repo>/{VENUE_PROFILE_DIR_RELATIVE};"
            "仓库布局若变了,请同步改 venue_profile._default_profile_dir)"
        )
    return profile_dir


def _resolve_profile_path(path_or_name) -> Path:
    text = str(path_or_name).strip()
    if not text:
        raise VenueProfileError("venue profile 名字/路径不能为空")
    # 含路径分隔符或以 .json 结尾 -> 按文件路径处理;否则按裸名处理。
    if "/" in text or "\\" in text or text.endswith(".json"):
        path = Path(text).expanduser()
        if path.suffix != ".json":
            raise VenueProfileError(
                f"venue profile 路径必须以 .json 结尾,拿到 {text!r}"
            )
        path = path.resolve()
        if not path.is_file():
            raise VenueProfileError(f"venue profile 文件不存在:{path}")
        return path
    if not _BARE_NAME_RE.match(text):
        raise VenueProfileError(
            f"venue profile 裸名只允许 [A-Za-z0-9_.-] 且以字母/数字开头,拿到 {text!r}"
        )
    profile_dir = _default_profile_dir()
    path = (profile_dir / (text + ".json")).resolve()
    if not path.is_file():
        available = sorted(p.stem for p in profile_dir.glob("*.json"))
        raise VenueProfileError(
            f"venue profile {text!r} 不存在(找过 {path});"
            f"目录里现有档案:{available}"
        )
    return path


# ---------------------------------------------------------------------------
# 对外入口。
# ---------------------------------------------------------------------------


def load_venue_profile(path_or_name):
    """读取并严格校验一份 venue profile,返回 ``(profile, meta)``。

    人话:``path_or_name`` 给裸名(如 ``"franco_rig_20260725"``)就去
    ``<repo>/configs/venue_profiles/<name>.json`` 找;给带 ``/`` 或以 ``.json`` 结尾的
    字符串/Path 就按文件路径读。任何 schema/类型/取值问题都抛 ``VenueProfileError``,
    绝不返回半份档案。

    返回:

    * ``profile``:恰好三个 section 的 dict ——
      ``{"mocap_noise": {...}, "transport": {...}, "physics": {...}}``,标量已规范成
      float/int/str,区间已规范成 ``(float, float)`` 元组(可直接塞进 EventCfg params /
      task.racket 覆写)。顶层 ``_comment``/``schema_version`` 是档案元信息,不进 profile。
    * ``meta``:``{"name", "path", "sha256", "schema_version"}`` —— name 是文件名去掉
      .json,path 是绝对路径字符串,sha256 是【文件原始字节】的十六进制摘要(进 run 记录
      用来逐字节对账),schema_version 是钉死的契约串。
    """
    path = _resolve_profile_path(path_or_name)
    raw = path.read_bytes()
    data = _parse_json_bytes(raw, path)

    top_keys = set(data)
    unknown = sorted(top_keys - _TOP_LEVEL_REQUIRED - _TOP_LEVEL_OPTIONAL)
    missing = sorted(_TOP_LEVEL_REQUIRED - top_keys)
    if unknown:
        raise VenueProfileError(
            f"venue profile {path} 顶层出现未知键 {unknown}"
            f"(必需:{sorted(_TOP_LEVEL_REQUIRED)};可选:{sorted(_TOP_LEVEL_OPTIONAL)})"
        )
    if missing:
        raise VenueProfileError(f"venue profile {path} 顶层缺少必需键 {missing}")

    version = data["schema_version"]
    if not isinstance(version, str) or version != SCHEMA_VERSION:
        raise VenueProfileError(
            f"venue profile {path} schema_version 必须逐字等于 {SCHEMA_VERSION!r},"
            f"拿到 {version!r}(schema 演进时 writer/validator/夹具必须同步换)"
        )
    if "_comment" in data and not isinstance(data["_comment"], str):
        raise VenueProfileError(
            f"venue profile {path} 的 _comment 必须是字符串,"
            f"拿到 {type(data['_comment']).__name__}"
        )

    profile = {
        section: _SECTION_VALIDATORS[section](data[section])
        for section in ("mocap_noise", "transport", "physics")
    }
    meta = {
        "name": path.stem,
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "schema_version": SCHEMA_VERSION,
    }
    return profile, meta
