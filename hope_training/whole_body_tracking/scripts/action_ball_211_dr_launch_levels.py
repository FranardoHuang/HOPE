#!/usr/bin/env python3
"""发射时"选哪一档随机性"的唯一权威 —— A211 与 C211 共用一份。

人话
====
在这个文件出现之前,"这一跑用哪一档 DR"是两个发射器里各写一行的**模块级常量**
(``TASK_PROFILE_ID = "...DRL0Learnability"``)。后果是 DR-L1 那两片 profile
(五条 plant 轴 + ``start_pose_ramp``,共六条随机性轴)**没有任何入口**:
契约、校验器、运行时、遥测、候选 manifest 全都齐了,就是选不中。

本模块把那个常量换成一个**发射时的显式选择**:

* ``--dr-level dr_l0``(默认)—— 解析结果与今天**逐字节相同**。四格归因跑的
  DR-L0 / DR-L0N 不受任何影响,这一点由下面的模块级断言和
  ``tests/test_action_ball_211_dr_launch_levels.py`` 的对拍测试守着。
* ``--dr-level dr_l1`` —— 解析到 DR-L1 那两片 profile、DR-L1 的候选 manifest、
  DR-L1 的 finalizer 合同,并把**六条轴各自的实际取值**摊开写进收据。

三条设计约束
============
1. **不新编一套数。** 每条轴的值都从两个**独立的活出处**取:
   ``training_contract.action_ball_dr_l{0,1}_contract_payload()``(代码侧)与
   tracked 候选 manifest(声明式工件)。两边逐字段对不上 = 当场拒收。
   这里一个数字都不手抄 —— 手抄就等于制造第三份会漂的副本。
2. **收据不许只写档名。** ``resolve()`` 返回的 ``axes`` 是六条轴的**实际取值**,
   发射器把它整块写进收据。只写 "dr_l1" 三个字的收据读起来无法复算。
3. **两档必须真的不同。** ``assert_levels_differ()`` 在模块导入期逐轴比对 L0 与
   L1 的解析结果;哪天有人把 fixture 或注册表改成两档同值,导入就炸 —— 否则
   "选 L1 真的换了值" 那条验收会变成恒真。

为什么 DR-L1 今天还发不出去
==========================
每一档都要有自己的 **lineage 工件**(内容寻址、跑一次 Isaac materialize 才有)。
DR-L0 的那份在仓里;DR-L1 的没有。所以 ``dr_l1`` 这一档在注册表里自陈
``lineage_materialized=False`` 并带一张 ``launch_blockers`` 清单,发射器在
``preflight_launchable()`` 上**明确拒收并把清单原样报出来**。这不是"忘了接",
是"入口接好了,后面那一步还没做",两者在收据里必须能分辨。
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Mapping


class DrLaunchLevelError(ValueError):
    """选档 / 解析 / 自陈核对失败时抛出。"""


# --------------------------------------------------------------------------- #
# 档位注册表
# --------------------------------------------------------------------------- #
DR_LEVEL_L0 = "dr_l0"
DR_LEVEL_L1 = "dr_l1"
LEVELS = (DR_LEVEL_L0, DR_LEVEL_L1)
# 默认必须是 L0:现役四格跑的就是这一档,默认值一变就是一次静默的实验改动。
DEFAULT_LEVEL = DR_LEVEL_L0
FAMILIES = ("A211", "C211")

# 六条随机性轴。前五条是 startup plant 事件,第六条是出生位姿斜坡。
AXIS_ORDER = (
    "physics_material",
    "add_joint_default_pos",
    "base_com",
    "randomize_link_mass",
    "randomize_pd_gains",
    "start_pose_ramp",
)
PLANT_EVENT_AXES = AXIS_ORDER[:5]
START_POSE_RAMP_AXIS = "start_pose_ramp"

TASK_PROFILE_DIR = "hope_training/whole_body_tracking/cfg/task/"

# 每一档:profile 名里的中缀、finalizer 合同身份、候选 manifest 的路径与身份、
# 以及"这一档的 lineage 工件今天存不存在"。
_LEVELS: dict[str, dict[str, Any]] = {
    DR_LEVEL_L0: {
        "profile_infix": "DRL0",
        "hard_contract_identity": "action_ball_dr_l0_exact_all_off_v1",
        "contract_payload_attr": "action_ball_dr_l0_contract_payload",
        "contract_sha256_attr": "action_ball_dr_l0_contract_sha256",
        "binds_start_pose_ramp": False,
        "manifest_source": (
            "configs/action_ball_n1_measured_20260803/"
            "action_ball_211_dr_l0_learnability_candidate.v1.json"
        ),
        "manifest_kind": "action_ball_211_dr_l0_learnability_candidate",
        "manifest_identity": "action_ball_211_dr_l0_learnability_candidate_v1",
        "manifest_status": "BOUND_FRESH_DIAGNOSTIC_LAUNCH",
        "lineage_materialized": True,
        "launch_blockers": (),
        "human_summary": (
            "整套随机性全关的因果对照:五条 startup plant 事件全部不装配,"
            "出生位姿没有斜坡"
        ),
    },
    DR_LEVEL_L1: {
        "profile_infix": "DRL1",
        "hard_contract_identity": "action_ball_dr_l1_measured_plant_restored_v1",
        "contract_payload_attr": "action_ball_dr_l1_contract_payload",
        "contract_sha256_attr": "action_ball_dr_l1_contract_sha256",
        "binds_start_pose_ramp": True,
        "manifest_source": (
            "configs/action_ball_n1_measured_20260805/"
            "action_ball_211_dr_l1_restored_plant_candidate.v1.json"
        ),
        "manifest_kind": "action_ball_211_dr_l1_restored_plant_candidate",
        "manifest_identity": "action_ball_211_dr_l1_restored_plant_candidate_v1",
        "manifest_status": "BOUND_MECHANISM_LANDED_LAUNCH_LINEAGE_PENDING",
        "lineage_materialized": False,
        "launch_blockers": (
            "dr_l1_lineage_artifact_missing: 这一档需要它自己的 lineage 工件"
            "(materialize_action_ball_{a211,c211}_lineage.py 各跑一次,产物 kind "
            "必须是本模块 LINEAGE_KINDS 里 dr_l1 那两行)。仓里今天只有 DR-L0 的那份。",
            "dr_l1_reward_and_policy_recipe_not_materialized: reward recipe 与 "
            "dynamic-ready policy recipe 都是按档内容寻址的,DR-L1 要各自重跑一次 "
            "materialize / recipe 阶段。",
            "dr_l1_grid_authority_absent: action_ball_211_four_grid_contract.py 的四格"
            "只封了 DR-L0 与 DR-L0N 两个身份。DR-L1 **按定义进不了那四格**"
            "(四格刻意只差 obs-noise 一根轴),要发就要另开一份属于它自己的格局权威。",
        ),
        "human_summary": (
            "把逻辑上本来就该开着的五条 plant 随机化恢复到 day-1 基线,"
            "并给出生位姿装一条声明式斜坡;观测腐蚀与执行器延迟仍然关着"
        ),
    },
}

# lineage kind 逐族逐档点名。**不要**用"前缀 + 档名"拼:C211 现役那份
# (..._direct_ball_split_ready_lineage_v4)里根本没有 dr_l0 字样,拼出来的名字
# 会和仓里真实存在的工件对不上,而且是那种"看起来很像"的对不上。
LINEAGE_KINDS: dict[tuple, str] = {
    ("A211", DR_LEVEL_L0): (
        "action_ball_a211_split_ready_online_question_dr_l0_lineage_v5"
    ),
    ("A211", DR_LEVEL_L1): (
        "action_ball_a211_split_ready_online_question_dr_l1_lineage_v1"
    ),
    ("C211", DR_LEVEL_L0): "action_ball_c211_direct_ball_split_ready_lineage_v4",
    ("C211", DR_LEVEL_L1): (
        "action_ball_c211_direct_ball_split_ready_dr_l1_lineage_v1"
    ),
}

# 候选 manifest 的 restored_axes 用的字段名 -> finalizer payload 的 event 字段名。
# 两边刻意不同名的只有 base_com 的 body/body_names 一处;其余同名也照样列出来,
# 因为这张表同时承担"这条轴至少要核对哪几个字段"的职责 —— 空表 = 这条轴没被核对。
_L1_AXIS_FIELD_MAP: dict[str, dict[str, str]] = {
    "physics_material": {
        "static_friction_range": "static_friction_range",
        "dynamic_friction_range": "dynamic_friction_range",
        "restitution_range": "restitution_range",
        "num_buckets": "num_buckets",
    },
    "add_joint_default_pos": {
        "pos_distribution_params": "pos_distribution_params",
        "operation": "operation",
    },
    "base_com": {
        "body": "body_names",
        "com_range": "com_range",
    },
    "randomize_link_mass": {
        "mass_distribution_params": "mass_distribution_params",
        "operation": "operation",
        "distribution": "distribution",
    },
    "randomize_pd_gains": {
        "stiffness_distribution_params": "stiffness_distribution_params",
        "damping_distribution_params": "damping_distribution_params",
        "operation": "operation",
        "distribution": "distribution",
    },
}
# DR-L1 的 manifest 用这个字符串表示"这条轴装配上了";DR-L0 用 JSON null。
L1_ACTIVE_MARKER = "active_day_one_baseline"


def canonical_sha256(value: Any) -> str:
    """与 training_contract._action_ball_canonical_sha256 同一套规范化。"""

    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DrLaunchLevelError("DR level value is not canonical JSON") from exc
    return hashlib.sha256(payload).hexdigest()


# --------------------------------------------------------------------------- #
# 纯查表的解析(不需要读任何文件)
# --------------------------------------------------------------------------- #
def validate_level(level: Any) -> str:
    if type(level) is not str or level not in _LEVELS:
        raise DrLaunchLevelError(
            "dr_launch_level must be exactly one of %r; got %r"
            % (list(LEVELS), level)
        )
    return level


def validate_family(family: Any) -> str:
    if type(family) is not str or family not in FAMILIES:
        raise DrLaunchLevelError(
            "task family must be exactly one of %r; got %r" % (list(FAMILIES), family)
        )
    return family


def task_profile_id(family: Any, level: Any) -> str:
    """例:("A211", "dr_l0") -> HOPEPingPongActionBallA211VendorV2N1DRL0Learnability"""

    family = validate_family(family)
    level = validate_level(level)
    return "HOPEPingPongActionBall%sVendorV2N1%sLearnability" % (
        family,
        _LEVELS[level]["profile_infix"],
    )


def task_profile_source(family: Any, level: Any) -> str:
    return TASK_PROFILE_DIR + task_profile_id(family, level) + ".yaml"


def parent_task_profile_id(family: Any) -> str:
    """两档共用的父 profile(叶子 defaults 里继承的那一份)。"""

    return "HOPEPingPongActionBall%sVendorV2N1Learnability" % validate_family(family)


def lineage_kind(family: Any, level: Any) -> str:
    family = validate_family(family)
    level = validate_level(level)
    kind = LINEAGE_KINDS.get((family, level))
    if type(kind) is not str or not kind:
        raise DrLaunchLevelError(
            "no lineage kind is registered for family %r at level %r" % (family, level)
        )
    return kind


def manifest_source(level: Any) -> str:
    return _LEVELS[validate_level(level)]["manifest_source"]


def hard_contract_identity(level: Any) -> str:
    return _LEVELS[validate_level(level)]["hard_contract_identity"]


def manifest_kind(level: Any) -> str:
    return _LEVELS[validate_level(level)]["manifest_kind"]


def manifest_identity(level: Any) -> str:
    return _LEVELS[validate_level(level)]["manifest_identity"]


def manifest_status(level: Any) -> str:
    return _LEVELS[validate_level(level)]["manifest_status"]


def lineage_materialized(level: Any) -> bool:
    return bool(_LEVELS[validate_level(level)]["lineage_materialized"])


def human_summary(level: Any) -> str:
    return _LEVELS[validate_level(level)]["human_summary"]


def binds_start_pose_ramp(level: Any) -> bool:
    return _LEVELS[validate_level(level)]["binds_start_pose_ramp"]


def level_of_task_profile(profile_id: Any) -> str:
    """反查:一个 profile 名属于哪一档。

    发射器用它做**反方向**的核对:正向是"这一档应该解析成哪个 profile",反向是
    "手上这个 profile 名到底属于哪一档"。两个方向都对上,才排除掉"名字看起来像
    但其实来自别处"的那一类。导入期自检也用它证明 profile -> 档 是单射。
    """

    for family in FAMILIES:
        for level in LEVELS:
            if task_profile_id(family, level) == profile_id:
                return level
    raise DrLaunchLevelError(
        "task profile %r does not belong to any registered DR launch level" % profile_id
    )


def launch_blockers(level: Any) -> tuple:
    return tuple(_LEVELS[validate_level(level)]["launch_blockers"])


def preflight_launchable(level: Any) -> None:
    """一档没有 lineage 工件就当场拒,并把清单原样报出来。

    这条门刻意**不是**"解析失败"。入口是通的、profile 解析得出来、六条轴的值也
    算得出来 —— 缺的是这一档自己的 materialize 产物。两种失败在收据里必须能分辨,
    否则下一个人会以为入口又没接。
    """

    level = validate_level(level)
    if _LEVELS[level]["lineage_materialized"]:
        return
    raise DrLaunchLevelError(
        "DR level %s has a launch entrance but no materialized lineage yet; "
        "outstanding: %s" % (level, " | ".join(launch_blockers(level)))
    )


# --------------------------------------------------------------------------- #
# 活值解析:代码侧 payload x 声明式 manifest,逐字段对拍
# --------------------------------------------------------------------------- #
def contract_payload(module: Any, level: Any, *, start_pose_ramp: Any = None) -> dict:
    """从 training_contract 模块取这一档的 resolved finalizer payload。"""

    level = validate_level(level)
    entry = _LEVELS[level]
    try:
        builder = getattr(module, entry["contract_payload_attr"])
    except AttributeError as exc:
        raise DrLaunchLevelError(
            "training contract module has no %s" % entry["contract_payload_attr"]
        ) from exc
    try:
        if entry["binds_start_pose_ramp"]:
            payload = builder(start_pose_ramp=start_pose_ramp)
        else:
            payload = builder()
    except Exception as exc:  # noqa: BLE001 - 任何解析失败都必须变成拒收
        raise DrLaunchLevelError(
            "cannot resolve the code-owned %s finalizer contract" % level
        ) from exc
    if type(payload) is not dict:
        raise DrLaunchLevelError("%s finalizer payload is not a dict" % level)
    if payload.get("identity") != entry["hard_contract_identity"]:
        raise DrLaunchLevelError(
            "%s finalizer payload identity is %r, expected %r"
            % (level, payload.get("identity"), entry["hard_contract_identity"])
        )
    return payload


def contract_sha256(module: Any, level: Any, *, start_pose_ramp: Any = None) -> str:
    """这一档 finalizer 合同的摘要,由 training_contract 自己算,不在这里手抄。"""

    level = validate_level(level)
    entry = _LEVELS[level]
    try:
        builder = getattr(module, entry["contract_sha256_attr"])
    except AttributeError as exc:
        raise DrLaunchLevelError(
            "training contract module has no %s" % entry["contract_sha256_attr"]
        ) from exc
    try:
        if entry["binds_start_pose_ramp"]:
            return builder(start_pose_ramp=start_pose_ramp)
        return builder()
    except Exception as exc:  # noqa: BLE001
        raise DrLaunchLevelError(
            "cannot resolve the code-owned %s finalizer contract digest" % level
        ) from exc


def default_start_pose_ramp(module: Any, level: Any) -> Any:
    """DR-L1 绑的是代码常量 ACTION_BALL_START_POSE_RAMP_FOUR_CELL,不是手抄的表。"""

    if not binds_start_pose_ramp(level):
        return None
    try:
        return copy.deepcopy(module.ACTION_BALL_START_POSE_RAMP_FOUR_CELL)
    except AttributeError as exc:
        raise DrLaunchLevelError(
            "training contract module has no ACTION_BALL_START_POSE_RAMP_FOUR_CELL"
        ) from exc


def _manifest_finalizer_state(manifest: Any, level: str) -> dict:
    if type(manifest) is not dict:
        raise DrLaunchLevelError("%s candidate manifest must be a dict" % level)
    entry = _LEVELS[level]
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != entry["manifest_kind"]
        or manifest.get("identity") != entry["manifest_identity"]
    ):
        raise DrLaunchLevelError(
            "%s candidate manifest schema/kind/identity differs" % level
        )
    state = manifest.get("required_post_finalizer_state")
    if type(state) is not dict:
        raise DrLaunchLevelError(
            "%s candidate manifest has no required_post_finalizer_state" % level
        )
    return state


def _resolve_plant_axes(payload: dict, manifest: dict, level: str) -> dict:
    """五条 startup plant 轴:payload 的 event_slots 与 manifest 逐字段对拍。"""

    state = _manifest_finalizer_state(manifest, level)
    slots = payload.get("event_slots")
    if type(slots) is not dict:
        raise DrLaunchLevelError("%s finalizer payload has no event_slots" % level)
    restored = manifest.get("restored_axes") if level == DR_LEVEL_L1 else None
    if level == DR_LEVEL_L1 and type(restored) is not dict:
        raise DrLaunchLevelError("DR-L1 candidate manifest has no restored_axes")
    axes: dict[str, Any] = {}
    for name in PLANT_EVENT_AXES:
        if name not in slots:
            raise DrLaunchLevelError(
                "%s finalizer payload does not carry the %s slot" % (level, name)
            )
        resolved = slots[name]
        declared = state.get("events.%s" % name, _MISSING)
        if declared is _MISSING:
            raise DrLaunchLevelError(
                "%s candidate manifest does not declare events.%s" % (level, name)
            )
        if level == DR_LEVEL_L0:
            if resolved is not None or declared is not None:
                raise DrLaunchLevelError(
                    "DR-L0 requires %s to be absent in both the finalizer payload "
                    "and the candidate manifest; got payload=%r manifest=%r"
                    % (name, resolved, declared)
                )
            axes[name] = None
            continue
        if declared != L1_ACTIVE_MARKER:
            raise DrLaunchLevelError(
                "DR-L1 candidate manifest must mark events.%s as %r; got %r"
                % (name, L1_ACTIVE_MARKER, declared)
            )
        if type(resolved) is not dict or not resolved:
            raise DrLaunchLevelError(
                "DR-L1 finalizer payload must carry an active %s event spec" % name
            )
        fields = _L1_AXIS_FIELD_MAP.get(name)
        if not fields:
            raise DrLaunchLevelError(
                "no cross-check field map is registered for axis %s" % name
            )
        declared_axis = restored.get(name)
        if type(declared_axis) is not dict:
            raise DrLaunchLevelError(
                "DR-L1 candidate manifest has no restored_axes.%s" % name
            )
        for manifest_key, contract_key in fields.items():
            if manifest_key not in declared_axis:
                raise DrLaunchLevelError(
                    "DR-L1 candidate manifest restored_axes.%s is missing %s"
                    % (name, manifest_key)
                )
            if contract_key not in resolved:
                raise DrLaunchLevelError(
                    "DR-L1 finalizer payload %s is missing %s" % (name, contract_key)
                )
            if declared_axis[manifest_key] != resolved[contract_key]:
                raise DrLaunchLevelError(
                    "DR-L1 axis %s disagrees between the candidate manifest "
                    "(%s=%r) and the code-owned finalizer contract (%s=%r)"
                    % (
                        name,
                        manifest_key,
                        declared_axis[manifest_key],
                        contract_key,
                        resolved[contract_key],
                    )
                )
        axes[name] = copy.deepcopy(resolved)
    return axes


def _resolve_start_pose_ramp(payload: dict, manifest: dict, level: str) -> Any:
    reset = payload.get("motion_reset_noise")
    if type(reset) is not dict:
        raise DrLaunchLevelError("%s finalizer payload has no motion_reset_noise" % level)
    ramp = reset.get(START_POSE_RAMP_AXIS)
    declared = manifest.get(START_POSE_RAMP_AXIS)
    if level == DR_LEVEL_L0:
        if ramp is not None or declared is not None:
            raise DrLaunchLevelError(
                "DR-L0 must not carry a start-pose ramp in either the finalizer "
                "payload or the candidate manifest"
            )
        return None
    if type(ramp) is not dict or ramp.get("enabled") is not True:
        raise DrLaunchLevelError(
            "DR-L1 finalizer payload must carry an enabled start_pose_ramp"
        )
    if type(declared) is not dict:
        raise DrLaunchLevelError("DR-L1 candidate manifest has no start_pose_ramp")
    derivation = declared.get("world_frame_derivation")
    if type(derivation) is not dict:
        raise DrLaunchLevelError(
            "DR-L1 candidate manifest start_pose_ramp has no world_frame_derivation"
        )
    pose = ramp.get("pose_range")
    if type(pose) is not dict:
        raise DrLaunchLevelError("DR-L1 start_pose_ramp has no pose_range")
    scalar_pairs = (
        ("kind", ramp.get("kind"), declared.get("kind")),
        ("ramp_steps", ramp.get("ramp_steps"), declared.get("ramp_steps")),
        (
            "hold_clock_owner",
            ramp.get("hold_clock_owner"),
            declared.get("hold_clock_owner"),
        ),
        ("pose_range.x", pose.get("x"), derivation.get("x_offset_m")),
        ("pose_range.y", pose.get("y"), derivation.get("y_offset_m")),
        ("pose_range.yaw", pose.get("yaw"), derivation.get("yaw_offset_rad")),
    )
    for name, resolved, expected in scalar_pairs:
        if resolved != expected:
            raise DrLaunchLevelError(
                "DR-L1 start_pose_ramp %s disagrees: finalizer contract %r vs "
                "candidate manifest %r" % (name, resolved, expected)
            )
    payload_sha256 = canonical_sha256(ramp)
    if declared.get("payload_sha256") != payload_sha256:
        raise DrLaunchLevelError(
            "DR-L1 start_pose_ramp payload SHA differs: resolved %s vs candidate "
            "manifest %r" % (payload_sha256, declared.get("payload_sha256"))
        )
    return copy.deepcopy(ramp)


_MISSING = object()


def resolve(
    level: Any,
    *,
    family: Any,
    contract_payload_document: Any,
    manifest_document: Any,
    contract_sha256: Any,
    manifest_file_sha256: Any,
) -> dict:
    """返回一整块可以直接写进收据的自陈:档名 + 身份 + 六条轴的实际取值。

    ``contract_payload_document`` 与 ``manifest_document`` 必须来自两个独立出处
    (代码侧 payload / tracked 候选 manifest)。喂同一份进来两边当然一致 ——
    这也是为什么发射器那一侧一个从 training_contract 取、一个从磁盘读。
    """

    level = validate_level(level)
    family = validate_family(family)
    entry = _LEVELS[level]
    payload = contract_payload_document
    if type(payload) is not dict:
        raise DrLaunchLevelError("%s finalizer payload must be a dict" % level)
    if payload.get("identity") != entry["hard_contract_identity"]:
        raise DrLaunchLevelError(
            "%s finalizer payload identity differs from the registry" % level
        )
    for name, digest in (
        ("contract", contract_sha256),
        ("candidate manifest file", manifest_file_sha256),
    ):
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise DrLaunchLevelError(
                "%s %s SHA must be 64 lowercase hex" % (level, name)
            )
    if canonical_sha256(payload) != contract_sha256:
        raise DrLaunchLevelError(
            "%s finalizer payload does not hash to the supplied contract SHA" % level
        )
    manifest = manifest_document
    resolved_manifest = (
        manifest.get("resolved_finalizer_contract")
        if type(manifest) is dict
        else None
    )
    if (
        type(resolved_manifest) is not dict
        or resolved_manifest.get("contract_sha256") != contract_sha256
        or resolved_manifest.get("hard_contract_identity")
        != entry["hard_contract_identity"]
    ):
        raise DrLaunchLevelError(
            "%s candidate manifest resolved_finalizer_contract differs from the "
            "code-owned contract" % level
        )
    axes = _resolve_plant_axes(payload, manifest, level)
    axes[START_POSE_RAMP_AXIS] = _resolve_start_pose_ramp(payload, manifest, level)
    if tuple(sorted(axes)) != tuple(sorted(AXIS_ORDER)):
        raise DrLaunchLevelError("resolved DR axis set differs from AXIS_ORDER")
    unsigned = {
        "schema_version": 1,
        "kind": "action_ball_211_dr_launch_level_v1",
        "dr_launch_level": level,
        "task_family": family,
        "task_profile": task_profile_id(family, level),
        "task_profile_source": task_profile_source(family, level),
        "parent_task_profile": parent_task_profile_id(family),
        "lineage_kind": lineage_kind(family, level),
        "hard_contract_identity": entry["hard_contract_identity"],
        "contract_sha256": contract_sha256,
        "candidate_manifest_source": entry["manifest_source"],
        "candidate_manifest_identity": entry["manifest_identity"],
        "candidate_manifest_file_sha256": manifest_file_sha256,
        "axis_order": list(AXIS_ORDER),
        # 六条轴的**实际取值**。只写档名的收据无法复算,所以这块是必需的。
        "axes": {name: copy.deepcopy(axes[name]) for name in AXIS_ORDER},
        "active_axis_count": sum(1 for name in AXIS_ORDER if axes[name] is not None),
        "lineage_materialized": bool(entry["lineage_materialized"]),
        "launch_blockers": list(launch_blockers(level)),
        "human_summary": human_summary(level),
    }
    return {**unsigned, "content_sha256": canonical_sha256(unsigned)}


def validate_declared(declared: Any, *, resolved: Mapping) -> dict:
    """收据自陈的那一块必须与重新解析的结果逐字节相同。

    谎报档名(写 dr_l0 却跑着 L1 的值)、只写档名不写取值、把 axes 抹成空 ——
    三种都在这里当场拒。
    """

    if type(resolved) is not dict or "content_sha256" not in resolved:
        raise DrLaunchLevelError("resolved DR level block is malformed")
    if type(declared) is not dict:
        raise DrLaunchLevelError("declared DR level block must be a dict")
    if set(declared) != set(resolved):
        raise DrLaunchLevelError(
            "declared DR level block keys differ: missing=%s extra=%s"
            % (
                sorted(set(resolved) - set(declared)),
                sorted(set(declared) - set(resolved)),
            )
        )
    unsigned = {key: value for key, value in declared.items() if key != "content_sha256"}
    if declared.get("content_sha256") != canonical_sha256(unsigned):
        raise DrLaunchLevelError("declared DR level block seal differs")
    if declared != resolved:
        differing = sorted(
            key for key in resolved if declared.get(key) != resolved.get(key)
        )
        raise DrLaunchLevelError(
            "declared DR level block differs from the re-resolved block in %r"
            % differing
        )
    return copy.deepcopy(dict(declared))


def assert_levels_differ(resolved_by_level: Mapping) -> None:
    """两档的解析结果必须**逐轴**都不同,而不是只有档名不同。

    这条断言存在的理由:验收里那句"选 L1 之后六条轴的值确实生效"如果被喂进两份
    同值的解析结果,就会变成恒真。这里把它钉死 —— 两档同值时直接炸。
    """

    if set(resolved_by_level) != set(LEVELS):
        raise DrLaunchLevelError(
            "level distinctness check needs exactly %r" % (list(LEVELS),)
        )
    low = resolved_by_level[DR_LEVEL_L0]
    high = resolved_by_level[DR_LEVEL_L1]
    for key in (
        "dr_launch_level",
        "task_profile",
        "task_profile_source",
        "lineage_kind",
        "hard_contract_identity",
        "contract_sha256",
        "candidate_manifest_source",
        "content_sha256",
    ):
        if low.get(key) == high.get(key):
            raise DrLaunchLevelError(
                "DR-L0 and DR-L1 must not share %s (%r)" % (key, low.get(key))
            )
    same = sorted(
        name
        for name in AXIS_ORDER
        if low["axes"].get(name) == high["axes"].get(name)
    )
    if same:
        raise DrLaunchLevelError(
            "DR-L0 and DR-L1 resolve these axes to the same value, so 'selecting "
            "DR-L1 really changes the plant' would be vacuously true: %r" % same
        )
    if low["active_axis_count"] != 0 or high["active_axis_count"] != len(AXIS_ORDER):
        raise DrLaunchLevelError(
            "DR-L0 must activate zero axes and DR-L1 must activate all six; got "
            "%r and %r" % (low["active_axis_count"], high["active_axis_count"])
        )


# --------------------------------------------------------------------------- #
# 导入期自检:注册表本身不许退化
# --------------------------------------------------------------------------- #
def _assert_registry_is_well_formed() -> None:
    if tuple(LEVELS) != tuple(sorted(_LEVELS)):
        raise RuntimeError("DR launch level registry and LEVELS disagree")
    if DEFAULT_LEVEL != DR_LEVEL_L0:
        raise RuntimeError(
            "the default DR launch level must stay DR-L0; the live four-cell "
            "attribution run is that level and a default change would move it "
            "silently"
        )
    profiles = set()
    lineages = set()
    for family in FAMILIES:
        for level in LEVELS:
            profiles.add(task_profile_id(family, level))
            lineages.add(lineage_kind(family, level))
            if level_of_task_profile(task_profile_id(family, level)) != level:
                raise RuntimeError("DR launch level reverse lookup is inconsistent")
    if len(profiles) != len(FAMILIES) * len(LEVELS):
        raise RuntimeError("two DR launch levels resolve to the same task profile")
    if len(lineages) != len(FAMILIES) * len(LEVELS):
        raise RuntimeError("two DR launch levels resolve to the same lineage kind")
    if len(set(entry["manifest_source"] for entry in _LEVELS.values())) != len(_LEVELS):
        raise RuntimeError("two DR launch levels share one candidate manifest")
    if len(
        set(entry["hard_contract_identity"] for entry in _LEVELS.values())
    ) != len(_LEVELS):
        raise RuntimeError("two DR launch levels share one hard-contract identity")
    for name in PLANT_EVENT_AXES:
        if not _L1_AXIS_FIELD_MAP.get(name):
            raise RuntimeError(
                "axis %s has no cross-check field map, so it would be accepted "
                "without ever being compared" % name
            )


_assert_registry_is_well_formed()
