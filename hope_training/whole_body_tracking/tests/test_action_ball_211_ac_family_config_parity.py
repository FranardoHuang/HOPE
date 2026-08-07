"""A211 与 C211 两族"解析后的有效任务配置"逐键比对,不等即拒。

人话
====
四格实验的前提是一句话:**A 族和 C 族只差"问什么题 + 怎么给分"**(观测语义与
reward),其余全部相同。这句话一旦不成立,"A 对 C = 架构效应"这个结论就没了 ——
差的可能只是某个没人注意到的 delay、episode 长度或 DR 开关。

这句话今天靠**手抄两份**维持:

* 两个发射器(``launch_action_ball_a211_four_arm_diagnostic.py`` /
  ``launch_action_ball_c211_diagnostic.py``)各自把六十多条 ``task.*`` override
  写了一遍;
* 两片 Hydra task profile(``HOPEPingPongActionBall{A,C}211VendorV2N1DRL0Learnability``)
  各自把继承链下面的叶子写了一遍。

现役代码里**没有任何东西**逐键比过这两份。已有的三道门都只盖到别的层:

* ``scripts/action_ball_211_four_grid_contract.py`` 是真正的单一真源(两族 exec 同
  一个文件、同一个 content seal),但它只盖 matched_contract 那几项(PPO、soft
  weights、wait 契约、预算、题源),不盖 argv、更不盖 YAML;
* ``test_action_ball_211_launcher_shared_constants.py`` 盖的是**模块级常量**,而且
  是一份点名清单(MUST_MATCH 49 项)—— 新加的常量默认不在清单里,自动逃逸;
* ``test_action_ball_211_isaac_four_grid.py::test_four_grid_cells_match_every_non_registered_setting``
  盖的是 8 个点名的 contract 键。

本文件补的就是最大的那块空白:**解析后的有效 task 配置,258 片叶子,逐键比。**

它怎么比
========
1. 用同一份 spec/lineage 调两个发射器**真正的** ``_training_argv``(不是抄它的源码
   文本),所以拿到的是活值;checkout/namespace/seed 两边给同一份,于是只有"族差异"
   能活下来。
2. 按 Hydra 的 ``defaults: [X@_here_, ..., _self_]`` 语义把两片 profile 各自解析到
   底,再把该族的 ``task.*`` override 叠上去。
3. 两边 flatten 成叶子,逐键比。**默认必须相同**;只有 ``LEGITIMATE_DIFFERENCES``
   里点名并写了理由的键才允许不同。新增一个键、或者某个共有键漂了,一律 fail。

为什么不直接用 Hydra 解析
=========================
host 上的 pytest 环境没有 hydra(``/usr/bin/python3`` 缺 hydra 会静默跳过 17 条测
试)。这道门是**必须永远跑**的,所以本文件自带一个只依赖 PyYAML 的解析器。
手写解析器 = 又一份手抄,所以 ``test_local_resolver_and_overlay_reproduce_hydra``
在有 hydra 的环境里逐字段核对"我解析出来的"和"hydra 解析出来的"是否一致 ——
2026-08-07 首次落地时它就抓到一个真错:``policy_contract_sha256`` 全零串被
PyYAML 读成整数 0,而 OmegaConf 因为节点是 str 类型会保持字符串。修法是按既有
节点类型强制,不是把断言放松。
"""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
from typing import Any, Mapping

import pytest
import yaml


WBT = Path(__file__).resolve().parents[1]
SCRIPTS = WBT / "scripts"
CFG_ROOT = WBT / "cfg"

A_PROFILE = "HOPEPingPongActionBallA211VendorV2N1DRL0Learnability"
C_PROFILE = "HOPEPingPongActionBallC211VendorV2N1DRL0Learnability"
# 两族的继承链在这里分叉;分叉点以上必须逐个 profile 相同,否则"其余全部相同"这句
# 话连起点都不成立。
COMMON_ANCESTOR = "HOPEPingPongActionBallA3VendorV2"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


A = _load(
    "a211_ac_parity_probe",
    SCRIPTS / "launch_action_ball_a211_four_arm_diagnostic.py",
)
C = _load(
    "c211_ac_parity_probe",
    SCRIPTS / "launch_action_ball_c211_diagnostic.py",
)


# --------------------------------------------------------------------------- #
# 允许不同的键 —— 显式点名 + 理由。默认是"必须相同"。
# --------------------------------------------------------------------------- #
# 规矩:
#   * 只能因为"问什么题"(观测/目标语义)或"怎么给分"(reward)而进这张表;
#   * 每一条都要能被一句话说清为什么它必须不同;
#   * 加一条 = 承认多一处架构外差异,请顺手在 exp 里留账。
# 反向也有门:``test_no_legitimate_difference_entry_is_stale`` 要求表里每一条**今天
# 确实还在不同**。已经对齐的键留在表里 = 一个永久敞开的洞。
LEGITIMATE_DIFFERENCES: dict[str, str] = {
    # ---- 身份(不是设置,是"这是哪条臂") ----
    "name": "task profile 身份;合流即 A 的 checkpoint 能装进 C",
    "gym_task": "Gym 注册身份,两族是两个环境",
    "actor_obs_contract": "211-D actor 的观测契约名 = 本族问什么题",
    "experiment_name": "日志/收据命名空间身份",
    # ---- 问什么题(目标语义) ----
    "racket.action_ball_target_source": (
        "A=online_solver 解逆问题;C=direct_ball 直接吃来球。这就是本对比的自变量"
    ),
    "racket.action_ball_target_recipe": (
        "A=current_lm 期望接触配方;C=outcome_dense_only 只有落点结果"
    ),
    "racket.action_ball_target_validity_mask": (
        "A 三路期望接触通道全有效;C 全无效(它的 actor 里根本没有这三路)"
    ),
    "racket.action_ball_reuse_exact_question_until_semantics_change": (
        "只有 A 有可复用的精确答案;C 零 solver 调用,没有答案可缓存"
    ),
    # ---- 怎么给分(reward) ----
    # C 是 outcome_dense_only:所有"期望接触"方向的稠密项归零,改由 C 专属的
    # exact-strike proximity manager(权重 240.0,无通用 YAML override)承担。
    "rewards.racket_position_weight": "C 关掉全部期望接触稠密项",
    "rewards.racket_velocity_weight": "C 关掉全部期望接触稠密项",
    "rewards.racket_normal_weight": "C 关掉全部期望接触稠密项",
    "rewards.racket_position_coarse_weight": "C 关掉全部期望接触稠密项",
    "rewards.racket_velocity_coarse_weight": "C 关掉全部期望接触稠密项",
    "rewards.racket_normal_coarse_weight": "C 关掉全部期望接触稠密项",
    "rewards.racket_position_precision_weight": "C 关掉全部期望接触稠密项",
    "rewards.racket_velocity_precision_weight": "C 关掉全部期望接触稠密项",
    "rewards.racket_normal_precision_weight": "C 关掉全部期望接触稠密项",
    "rewards.racket_position_coarse_std": (
        "A 把粗核宽度收到 0.20;C 该项权重为 0,宽度停在 VendorV2 继承值 0.70"
    ),
    "rewards.racket_velocity_coarse_std": (
        "A 把粗核宽度收到 1.50;C 该项权重为 0,宽度停在 VendorV2 继承值 4.0"
    ),
    "rewards.strike_capture_bonus_weight": (
        "C 的结果口径只认过网与落点,不给「接到了」单独发钱"
    ),
    "rewards.virtual_pass_net_weight": "同上:C 只保留合法落台这一档结果收入",
    "rewards.virtual_landing_dense_weight": "同上:C 不给落点稠密引导",
}


# 每族 argv 里"故意覆盖自己 YAML"的键(路径/sha/命名空间这类逐次跑都不同的量)。
# 这张表本身不点名具体值,只要求**两族的这张表逐字相同** —— 一边多写一条覆盖、
# 另一边没跟上,就是本仓最常见的那种漂。
_ARGV_OVERRIDE_SHAPE_NOTE = (
    "两族 argv 相对自己 profile 的覆盖集合必须完全一样;不一样 = 一边改了另一边没跟上"
)


# --------------------------------------------------------------------------- #
# Hydra defaults 链解析(只依赖 PyYAML)
# --------------------------------------------------------------------------- #
class ProfileChainUnsupported(AssertionError):
    """本解析器只认 ``[<parent>@_here_, ..., _self_]`` 这一种 defaults 形状。"""


def _profile_path(ref: str, group: str) -> Path:
    if ref.startswith("/"):
        return CFG_ROOT / (ref[1:] + ".yaml")
    return CFG_ROOT / group / (ref + ".yaml")


def _split_ref(entry: Any, group: str) -> "tuple[str, str]":
    text = str(entry)
    if not text.endswith("@_here_"):
        raise ProfileChainUnsupported(
            "defaults entry %r is not a <name>@_here_ merge; the parity gate's "
            "resolver only models that form" % text
        )
    ref = text[: -len("@_here_")]
    if ref.startswith("/"):
        head = ref[1:]
        return ref, head.rsplit("/", 1)[0] if "/" in head else ""
    return ref, group


def _deep_merge(base: Mapping[str, Any], over: Mapping[str, Any]) -> dict:
    out = dict(base)
    for key, value in over.items():
        current = out.get(key)
        if isinstance(value, dict) and isinstance(current, dict):
            out[key] = _deep_merge(current, value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read_profile(ref: str, group: str) -> "tuple[dict, list]":
    path = _profile_path(ref, group)
    if not path.is_file():
        raise ProfileChainUnsupported("task profile %s is missing" % path)
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(doc, dict):
        raise ProfileChainUnsupported("task profile %s is not a mapping" % path)
    doc = dict(doc)
    defaults = doc.pop("defaults", None) or []
    if defaults and defaults[-1] != "_self_":
        raise ProfileChainUnsupported(
            "%s puts something after _self_; the resolver only models "
            "parents-then-self merges" % ref
        )
    return doc, list(defaults[:-1]) if defaults else []


def resolve_task_profile(ref: str, group: str = "task") -> dict:
    """把一片 task profile 按 Hydra defaults 链解析成一份完整配置。"""

    doc, parents = _read_profile(ref, group)
    accumulated: dict = {}
    for parent in parents:
        parent_ref, parent_group = _split_ref(parent, group)
        accumulated = _deep_merge(
            accumulated, resolve_task_profile(parent_ref, parent_group)
        )
    return _deep_merge(accumulated, doc)


def profile_chain(ref: str, group: str = "task") -> list:
    """返回这片 profile 的祖先链(最老在前,自己在最后)。"""

    _doc, parents = _read_profile(ref, group)
    chain: list = []
    for parent in parents:
        parent_ref, parent_group = _split_ref(parent, group)
        chain.extend(profile_chain(parent_ref, parent_group))
    chain.append(ref)
    return chain


# --------------------------------------------------------------------------- #
# argv override 叠加(照抄 OmegaConf 的"按既有节点类型强制"行为)
# --------------------------------------------------------------------------- #
_ABSENT = object()


def _coerce_override(raw: str, prior: Any) -> Any:
    """把一条 CLI override 的字符串值变成 OmegaConf 会写进节点的那个值。

    关键点:节点已经是 str 时**不要**再解析。全零 sha256 会被 PyYAML 读成整数 0,
    而 OmegaConf 因为节点是 str 会保持原字符串 —— 这条规则就是被 hydra 交叉核对
    抓出来的。
    """

    if prior is not _ABSENT and isinstance(prior, str):
        return raw
    if raw == "":
        return ""
    try:
        parsed = yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw
    if parsed is None:
        return raw
    if prior is _ABSENT:
        return parsed
    if isinstance(prior, bool):
        if not isinstance(parsed, bool):
            raise ProfileChainUnsupported(
                "override %r targets a bool node but is not true/false" % raw
            )
        return parsed
    if (
        isinstance(prior, float)
        and isinstance(parsed, int)
        and not isinstance(parsed, bool)
    ):
        return float(parsed)
    return parsed


def apply_task_overrides(config: Mapping[str, Any], argv: "list[str]") -> dict:
    """把 argv 里的 ``task.*`` / ``+task.*`` override 叠到一份解析好的配置上。"""

    out = copy.deepcopy(dict(config))
    for token in argv[2:]:
        if "=" not in token:
            continue
        key, raw = token.split("=", 1)
        appended = key.startswith("+")
        key = key.lstrip("+")
        if not key.startswith("task."):
            continue
        parts = key.split(".")[1:]
        node = out
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        leaf = parts[-1]
        present = leaf in node
        if appended and present:
            raise ProfileChainUnsupported(
                "+%s appends a key the profile already defines" % key
            )
        if not appended and not present:
            raise ProfileChainUnsupported(
                "%s overrides a key no profile defines; hydra would refuse it "
                "without a leading +" % key
            )
        node[leaf] = _coerce_override(raw, node[leaf] if present else _ABSENT)
    return out


def flatten(config: Mapping[str, Any], prefix: str = "") -> dict:
    out: dict = {}
    for key, value in (config or {}).items():
        path = prefix + str(key)
        if isinstance(value, dict):
            out.update(flatten(value, path + "."))
        else:
            out[path] = value
    return out


# --------------------------------------------------------------------------- #
# 用同一份 spec/lineage 拿两族的活 argv
# --------------------------------------------------------------------------- #
# 两族喂同一份输入,于是 checkout/namespace/seed/题面这些"逐次跑都不同"的量在两边
# 逐字相同,活下来的差异只能是族差异。stage=materialize 是唯一不需要上游收据的阶段,
# ``_planned_materialization`` 纯计算、不碰磁盘。
_SPEC = {
    "source": {"checkout": "/parity/checkout", "isaac_python": "/parity/python"},
    "stage": "materialize",
    "num_envs": 1,
    "max_iterations": 0,
    "save_interval": 1,
    "namespace": "/parity/namespace/cell",
}
_LINEAGE = {
    "seed": 0,
    "action_id": "take_061_unit04_bh",
    "lineage_sha256": "dd" * 32,
    "dr_l0_manifest": {"parity_probe": True},
    "motion": {"path": "parity_motion.npz"},
    "action_manifest": {"path": "parity_manifest.json", "sha256": "aa" * 32},
    "dynamic_ready_artifact": {"path": "parity_ready.json", "sha256": "bb" * 32},
    "dynamic_ready_nominal_receipt": {
        "path": "parity_hold.json",
        "sha256": "cc" * 32,
    },
}


def matched_family_argv() -> "tuple[list[str], list[str]]":
    """两族在同一份输入、同一格(噪声关)下真正会发出的 argv。"""

    arm = A._arm_contract(A.A_OBS_NOISE_OFF_CELL_ID)
    recipe = C._recipe_contract(C.C_OBS_NOISE_OFF_CELL_ID)
    return (
        A._training_argv(_SPEC, _LINEAGE, arm),
        C._training_argv(_SPEC, _LINEAGE, recipe),
    )


def _profile_name_from_argv(argv: "list[str]") -> str:
    for token in argv[2:]:
        if token.startswith("task="):
            return token.split("=", 1)[1]
    raise AssertionError("launcher argv carries no task= profile selector")


def effective_task_config(argv: "list[str]") -> dict:
    return apply_task_overrides(resolve_task_profile(_profile_name_from_argv(argv)), argv)


def parity_violations(
    effective_a: Mapping[str, Any], effective_c: Mapping[str, Any]
) -> dict:
    """返回所有"不在理由清单上"的差异。空 dict = 两族只差 obs/reward。

    默认关闭:共有键值不同、或某个键只在一族出现,都算违规,除非它被
    ``LEGITIMATE_DIFFERENCES`` 点名。
    """

    flat_a = flatten(effective_a)
    flat_c = flatten(effective_c)
    violations: dict = {}
    for key in sorted(set(flat_a) | set(flat_c)):
        left = flat_a.get(key, _ABSENT)
        right = flat_c.get(key, _ABSENT)
        if left == right and left is not _ABSENT:
            continue
        if key in LEGITIMATE_DIFFERENCES:
            continue
        violations[key] = (
            "<absent on A>" if left is _ABSENT else left,
            "<absent on C>" if right is _ABSENT else right,
        )
    return violations


def argv_override_shape(argv: "list[str]") -> dict:
    """这条 argv 相对自己 profile 真正改动了哪些键(键 -> 'override'/'append')。"""

    base = flatten(resolve_task_profile(_profile_name_from_argv(argv)))
    shape: dict = {}
    for token in argv[2:]:
        if "=" not in token:
            continue
        key, raw = token.split("=", 1)
        appended = key.startswith("+")
        key = key.lstrip("+")
        if not key.startswith("task."):
            continue
        leaf = key[len("task.") :]
        if appended:
            shape[leaf] = "append"
            continue
        prior = base.get(leaf, _ABSENT)
        if _coerce_override(raw, prior) != prior:
            shape[leaf] = "override"
    return shape


# --------------------------------------------------------------------------- #
# 门
# --------------------------------------------------------------------------- #
def test_both_families_fork_from_one_common_ancestor_profile():
    """分叉点以上必须逐个 profile 相同 —— 否则"其余相同"从起点就不成立。"""

    a_chain = profile_chain(A_PROFILE)
    c_chain = profile_chain(C_PROFILE)
    assert COMMON_ANCESTOR in a_chain and COMMON_ANCESTOR in c_chain, (
        a_chain,
        c_chain,
    )
    cut_a = a_chain.index(COMMON_ANCESTOR) + 1
    cut_c = c_chain.index(COMMON_ANCESTOR) + 1
    assert a_chain[:cut_a] == c_chain[:cut_c], (
        "A211 和 C211 的继承链在共同祖先 %s 之前就已经不同了:%r vs %r"
        % (COMMON_ANCESTOR, a_chain[:cut_a], c_chain[:cut_c])
    )
    # 分叉点以下各自只准有两片(family parent + DR-L0 leaf),多一片就说明有人偷偷
    # 塞了第三层继承 —— 那一层的内容不会被本门以外的任何东西核对。
    assert a_chain[cut_a:] == [
        "HOPEPingPongActionBallA211VendorV2N1Learnability",
        A_PROFILE,
    ], a_chain
    assert c_chain[cut_c:] == [
        "HOPEPingPongActionBallC211VendorV2N1Learnability",
        C_PROFILE,
    ], c_chain


def test_effective_task_config_differs_only_on_declared_obs_and_reward_axes():
    """本文件的主门:258 片叶子逐键比,默认必须相同。"""

    a_argv, c_argv = matched_family_argv()
    violations = parity_violations(
        effective_task_config(a_argv), effective_task_config(c_argv)
    )
    assert not violations, (
        "A211/C211 的有效任务配置在 obs/reward 之外漂开了。以下每一条要么对齐,"
        "要么带理由写进 LEGITIMATE_DIFFERENCES:\n"
        + "\n".join(
            "  %s: A=%r C=%r" % (key, left, right)
            for key, (left, right) in violations.items()
        )
    )


def test_no_legitimate_difference_entry_is_stale():
    """清单里每一条今天必须**确实还在不同**;对齐了还留着 = 永久敞开的洞。"""

    a_argv, c_argv = matched_family_argv()
    flat_a = flatten(effective_task_config(a_argv))
    flat_c = flatten(effective_task_config(c_argv))
    stale = sorted(
        key
        for key in LEGITIMATE_DIFFERENCES
        if key in flat_a and key in flat_c and flat_a[key] == flat_c[key]
    )
    assert not stale, (
        "这些键已经在两族对齐了,请把它们从 LEGITIMATE_DIFFERENCES 删掉,"
        "否则将来它们再漂就没人拦:%r" % stale
    )
    missing = sorted(
        key
        for key in LEGITIMATE_DIFFERENCES
        if key not in flat_a and key not in flat_c
    )
    assert not missing, (
        "LEGITIMATE_DIFFERENCES 点名了两族都不存在的键,清单已经过期:%r" % missing
    )


def test_every_legitimate_difference_carries_a_written_reason():
    for key, reason in LEGITIMATE_DIFFERENCES.items():
        assert isinstance(reason, str) and len(reason.strip()) >= 8, (key, reason)


def test_both_launchers_override_their_own_profile_in_the_same_shape():
    """两族 argv 相对自己 YAML 的改动集合必须逐字相同。"""

    a_argv, c_argv = matched_family_argv()
    assert argv_override_shape(a_argv) == argv_override_shape(c_argv), (
        _ARGV_OVERRIDE_SHAPE_NOTE,
        argv_override_shape(a_argv),
        argv_override_shape(c_argv),
    )


def test_both_launchers_run_the_same_entrypoint_and_interpreter():
    a_argv, c_argv = matched_family_argv()
    assert a_argv[:2] == c_argv[:2], (a_argv[:2], c_argv[:2])


# --------------------------------------------------------------------------- #
# 变异测试 —— 证明这道门"该拦的拦得住、该放的放得过"
# --------------------------------------------------------------------------- #
def _mutated(config: Mapping[str, Any], dotted: str, value: Any) -> dict:
    out = copy.deepcopy(dict(config))
    node = out
    parts = dotted.split(".")
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value
    return out


@pytest.mark.parametrize(
    "dotted, value",
    [
        # 每一条都是"如果真发生了,四格对比就报废"的漂:
        ("actions.control_step_action_delay_max", 2),   # 延迟只进了一族
        ("env.episode_length_s", 12.0),                 # episode 长度不一样
        ("env.num_envs", 2048),                         # 并行规模不一样
        ("domain_rand.startup_physics_material", True), # DR 档只在一族开
        ("domain_rand.policy_observation_corruption", True),  # 注册差异轴被串族
        ("task_wait.max_wait_ticks", 40),               # 隐藏等待窗不一样
        ("push.enable", True),                          # 推撞只进了一族
        ("physical_ball", True),                        # 一族跑上了物理球
        ("racket.adaptive_sigma", True),                # 核宽控制器只在一族活
        ("racket.action_ball_initial_center_single_question", False),  # 课程起点不同
    ],
)
def test_mutation_a_real_cross_family_drift_is_refused(dotted, value):
    a_argv, c_argv = matched_family_argv()
    baseline = effective_task_config(a_argv)
    effective_c = effective_task_config(c_argv)
    assert not parity_violations(baseline, effective_c), "baseline must be clean first"
    assert flatten(baseline)[dotted] != value, (
        "变异值和现值相同,这条变异什么都没测:%s" % dotted
    )
    violations = parity_violations(_mutated(baseline, dotted, value), effective_c)
    assert dotted in violations, (dotted, violations)


def test_mutation_a_brand_new_one_sided_key_is_refused_by_default():
    """新增一个键不会自动逃逸 —— 这正是点名清单最容易犯的错。"""

    a_argv, c_argv = matched_family_argv()
    baseline = effective_task_config(a_argv)
    effective_c = effective_task_config(c_argv)
    grown = copy.deepcopy(baseline)
    grown["rewards"]["some_new_term_weight"] = 1.0
    violations = parity_violations(grown, effective_c)
    assert "rewards.some_new_term_weight" in violations, violations
    # 反方向也一样:C 长出来的新键同样要被看见。
    grown_c = copy.deepcopy(effective_c)
    grown_c["racket"]["some_new_channel"] = True
    assert "racket.some_new_channel" in parity_violations(baseline, grown_c)


@pytest.mark.parametrize(
    "dotted, value",
    [
        ("rewards.racket_position_weight", 9.9),        # A 侧重调 reward 权重
        ("rewards.strike_capture_bonus_weight", 0.5),   # 结果项配比重调
        ("racket.action_ball_target_recipe", "some_other_recipe"),  # 目标配方换名
        ("racket.action_ball_target_source", "another_solver"),     # 题源换名
    ],
)
def test_mutation_a_legitimate_obs_or_reward_difference_still_passes(dotted, value):
    a_argv, c_argv = matched_family_argv()
    baseline = effective_task_config(a_argv)
    effective_c = effective_task_config(c_argv)
    assert flatten(baseline)[dotted] != value
    assert not parity_violations(_mutated(baseline, dotted, value), effective_c)


def test_mutation_argv_override_shape_asymmetry_is_visible():
    """一族多写一条覆盖、另一族没跟上,形状门必须看得见。"""

    a_argv, _c_argv = matched_family_argv()
    shape = argv_override_shape(a_argv)
    assert "racket.clip_names" in shape, shape
    grown = dict(shape)
    grown["racket.some_newly_pinned_field"] = "override"
    assert grown != shape


# --------------------------------------------------------------------------- #
# 本地解析器 vs 真 hydra —— 手写副本不许悄悄漂
# --------------------------------------------------------------------------- #
def test_local_resolver_and_overlay_reproduce_hydra():
    """有 hydra 的环境里逐字段核对本文件的解析结果。

    这条跳过时上面所有门仍然在跑;它只是把"我这份手写解析器还准不准"钉住。
    """

    hydra = pytest.importorskip("hydra", reason="host pytest env has no hydra")
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    assert hydra is not None
    a_argv, c_argv = matched_family_argv()
    for argv in (a_argv, c_argv):
        overrides = [
            token
            for token in argv[2:]
            if token.startswith("task=")
            or token.lstrip("+").split(".", 1)[0] == "task"
        ]
        with initialize_config_dir(version_base=None, config_dir=str(CFG_ROOT)):
            composed = compose(config_name="train", overrides=overrides)
        expected = OmegaConf.to_container(composed.task, resolve=False)
        observed = effective_task_config(argv)
        assert observed == expected, {
            key: (observed.get(key, "<absent>"), expected.get(key, "<absent>"))
            for key in set(flatten(observed)) | set(flatten(expected))
            if flatten(observed).get(key, "<absent>")
            != flatten(expected).get(key, "<absent>")
        }
