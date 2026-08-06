"""225 家族已整族退役 —— 这里是"它回不来"的机器证据。

人话
====
2026-08-03 的 A225/C225 是 211 的前身:actor 225 维、critic 318 维。211 把
actor 里那 15 维 raw teacher-base 删掉、补一维 ``task_valid``,得到 211/319。
两族**宽度不同、语义不同、normalizer 和 checkpoint 都不通用**,所以 225 不是
"旧一点的 211",是另一套 ABI。

2026-08-06 盘点确认 225 **整族不可达**,四道各自独立的门任何一道都足以拦死它:

1. **gym 注册表里没有它**。``config/agibot_a3/__init__.py`` 只注册了 A211/C211;
   ``HOPE-PingPong-ActionBall-A225Learnability-AgibotA3-v0`` 和 C225 那条根本
   不存在,``gym.make`` 在任何 obs_mode 逻辑跑到之前就 ``NameNotFound``。
   (这一条由 ``test_action_ball_task_config.py`` 盯着,本文件不重复。)
2. **没有对应的 EnvCfg 类**。``hope_env_cfg.py`` 里没有任何
   ``HOPEPingPongActionBallA225/C225...EnvCfg``,也没有任何 ``obs_mode`` 默认值
   等于 ``action_ball_a225``/``action_ball_c225``。
3. **``train.py`` 拒 actor 合同**:``configured_actor_contract in {四个 legacy}``
   → ``_OverrideError``。
4. **``MotionOnPolicyRunner.__init__`` 拒 obs_mode**:同一组四个名字 →
   ``RuntimeError``,而且在 ``super().__init__`` **之前**,拒得最早。

两个 225 发射器自己的 argv 里写死了 ``task.actor_obs_contract=action_ball_a225``
(或 c225),所以它们连第 3 道门都过不去 —— 发射器是结构性死代码,不是"暂时不用"。
2026-08-06 把它们连同两个零调用点的 225 trainability validator、两份 task yaml、
三个只测这些死文件的测试一起删了(约 9.5k 行)。

为什么还要留这个测试文件
========================
删掉发射器 = 删掉了唯一会踩到第 3、4 道门的代码路径。核对过:**这两道门在 2026-08-06
之前一个测试都没有**。也就是说,现在谁把 ``train.py:3325`` 那个 set 清空、或者把
runner 里那个 ``if`` 删掉,全仓没有任何东西会红 —— 而那正是"225 不可达"这个结论的
承重墙。所以退役和这份收据必须同批交付。

不做"第三份手抄"
================
本文件**不**自己抄一份 legacy 名单。唯一权威是
``action_ball_a211_trainability._LEGACY_MODES``(纯 stdlib、可直接 import,而且被
``test_action_ball_a225_trainability.py`` 拿真调用 ``validate_action_ball_211_cfg_trainability``
逐个跑过,是活值不是字面量)。这里只证明 ``train.py`` 与 ``my_on_policy_runner.py``
两处内联 set 跟它逐字相同。

读法用 AST 直接取那两个 ``if X in {...}`` 判据本身
==================================================
不是在源码里 grep ``"action_ball_a225"`` —— 那种粗检查在 set 被清空、甚至整个 ``if``
被删掉之后照样通过(名字还在 docstring 和错误信息里)。这里定位的是**判据节点自己**:
``Compare(left=Name(<变量名>), ops=[In], comparators=[Set(...)])``,门没了就取不到,
名单少一个就不等。``train.py`` import 会拉起 Isaac,所以只解析不 import。
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest


WBT = Path(__file__).resolve().parents[1]
SCRIPTS = WBT / "scripts"
TRACKING = (
    WBT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
)
UTILS = (
    WBT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "utils"
)


# ---------------------------------------------------------------------------
# 1. 删掉的东西要保持删掉
# ---------------------------------------------------------------------------

RETIRED_PATHS = (
    SCRIPTS / "launch_action_ball_a225_four_arm_diagnostic.py",
    SCRIPTS / "launch_action_ball_c225_diagnostic.py",
    SCRIPTS / "materialize_action_ball_a225_lineage.py",
    TRACKING / "action_ball_225_trainability.py",
    TRACKING / "action_ball_c225_trainability.py",
    WBT / "cfg" / "task" / "HOPEPingPongActionBallA225VendorV2N1Learnability.yaml",
    WBT / "cfg" / "task" / "HOPEPingPongActionBallC225VendorV2N1Learnability.yaml",
)


@pytest.mark.parametrize("path", RETIRED_PATHS, ids=[p.name for p in RETIRED_PATHS])
def test_retired_225_file_stays_deleted(path):
    """复活其中任何一个都必须是一次自觉的决定,不能是复制粘贴时顺手带回来。

    真要重开 225,得先把 gym 注册、EnvCfg 类、train.py 与 runner 的两道门一起改回去
    —— 那时候这个测试红了正好提醒:这不是加一个文件,是重开一整套 ABI。
    """

    assert not path.exists(), f"225 家族已退役,但这个文件回来了: {path}"


def test_no_env_cfg_declares_a_225_obs_mode():
    """第 2 道门:没有任何 EnvCfg 能产出 225 的 obs_mode。

    只看 ``obs_mode: str = "..."`` 这种带默认值的 annotated assignment ——
    那正是 ``gym.make`` 之后 ``env.cfg.obs_mode`` 的来源。
    """

    env_cfg = TRACKING / "config" / "agibot_a3" / "hope_env_cfg.py"
    tree = ast.parse(env_cfg.read_text(encoding="utf-8"))
    declared = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.AnnAssign) or node.value is None:
            continue
        target = node.target
        if not isinstance(target, ast.Name) or target.id != "obs_mode":
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            declared.add(node.value.value)
    assert declared, "hope_env_cfg.py 里一个 obs_mode 默认值都没解析到,读法坏了"
    assert not declared & {"action_ball_a225", "action_ball_c225"}


# ---------------------------------------------------------------------------
# 2. 让 225 不可达的那两道门 —— 跟唯一权威逐字核对
# ---------------------------------------------------------------------------


def _live_legacy_modes() -> frozenset:
    """从 A211 trainability 叶子取**活值**(纯 stdlib,直接 import,不是解析字面量)。"""

    path = TRACKING / "action_ball_a211_trainability.py"
    spec = importlib.util.spec_from_file_location("a211_trainability_legacy_authority", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return frozenset(module._LEGACY_MODES)


def _membership_set_literal(path: Path, variable: str) -> frozenset:
    """返回 ``if <variable> in {...}`` 里那个 set 判据本身。

    找不到就是门没了 —— 直接失败,不做任何"可能改名了"的宽容处理。
    """

    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare) or len(node.ops) != 1:
            continue
        if not isinstance(node.ops[0], ast.In):
            continue
        if not isinstance(node.left, ast.Name) or node.left.id != variable:
            continue
        if not isinstance(node.comparators[0], ast.Set):
            continue
        found.append(frozenset(ast.literal_eval(node.comparators[0])))
    assert found, f"{path}: 找不到 `if {variable} in {{...}}` 这道门"
    assert len(found) == 1, f"{path}: `{variable} in {{...}}` 出现了 {len(found)} 次"
    return found[0]


LEGACY_GATES = (
    (SCRIPTS / "train.py", "configured_actor_contract"),
    (UTILS / "my_on_policy_runner.py", "runtime_obs_mode"),
)


@pytest.mark.parametrize(
    "path,variable",
    LEGACY_GATES,
    ids=[f"{p.name}:{v}" for p, v in LEGACY_GATES],
)
def test_legacy_gate_still_names_every_live_legacy_mode(path, variable):
    """两道门必须跟 ``_LEGACY_MODES`` 逐字相同 —— 少一个名字就是开了一条缝。

    2026-08-06 之前这两道门零测试覆盖。删掉 225 发射器等于删掉了唯一会踩到它们的
    代码路径,所以这份核对是退役的承重部分,不是装饰。
    """

    assert _membership_set_literal(path, variable) == _live_legacy_modes()


def test_the_legacy_authority_still_covers_both_225_contracts():
    """权威名单自己也得盯着:a225/c225 掉出去了,上面两条就变成"两边一起漏"。"""

    assert {"action_ball_a225", "action_ball_c225"} <= _live_legacy_modes()
