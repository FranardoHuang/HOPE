"""三份安全词表在全仓十几个手抄副本之间必须逐字相同 —— 机器来核对。

人话
====
这个仓库反复出同一种错:**同一个事实存了多份,改一份,剩下几份静默过期**。
本 session 已经踩到过四次(四处 ``init_noise_std`` 硬钉、两条 soft-limit 通道带宽不
一致、DR-L0 关轴删载体、``table_termination`` 广相/精确判据错位)。

这个文件盯的是三份**安全语义词表**。它们决定"什么算把人摔了 / 什么算撞桌子 /
参考包络违规算哪几项",也就是每条臂的**终止面**。全仓各自手抄了这么多份:

* **硬安全终止并集**(5 项):10 个持有者,横跨 production 包、四个发射器、三个
  oracle 脚本和 audit 工具。改一处 = 那一条臂的死法跟别人不一样,而且没有任何
  现役代码会发现。
* **参考包络终止项**(3 项):9 个持有者。名字各不相同(``REFERENCE_GUARD_REASONS``
  是"守卫为什么开火"、``PROHIBITED_HOLD_REFERENCE_TERMINATIONS`` 是"hold 期间禁用
  哪几项"、``EXACT_PHASE_FIDELITY_REASON_ORDER`` 是跨引擎复盘的理由顺序),但指的
  是同一个三元组。
* **终止 outcome 词表**(7 项):2 个现役持有者。第三份
  ``action_ball_runtime.FROZEN_TERMINAL_OUTCOMES`` 零读者,2026-08-06 删掉了。

为什么不做"单一真源 + import"
=============================
做不了,而且硬做会**放宽** fail-closed 门:

* ``action_ball_evaluation_inbox.py`` 的模块 docstring 自陈 "dependency-light so the
  inbox can be audited and tested on a CPU-only host"。它和 ``action_ball_evaluation.py``
  都是纯 stdlib、按文件路径 exec 加载的;改成包内 import 会拽进
  ``whole_body_tracking.tasks.tracking.mdp.__init__``,而那里第一行就是
  ``from isaaclab.envs.mdp import *``。CPU-only 审计当场没了。
* 四个发射器把自己**加载过的每个文件**都钉进 ``RUNTIME_SOURCE_PATHS`` 并算进 launch
  claim。新引一个共享常量模块,就必须同批把它加进四边的 tracked-source 钉子表,
  否则等于让一个没被 provenance 覆盖的文件参与决策。

所以按"dependency-free 模块必须手抄 → 加跨模块一致性断言"办:副本留着,但漂移
在 host 测试就红,不必先烧一次 Pod 时间。

读法用 AST 直接取字面量,不 import 任何持有者
============================================
一半持有者是发射器/oracle 脚本,import 它们会拉起 torch 甚至 Kit。这里只解析源码取
模块级赋值的字面量,所以在任何 CPU host 上都能跑,也不会因为某个脚本的 import 副作用
把测试变成集成测试。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


WBT = Path(__file__).resolve().parents[1]
REPO = WBT.parents[1]
MDP = (
    WBT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
)
UTILS = (
    WBT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "utils"
)
SCRIPTS = WBT / "scripts"
MUJOCO_NATIVE = WBT / "mujoco_native"


def _module_literal(path: Path, name: str):
    """Return the literal value of one module-level ``NAME = <literal>``.

    Deliberately AST-only: several holders are launcher scripts whose import
    side effects would drag torch/Kit into a host-side unit test.
    """

    assert path.is_file(), f"holder file is missing: {path}"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == name:
            found.append(ast.literal_eval(node.value))
    assert found, f"{path}: module-level literal {name} is gone"
    assert len(found) == 1, f"{path}: {name} is assigned {len(found)} times"
    return found[0]


# ---------------------------------------------------------------------------
# 1. 硬安全终止并集 —— 每条臂"允许怎么死"的那一份表
# ---------------------------------------------------------------------------

HARD_SAFETY_TERMINATIONS = (
    "base_fell_tilt",
    "base_too_low",
    "joint_actual_forbidden",
    "joint_qdes_forbidden",
    "robot_hit_table",
)

HARD_SAFETY_HOLDERS = (
    (UTILS / "effective_reward_recipe.py",
     "ACTION_BALL_HARD_SAFETY_TERMINATION_TERMS"),
    (MDP / "hope_rewards.py",
     "_ACTION_BALL_HARD_SAFETY_TERMINATIONS"),
    (SCRIPTS / "audit_reward_run.py",
     "HARD_SAFETY_TERMINATION_TERMS"),
    (SCRIPTS / "action_ball_c211_live_oracle.py",
     "HARD_TERMINATIONS"),
    (SCRIPTS / "action_ball_c211_oracle_evidence.py",
     "HARD_TERMINATIONS"),
    (SCRIPTS / "sweep_action_ball_a211_physical_ready_qdes.py",
     "HARD_TERMINATIONS"),
    (SCRIPTS / "launch_action_ball_a211_four_arm_diagnostic.py",
     "HARD_TERMINATION_UNION"),
    (SCRIPTS / "launch_action_ball_a225_four_arm_diagnostic.py",
     "HARD_TERMINATION_UNION"),
    (SCRIPTS / "launch_action_ball_c211_diagnostic.py",
     "HARD_TERMINATION_UNION"),
    (SCRIPTS / "launch_action_ball_c225_diagnostic.py",
     "HARD_TERMINATION_UNION"),
)


@pytest.mark.parametrize(
    "path,name",
    HARD_SAFETY_HOLDERS,
    ids=[f"{p.name}:{n}" for p, n in HARD_SAFETY_HOLDERS],
)
def test_hard_safety_termination_union_is_identical_everywhere(path, name):
    assert tuple(_module_literal(path, name)) == HARD_SAFETY_TERMINATIONS


def test_hard_safety_holder_count_is_pinned():
    """新增一份手抄副本必须同批登记进来,否则这个门就只盯着旧的十份。"""

    assert len(HARD_SAFETY_HOLDERS) == 10
    assert len({path for path, _name in HARD_SAFETY_HOLDERS}) == 10


# ---------------------------------------------------------------------------
# 2. 参考包络终止项 —— hold 期间禁用、守卫开火理由、跨引擎复盘顺序,同一个三元组
# ---------------------------------------------------------------------------

REFERENCE_ENVELOPE_TERMS = ("anchor_pos", "anchor_ori", "ee_body_pos")

REFERENCE_ENVELOPE_HOLDERS = (
    (UTILS / "effective_reward_recipe.py",
     "ACTION_BALL_REFERENCE_ENVELOPE_TERMINATION_TERMS"),
    (MDP / "action_ball_reference_guard.py", "REFERENCE_GUARD_REASONS"),
    (MDP / "hope_commands.py", "_REFERENCE_TERMINATION_TERMS"),
    (MUJOCO_NATIVE / "vec_env.py", "EXACT_PHASE_FIDELITY_REASON_ORDER"),
    (SCRIPTS / "isaac_bank_exam.py", "GUARD_TERMS"),
    (SCRIPTS / "termination_contract.py", "TRACKING_TERMS"),
    (SCRIPTS / "audit_reward_run.py", "REFERENCE_ENVELOPE_TERMINATION_TERMS"),
    (SCRIPTS / "launch_action_ball_a211_four_arm_diagnostic.py",
     "PROHIBITED_HOLD_REFERENCE_TERMINATIONS"),
    (SCRIPTS / "launch_action_ball_c211_diagnostic.py",
     "PROHIBITED_HOLD_REFERENCE_TERMINATIONS"),
)


@pytest.mark.parametrize(
    "path,name",
    REFERENCE_ENVELOPE_HOLDERS,
    ids=[f"{p.name}:{n}" for p, n in REFERENCE_ENVELOPE_HOLDERS],
)
def test_reference_envelope_terms_are_identical_everywhere(path, name):
    """顺序也算 —— ``EXACT_PHASE_FIDELITY_REASON_ORDER`` 的语义就是顺序。

    如果将来真有一处需要跟其他八处不一样,正确做法是把它从这张表里拆出去并写清
    为什么,不是把这个测试删掉。
    """

    assert tuple(_module_literal(path, name)) == REFERENCE_ENVELOPE_TERMS


def test_reference_envelope_holder_count_is_pinned():
    assert len(REFERENCE_ENVELOPE_HOLDERS) == 9
    assert len({path for path, _name in REFERENCE_ENVELOPE_HOLDERS}) == 9


# ---------------------------------------------------------------------------
# 3. 终止 outcome 词表 —— frozen evaluator 与它的 inbox 传输层
# ---------------------------------------------------------------------------

TERMINAL_OUTCOMES = (
    "legal_return",
    "safe_nonreturn",
    "table_hit",
    "fall",
    "collision",
    "joint_qdes_limit",
    "joint_actual_limit",
)

TERMINAL_OUTCOME_HOLDERS = (
    (MDP / "action_ball_evaluation.py", "TERMINAL_OUTCOMES"),
    (MDP / "action_ball_evaluation_inbox.py", "TERMINAL_OUTCOMES"),
)


@pytest.mark.parametrize(
    "path,name",
    TERMINAL_OUTCOME_HOLDERS,
    ids=[f"{p.name}:{n}" for p, n in TERMINAL_OUTCOME_HOLDERS],
)
def test_terminal_outcome_vocabulary_is_identical_everywhere(path, name):
    assert tuple(_module_literal(path, name)) == TERMINAL_OUTCOMES


def test_the_deleted_third_copy_stays_deleted():
    """``action_ball_runtime.FROZEN_TERMINAL_OUTCOMES`` 零读者,2026-08-06 删除。

    它跟上面两份逐字相同,但谁都不读它 —— 留着只会让下一个人以为改它有用。
    """

    source = (MDP / "action_ball_runtime.py").read_text(encoding="utf-8")
    assert "FROZEN_TERMINAL_OUTCOMES = (" not in source


# ---------------------------------------------------------------------------
# 4. 每个 recipe 的目标有效位 —— 哪几列算进 reward 的 eligible 分母
# ---------------------------------------------------------------------------
#
# 三份都活着,而且是**同一个决策**的三份手抄:
#
# * ``hope_commands._ACTION_BALL_TARGET_VALIDITY_BY_RECIPE`` 是现役那份 ——
#   ``_action_ball_target_metric_eligibility`` 拿它决定 position/velocity/face
#   三列哪几列进 eligible 分母;
# * ``action_ball_fixed_question_tape.TARGET_VALIDITY_BY_RECIPE`` 是磁带自述的那份;
# * ``materialize_action_ball_n1_fixed_tape_variants.VALIDITY`` 是磁带**生产者**的那份。
#
# 改一份就是"磁带以为自己是 analytic_full、运行时按 no_velocity 记分",而且不会报错,
# 只会让某个 reward 组的分母悄悄变了。现役零机器核对。
#
# 这三份**不能**改成单一真源 import:``hope_commands.py`` 在
# ``pin_action_ball_profile_contracts.SOLVER_SOURCES`` 七件套里逐字节钉着,materializer
# 那份则被 bundle build report 的 producer sha 钉着 —— 动任何一个字节都会让待跑的
# A211/C211 materialize 直接 BundleError。所以这里只读、不改,用 AST 核对。

TARGET_VALIDITY_BY_RECIPE = {
    "current_lm": (True, True, True),
    "analytic_full": (True, True, True),
    "analytic_no_velocity": (True, False, True),
    "teacher_pos_face_no_velocity": (True, False, True),
    "outcome_dense_only": (False, False, False),
}

TARGET_VALIDITY_HOLDERS = (
    (MDP / "hope_commands.py", "_ACTION_BALL_TARGET_VALIDITY_BY_RECIPE"),
    (MDP / "action_ball_fixed_question_tape.py", "TARGET_VALIDITY_BY_RECIPE"),
    (SCRIPTS / "materialize_action_ball_n1_fixed_tape_variants.py", "VALIDITY"),
)


@pytest.mark.parametrize(
    "path,name",
    TARGET_VALIDITY_HOLDERS,
    ids=[f"{p.name}:{n}" for p, n in TARGET_VALIDITY_HOLDERS],
)
def test_target_validity_by_recipe_is_identical_everywhere(path, name):
    assert _module_literal(path, name) == TARGET_VALIDITY_BY_RECIPE


def test_target_validity_holder_count_is_pinned():
    assert len(TARGET_VALIDITY_HOLDERS) == 3
    assert len({path for path, _name in TARGET_VALIDITY_HOLDERS}) == 3
