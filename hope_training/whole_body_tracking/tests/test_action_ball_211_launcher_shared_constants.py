"""A211 与 C211 两个发射器之间"必须相同"和"必须不同"的常量,机器来核对。

人话
====
`launch_action_ball_a211_four_arm_diagnostic.py` 和 `launch_action_ball_c211_diagnostic.py`
是两个各自独立 exec 的发射器(连 ``LaunchRefused`` 都是两个不同的类)。它们里面有
**两类**同名模块级常量,含义完全相反:

* **必须逐字相同**的那一类(路径、ABI 宽度、动作身份、终止并集、GPU 共址上限……)。
  改一边忘另一边 = 两条臂跑的不是同一个实验,而现役代码里**没有任何东西**会发现。
* **必须各不相同**的那一类(contract 名、normalizer 身份、lineage、收据文件名……)。
  把它们"顺手统一" = 两条臂身份合流,A 的 checkpoint 能装进 C,是更严重的错误。

本文件就是这两条规矩的唯一机器执行点。

它替代了什么
============
2026-08-06 删除了 `scripts/action_ball_211_launcher_shared.py`(577 行)。那个文件
2026-08-05 建立时的自述是"**本轮只建库,不改发射器**,接线是下一步",而接线从未发生:
全仓对它的引用数为 0(文件名、``BoundLauncherHelpers``、``BoundGpuAdmission``、
``bind_admission``、``WAIT_SCHEDULE_KWARGS`` 五个符号在仓库里都只出现在它自己那一份里)。
于是它成了同一批常量的**第三份**手抄副本——删除当天实测 49/49 仍与 A/C 逐字相同,
也就是说它还没漂,但没有任何机制拦得住它漂;等它漂了再有人按那份 docstring 去"接线",
就会把过期值静默灌进两个发射器。

删掉副本本身不够:删完就退回"A、C 两份,谁都不核对"。所以那 577 行里**唯一有价值的
知识**——哪些常量必须相同、哪些必须不同——搬到了这里,变成断言。相对删除前,这是净增
一道保护,不是减。

真要做单一真源 import 的话
==========================
那是另一件事,而且不能顺手做:两个发射器把自己**加载过的每个 helper 文件**都钉进
``RUNTIME_SOURCE_PATHS`` 并算进 launch claim。新引一个共享模块,就必须同批把它加进两边
的 tracked-source 钉子表,否则等于让一个未被 provenance 覆盖的文件参与决策——那是**放宽**
fail-closed 门。要做就连钉子表和 claim 测试一起做,别只搬常量。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
MUJOCO_NATIVE = Path(__file__).resolve().parents[1] / "mujoco_native"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


A = _load(
    "a211_shared_constants_probe",
    SCRIPTS / "launch_action_ball_a211_four_arm_diagnostic.py",
)
C = _load(
    "c211_shared_constants_probe",
    SCRIPTS / "launch_action_ball_c211_diagnostic.py",
)


# 逐字相同 —— 取自被删共享库在 2026-08-06 实测通过的 49 项(它自己的 SCRIPT_DIR 不算,
# 那是 ``Path(__file__)`` 派生量,三处取值本来就相同)。
MUST_MATCH = (
    "ACTION_BALL_COMMAND_SOURCE",
    "ACTION_BALL_SAMPLING_SOURCE",
    "ACTION_ID",
    "ACTION_UID",
    "ACTOR_WIDTH",
    "ADMISSION_FILE",
    "BASE_FILE",
    "BASE_SOURCE",
    "COLOCATED_STAGES",
    "COLOCATION_SPEC_KEY",
    "CRITIC_WIDTH",
    "EXACT_GROUP_FILE",
    "FOUR_GRID_BARRIER_FILE",
    "FOUR_GRID_BARRIER_SOURCE",
    "FOUR_GRID_FILE",
    "FOUR_GRID_SOURCE",
    "FRAME0_LIVE_RECEIPT_KIND",
    "HARD_TERMINATION_UNION",
    "KIT_LAUNCHER_SOURCE",
    "MAX_COLOCATED_PROCESSES_PER_GPU",
    "OLD_VALIDATOR_FILE",
    "OLD_VALIDATOR_SOURCE",
    "PHYSICAL_BALL_SEMANTICS",
    "PHYSICAL_FALL_PHASES",
    "PHYSICAL_FALL_REASONS",
    "PIN_KEYS",
    "POLICY_DT_S",
    "PRELONG_GATE_FILE",
    "PRELONG_GATE_SOURCE",
    "PRELONG_REWARD_RECIPE_SHA_ENV",
    "PRELONG_SEMANTICS_ENABLE_ENV",
    "PRELONG_SEMANTICS_FILE",
    "PRELONG_SEMANTICS_SOURCE",
    "PROHIBITED_HOLD_REFERENCE_TERMINATIONS",
    "RECIPE_SENTINEL_POLICY_SHA256",
    "REWARD_MATERIALIZATION_PROFILE",
    "REWARD_PPO_ECONOMY_ENABLE_ENV",
    "SCHEMA_VERSION",
    "STRICT_HARD_TERMINATION_UNION",
    "TASK_REVEAL_REACHED_COUNTER",
    "TASK_WAIT_FILE",
    "TASK_WAIT_SOURCE",
    "TASK_WAIT_STARTED_COUNTER",
    "TEACHER_ID",
    "TRAIN_SOURCE",
    "UPDATE_PROFILE_ENV",
    "UPDATE_PROFILE_JSON_PREFIX",
    "_DIRECT_FRAME0_ROBUST_MINIMUM_SLACKS",
)

# 必须各不相同 —— 这是"每条臂的身份",合流即错。取自被删共享库 docstring 的两段清单。
MUST_DIFFER = (
    "ACTOR_CONTRACT",
    "ACTOR_NORMALIZER_IDENTITY",
    "CLAIM_KIND",
    "CRITIC_CONTRACT",
    "CRITIC_NORMALIZER_IDENTITY",
    "EXPERIMENT_NAME",
    "GYM_TASK_ID",
    "LAUNCHER_SOURCE",
    "LINEAGE_KIND",
    "MATERIALIZATION_KIND",
    "ORACLE32_KIND",
    "POLICY_MATERIALIZATION_KIND",
    "POLICY_RECIPE_FILENAME",
    "RESULT_KIND",
    "RETAINED_TASK_PROFILE_PARENT_SOURCE",
    "REWARD_RECIPE_FILENAME",
    "SCALE4096_TERMINAL_ACCEPTANCE_KIND",
    "SPEC_KIND",
    "TARGET_SEMANTICS",
    "TASK_PROFILE_ID",
    "TASK_PROFILE_SOURCE",
    "TRAINABILITY_CONTRACT",
)


@pytest.mark.parametrize("name", MUST_MATCH)
def test_shared_launcher_constant_is_identical_on_both_arms(name):
    assert hasattr(A, name), "A211 launcher lost %s" % name
    assert hasattr(C, name), "C211 launcher lost %s" % name
    assert getattr(A, name) == getattr(C, name), (
        "%s drifted between the A211 and C211 launchers; it is a shared fact, "
        "not an arm identity" % name
    )


@pytest.mark.parametrize("name", MUST_DIFFER)
def test_arm_identity_constant_never_merges_across_arms(name):
    assert hasattr(A, name), "A211 launcher lost %s" % name
    assert hasattr(C, name), "C211 launcher lost %s" % name
    assert getattr(A, name) != getattr(C, name), (
        "%s became identical on both arms; A/C must keep separate lineage, "
        "normalizer and checkpoint identities" % name
    )


def test_the_deleted_third_copy_did_not_come_back():
    """任何人重建 scripts/action_ball_211_launcher_shared.py 都要先看见这条。"""

    assert not (SCRIPTS / "action_ball_211_launcher_shared.py").exists(), (
        "action_ball_211_launcher_shared.py 于 2026-08-06 因零调用点被删。要重建它,"
        "必须同批完成接线(两个发射器真的 import 它)并把它加进两边的 "
        "RUNTIME_SOURCE_PATHS,否则它只会再次变成一份没人核对的手抄副本。"
    )


def test_the_211_abi_widths_have_one_value_across_every_owner():
    """211/319 在四个模块里各写了一份,这里是唯一逐位核对的地方。

    ``mujoco_native/action_ball_211_abi.py`` 刻意 dependency-free、
    ``action_ball_c211_oracle_evidence.py`` 也自带一份,所以物理上必须手抄;
    手抄就必须有跨模块断言,否则宽度漂了要等到 tensor 构造才炸。
    """

    abi = _load("a211_abi_width_probe", MUJOCO_NATIVE / "action_ball_211_abi.py")
    evidence = _load(
        "c211_oracle_evidence_width_probe",
        SCRIPTS / "action_ball_c211_oracle_evidence.py",
    )
    owners = {
        "a211_launcher": (A.ACTOR_WIDTH, A.CRITIC_WIDTH),
        "c211_launcher": (C.ACTOR_WIDTH, C.CRITIC_WIDTH),
        "mujoco_native_abi": (abi.ACTOR_WIDTH, abi.CRITIC_WIDTH),
        "c211_oracle_evidence": (evidence.ACTOR_WIDTH, evidence.CRITIC_WIDTH),
    }
    assert set(owners.values()) == {(211, 319)}, owners
